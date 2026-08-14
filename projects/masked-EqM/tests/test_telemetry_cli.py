"""Tests for the telemetry CLI, and in particular for ``doctor``.

The tests are written the way the tool is meant to be used: every fixture is
built through the *real* producer API (:class:`telemetry.RunRecorder`), never by
hand-writing JSON.  A test that fabricates its own log format tests the test's
idea of the format, not the writer's -- which is exactly how the previous
telemetry drifted from its own analyzer.

Each defect is introduced the way it actually occurs in the field:

* an unsealed run  -- ``__enter__`` without ``__exit__`` (SIGKILL / node death);
* a seq gap        -- a whole line removed from a written stream (record loss);
* a torn tail      -- a truncated final line (a hard kill mid-write);
* confounded arms  -- two arms of one comparison group planned to different
  step counts, and two arms built on different git shas;
* a mutated spec   -- ``spec.json`` edited after minting, so it no longer hashes
  to its own ``run_uid``.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telemetry import RunRecorder, RunSpec  # noqa: E402
from telemetry import cli  # noqa: E402


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

def _spec(arm, seed=0, *, campaign="ctest", phase="p1", planned=10,
          git_sha="a" * 40, **params):
    return RunSpec(campaign=campaign, phase=phase, arm=arm, seed=seed,
                   git_sha=git_sha, planned_steps=planned, params=dict(params))


def _record(root, spec, *, job_id, steps=None, seal=True, attempt=0):
    """Run one execution end to end through the real recorder."""
    steps = spec.planned_steps if steps is None else steps
    recorder = RunRecorder(root, spec, planned_steps=spec.planned_steps,
                           job_id=job_id, attempt=attempt, mirror_stderr=False,
                           install_signal_handlers=False)
    if not seal:
        # Enter without exiting: this is what a SIGKILL leaves behind -- a START
        # and a prefix of the stream, with no terminal record and no way for the
        # process itself to ever write one.
        recorder.__enter__()
        for step in range(steps):
            recorder.progress(step, kind="grad", grad_norm=1.0 + step)
        return recorder
    with recorder as run:
        for step in range(steps):
            run.progress(step, kind="grad", grad_norm=1.0 + step, clipped=False)
        run.set_last_step(steps - 1)
    return recorder


def _stream_path(root, spec, exec_id_fragment=None):
    events = os.path.join(root, spec.slug(), "events")
    names = sorted(os.listdir(events))
    if exec_id_fragment:
        names = [n for n in names if exec_id_fragment in n]
    return os.path.join(events, names[0])


def _doctor(root, **kw):
    """Run doctor in --json mode; return (exit_code, parsed report)."""
    import io
    import contextlib

    argv = ["--root", str(root), "doctor", "--json"]
    for key, value in kw.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(argv)
    return code, json.loads(buffer.getvalue())


def _codes(report):
    return {f["code"] for f in report["findings"]}


def _run(argv):
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(argv)
    return code, buffer.getvalue()


# ---------------------------------------------------------------------------
# empty root
# ---------------------------------------------------------------------------

def test_empty_root_every_command_survives(tmp_path):
    """No subcommand may crash on a root that does not exist yet.

    This is the state of the system before the first run, and a diagnostic that
    dies here is a diagnostic nobody installs.
    """
    root = str(tmp_path / "never_created")
    for argv in (["--root", root, "ls"],
                 ["--root", root, "ls", "--json"],
                 ["--root", root, "doctor"],
                 ["--root", root, "gate"],
                 ["--root", root, "gate", "--json"]):
        code, out = _run(argv)
        assert code == 0, (argv, out)
        assert out.strip()

    code, report = _doctor(root)
    assert code == 0
    assert report["findings"] == []
    assert report["summary"] == {"ERROR": 0, "WARN": 0, "INFO": 0}


def test_empty_but_existing_root(tmp_path):
    root = str(tmp_path / "telemetry")
    os.makedirs(root)
    code, report = _doctor(root)
    assert code == 0 and report["runs"] == 0


# ---------------------------------------------------------------------------
# the clean case must be clean -- otherwise every positive below is meaningless
# ---------------------------------------------------------------------------

def test_healthy_campaign_is_clean_and_exits_zero(tmp_path):
    root = str(tmp_path / "telemetry")
    for seed in (0, 1):
        _record(root, _spec("baseline", seed), job_id=f"100{seed}")
        _record(root, _spec("treatment", seed), job_id=f"200{seed}")

    code, report = _doctor(root)
    assert code == 0, json.dumps(report["findings"], indent=2)
    assert report["summary"]["ERROR"] == 0
    assert report["runs"] == 4 and report["executions"] == 4

    # ... and the gate admits all four.
    code, out = _run(["--root", root, "gate", "--json"])
    gate = json.loads(out)
    assert len(gate["admitted"]) == 4
    assert gate["quarantined"] == []


# ---------------------------------------------------------------------------
# individual defects
# ---------------------------------------------------------------------------

def test_detects_run_with_start_and_no_end(tmp_path):
    root = str(tmp_path / "telemetry")
    _record(root, _spec("baseline"), job_id="1")
    _record(root, _spec("dead"), job_id="2", seal=False)

    code, report = _doctor(root)
    assert "unsealed" in _codes(report)
    unsealed = [f for f in report["findings"] if f["code"] == "unsealed"]
    assert len(unsealed) == 1
    # Freshly written, so it still looks alive: a WARNING, not an ERROR.
    assert unsealed[0]["severity"] == "WARN"
    assert "no END" in unsealed[0]["message"]
    assert unsealed[0]["action"]


def test_unsealed_and_stale_is_an_error(tmp_path):
    """A stream that stopped emitting is a wedged job, not a slow one."""
    root = str(tmp_path / "telemetry")
    _record(root, _spec("dead"), job_id="2", seal=False)

    # Threshold of zero hours: everything already written is 'stale'.
    code, report = _doctor(root, stale_hours=0)
    assert code == 1
    codes = _codes(report)
    assert "stale_heartbeat" in codes
    assert any(f["code"] == "unsealed" and f["severity"] == "ERROR"
               for f in report["findings"])


def test_detects_seq_gap(tmp_path):
    """A missing line is positive proof of record loss."""
    root = str(tmp_path / "telemetry")
    spec = _spec("gappy")
    _record(root, spec, job_id="3")
    path = _stream_path(root, spec)
    with open(path) as handle:
        lines = handle.readlines()
    del lines[3]                       # drop one PROGRESS record entirely
    with open(path, "w") as handle:
        handle.writelines(lines)

    code, report = _doctor(root)
    assert code == 1
    gaps = [f for f in report["findings"] if f["code"] == "seq_gap"]
    assert len(gaps) == 1 and gaps[0]["severity"] == "ERROR"
    assert "LOST" in gaps[0]["message"]

    # and the gate must refuse to aggregate it
    _, out = _run(["--root", root, "gate", "--json"])
    gate = json.loads(out)
    assert gate["admitted"] == []
    assert gate["quarantined"][0]["reason"] == "seq_gaps"


def test_detects_duplicate_seq(tmp_path):
    """Two writers on one stream: the records interleave."""
    root = str(tmp_path / "telemetry")
    spec = _spec("doubled")
    _record(root, spec, job_id="4")
    path = _stream_path(root, spec)
    with open(path) as handle:
        lines = handle.readlines()
    lines.insert(2, lines[2])
    with open(path, "w") as handle:
        handle.writelines(lines)

    code, report = _doctor(root)
    assert code == 1
    assert "duplicate_seq" in _codes(report)


def test_detects_torn_final_line(tmp_path):
    """A hard kill mid-write leaves half a JSON object on the last line."""
    root = str(tmp_path / "telemetry")
    spec = _spec("torn")
    _record(root, spec, job_id="5")
    path = _stream_path(root, spec)
    with open(path, "a") as handle:
        handle.write('{"v":1,"ts":"2026-08-14T00:00:00.000Z","run_uid"')

    code, report = _doctor(root)
    assert code == 1
    issues = [f for f in report["findings"] if f["code"] == "parse_issue"]
    assert issues and issues[0]["severity"] == "ERROR"
    assert "invalid JSON" in issues[0]["message"]

    # `validate` must point at the exact line.
    code, out = _run(["--root", root, "validate", path])
    assert code == 1
    assert "1 bad line" in out
    assert "line " in out


def test_detects_confounded_planned_steps(tmp_path):
    """Two arms planned to different horizons have no shared 'late' window."""
    root = str(tmp_path / "telemetry")
    _record(root, _spec("short_arm", planned=10), job_id="6")
    _record(root, _spec("long_arm", planned=20), job_id="7")

    code, report = _doctor(root)
    assert code == 1
    found = [f for f in report["findings"]
             if f["code"] == "confounded_planned_steps"]
    assert len(found) == 1
    assert found[0]["severity"] == "ERROR"
    assert "short_arm" in found[0]["message"] and "long_arm" in found[0]["message"]


def test_detects_confounded_git_sha(tmp_path):
    root = str(tmp_path / "telemetry")
    _record(root, _spec("armA", git_sha="a" * 40), job_id="8")
    _record(root, _spec("armB", git_sha="b" * 40), job_id="9")

    code, report = _doctor(root)
    assert code == 1
    assert "confounded_git_sha" in _codes(report)


def test_arms_in_different_phases_are_not_a_comparison_group(tmp_path):
    """Different planned_steps across PHASES is normal, not a confound."""
    root = str(tmp_path / "telemetry")
    _record(root, _spec("armA", phase="p1", planned=10), job_id="10")
    _record(root, _spec("armB", phase="p1", planned=10), job_id="11")
    _record(root, _spec("armA", phase="p2", planned=20), job_id="12")
    _record(root, _spec("armB", phase="p2", planned=20), job_id="13")

    code, report = _doctor(root)
    assert code == 0, json.dumps(report["findings"], indent=2)


def test_detects_mutated_spec(tmp_path):
    """A spec edited after minting no longer hashes to its own run_uid."""
    root = str(tmp_path / "telemetry")
    spec = _spec("tampered", fd_k=1)
    _record(root, spec, job_id="14")
    spec_path = os.path.join(root, spec.slug(), "spec.json")
    with open(spec_path) as handle:
        data = json.load(handle)
    data["params"]["fd_k"] = 4            # the classic hand-corrected results row
    with open(spec_path, "w") as handle:
        json.dump(data, handle)

    code, report = _doctor(root)
    assert code == 1
    found = [f for f in report["findings"] if f["code"] == "spec_hash_mismatch"]
    assert len(found) == 1 and found[0]["severity"] == "ERROR"


def test_detects_multiple_attempts_and_job_reuse(tmp_path):
    """A requeue reuses the job id; both facts must be statable."""
    root = str(tmp_path / "telemetry")
    spec = _spec("requeued")
    _record(root, spec, job_id="99", attempt=0, steps=4, seal=False)
    _record(root, spec, job_id="99", attempt=1)

    code, report = _doctor(root)
    codes = _codes(report)
    assert "multiple_attempts" in codes
    assert "job_id_reused" in codes
    attempts = [f for f in report["findings"] if f["code"] == "multiple_attempts"]
    assert attempts[0]["severity"] == "INFO"


def test_detects_job_id_collision_across_logical_runs(tmp_path):
    """One job id claimed by two different experiments is unresolvable."""
    root = str(tmp_path / "telemetry")
    _record(root, _spec("armA"), job_id="777")
    _record(root, _spec("armB"), job_id="777")

    code, report = _doctor(root)
    assert code == 1
    assert "job_id_collision" in _codes(report)


def test_detects_empty_run_directory(tmp_path):
    root = str(tmp_path / "telemetry")
    spec = _spec("neverran")
    os.makedirs(os.path.join(root, spec.slug(), "events"))
    with open(os.path.join(root, spec.slug(), "spec.json"), "w") as handle:
        json.dump(spec.to_dict(), handle)

    code, report = _doctor(root)
    assert "no_executions" in _codes(report)


def test_detects_unreadable_spec(tmp_path):
    root = str(tmp_path / "telemetry")
    spec = _spec("broken")
    _record(root, spec, job_id="15")
    with open(os.path.join(root, spec.slug(), "spec.json"), "w") as handle:
        handle.write("{not json")

    code, report = _doctor(root)
    assert "spec_unreadable" in _codes(report)


def test_reconstructed_streams_are_a_note_not_an_error(tmp_path):
    """A migrated legacy run has no START *by construction*.

    Reporting that as an ERROR once per execution buries the real findings under
    a restatement of "this predates the telemetry system" -- on the live root
    that was 993 of 1301 errors.  It must be one aggregate note instead.
    """
    from telemetry.emit import JsonlSink
    from telemetry.schema import EventType, make_record

    root = str(tmp_path / "telemetry")
    _record(root, _spec("native"), job_id="1")

    spec = _spec("legacy")
    run_dir = os.path.join(root, spec.slug())
    os.makedirs(os.path.join(run_dir, "events"))
    with open(os.path.join(run_dir, "spec.json"), "w") as handle:
        json.dump(spec.to_dict(), handle)
    exec_id = f"{spec.run_uid}:88:a0"
    sink = JsonlSink(os.path.join(run_dir, "events", f"{exec_id}.jsonl"))
    sink.write(make_record(
        run_uid=spec.run_uid, exec_id=exec_id, seq=0, event=EventType.NOTICE,
        payload={"code": "reconstruction_header", "level": "info",
                 "provenance": "reconstructed", "message": "migrated"}))
    sink.write(make_record(
        run_uid=spec.run_uid, exec_id=exec_id, seq=1, event=EventType.END,
        payload={"status": "completed", "last_step": 9, "planned_steps": 10,
                 "inferred": True, "provenance": "reconstructed"}))
    sink.close()

    code, report = _doctor(root)
    codes = _codes(report)
    assert "no_start" not in codes          # not an error for a declared migration
    assert "inferred_end" not in codes      # nor is its second-hand terminal record
    note = [f for f in report["findings"]
            if f["code"] == "reconstructed_executions"]
    assert len(note) == 1 and note[0]["severity"] == "INFO"
    assert "1 execution" in note[0]["message"]
    assert code == 0


def test_migrated_inventory_is_not_treated_as_a_designed_comparison(tmp_path):
    """One arm per historical job is an inventory, not an A/B group.

    The migrated ledger assigns a distinct arm to every legacy job, so a phase
    can hold >100 single-run 'arms' with heterogeneous shas.  Calling that an
    ERROR makes the headline error count meaningless, so it is downgraded and
    the caveat is stated in the finding itself.
    """
    root = str(tmp_path / "telemetry")
    for i in range(20):
        _record(root, _spec(f"legacy_job_{i:02d}", git_sha=f"{i:040x}"),
                job_id=str(1000 + i))

    code, report = _doctor(root)
    sha = [f for f in report["findings"] if f["code"] == "confounded_git_sha"]
    assert len(sha) == 1
    assert sha[0]["severity"] == "WARN"
    assert "migrated inventory" in sha[0]["message"]
    assert code == 0

    # ... but the same heterogeneity across a SMALL designed group is an error.
    root2 = str(tmp_path / "designed")
    _record(root2, _spec("armA", git_sha="a" * 40), job_id="1")
    _record(root2, _spec("armB", git_sha="b" * 40), job_id="2")
    code2, report2 = _doctor(root2)
    assert code2 == 1
    assert [f["severity"] for f in report2["findings"]
            if f["code"] == "confounded_git_sha"] == ["ERROR"]


def test_doctor_per_check_cap_keeps_the_report_readable(tmp_path):
    root = str(tmp_path / "telemetry")
    for i in range(12):
        _record(root, _spec("dead", seed=i), job_id=str(i), seal=False)

    def shown(text):
        return sum(1 for line in text.splitlines()
                   if line.strip().startswith("[unsealed]"))

    _, out = _run(["--root", root, "doctor", "--per-check", "3"])
    assert shown(out) == 3
    assert "9 further [unsealed] finding(s) withheld" in out
    # the count is still reported in full, only the instances are elided
    assert "unsealed" in out and "12" in out

    _, full = _run(["--root", root, "doctor", "--per-check", "0"])
    assert shown(full) == 12
    assert "withheld" not in full


# ---------------------------------------------------------------------------
# everything at once -- the realistic messy campaign
# ---------------------------------------------------------------------------

@pytest.fixture()
def broken_campaign(tmp_path):
    root = str(tmp_path / "telemetry")
    _record(root, _spec("baseline", 0), job_id="1")
    _record(root, _spec("dead", 0), job_id="2", seal=False)

    gap_spec = _spec("gappy", 0)
    _record(root, gap_spec, job_id="3")
    path = _stream_path(root, gap_spec)
    with open(path) as handle:
        lines = handle.readlines()
    del lines[3]
    with open(path, "w") as handle:
        handle.writelines(lines)

    torn_spec = _spec("torn", 0)
    _record(root, torn_spec, job_id="4")
    with open(_stream_path(root, torn_spec), "a") as handle:
        handle.write('{"v":1,"ts":"2026-08-14T00')

    _record(root, _spec("long_arm", 0, planned=20), job_id="5")
    return root


def test_doctor_finds_every_defect_and_exits_one(broken_campaign):
    code, report = _doctor(broken_campaign)
    assert code == 1
    codes = _codes(report)
    for expected in ("unsealed", "seq_gap", "parse_issue",
                     "confounded_planned_steps"):
        assert expected in codes, (expected, sorted(codes))
    assert report["summary"]["ERROR"] >= 3
    # every finding is actionable
    assert all(f["action"].strip() for f in report["findings"])
    # findings are grouped by severity, errors first
    order = [cli._SEVERITY_ORDER[f["severity"]] for f in report["findings"]]
    assert order == sorted(order)


def test_doctor_human_output_leads_with_the_summary(broken_campaign):
    code, out = _run(["--root", broken_campaign, "doctor"])
    assert code == 1
    lines = [line for line in out.splitlines() if line.strip()]
    assert "telemetry doctor" in lines[0]
    assert "error(s)" in lines[2]
    assert "ERRORS" in out
    assert "\033[" not in out           # no ANSI without --color


def test_gate_separates_admitted_from_quarantined(broken_campaign):
    code, out = _run(["--root", broken_campaign, "gate", "--json"])
    assert code == 0
    gate = json.loads(out)
    admitted = {a["arm"] for a in gate["admitted"]}
    quarantined = {q["arm"]: q["reason"] for q in gate["quarantined"]}
    assert "baseline" in admitted
    assert quarantined["dead"] == "no_end"
    assert quarantined["gappy"] == "seq_gaps"
    assert quarantined["torn"] == "parse_issues"

    code, out = _run(["--root", broken_campaign, "gate"])
    assert "ADMITTED" in out and "QUARANTINED" in out


# ---------------------------------------------------------------------------
# ls / show / tail / validate / dispatchers
# ---------------------------------------------------------------------------

def test_ls_is_stable_and_aligned(broken_campaign):
    code, out = _run(["--root", broken_campaign, "ls"])
    assert code == 0
    assert "RUN_UID" in out and "STEP/PLAN" in out
    assert "5 run(s)" in out
    _, again = _run(["--root", broken_campaign, "ls"])
    assert out == again                    # deterministic ordering

    code, out = _run(["--root", broken_campaign, "ls", "--json"])
    rows = json.loads(out)
    assert [r["arm"] for r in rows] == sorted(r["arm"] for r in rows)
    assert all(len(r["run_uid"]) == 17 for r in rows)

    code, out = _run(["--root", broken_campaign, "ls", "--arm", "baseline"])
    assert "1 run(s)" in out


def test_show_reports_detail_and_seq_integrity(broken_campaign):
    _, out = _run(["--root", broken_campaign, "ls", "--json"])
    rows = json.loads(out)
    gappy = next(r for r in rows if r["arm"] == "gappy")

    code, out = _run(["--root", broken_campaign, "show", gappy["run_uid"]])
    assert code == 0
    assert "spec hash OK" in out
    assert "GAPS" in out
    assert "first events" in out
    assert "START" in out

    # prefix resolution works
    code, out = _run(["--root", broken_campaign, "show", gappy["run_uid"][:9]])
    assert code == 0


def test_show_unknown_target_is_an_error_not_a_crash(broken_campaign):
    code, _out = _run(["--root", broken_campaign, "show", "rdeadbeefdeadbeef"])
    assert code == 2


def test_tail(broken_campaign):
    _, out = _run(["--root", broken_campaign, "ls", "--json"])
    uid = json.loads(out)[0]["run_uid"]
    code, out = _run(["--root", broken_campaign, "tail", uid, "-n", "3"])
    assert code == 0
    assert out.count("#") >= 3


def test_validate_whole_root(broken_campaign):
    code, out = _run(["--root", broken_campaign, "validate", broken_campaign])
    assert code == 1
    assert "bad line" in out
    assert "OK" in out                     # the healthy streams are reported too


def test_validate_missing_path(tmp_path):
    code, _ = _run(["--root", str(tmp_path), "validate", str(tmp_path / "nope")])
    assert code == 2


def test_dispatchers_degrade_gracefully(tmp_path, capsys):
    """The sibling modules may not exist yet; that is a message, not a crash."""
    for name in ("reconcile", "migrate", "contradictions", "ledger"):
        code = cli.main(["--root", str(tmp_path), name])
        # 0 = the sibling ran; 2 = it parsed our forwarded args and wants more;
        # 3 = it does not exist yet.  Never an uncaught exception or a SystemExit
        # escaping from inside a library call.
        assert code in (0, 1, 2, 3), name
        captured = capsys.readouterr()
        if code == 3:
            assert name in captured.err


def test_no_subcommand_prints_help(tmp_path):
    code, out = _run(["--root", str(tmp_path)])
    assert code == 0 and "doctor" in out
