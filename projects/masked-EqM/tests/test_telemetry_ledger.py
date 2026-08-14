"""Tests for the ledger fold, the sacct reconciler, and the legacy views.

Everything here runs against synthetic event logs written through the *real*
producer API (:func:`telemetry.emit.open_writer`) into a tmpdir.  No cluster, no
network, no ``sacct``: the scheduler is injected as a runner callable, which is
the only reason the reconciler's failure paths (timeouts, partial responses,
aged-out jobs) are testable at all.  A reconciler whose error handling can only
be exercised by breaking a real cluster is a reconciler whose error handling is
never exercised.

Each test that pins down one of the eight audited defects in
``scripts/cluster/reconcile_pipeline.py`` names it in its docstring, so the
mapping from audit finding to regression test stays greppable.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telemetry import RunSpec  # noqa: E402
from telemetry.emit import open_writer  # noqa: E402
from telemetry.ledger import build_ledger, fold  # noqa: E402
from telemetry.read import load_campaign  # noqa: E402
from telemetry.reconcile import (  # noqa: E402
    Disagreement, SacctUnavailable, apply_to_pipeline, build_sacct_command,
    classify, file_transaction, job_id_aliases, normalize_job_id, parse_elapsed,
    parse_exit_code, parse_sacct, query_sacct, reconcile,
    sbatch_exit_code_trustworthy,
)
from telemetry.schema import EventType, RunStatus  # noqa: E402
from telemetry import views  # noqa: E402


# -- fixtures ----------------------------------------------------------------

def make_spec(arm: str = "btm_vector", seed: int = 0, **params) -> RunSpec:
    return RunSpec(campaign="btm", phase="II-A", arm=arm, seed=seed,
                   git_sha="0f9658" + "0" * 34, planned_steps=100,
                   params={"lr": 1e-4, **params})


def write_run(root, spec, *, job_id, attempt=0, steps=(0, 50), end_status=None,
              artifacts=(), evals=(), start_extra=None, inferred=False):
    """Write one execution's stream through the real writer.

    Returns the exec_id.  ``end_status=None`` writes no END, which is how a
    "process vanished" stream is simulated -- the only faithful simulation,
    since the defining property of that failure is the absence of a record.
    """
    writer = open_writer(str(root), spec, job_id=job_id, attempt=attempt,
                         mirror_stderr=False)
    payload = {
        "campaign": spec.campaign, "phase": spec.phase, "arm": spec.arm,
        "seed": spec.seed, "git_sha": spec.git_sha,
        "planned_steps": spec.planned_steps, "config": dict(spec.params),
        "job_id": str(job_id), "attempt": attempt, "world_size": 4,
        "slurm": {"slurm_job_partition": "seas_gpu", "slurm_job_nodelist": "holygpu01"},
    }
    payload.update(start_extra or {})
    writer.emit(EventType.START, payload)
    for step in steps:
        writer.emit(EventType.PROGRESS, {"step": step, "kind": "grad",
                                         "grad_norm": 1.0 + step})
    for step, path in artifacts:
        writer.emit(EventType.ARTIFACT, {"step": step, "kind": "checkpoint",
                                         "path": path, "bytes": 1024})
    for record in evals:
        writer.emit(EventType.EVAL, dict(record))
    if end_status is not None:
        writer.emit(EventType.END, {
            "status": RunStatus(end_status).value,
            "last_step": max(steps) if steps else -1,
            "planned_steps": spec.planned_steps,
            "truncated": end_status != RunStatus.COMPLETED,
            "wall_seconds": 123.5, "inferred": inferred,
        })
    exec_id = writer.exec_id
    writer.close()
    return exec_id


def sacct_line(job_id, raw, state, exit_code="0:0", elapsed="00:10:00",
               start="2026-08-13T01:00:00", end="2026-08-13T01:10:00",
               nodelist="holygpu01", partition="seas_gpu"):
    return "|".join([str(job_id), str(raw), state, exit_code, elapsed,
                     start, end, nodelist, partition])


def runner_returning(text, *, calls=None):
    def runner(argv, timeout):
        if calls is not None:
            calls.append((list(argv), timeout))
        return text
    return runner


# -- ledger: determinism -----------------------------------------------------

def test_fold_is_deterministic_and_byte_identical(tmp_path):
    """Regenerating the ledger from the same logs must produce identical bytes.

    This is the property that lets the ledger be committed: a diff then means
    the cluster did something, never that the renderer is nondeterministic.
    """
    root = tmp_path / "telemetry"
    for arm, seed, job in (("btm_vector", 0, "101"), ("btm_scalar", 1, "102"),
                           ("btm_scalar", 0, "103")):
        spec = make_spec(arm, seed)
        write_run(root, spec, job_id=job, steps=(0, 50, 99),
                  end_status=RunStatus.COMPLETED,
                  artifacts=((50, "/scratch/ck50.pt"),),
                  evals=({"step": 99, "kind": "fid", "fid": 13.4},))

    first = build_ledger(str(root))
    second = build_ledger(str(root))
    assert first.dumps() == second.dumps()
    assert first.render_markdown() == second.render_markdown()

    # And identical across independent loads of the campaign object, i.e. the
    # fold itself is a pure function of the parsed log.
    assert fold(load_campaign(str(root))).dumps() == first.dumps()

    # No wall-clock leakage: the rendered bytes contain no "generated_at".
    assert "generated_at" not in first.dumps()

    # Total order: run ordering is by (campaign, phase, arm, seed, run_uid).
    arms = [(r.arm, str(r.seed)) for r in first.runs]
    assert arms == sorted(arms)


def _scrub_timestamps(value):
    """Blank every clock-derived field, leaving only the fold's structure.

    Needed because two logs written at different instants legitimately carry
    different ``ts`` values; what must not differ is the *shape and ordering*
    the fold produces from them.
    """
    keys = {"at", "started_at", "ended_at", "wall_seconds"}
    if isinstance(value, dict):
        return {k: ("<ts>" if k in keys else _scrub_timestamps(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_timestamps(v) for v in value]
    return value


def test_fold_is_order_independent_of_filesystem_listing(tmp_path):
    """Two roots with the same content in a different write order agree.

    Guards the specific way determinism is usually lost: ordering that falls
    through to ``os.listdir``.
    """
    order_a = [("a1", 0, "201"), ("a2", 1, "202")]
    order_b = list(reversed(order_a))
    renders = []
    for index, order in enumerate((order_a, order_b)):
        root = tmp_path / f"tel{index}"
        for arm, seed, job in order:
            write_run(root, make_spec(arm, seed), job_id=job,
                      end_status=RunStatus.COMPLETED)
        renders.append(_scrub_timestamps(build_ledger(str(root)).to_json()))
    assert renders[0] == renders[1]
    assert [r["arm"] for r in renders[0]["runs"]] == ["a1", "a2"]


def test_ledger_records_timeline_provenance_and_artifacts(tmp_path):
    root = tmp_path / "telemetry"
    spec = make_spec()
    write_run(root, spec, job_id="301", steps=(0, 50),
              end_status=RunStatus.TIMEOUT,
              artifacts=((50, "/scratch/ck50.pt"), (0, "/scratch/ck0.pt")))
    ledger = build_ledger(str(root))
    execution = ledger.runs[0].executions[0]
    assert execution.status == "timeout"
    assert execution.git_sha == spec.git_sha
    assert execution.world_size == 4
    assert execution.partition == "seas_gpu"
    assert execution.wall_seconds == pytest.approx(123.5)
    assert [t.event for t in execution.timeline] == ["START", "END"]
    assert [t.source for t in execution.timeline] == ["run", "run"]
    assert [a.step for a in execution.artifacts] == [0, 50]  # sorted by step
    # Stream path is relative to the root: absolute paths would be machine
    # specific and would break byte-identity across machines.
    assert not os.path.isabs(execution.stream)


def test_ledger_reports_record_loss_as_a_defect(tmp_path):
    """A seq gap is positive proof of record loss and must be surfaced."""
    root = tmp_path / "telemetry"
    spec = make_spec()
    exec_id = write_run(root, spec, job_id="401", end_status=RunStatus.COMPLETED)
    stream = root / spec.slug() / "events" / f"{exec_id}.jsonl"
    lines = stream.read_text().splitlines()
    stream.write_text("\n".join(lines[:1] + lines[2:]) + "\n")

    execution = build_ledger(str(root)).runs[0].executions[0]
    assert any("record loss" in d for d in execution.defects)
    assert "Stream defects" in build_ledger(str(root)).render_markdown()


def test_ledger_open_executions_are_the_reconcilers_worklist(tmp_path):
    root = tmp_path / "telemetry"
    write_run(root, make_spec("done", 0), job_id="501",
              end_status=RunStatus.COMPLETED)
    write_run(root, make_spec("open", 0), job_id="502", end_status=None)
    ledger = build_ledger(str(root))
    assert [e.job_id for e in ledger.open_executions()] == ["502"]


# -- sacct parsing -----------------------------------------------------------

def test_parse_exit_code_splits_code_and_signal():
    """Defect (f): "0:0" was stored opaquely; 9:0 and 0:9 are different events."""
    assert parse_exit_code("0:0") == parse_exit_code("0:0")
    assert parse_exit_code("0:0").code == 0 and parse_exit_code("0:0").signal == 0
    assert parse_exit_code("0:0").killed is False
    killed = parse_exit_code("0:9")
    assert (killed.code, killed.signal, killed.killed) == (0, 9, True)
    exited = parse_exit_code("9:0")
    assert (exited.code, exited.signal, exited.killed) == (9, 0, False)
    unknown = parse_exit_code("")
    assert unknown.code is None and unknown.signal is None


def test_parse_elapsed_handles_day_prefix():
    assert parse_elapsed("00:10:00") == 600
    assert parse_elapsed("1-02:00:00") == 93600
    assert parse_elapsed("garbage") is None


def test_array_task_id_normalization_and_lookup():
    """Defect (c): "34719790_2" never matched JobIDRaw's numeric id.

    The ledger's array form, the slugified exec_id form, and the raw numeric id
    must all resolve to the same sacct row.
    """
    assert normalize_job_id("34719790_2") == "34719790_2"
    # exec_id slugifies "_" to "-"; both must fold to one key.
    assert normalize_job_id("34719790-2") == "34719790_2"
    # Step rows fold onto their parent.
    assert normalize_job_id("34719790_2.batch") == "34719790_2"
    assert normalize_job_id("34719790.extern") == "34719790"
    assert job_id_aliases("34719790_2") == ["34719790_2", "34719790"]

    text = "\n".join([
        sacct_line("34719790_2", "34719999", "COMPLETED"),
        sacct_line("34719790_2.batch", "34719999.batch", "COMPLETED"),
    ])
    rows = parse_sacct(text)
    for key in ("34719790_2", "34719999", "34719790"):
        assert key in rows, f"array row unreachable via {key!r}"
    assert rows["34719790_2"].state == "completed"
    assert rows["34719790_2"].terminal


def test_sacct_state_words_and_terminal_set():
    """Defect (d): boot_fail/revoked/special_exit left jobs active forever."""
    text = "\n".join([
        sacct_line("1", "1", "BOOT_FAIL", exit_code="0:0"),
        sacct_line("2", "2", "REVOKED"),
        sacct_line("3", "3", "SPECIAL_EXIT"),
        sacct_line("4", "4", "CANCELLED by 501"),
        sacct_line("5", "5", "REQUEUED"),
        sacct_line("6", "6", "SUSPENDED"),
        sacct_line("7", "7", "OUT_OF_MEMORY"),
        sacct_line("8", "8", "DEADLINE"),
    ])
    rows = parse_sacct(text)
    for job in ("1", "2", "3", "4", "7", "8"):
        assert rows[job].terminal, f"{rows[job].state} should be terminal"
        assert rows[job].recognized
    # REQUEUED / SUSPENDED are recognized but genuinely NOT terminal: a job
    # about to run again must not be sealed.
    for job in ("5", "6"):
        assert rows[job].recognized and not rows[job].terminal
    assert rows["4"].state == "cancelled"       # "CANCELLED by 501"
    assert rows["1"].status is RunStatus.LOST   # boot fail: nothing observed it


def test_build_sacct_command_always_bounds_the_window():
    """Defect (b): with no --starttime, sacct only sees jobs from midnight."""
    command = build_sacct_command(["1", "2"], starttime="2026-08-12T00:00:00")
    assert "--starttime=2026-08-12T00:00:00" in command
    assert "--endtime=now" in command
    assert "JobID,JobIDRaw" in command  # defect (c)


def test_query_sacct_short_circuits_on_empty_active_runs():
    """Defect (h): ",".join([]) produced `sacct -X -j --format=...`."""
    calls = []
    rows, errors = query_sacct([], ssh=None, starttime="now-1days",
                               runner=runner_returning("", calls=calls))
    assert rows == {} and errors == []
    assert calls == [], "no command may be built for an empty id set"
    with pytest.raises(ValueError):
        build_sacct_command([], starttime="now-1days")


def test_query_sacct_survives_a_timeout_with_partial_progress():
    """Defect (e): a hung SSH ControlMaster must not hang the reconciler."""
    calls = []

    def flaky(argv, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise SacctUnavailable("command timed out after 1.0s")
        return sacct_line("2", "2", "COMPLETED")

    rows, errors = query_sacct(["1", "2"], ssh=None, starttime="now-1days",
                               timeout=1.0, runner=flaky, chunk_size=1)
    assert errors and "timed out" in errors[0]
    assert "2" in rows, "the surviving chunk's rows must be kept"
    assert all(t == 1.0 for t in calls), "every call must carry the timeout"


# -- reconcile: partial responses --------------------------------------------

def test_partial_sacct_response_still_reconciles_the_returned_jobs(tmp_path):
    """Defect (a): one aged-out job used to abort the entire reconcile."""
    root = tmp_path / "telemetry"
    write_run(root, make_spec("aged_out", 0), job_id="601", end_status=None)
    write_run(root, make_spec("present", 0), job_id="602", end_status=None)

    # sacct answers about 602 only -- 601 has aged out of retention.
    result = reconcile(str(root), ssh=None,
                       runner=runner_returning(sacct_line("602", "602", "COMPLETED")))

    by_job = {f.job_id: f for f in result.findings}
    assert by_job["601"].disagreement is Disagreement.MISSING_FROM_SACCT
    assert by_job["602"].disagreement is Disagreement.LOST_TELEMETRY
    assert result.appended == 2, "both jobs got an OBSERVED record"

    # The missing job's own record stands; nothing was invented for it.
    ledger = build_ledger(str(root))
    missing = [e for e in ledger.executions() if e.job_id == "601"][0]
    assert missing.status == "running"
    assert missing.observations[-1]["disagreement"] == "missing_from_sacct"


def test_reconcile_with_no_open_executions_is_a_no_op(tmp_path):
    """Defect (h), end to end: nothing open means no command and no error."""
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="701", end_status=RunStatus.COMPLETED)
    calls = []
    result = reconcile(str(root), ssh=None,
                       runner=runner_returning("", calls=calls))
    assert result.queried == [] and result.findings == [] and calls == []
    assert result.appended == 0


def test_reconcile_ignores_local_runs(tmp_path):
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id=None, end_status=None)  # -> "local"
    calls = []
    result = reconcile(str(root), ssh=None, runner=runner_returning("", calls=calls))
    assert result.queried == [] and calls == []


# -- reconcile: the disagreement classifications -----------------------------

def test_classification_lost_telemetry(tmp_path):
    """sacct COMPLETED + no END = the run never sealed itself."""
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="801", steps=(0, 50), end_status=None)
    result = reconcile(str(root), ssh=None,
                       runner=runner_returning(sacct_line("801", "801", "COMPLETED")))
    finding = result.findings[0]
    assert finding.disagreement is Disagreement.LOST_TELEMETRY
    assert finding.observed_status == RunStatus.LOST.value
    assert "no END record" in finding.detail


def test_classification_self_misreport(tmp_path):
    """END: completed + sacct TIMEOUT = the run misreported itself."""
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="802", steps=(0, 99),
              end_status=RunStatus.COMPLETED)
    # A sealed stream is not in the default work list, so ask explicitly -- which
    # is exactly how a human re-checks a suspicious "completed" run.
    result = reconcile(str(root), ssh=None, extra_job_ids=["802"],
                       runner=runner_returning(sacct_line("802", "802", "TIMEOUT")))
    findings = [f for f in result.findings if f.job_id == "802"]
    assert findings and findings[0].disagreement is Disagreement.SELF_MISREPORT
    assert result.appended == 1
    # ...and directly, via classify(), on the sealed stream itself:
    campaign = load_campaign(str(root))
    log = campaign.runs[0].executions[0]
    row = parse_sacct(sacct_line("802", "802", "TIMEOUT"))["802"]
    finding = classify(log, row)
    assert finding.disagreement is Disagreement.SELF_MISREPORT
    assert finding.self_status == "completed"
    assert finding.observed_status == "timeout"


def test_classification_untrustworthy_exit(tmp_path):
    """sacct COMPLETED on a `| tee` sbatch carries no information."""
    trusted, reason = sbatch_exit_code_trustworthy(
        "python train.py 2>&1 | tee $LOG\n")
    assert trusted is False and "PIPESTATUS" in reason
    assert sbatch_exit_code_trustworthy(
        "set -euo pipefail\npython train.py | tee $LOG\n")[0] is True
    assert sbatch_exit_code_trustworthy(
        "python train.py | tee $LOG\nexit ${PIPESTATUS[0]}\n")[0] is True
    assert sbatch_exit_code_trustworthy("python train.py > log 2>&1\n")[0] is True

    root = tmp_path / "telemetry"
    sbatch = tmp_path / "job.sbatch"
    sbatch.write_text("#!/bin/bash\npython train.py 2>&1 | tee $LOG\n")
    write_run(root, make_spec(), job_id="803", steps=(0, 50), end_status=None,
              start_extra={"sbatch_path": str(sbatch)})
    result = reconcile(str(root), ssh=None,
                       runner=runner_returning(sacct_line("803", "803", "COMPLETED")))
    finding = result.findings[0]
    assert finding.disagreement is Disagreement.UNTRUSTWORTHY_EXIT
    assert finding.trusted_exit_code is False
    # The untrustworthy COMPLETED must NOT be promoted to an observed status.
    assert finding.observed_status is None


def test_classification_agree_and_still_active(tmp_path):
    root = tmp_path / "telemetry"
    write_run(root, make_spec("a", 0), job_id="804", steps=(0, 99),
              end_status=RunStatus.TIMEOUT)
    campaign = load_campaign(str(root))
    log = campaign.runs[0].executions[0]
    assert classify(log, parse_sacct(sacct_line("804", "804", "TIMEOUT"))["804"]
                    ).disagreement is Disagreement.AGREE
    assert classify(log, parse_sacct(sacct_line("804", "804", "RUNNING"))["804"]
                    ).disagreement is Disagreement.STILL_ACTIVE


def test_classification_no_stream():
    """A job the scheduler knows but telemetry never saw."""
    row = parse_sacct(sacct_line("805", "805", "FAILED", exit_code="1:0"))["805"]
    finding = classify(None, row)
    assert finding.disagreement is Disagreement.NO_STREAM
    assert finding.row.exit_status.code == 1


def test_still_active_appends_nothing(tmp_path):
    """A running job is not news; the log must not grow on every poll."""
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="806", end_status=None)
    result = reconcile(str(root), ssh=None,
                       runner=runner_returning(sacct_line("806", "806", "RUNNING")))
    assert result.findings[0].disagreement is Disagreement.STILL_ACTIVE
    assert result.appended == 0


def test_reconcile_is_idempotent(tmp_path):
    """Re-running with unchanged facts appends nothing the second time."""
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="807", steps=(0, 50), end_status=None)
    runner = runner_returning(sacct_line("807", "807", "COMPLETED"))
    first = reconcile(str(root), ssh=None, runner=runner, ts="2026-08-14T00:00:00Z")
    second = reconcile(str(root), ssh=None, runner=runner, ts="2026-08-14T01:00:00Z")
    assert first.appended == 1
    assert second.appended == 0, "an unchanged finding must not be re-appended"
    ledger = build_ledger(str(root))
    assert len(ledger.executions()[0].observations) == 1


def test_reconcile_never_mutates_history(tmp_path):
    """Appending an OBSERVED must leave every prior byte untouched."""
    root = tmp_path / "telemetry"
    spec = make_spec()
    exec_id = write_run(root, spec, job_id="808", steps=(0, 50), end_status=None)
    stream = root / spec.slug() / "events" / f"{exec_id}.jsonl"
    before = stream.read_text()
    reconcile(str(root), ssh=None,
              runner=runner_returning(sacct_line("808", "808", "FAILED", "1:0")))
    after = stream.read_text()
    assert after.startswith(before), "history was rewritten, not appended to"
    appended = json.loads(after[len(before):].strip())
    assert appended["event"] == "OBSERVED"
    # seq continues the stream: a gap would be indistinguishable from loss.
    assert appended["seq"] == len(before.strip().splitlines())


def test_seal_lost_appends_an_inferred_end(tmp_path):
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="809", steps=(0, 50), end_status=None)
    result = reconcile(str(root), ssh=None, seal_lost=True,
                       runner=runner_returning(sacct_line("809", "809", "COMPLETED")))
    assert result.sealed
    execution = build_ledger(str(root)).executions()[0]
    assert execution.status == "lost" and execution.inferred_end is True


# -- transactional update (defect g) -----------------------------------------

def test_file_transaction_rereads_after_locking(tmp_path):
    """Defect (g): the old code computed from a read taken before the SSH.

    The invariant is that no observation used to compute the new value predates
    the lock.  Here a concurrent writer edits the file after the transaction's
    *caller* has already looked at it; the transaction must still see the edit.
    """
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps({"active_runs": [], "phase": "II-A"}))
    stale = json.loads(path.read_text())          # the caller's stale snapshot

    # A concurrent editor lands between the stale read and the transaction.
    path.write_text(json.dumps({"active_runs": [{"job_id": "999"}],
                                "phase": "II-A", "next_action": "added"}))

    with file_transaction(str(path)) as state:
        assert state["active_runs"] == [{"job_id": "999"}], \
            "transaction did not re-read after acquiring the lock"
        state["completed_runs"] = []

    final = json.loads(path.read_text())
    assert final["next_action"] == "added", "concurrent edit was clobbered"
    assert "next_action" not in stale


def test_apply_to_pipeline_moves_only_terminal_runs(tmp_path):
    root = tmp_path / "telemetry"
    write_run(root, make_spec("done", 0), job_id="901", steps=(0, 50), end_status=None)
    write_run(root, make_spec("live", 0), job_id="902", end_status=None)
    text = "\n".join([sacct_line("901", "901", "COMPLETED"),
                      sacct_line("902", "902", "RUNNING")])
    result = reconcile(str(root), ssh=None, runner=runner_returning(text))

    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(json.dumps({
        "phase": "II-A",
        "active_runs": [{"job_id": "901", "run_id": "a"},
                        {"job_id": "902", "run_id": "b"}],
    }))
    counts = apply_to_pipeline(str(pipeline), result)
    state = json.loads(pipeline.read_text())
    assert counts == {"moved": 1, "updated": 0, "kept": 1}
    assert [r["job_id"] for r in state["active_runs"]] == ["902"]
    moved = state["completed_runs"][0]
    assert moved["disagreement"] == "lost_telemetry"
    assert moved["exit_code"] == 0 and moved["exit_signal"] == 0
    assert state["phase"] == "II-A", "human-authored keys must survive"


# -- views -------------------------------------------------------------------

def test_pipeline_view_partitions_by_the_runs_own_record(tmp_path):
    root = tmp_path / "telemetry"
    write_run(root, make_spec("done", 0), job_id="1001", steps=(0, 99),
              end_status=RunStatus.COMPLETED,
              start_extra={"description": "d", "gate": "g",
                           "sbatch_path": "slurm/jobs/x.sbatch",
                           "expected_runtime": "~4h"})
    write_run(root, make_spec("open", 0), job_id="1002_3", end_status=None)

    view = views.pipeline_view(root=str(root))
    assert len(view["active_runs"]) == 1 and len(view["completed_runs"]) == 1
    active = view["active_runs"][0]
    # Array job id round-trips out of the slugified exec_id.
    assert active["job_id"] == "1002_3"
    done = view["completed_runs"][0]
    for field in views.PIPELINE_FIELDS:
        assert field in done, f"AGENTS.md requires {field}"
    assert done["partition"] == "seas_gpu" and done["gate"] == "g"
    assert done["status"] == "completed"


def test_pipeline_view_surfaces_the_disagreement(tmp_path):
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="1003", steps=(0, 50), end_status=None)
    reconcile(str(root), ssh=None,
              runner=runner_returning(sacct_line("1003", "1003", "TIMEOUT")))
    view = views.pipeline_view(root=str(root))
    entry = view["active_runs"][0]
    assert entry["status"] == "running", "the run's own record is not overwritten"
    assert entry["scheduler_state"] == "timeout"
    assert entry["disagreement"] == "lost_telemetry"


def test_results_rows_and_tsv_shape(tmp_path):
    root = tmp_path / "telemetry"
    write_run(root, make_spec("v10", 0), job_id="1004", steps=(0, 99),
              end_status=RunStatus.COMPLETED,
              artifacts=((50, "/scratch/ck50.pt"), (99, "/scratch/ck99.pt")),
              evals=({"step": 99, "kind": "fid", "fid": 13.4, "report": True,
                      "passed": True, "notes": "beats vanilla"},),
              start_extra={"gate": "Phase 1"})
    rows = views.results_rows(root=str(root))
    assert [r["metric_name"] for r in rows] == ["fid.fid"]
    row = rows[0]
    assert row["metric_value"] == pytest.approx(13.4)
    assert row["checkpoint"] == "/scratch/ck99.pt"
    assert row["pass"] == "true" and row["gate"] == "Phase 1"
    tsv = views.render_tsv(rows)
    header = tsv.splitlines()[0].split("\t")
    assert tuple(header) == views.RESULTS_COLUMNS
    assert len(tsv.splitlines()[1].split("\t")) == len(views.RESULTS_COLUMNS)


def test_views_check_mode_detects_drift_without_writing(tmp_path):
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="1005", steps=(0, 99),
              end_status=RunStatus.COMPLETED,
              evals=({"step": 99, "kind": "fid", "fid": 13.4},))
    results = tmp_path / "results_variants.tsv"
    pipeline = tmp_path / "pipeline.json"

    # Nothing committed yet -> drift, and --check writes nothing.
    assert views.check_text(str(results), views.render_tsv(
        views.results_rows(root=str(root)))).drifted
    assert not results.exists()

    views.write_atomic(str(results), views.render_tsv(views.results_rows(root=str(root))))
    assert not views.check_text(
        str(results), views.render_tsv(views.results_rows(root=str(root)))).drifted

    # A hand edit is drift.
    results.write_text(results.read_text().replace("13.4", "99.9"))
    drift = views.check_text(str(results),
                             views.render_tsv(views.results_rows(root=str(root))))
    assert drift.drifted and any("99.9" in line for line in drift.diff)

    # pipeline.json: only the two view-owned keys are compared.
    pipeline.write_text(json.dumps({"phase": "II-A", "next_action": "human prose"}))
    view = views.pipeline_view(root=str(root))
    assert views.check_pipeline(str(pipeline), view).drifted
    views.update_pipeline_file(str(pipeline), view)
    assert not views.check_pipeline(str(pipeline), view).drifted
    state = json.loads(pipeline.read_text())
    assert state["next_action"] == "human prose"


def test_views_cli_check_returns_nonzero_on_drift(tmp_path, capsys):
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="1006", end_status=RunStatus.COMPLETED)
    results = tmp_path / "results_variants.tsv"
    results.write_text("garbage\n")
    code = views.main(["--root", str(root), "--results", str(results), "--check"])
    assert code == 1
    assert "DRIFT" in capsys.readouterr().out
    assert results.read_text() == "garbage\n", "--check must not write"


def test_ledger_cli_check_round_trips(tmp_path, capsys):
    root = tmp_path / "telemetry"
    write_run(root, make_spec(), job_id="1007", end_status=RunStatus.COMPLETED)
    from telemetry import ledger as ledger_module
    out = tmp_path / "LEDGER.md"
    assert ledger_module.main(["--root", str(root), "--out", str(out)]) == 0
    assert ledger_module.main(["--root", str(root), "--out", str(out),
                               "--check"]) == 0
    out.write_text(out.read_text() + "hand edit\n")
    assert ledger_module.main(["--root", str(root), "--out", str(out),
                               "--check"]) == 1
    assert "DRIFT" in capsys.readouterr().out
