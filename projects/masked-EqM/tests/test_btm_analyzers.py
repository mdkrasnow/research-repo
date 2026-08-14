"""Regression tests for the LEGACY BTM analysis path.

Each test encodes one identity/join/windowing defect that was verified by
execution against synthetic inputs, and each FAILS on the pre-fix code:

  analyze_image.py
    T2   early/mid/late computed run-relatively, so a preempted run's early
         training was pooled as "late" with a complete run's real late window
    T3   `btm_scalar_fd_directional` matches inside
         `btm_scalar_fd_directional4`, collapsing K=1 and K=4 into one arm
         whose "std over seeds" is really a K=1-vs-K=4 arm difference; the
         matching loop also had no `break` and tested the FULL PATH
    -    `unclipped_grad_norm` is never logged, so the "unclipped max" column
         silently duplicated the grad-norm column
    -    window edges inclusive on both sides double-counted interior-edge
         records, and `n` pooled the grad record with the checklist record

  analyze_toy.py
    T12  the group key drops `tc`, so tc_sweep rows silently overwrite
    T13  crashed seeds (NaN mass_mae) vanish from both the seed count and the
         stability denominator; `n` counted records, not distinct seeds
    T14  a missing negative control DELETED its gate condition and left PASS
    -    a bare json.loads killed the whole gate on one partial line

  scripts/cluster/reconcile_pipeline.py
    -    `--format=JobIDRaw` cannot match an array-task ledger id (`123_4`),
         so the lookup missed and the script aborted reconciliation entirely
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJECT))
BTM = os.path.join(PROJECT, "experiments", "btm")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze_image = _load("btm_analyze_image",
                      os.path.join(BTM, "analyze_image.py"))
analyze_toy = _load("btm_analyze_toy", os.path.join(BTM, "analyze_toy.py"))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def grad_record(step, grad_norm=1.0, clipped=False, **kw):
    """The record train.py writes BEFORE opt.step()."""
    rec = {"step": step, "grad_norm": grad_norm, "clipped": clipped,
           "learning_rate": 1e-4, "loss": 0.5}
    rec.update(kw)
    return rec


def probe_record(step, **kw):
    """The checklist record train.py writes AFTER opt.step(), same step."""
    rec = {"step": step, "delta_theta_norm": 0.01, "probe_delta_L": -1e-4,
           "P_t": 1e-4, "eta_func": 1e-2}
    rec.update(kw)
    return rec


def write_run(tmp_path, tag, records):
    d = tmp_path / tag / "000-EqM-B-2"
    d.mkdir(parents=True)
    p = d / "gradient_metrics.jsonl"
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return p


def write_manifest(tmp_path, entries):
    p = tmp_path / "manifest.jsonl"
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    return p


def run_image_analyzer(tmp_path, roots, manifest, extra=()):
    cmd = [sys.executable, os.path.join(BTM, "analyze_image.py")] + \
        [str(r) for r in roots] + ["--manifest", str(manifest)] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True)


# --------------------------------------------------------------------------
# analyze_image.py — BUG T2: run-relative windows
# --------------------------------------------------------------------------

def test_T2_truncated_run_is_not_pooled_into_the_late_window(tmp_path):
    """A run preempted at 3000/20000 must not contribute a "late" row.

    Pre-fix, `windows()` split the run's OWN range, so steps 2050-3000 were
    labelled "late" and averaged beside a complete run's 13350-20000 -- exactly
    inverting the campaign's late-degradation hypothesis for truncated arms.
    """
    # complete run: clip rate rises to 90% late (the failure signature)
    complete = []
    for s in range(50, 20001, 50):
        complete.append(grad_record(s, clipped=(s > 13400 and s % 100 == 0)))
    # truncated run of the SAME arm, killed at 3000, healthy throughout
    truncated = [grad_record(s, clipped=False) for s in range(50, 3001, 50)]

    write_run(tmp_path, "btm_IIA_btm_scalar_exact_s0", complete)
    write_run(tmp_path, "btm_IIA_btm_scalar_exact_s1", truncated)
    manifest = write_manifest(tmp_path, [
        {"run_tag": "btm_IIA_btm_scalar_exact_s0", "arm": "btm_scalar_exact",
         "fd_k": 1, "max_steps": 20000},
        {"run_tag": "btm_IIA_btm_scalar_exact_s1", "arm": "btm_scalar_exact",
         "fd_k": 1, "max_steps": 20000},
    ])
    res = run_image_analyzer(tmp_path, [tmp_path], manifest)
    assert res.returncode == 0, res.stderr
    out = res.stdout

    # The aggregate must be over the ONE complete seed, not two.
    key = out.split("Key mechanistic comparison")[1].split("Incomplete runs")[0]
    late_rows = [l for l in key.splitlines()
                 if l.startswith("|") and "scalar exact" in l]
    assert len(late_rows) == 1
    assert late_rows[0].split("|")[2].strip() == "1", late_rows[0]

    # Incompleteness must be VISIBLE.
    inc = out.split("Incomplete runs")[1]
    assert "btm_IIA_btm_scalar_exact_s1" in inc
    assert "3000" in inc and "20000" in inc
    assert "btm_IIA_btm_scalar_exact_s0" not in inc


def test_T2_windows_are_absolute_thirds_of_the_planned_budget(tmp_path):
    rows = [grad_record(s) for s in range(50, 3001, 50)]
    got = analyze_image.windows(rows, 20000)
    # A 3000-step run touches ONLY the early window of a 20000-step budget.
    assert [name for name, _, _ in got] == ["early"]


# --------------------------------------------------------------------------
# analyze_image.py — BUG T3: K=1 / K=4 collapse and path-based matching
# --------------------------------------------------------------------------

def test_T3_k1_and_k4_are_distinct_arms(tmp_path):
    """`btm_scalar_fd_directional` is a substring of `..._directional4`."""
    rows = [grad_record(s) for s in range(50, 20001, 50)]
    write_run(tmp_path, "btm_IIA_btm_scalar_fd_directional_s0", rows)
    write_run(tmp_path, "btm_IIA_btm_scalar_fd_directional4_s0", rows)
    manifest = write_manifest(tmp_path, [
        {"run_tag": "btm_IIA_btm_scalar_fd_directional_s0",
         "arm": "btm_scalar_fd_directional", "fd_k": 1, "max_steps": 20000},
        {"run_tag": "btm_IIA_btm_scalar_fd_directional4_s0",
         "arm": "btm_scalar_fd_directional4", "fd_k": 4, "max_steps": 20000},
    ])
    res = run_image_analyzer(tmp_path, [tmp_path], manifest)
    assert res.returncode == 0, res.stderr
    key = res.stdout.split("Key mechanistic comparison")[1]
    labels = {l.split("|")[1].strip() for l in key.splitlines()
              if l.startswith("|") and "FD directional" in l}
    assert labels == {"D  FD directional K=1", "D  FD directional K=4"}, labels
    # and neither bucket may claim two seeds
    for line in key.splitlines():
        if line.startswith("|") and "FD directional" in line:
            assert line.split("|")[2].strip() == "1", line


def test_T3_arm_resolution_ignores_the_containing_path(tmp_path):
    """Pointing the analyzer at a root named after an arm must not relabel."""
    # a root directory whose NAME contains another arm's name
    root = tmp_path / "btm_vector_sweep"
    root.mkdir()
    rows = [grad_record(s) for s in range(50, 20001, 50)]
    write_run(root, "btm_IIA_btm_scalar_exact_s0", rows)
    manifest = write_manifest(tmp_path, [])  # deliberately empty -> tag fallback
    res = run_image_analyzer(tmp_path, [root], manifest,
                             extra=["--planned-steps", "20000"])
    assert res.returncode == 0, res.stderr
    assert "G  scalar exact" in res.stdout
    assert "V  vector" not in res.stdout


def test_T3_unresolvable_arm_is_a_hard_error(tmp_path):
    rows = [grad_record(s) for s in range(50, 20001, 50)]
    write_run(tmp_path, "some_unlabelled_run_s0", rows)
    manifest = write_manifest(tmp_path, [])
    res = run_image_analyzer(tmp_path, [tmp_path], manifest,
                             extra=["--planned-steps", "20000"])
    assert res.returncode != 0
    assert "could not resolve an arm" in res.stderr
    assert "some_unlabelled_run_s0" in res.stderr


# --------------------------------------------------------------------------
# analyze_image.py — absent metrics must not be silently substituted
# --------------------------------------------------------------------------

def test_absent_unclipped_and_param_norm_are_reported_absent(tmp_path):
    rows = []
    for s in range(50, 20001, 50):
        rows.append(grad_record(s, grad_norm=7.0))
        rows.append(probe_record(s))
    write_run(tmp_path, "btm_IIA_btm_vector_s0", rows)
    manifest = write_manifest(tmp_path, [
        {"run_tag": "btm_IIA_btm_vector_s0", "arm": "btm_vector", "fd_k": 1,
         "max_steps": 20000}])
    res = run_image_analyzer(tmp_path, [tmp_path], manifest)
    assert res.returncode == 0, res.stderr
    table_c = res.stdout.split("Table D")[0]
    body = [l for l in table_c.splitlines()
            if l.startswith("|") and "vector" in l]
    assert body
    for line in body:
        # unclipped column must say "not logged", never repeat 7 (the grad norm)
        assert "not logged" in line, line
        assert "param_norm absent" in line, line


def test_nonfinite_count_is_actually_computed(tmp_path):
    rows = [grad_record(s) for s in range(50, 20001, 50)]
    rows.append(grad_record(19950, grad_norm=float("inf")))
    write_run(tmp_path, "btm_IIA_btm_vector_s0", rows)
    manifest = write_manifest(tmp_path, [
        {"run_tag": "btm_IIA_btm_vector_s0", "arm": "btm_vector", "fd_k": 1,
         "max_steps": 20000}])
    res = run_image_analyzer(tmp_path, [tmp_path], manifest)
    assert res.returncode == 0, res.stderr
    assert "nonfinite" in res.stdout.split("Table D")[0]
    late = [l for l in res.stdout.split("Table D")[0].splitlines()
            if l.startswith("|") and "late" in l]
    assert late and any(c.strip() == "1" for c in late[0].split("|")), late


# --------------------------------------------------------------------------
# analyze_image.py — half-open windows and separated record populations
# --------------------------------------------------------------------------

def test_interior_window_edge_record_is_counted_once():
    """61 records, one landing exactly on an interior edge, must total 61."""
    rows = [grad_record(s) for s in range(0, 3001, 50)]  # 61 records
    got = analyze_image.windows(rows, 3000)
    total = sum(len(sel) for _, _, sel in got)
    assert total == len(rows) == 61
    # step 1000 and step 2000 are the interior edges
    assert sum(1 for _, _, sel in got for r in sel if r["step"] == 1000) == 1
    assert sum(1 for _, _, sel in got for r in sel if r["step"] == 2000) == 1


def test_grad_and_probe_records_are_counted_separately(tmp_path):
    """20 optimizer steps => n_grad=20, n_probe=20, never a pooled 40."""
    rows = []
    for s in range(1000, 20001, 1000):   # 20 steps, all in one budget
        rows.append(grad_record(s))
        rows.append(probe_record(s))
    write_run(tmp_path, "btm_IIA_btm_vector_s0", rows)
    manifest = write_manifest(tmp_path, [
        {"run_tag": "btm_IIA_btm_vector_s0", "arm": "btm_vector", "fd_k": 1,
         "max_steps": 20000}])
    out = tmp_path / "recs.json"
    res = run_image_analyzer(tmp_path, [tmp_path], manifest,
                             extra=["--out", str(out)])
    assert res.returncode == 0, res.stderr
    recs = json.loads(out.read_text())["complete"]
    assert recs
    assert sum(r["n_grad"] for r in recs) == 20
    assert sum(r["n_probe"] for r in recs) == 20
    for r in recs:
        assert "n" not in r  # the pooled count is gone


def test_update_over_param_never_mixes_populations(tmp_path):
    """param_norm logged on the GRAD record only must not be divided into a
    delta_theta_norm median taken over the PROBE record population."""
    rows = []
    for s in range(1000, 20001, 1000):
        rows.append(grad_record(s))
        rows.append(probe_record(s, param_norm=100.0))
    write_run(tmp_path, "btm_IIA_btm_vector_s0", rows)
    manifest = write_manifest(tmp_path, [
        {"run_tag": "btm_IIA_btm_vector_s0", "arm": "btm_vector", "fd_k": 1,
         "max_steps": 20000}])
    out = tmp_path / "recs.json"
    res = run_image_analyzer(tmp_path, [tmp_path], manifest,
                             extra=["--out", str(out)])
    assert res.returncode == 0, res.stderr
    for r in json.loads(out.read_text())["complete"]:
        assert r["param_norm_available"]
        assert r["update_over_param"] == pytest.approx(0.01 / 100.0, rel=1e-6)


# --------------------------------------------------------------------------
# analyze_toy.py — BUG T12: tc dropped from the group key
# --------------------------------------------------------------------------

def toy_row(arm, seed, mae, tc=0.9, K=1, eps=1e-3, geom="ring",
            stage="tc_sweep", stable=True, **kw):
    rec = {"stage": stage, "mass_mae": mae, "stable": stable,
           "unresolved_frac": 0.0, "R_overall_median_rel": 0.0,
           "config": {"arm": arm, "K": K, "eps_fd": eps, "tc": tc,
                      "geometry": geom, "seed": seed}}
    rec.update(kw)
    return rec


def test_T12_tc_sweep_rows_do_not_overwrite_each_other():
    rows = [toy_row("btm_scalar_fd_directional", s, 0.001, tc=0.5)
            for s in range(3)]
    rows += [toy_row("btm_scalar_fd_directional", s, 0.040, tc=0.9)
             for s in range(3)]
    table, summary = analyze_toy.table_a(rows, stage="tc_sweep")
    assert len(summary) == 2, summary
    assert {v["tc"] for v in summary.values()} == {0.5, 0.9}
    assert "| tc |" in table
    # both distinct MAEs must be visible in the rendered table
    assert "0.0010" in table and "0.0400" in table


# --------------------------------------------------------------------------
# analyze_toy.py — BUG T13: crashed seeds vanish; n counts records
# --------------------------------------------------------------------------

def crashed_row(arm, seed, **kw):
    """Exactly what toy_parallel.py writes for a failed seed."""
    return {"stage": "main", "mass_mae": float("nan"), "stable": False,
            "error": "RuntimeError: boom",
            "config": {"arm": arm, "K": 1, "eps_fd": 1e-3, "tc": 0.9,
                       "geometry": "ring", "seed": seed}}


def test_T13_crashed_seeds_are_counted_and_break_the_stability_gate():
    rows = [toy_row("btm_scalar_fd_directional", s, 0.002, stage="main")
            for s in range(10)]
    rows += [crashed_row("btm_scalar_fd_directional", s) for s in (10, 11)]
    rows += [toy_row("btm_vector", s, 0.002, stage="main") for s in range(10)]
    _, summary = analyze_toy.table_a(rows, stage="main")
    d = [v for k, v in summary.items()
         if k.startswith("btm_scalar_fd_directional|")][0]
    assert d["n"] == 12, d            # pre-fix: 10
    assert d["n_crashed"] == 2, d     # pre-fix: absent
    assert d["n_stable"] == 10, d
    g = analyze_toy.gate(summary, rows)
    c = g["checks"]["2_stable_across_seeds"]
    assert c["pass"] is False, c      # pre-fix: True ("10/10 stable")
    assert "2 crashed" in c["detail"]
    assert g["verdict"] != "PASS"


def test_T13_duplicate_records_do_not_inflate_the_seed_count():
    """Passing the same path twice must not turn 5 seeds into 10."""
    rows = [toy_row("btm_scalar_fd_directional", s, 0.002, stage="main")
            for s in range(5)] * 2
    _, summary = analyze_toy.table_a(rows, stage="main")
    d = list(summary.values())[0]
    assert d["n"] == 5, d             # pre-fix: 10


# --------------------------------------------------------------------------
# analyze_toy.py — BUG T14: missing control silently removes a gate condition
# --------------------------------------------------------------------------

def _passing_rows():
    rows = [toy_row("btm_vector", s, 0.001, stage="main") for s in range(12)]
    rows += [toy_row("btm_scalar_fd_directional", s, 0.002, stage="main")
             for s in range(12)]
    rows += [{"stage": "fd_calibration", "rel_rmse": 0.01,
              "nonfinite_frac": 0.0}]
    return rows


def test_T14_missing_negative_control_makes_the_verdict_inconclusive():
    rows = _passing_rows()   # no eqm_legacy_* records at all
    _, summary = analyze_toy.table_a(rows, stage="main")
    g = analyze_toy.gate(summary, rows)
    assert "4_beats_legacy_eqm" in g["checks"]      # pre-fix: absent
    assert g["checks"]["4_beats_legacy_eqm"]["measured"] is False
    assert g["verdict"] == "INCONCLUSIVE"           # pre-fix: "PASS"
    assert "4_beats_legacy_eqm" in g["unmeasured"]


def test_T14_present_negative_control_allows_a_pass():
    rows = _passing_rows()
    rows += [toy_row("eqm_legacy_vector", s, 0.50, stage="main")
             for s in range(12)]
    _, summary = analyze_toy.table_a(rows, stage="main")
    g = analyze_toy.gate(summary, rows)
    assert g["checks"]["4_beats_legacy_eqm"]["measured"] is True
    assert g["verdict"] == "PASS", g


def test_T14_hardcoded_check_is_labelled_not_measured():
    rows = _passing_rows()
    rows += [toy_row("eqm_legacy_vector", s, 0.50, stage="main")
             for s in range(12)]
    _, summary = analyze_toy.table_a(rows, stage="main")
    g = analyze_toy.gate(summary, rows)
    c = g["checks"]["5_no_mixed_derivative"]
    assert c["measured"] is False                   # pre-fix: reported as PASS
    assert "NOT MEASURED" in c["detail"]
    assert "5_no_mixed_derivative" in g["not_measured_here"]


def test_T14_missing_V_or_D_is_inconclusive_not_incomplete():
    rows = [toy_row("btm_vector", s, 0.001, stage="main") for s in range(3)]
    _, summary = analyze_toy.table_a(rows, stage="main")
    assert analyze_toy.gate(summary, rows)["verdict"] == "INCONCLUSIVE"


# --------------------------------------------------------------------------
# analyze_toy.py — one partial line must not kill the gate
# --------------------------------------------------------------------------

def test_partial_jsonl_line_is_counted_not_fatal(tmp_path):
    p = tmp_path / "records.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps(toy_row("btm_vector", 0, 0.001, stage="main")) + "\n")
        f.write('{"stage": "main", "mass_ma')  # in-flight partial flush
    rows, bad = analyze_toy.load([str(p)])     # pre-fix: JSONDecodeError
    assert len(rows) == 1
    assert len(bad) == 1
    assert str(p) in bad[0]


# --------------------------------------------------------------------------
# reconcile_pipeline.py — array-task ids and single-job robustness
# --------------------------------------------------------------------------

RECONCILE = os.path.join(REPO, "scripts", "cluster", "reconcile_pipeline.py")


def fake_ssh(tmp_path, stdout, name="ssh.sh", raw_stdout=None):
    """A stand-in for scripts/cluster/ssh.sh that emulates sacct.

    Crucially FORMAT-AWARE: real sacct prints the ledger's `<job>_<task>` form
    only for `JobID`.  Under `JobIDRaw` an array task comes back as its
    EXPANDED numeric id (39090540_3 -> 39090543), which is precisely why the
    pre-fix lookup missed and the whole reconciliation aborted.  `raw_stdout`
    is what the stub returns when the caller asks for JobIDRaw.
    """
    p = tmp_path / name
    p.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  *JobIDRaw*) cat <<'EOF'\n" + (raw_stdout if raw_stdout is not None
                                         else stdout) + "\nEOF\n"
        "  ;;\n"
        "  *) cat <<'EOF'\n" + stdout + "\nEOF\n"
        "  ;;\n"
        "esac\n")
    p.chmod(0o755)
    return p


def write_pipeline(tmp_path, active):
    p = tmp_path / "pipeline.json"
    p.write_text(json.dumps({"phase": "2", "active_runs": active,
                             "completed_runs": []}, indent=2))
    return p


def run_reconcile(pipeline, ssh, extra=()):
    return subprocess.run(
        [sys.executable, RECONCILE, str(pipeline), "--ssh", str(ssh)]
        + list(extra), capture_output=True, text=True)


def test_array_task_ledger_id_is_matched(tmp_path):
    """`JobIDRaw` prints 39090543 for ledger id 39090540_3 -> lookup missed."""
    pipeline = write_pipeline(tmp_path, [
        {"run_id": "arr3", "job_id": "39090540_3", "status": "running"},
        {"run_id": "plain", "job_id": "39090739", "status": "running"},
    ])
    ssh = fake_ssh(
        tmp_path,
        # what sacct --format=JobID,... returns
        "39090540_3|COMPLETED|0:0|10:00:00|2026-08-14T01:00:00\n"
        "39090739|RUNNING|0:0|02:00:00|Unknown",
        # what sacct --format=JobIDRaw,... returns: the array task is expanded
        raw_stdout="39090543|COMPLETED|0:0|10:00:00|2026-08-14T01:00:00\n"
                   "39090739|RUNNING|0:0|02:00:00|Unknown")
    res = run_reconcile(pipeline, ssh)
    assert res.returncode == 0, res.stderr     # pre-fix: RuntimeError
    state = json.loads(pipeline.read_text())
    assert [r["run_id"] for r in state["completed_runs"]] == ["arr3"]
    assert [r["run_id"] for r in state["active_runs"]] == ["plain"]


def test_one_missing_job_does_not_abort_the_whole_reconciliation(tmp_path):
    pipeline = write_pipeline(tmp_path, [
        {"run_id": "gone", "job_id": "11111111", "status": "running"},
        {"run_id": "done", "job_id": "22222222", "status": "running"},
    ])
    ssh = fake_ssh(tmp_path,
                   "22222222|COMPLETED|0:0|01:00:00|2026-08-14T01:00:00")
    res = run_reconcile(pipeline, ssh)
    assert res.returncode == 0, res.stderr     # pre-fix: RuntimeError
    state = json.loads(pipeline.read_text())
    assert [r["run_id"] for r in state["completed_runs"]] == ["done"]
    (still,) = state["active_runs"]
    assert still["run_id"] == "gone" and still["status"] == "unknown"
    assert "unknown=1" in res.stdout


@pytest.mark.parametrize("state_name", ["BOOT_FAIL", "REVOKED", "SPECIAL_EXIT"])
def test_missing_terminal_states_are_terminal(tmp_path, state_name):
    pipeline = write_pipeline(tmp_path, [
        {"run_id": "x", "job_id": "33333333", "status": "running"}])
    ssh = fake_ssh(tmp_path,
                   f"33333333|{state_name}|1:0|00:03:00|2026-08-14T01:00:00")
    res = run_reconcile(pipeline, ssh)
    assert res.returncode == 0, res.stderr
    state = json.loads(pipeline.read_text())
    assert len(state["completed_runs"]) == 1   # pre-fix: stayed active
    assert state["active_runs"] == []


def test_empty_active_runs_does_not_query_every_job(tmp_path):
    pipeline = write_pipeline(tmp_path, [])
    # an ssh stub that FAILS if invoked at all
    p = tmp_path / "ssh.sh"
    p.write_text("#!/bin/sh\nexit 42\n")
    p.chmod(0o755)
    res = run_reconcile(pipeline, p)
    assert res.returncode == 0, res.stderr     # pre-fix: `sacct -j ` on all jobs
    assert "active=0" in res.stdout


def test_sacct_query_has_a_timeout(tmp_path):
    pipeline = write_pipeline(tmp_path, [
        {"run_id": "x", "job_id": "44444444", "status": "running"}])
    p = tmp_path / "ssh.sh"
    p.write_text("#!/bin/sh\nsleep 30\n")
    p.chmod(0o755)
    res = run_reconcile(pipeline, p, extra=["--timeout", "2"])
    assert res.returncode != 0                 # pre-fix: hung indefinitely
    assert "TimeoutExpired" in res.stderr


def test_reconcile_uses_JobID_not_JobIDRaw(tmp_path):
    src = open(RECONCILE).read()
    assert "--format=JobIDRaw" not in src
    assert "--format=JobID," in src
