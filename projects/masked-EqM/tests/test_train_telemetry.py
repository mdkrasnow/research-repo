"""Tests for train.py's telemetry wiring.  No GPU, no cluster, no dataset.

Three properties are checked, each corresponding to one defect the instrumentation
was added to close:

1. **The kind discriminator is applied, and is applied correctly.**  Checked both
   structurally (every ``progress``/``evaluation`` call in ``train.py`` passes a
   literal ``kind``; the pre-step gradient record and the post-step probe record
   use *different* literals, and sit on the correct side of ``opt.step()``) and
   behaviourally (two records sharing a step number remain separable on the
   stream, so a per-kind count is right where an undiscriminated count doubles).
2. **The fallback path works without ``EQM_RUN_SPEC``.**  An uninstrumented job
   must still train, with its identity explicitly marked ``inferred``.
3. **The experiment directory name is a deterministic function of the spec**, not
   of how many sibling directories happen to exist -- with legacy directories
   adopted rather than shadowed.
"""

import ast
import json
import os
import sys
from argparse import Namespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_utils
from telemetry import RunSpec, RunStatus

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(PROJECT_ROOT, "train.py")


def make_args(**overrides):
    """A minimal argparse-shaped stand-in for the real trainer args."""
    base = dict(
        model="EqM-B/2", path_type="Linear", prediction="velocity",
        loss_weight=None, ebm="direct", image_size=256, num_classes=1000,
        epochs=80, global_batch_size=256, global_seed=0, vae="ema",
        uncond=True, disp=False, weight_decay=0.0, max_grad_norm=6.87141,
        adaptive_clip=False, adaptive_clip_factor=4.0,
        adaptive_clip_ema_decay=0.99, exact_fwrev=False, gp_lambda=0.0,
        energy_zloss_lambda=0.0, wfb_backward=False, wfb_rho=1e-4, wfb_k=12,
        wfb_alpha=0.5, reset_adam_state=False, btm_mode="btm_scalar_fd_directional",
        btm_interpolant="self_stopping", btm_tc=0.8, btm_kappa=0.8,
        fd_eps=1e-3, fd_k=4, fd_direction="rademacher",
        energy_difference_fp32=True, train_eps=None, sample_eps=None,
        corruption_mode="gaussian", mask_prob=0.5, fourier_cutoff=0.25,
        blur_sigma=1.0, downsample_factor=4.0, gaussian_weight=1.0,
        mask_weight=0.0, blur_weight=0.0, fourier_weight=0.0,
        downsample_weight=0.0, structured_mask_weight=0.0, max_steps=20000,
        results_dir="results",
    )
    base.update(overrides)
    return Namespace(**base)


def a_spec(**overrides):
    fields = dict(campaign="btm", arm="btm_scalar_fd_directional", seed=0,
                  git_sha="deadbeef", phase="II", planned_steps=20000,
                  params={"fd_k": 4})
    fields.update(overrides)
    return RunSpec(**fields)


#################################################################################
#                   1. the kind discriminator (structure + behaviour)           #
#################################################################################

def _telemetry_calls(tree):
    """Every ``telemetry_run.<method>(...)`` call in the module, with its line."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "telemetry_run"):
            out.append((func.attr, node))
    return out


def test_train_py_parses():
    with open(TRAIN_PY, encoding="utf-8") as handle:
        ast.parse(handle.read())


def test_every_measurement_call_carries_a_literal_kind():
    with open(TRAIN_PY, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    calls = _telemetry_calls(tree)
    measurements = [(name, node) for name, node in calls
                    if name in ("progress", "evaluation")]
    assert measurements, "train.py emits no progress/evaluation records at all"
    for name, node in measurements:
        kinds = [kw for kw in node.keywords if kw.arg == "kind"]
        assert len(kinds) == 1, (
            f"{name}() at line {node.lineno} must pass exactly one kind= "
            "(it is the discriminator that keeps record shapes separable)")
        assert isinstance(kinds[0].value, ast.Constant) and isinstance(
            kinds[0].value.value, str) and kinds[0].value.value, (
            f"{name}() at line {node.lineno} must pass a non-empty literal kind")


def test_grad_and_probe_kinds_are_distinct_and_straddle_opt_step():
    """The two records that used to be indistinguishable now differ by kind.

    Additionally pins the step-semantics convention: kind="grad" is emitted
    BEFORE ``opt.step()`` (it describes the gradient that step applies) and
    kind="probe" AFTER it (it measures that step's effect).  Both carry
    ``train_steps + 1``.
    """
    with open(TRAIN_PY, encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)
    kinds = {}
    for name, node in _telemetry_calls(tree):
        if name != "progress":
            continue
        kind = [kw.value.value for kw in node.keywords if kw.arg == "kind"][0]
        kinds.setdefault(kind, []).append(node.lineno)
    assert "grad" in kinds and "probe" in kinds
    assert kinds["grad"] != kinds["probe"], "grad and probe must not share a call"

    opt_step_lines = [i + 1 for i, line in enumerate(source.splitlines())
                      if line.strip() == "opt.step()"]
    assert len(opt_step_lines) == 1
    opt_step = opt_step_lines[0]
    assert all(line < opt_step for line in kinds["grad"]), \
        "kind='grad' must be emitted before opt.step()"
    assert all(line > opt_step for line in kinds["probe"]), \
        "kind='probe' must be emitted after opt.step()"


def test_both_training_paths_are_wrapped_in_a_recorder():
    with open(TRAIN_PY, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)}
    for name in ("main", "main_forward_backwards_direct"):
        body = functions[name]
        withs = [n for n in ast.walk(body) if isinstance(n, ast.With)
                 and any(isinstance(item.context_expr, ast.Name)
                         and item.context_expr.id == "telemetry_run"
                         for item in n.items)]
        assert withs, f"{name}() does not wrap its loop in the recorder context"
        loops = [n for n in ast.walk(withs[0]) if isinstance(n, ast.For)]
        assert loops, f"{name}()'s recorder context contains no training loop"


def test_records_of_two_kinds_at_one_step_stay_separable(tmp_path):
    """The behavioural half: same step, two shapes, still countable per kind."""
    spec = a_spec()
    root = str(tmp_path / "telemetry")
    from telemetry import RunRecorder

    with RunRecorder(root, spec, planned_steps=3, job_id="12345",
                     mirror_stderr=False, install_signal_handlers=False) as run:
        for step in (1, 2, 3):
            run.progress(step, kind="grad", grad_norm=1.0 * step, clipped=False)
            run.progress(step, kind="probe", delta_theta_norm=0.5 * step,
                         probe_delta_L=-0.01)
        run.set_last_step(3)

    events_dir = os.path.join(root, spec.slug(), "events")
    (stream,) = [os.path.join(events_dir, n) for n in os.listdir(events_dir)]
    records = [json.loads(line) for line in open(stream, encoding="utf-8")]

    progress = [r for r in records if r["event"] == "PROGRESS"]
    assert len(progress) == 6                      # the undiscriminated count
    assert len([r for r in progress if r["kind"] == "grad"]) == 3
    assert len([r for r in progress if r["kind"] == "probe"]) == 3
    # Disjoint payload keys: the two populations must not be poolable by accident.
    grad_keys = {k for r in progress if r["kind"] == "grad" for k in r}
    probe_keys = {k for r in progress if r["kind"] == "probe" for k in r}
    assert "grad_norm" in grad_keys and "grad_norm" not in probe_keys
    assert "delta_theta_norm" in probe_keys and "delta_theta_norm" not in grad_keys
    # And the headline defect is closed: there is exactly one terminal record.
    ends = [r for r in records if r["event"] == "END"]
    assert len(ends) == 1
    assert ends[0]["status"] == RunStatus.COMPLETED.value
    assert ends[0]["truncated"] is False
    assert ends[0]["last_step"] == 3


def test_truncated_run_is_distinguishable_from_a_complete_one(tmp_path):
    spec = a_spec(seed=1)
    root = str(tmp_path / "telemetry")
    from telemetry import RunRecorder

    with RunRecorder(root, spec, planned_steps=20000, job_id="7",
                     mirror_stderr=False, install_signal_handlers=False) as run:
        run.progress(3000, kind="grad", grad_norm=1.0)
        run.set_last_step(3000)

    events_dir = os.path.join(root, spec.slug(), "events")
    (stream,) = [os.path.join(events_dir, n) for n in os.listdir(events_dir)]
    end = [json.loads(l) for l in open(stream, encoding="utf-8")
           if json.loads(l)["event"] == "END"][0]
    assert end["truncated"] is True
    assert end["last_step"] == 3000 and end["planned_steps"] == 20000


#################################################################################
#                          2. the uninstrumented fallback                       #
#################################################################################

def test_spec_is_stated_when_the_launcher_exported_one():
    spec = a_spec()
    env = {"EQM_RUN_SPEC": spec.to_env()}
    resolved, provenance = train_utils.resolve_run_spec(make_args(), env=env)
    assert provenance == "stated"
    assert resolved.run_uid == spec.run_uid
    assert resolved.params.get("provenance") is None


def test_stated_spec_is_not_mutated_by_the_trainers_planned_steps():
    """Patching planned_steps into a stated spec would move its run_uid."""
    spec = a_spec()
    env = {"EQM_RUN_SPEC": spec.to_env()}
    resolved, _ = train_utils.resolve_run_spec(
        make_args(), planned_steps=999, env=env)
    assert resolved.run_uid == spec.run_uid
    assert resolved.planned_steps == spec.planned_steps


def test_spec_is_inferred_and_marked_when_env_is_absent():
    args = make_args()
    resolved, provenance = train_utils.resolve_run_spec(
        args, planned_steps=20000, env={"EQM_GIT_SHA": "cafe1234"})
    assert provenance == train_utils.INFERRED_PROVENANCE
    assert resolved.params["provenance"] == train_utils.INFERRED_PROVENANCE
    assert resolved.git_sha == "cafe1234"
    assert resolved.seed == args.global_seed
    assert resolved.arm == "btm_scalar_fd_directional_k4"
    # Identity-bearing: an inferred run cannot silently collide with a stated one.
    stated = a_spec()
    assert resolved.run_uid != stated.run_uid


def test_a_malformed_env_spec_degrades_to_inferred_rather_than_crashing():
    resolved, provenance = train_utils.resolve_run_spec(
        make_args(), env={"EQM_RUN_SPEC": "{not json", "EQM_GIT_SHA": "abc"})
    assert provenance == train_utils.INFERRED_PROVENANCE
    assert resolved is not None


def test_inferred_arm_names_separate_the_backward_modes():
    assert train_utils.infer_arm_name(
        make_args(btm_mode=None, ebm="direct", exact_fwrev=True)) == "exact_fwrev"
    assert train_utils.infer_arm_name(
        make_args(btm_mode=None, ebm="direct", wfb_backward=True,
                  wfb_alpha=1.0)) == "wfb_alpha1.0"
    assert train_utils.infer_arm_name(
        make_args(btm_mode=None, ebm="none")) == "ebm_none"
    assert train_utils.infer_arm_name(
        make_args(btm_mode=None, ebm="forward-backwards-direct")) == "fb_direct"


def test_recorder_without_a_spec_is_a_working_no_op():
    """Telemetry misconfiguration must never be able to stop training."""
    run = train_utils.open_run_recorder(0, "/nonexistent/root", None)
    with run as handle:
        handle.progress(1, kind="grad", grad_norm=1.0)
        handle.evaluation(1, kind="target_match", target_cosine=0.5)
        handle.artifact("/no/such/file.pt", step=1)
        handle.notice("hello", level="warn")
        handle.heartbeat(1)
        handle.set_last_step(1)


def test_recorder_construction_failure_degrades_to_no_op(monkeypatch, capsys):
    import telemetry

    def boom(*_a, **_k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(telemetry, "recorder_for_rank", boom)
    run = train_utils.open_run_recorder(0, "/nonexistent/root", a_spec())
    with run as handle:
        handle.progress(1, kind="grad", grad_norm=1.0)
    assert "[telemetry]" in capsys.readouterr().out


def test_a_bad_payload_warns_but_does_not_kill_training(tmp_path, capsys):
    """A record the schema rejects must cost a warning, not the run."""
    run = train_utils.open_run_recorder(
        0, str(tmp_path), a_spec(), planned_steps=2, job_id="1",
        mirror_stderr=False, install_signal_handlers=False)
    with run as handle:
        handle.progress(1, kind="grad", seq=123)   # "seq" is a reserved envelope key
        handle.progress(2, kind="grad", grad_norm=1.0)   # still works afterwards
        handle.set_last_step(2)
    assert "[telemetry]" in capsys.readouterr().out
    events_dir = os.path.join(str(tmp_path), a_spec().slug(), "events")
    (stream,) = [os.path.join(events_dir, n) for n in os.listdir(events_dir)]
    records = [json.loads(line) for line in open(stream, encoding="utf-8")]
    assert [r["step"] for r in records if r["event"] == "PROGRESS"] == [2]
    assert len([r for r in records if r["event"] == "END"]) == 1


def test_the_terminal_record_survives_a_crashing_training_loop(tmp_path):
    spec = a_spec(seed=3)
    run = train_utils.open_run_recorder(
        0, str(tmp_path), spec, planned_steps=100, job_id="9",
        mirror_stderr=False, install_signal_handlers=False)
    with pytest.raises(RuntimeError):
        with run as handle:
            handle.progress(1, kind="grad", grad_norm=1.0)
            handle.set_last_step(1)
            raise RuntimeError("CUDA out of memory")
    events_dir = os.path.join(str(tmp_path), spec.slug(), "events")
    (stream,) = [os.path.join(events_dir, n) for n in os.listdir(events_dir)]
    end = [json.loads(l) for l in open(stream, encoding="utf-8")
           if json.loads(l)["event"] == "END"][0]
    assert end["status"] == RunStatus.CRASHED.value
    assert end["truncated"] is True
    assert "CUDA out of memory" in end["error"]


def test_non_zero_ranks_get_a_no_op_recorder(tmp_path):
    run = train_utils.open_run_recorder(1, str(tmp_path), a_spec(), planned_steps=5)
    with run as handle:
        handle.progress(1, kind="grad", grad_norm=1.0)
    assert not os.listdir(tmp_path), "a non-zero rank must not write telemetry"


#################################################################################
#                     3. deterministic experiment-directory naming              #
#################################################################################

def test_experiment_dir_is_independent_of_sibling_count():
    args = make_args()
    spec = a_spec()
    empty = train_utils.resolve_experiment_dir("results", args, spec, existing=[])
    crowded = train_utils.resolve_experiment_dir(
        "results", args, spec,
        existing=["000-something-else", "001-another", "002-yet-another"])
    assert empty == crowded
    assert spec.run_uid in os.path.basename(empty)


def test_experiment_dir_is_a_function_of_the_spec():
    args = make_args()
    same = {train_utils.resolve_experiment_dir("results", args, a_spec(), existing=[])
            for _ in range(5)}
    assert len(same) == 1
    other = train_utils.resolve_experiment_dir(
        "results", args, a_spec(seed=1), existing=[])
    assert other != same.pop(), "different specs must not share a directory"


def test_requeue_lands_back_in_its_own_directory():
    """The defect: a requeue used to create 001-... beside 000-... ."""
    args = make_args()
    spec = a_spec()
    first = train_utils.resolve_experiment_dir("results", args, spec, existing=[])
    second = train_utils.resolve_experiment_dir(
        "results", args, spec, existing=[os.path.basename(first)])
    assert first == second


def test_legacy_directories_are_adopted_not_shadowed():
    """A run resuming into an existing on-disk results dir must keep using it."""
    args = make_args()
    stem = train_utils.experiment_stem(args)
    legacy = f"000-{stem}"
    resolved = train_utils.resolve_experiment_dir(
        "results", args, a_spec(), existing=[legacy])
    assert os.path.basename(resolved) == legacy


def test_experiment_dir_falls_back_without_a_spec():
    args = make_args()
    resolved = train_utils.resolve_experiment_dir("results", args, None, existing=[])
    assert os.path.basename(resolved) == f"000-{train_utils.experiment_stem(args)}"


def test_experiment_dir_reads_the_filesystem_when_not_injected(tmp_path):
    args = make_args()
    spec = a_spec()
    stem = train_utils.experiment_stem(args)
    os.makedirs(tmp_path / f"000-{stem}")
    os.makedirs(tmp_path / "unrelated-dir")
    resolved = train_utils.resolve_experiment_dir(str(tmp_path), args, spec)
    assert os.path.basename(resolved) == f"000-{stem}"


#################################################################################
#                       misc: the small measurement helpers                     #
#################################################################################

def test_is_nonfinite():
    assert train_utils.is_nonfinite(float("nan"))
    assert train_utils.is_nonfinite(float("inf"))
    assert not train_utils.is_nonfinite(0.0)
    assert not train_utils.is_nonfinite(None)
    assert not train_utils.is_nonfinite("not a number")


def test_memory_probes_never_raise():
    for value in (train_utils.peak_gpu_memory_bytes(), train_utils.host_rss_bytes()):
        assert value is None or isinstance(value, int)


def test_telemetry_root_is_overridable(tmp_path):
    assert train_utils.telemetry_root_for("results", env={}) == \
        os.path.join("results", "_telemetry")
    assert train_utils.telemetry_root_for(
        "results", env={"EQM_TELEMETRY_ROOT": "/pool"}) == "/pool"
