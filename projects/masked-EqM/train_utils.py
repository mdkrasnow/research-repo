"""Argument parsing helpers, plus the telemetry glue used by ``train.py``.

Everything telemetry-related that does not need a GPU, a dataset or a process
group lives here rather than inline in ``train.py``, so that it can be tested
without running training.  ``train.py`` keeps only the call sites.
"""

import os
import re
import subprocess
from glob import glob


def none_or_str(value):
    if value == 'None':
        return None
    return value


#################################################################################
#                      Telemetry glue (importable, torch-free)                  #
#################################################################################
#
# STEP-SEMANTICS CONVENTION (applies to every emission from train.py):
#
#   Both the pre-``opt.step()`` gradient record and the post-``opt.step()``
#   probe record are labelled with ``step = train_steps + 1``: the 1-based index
#   of the optimizer step being taken on this iteration.  ``kind="grad"``
#   measures the INPUT of that step (the gradient it is about to apply);
#   ``kind="probe"`` measures the EFFECT of that same step (parameter
#   displacement and held-out field motion caused by it).  They deliberately
#   share a step number -- they are two measurements of one optimizer step --
#   and are told apart by ``kind``, never by position in the file.  This is also
#   exactly the numbering the legacy ``gradient_metrics.jsonl`` uses, so the
#   dual-written legacy stream is unchanged byte-for-byte.

#: Marker placed in an inferred spec's params so that a consumer can tell a run
#: whose identity was STATED by the launcher from one whose identity was GUESSED
#: by the trainer.  It is identity-bearing on purpose: an inferred run must not
#: collide with the properly-launched run it approximates.
INFERRED_PROVENANCE = "inferred"

_SLUG_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def resolve_git_sha(default="unknown", env=None):
    """Best-effort commit id: launcher-provided first, then ``git``, then unknown.

    The launcher's value wins because it is the sha the job was *submitted* at,
    which is what the job actually checked out; asking ``git`` inside a working
    tree that has moved on would report a different commit for the same run.
    """
    env = os.environ if env is None else env
    for key in ("EQM_GIT_SHA", "GIT_SHA", "SLURM_JOB_GIT_SHA"):
        value = env.get(key)
        if value:
            return value.strip()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip() or default
    except Exception:
        return default


def infer_arm_name(args):
    """A stable, human-readable name for the training arm implied by ``args``.

    Only used when no launcher stated one.  Ordered most-specific-first so that
    e.g. a BTM run is never labelled merely by its ``--ebm`` value.
    """
    btm_mode = getattr(args, "btm_mode", None)
    if btm_mode:
        parts = [str(btm_mode)]
        fd_k = getattr(args, "fd_k", None)
        if fd_k is not None:
            parts.append(f"k{fd_k}")
        return "_".join(parts)
    if getattr(args, "ebm", "none") == "forward-backwards-direct":
        return "fb_direct"
    if getattr(args, "wfb_backward", False):
        return f"wfb_alpha{getattr(args, 'wfb_alpha', '')}"
    if getattr(args, "exact_fwrev", False):
        return "exact_fwrev"
    return f"ebm_{getattr(args, 'ebm', 'none')}"


#: Argparse attributes folded into an inferred spec's ``params``.  These are the
#: knobs that change what is computed; scheduler/IO knobs are excluded because
#: ``telemetry.ids.canonicalize`` would drop most of them anyway and the rest
#: (paths, worker counts) must not participate in identity.
_INFERRED_PARAM_KEYS = (
    "model", "image_size", "num_classes", "epochs", "global_batch_size",
    "vae", "ebm", "uncond", "disp", "weight_decay", "max_grad_norm",
    "adaptive_clip", "adaptive_clip_factor", "adaptive_clip_ema_decay",
    "exact_fwrev", "gp_lambda", "energy_zloss_lambda", "wfb_backward",
    "wfb_rho", "wfb_k", "wfb_alpha", "reset_adam_state", "btm_mode",
    "btm_interpolant", "btm_tc", "btm_kappa", "fd_eps", "fd_k", "fd_direction",
    "energy_difference_fp32", "path_type", "prediction", "loss_weight",
    "train_eps", "sample_eps", "corruption_mode", "mask_prob",
    "fourier_cutoff", "blur_sigma", "downsample_factor", "gaussian_weight",
    "mask_weight", "blur_weight", "fourier_weight", "downsample_weight",
    "structured_mask_weight", "max_steps",
)


def build_inferred_run_spec(args, planned_steps=None, env=None):
    """Reconstruct a plausible :class:`telemetry.ids.RunSpec` from ``args``.

    Used only when ``EQM_RUN_SPEC`` is absent (a hand-run job, an sbatch written
    before the launcher existed).  ``RunSpec.from_env`` deliberately refuses to
    guess; this function guesses *loudly* instead, marking the result
    ``provenance: "inferred"`` so no analysis can mistake it for a stated
    identity.  Refusing to run at all would be worse: telemetry must never be
    able to prevent a training job from training.
    """
    from telemetry import RunSpec

    env = os.environ if env is None else env
    params = {}
    for key in _INFERRED_PARAM_KEYS:
        if hasattr(args, key):
            value = getattr(args, key)
            if isinstance(value, (set, frozenset)):
                value = sorted(value)
            params[key] = value
    params["provenance"] = INFERRED_PROVENANCE
    return RunSpec(
        campaign=env.get("EQM_CAMPAIGN", "unregistered"),
        arm=infer_arm_name(args),
        seed=int(getattr(args, "global_seed", 0)),
        git_sha=resolve_git_sha(env=env),
        phase=env.get("EQM_PHASE", ""),
        planned_steps=planned_steps,
        params=params,
    )


def resolve_run_spec(args, planned_steps=None, env=None):
    """``(spec, provenance)`` for this process.

    ``provenance`` is ``"stated"`` when the launcher exported ``EQM_RUN_SPEC``
    and ``"inferred"`` when it did not.  Never raises: a caller that cannot get
    a spec at all gets ``(None, "unavailable")`` and must degrade to a no-op
    recorder.
    """
    env = os.environ if env is None else env
    try:
        from telemetry import RunSpec
    except Exception:
        return None, "unavailable"
    if env.get("EQM_RUN_SPEC"):
        try:
            # Returned VERBATIM: planned_steps is identity-bearing, so patching
            # the trainer's own estimate into a stated spec would move its
            # run_uid away from the one the launcher registered.  The recorder
            # takes planned_steps as a separate argument for exactly this reason.
            return RunSpec.from_env(env), "stated"
        except Exception:
            # A malformed EQM_RUN_SPEC is a launcher bug, not a reason to lose a
            # GPU-hour: fall through to inference and let the marker say so.
            pass
    try:
        return build_inferred_run_spec(args, planned_steps=planned_steps, env=env), INFERRED_PROVENANCE
    except Exception:
        return None, "unavailable"


def experiment_stem(args):
    """The configuration-describing tail of an experiment directory name."""
    model_string_name = str(args.model).replace("/", "-")
    return (f"{model_string_name}-{args.path_type}-{args.prediction}-"
            f"{args.loss_weight}-ebm-{args.ebm}")


def experiment_dir_prefix(spec):
    """Deterministic directory prefix: the run's content hash, not a counter.

    The counter this replaces (``len(glob(results_dir + '/*'))``) made the name
    a function of *how many entries already existed*, so a SLURM requeue -- which
    reuses the job id and therefore the results dir -- created ``001-...``
    beside ``000-...`` and an analyzer globbing recursively counted one logical
    run twice.  A content hash is idempotent under requeue by construction.
    """
    if spec is None:
        return "000"
    return _SLUG_SAFE.sub("-", str(spec.run_uid))


def resolve_experiment_dir(results_dir, args, spec=None, existing=None):
    """Pick this run's experiment directory, deterministically.

    Backward compatibility is preserved by *adoption*: if a directory with this
    exact configuration stem already exists under ``results_dir`` -- including
    the legacy ``000-...`` names, and including the directory a previous attempt
    of this same run created -- it is reused rather than shadowed.  That keeps
    ``--ckpt`` resumes and in-flight runs writing where they already write, and
    it is what makes a requeue land back in its own directory instead of forking
    a new one.

    ``existing`` is injectable so this is testable without a filesystem.
    """
    stem = experiment_stem(args)
    if existing is None:
        existing = sorted(os.path.basename(p.rstrip("/"))
                          for p in glob(f"{results_dir}/*-{stem}"))
    else:
        existing = sorted(existing)
    preferred = f"{experiment_dir_prefix(spec)}-{stem}"
    if preferred in existing:
        return os.path.join(results_dir, preferred)
    legacy = [name for name in existing if name.endswith(f"-{stem}")]
    if legacy:
        # Deterministic tie-break: lexicographically first, so every rank and
        # every attempt agrees on the same answer without coordination.
        return os.path.join(results_dir, legacy[0])
    return os.path.join(results_dir, preferred)


def telemetry_root_for(results_dir, env=None):
    """Where event streams live.  Overridable so a campaign can pool them."""
    env = os.environ if env is None else env
    return env.get("EQM_TELEMETRY_ROOT") or os.path.join(results_dir, "_telemetry")


def open_run_recorder(rank, telemetry_root, spec, planned_steps=None,
                      logger=None, **kwargs):
    """Build a recorder that cannot take the training run down with it.

    Any failure -- missing package, unwritable root, malformed spec -- degrades
    to a no-op recorder plus a loud message.  Telemetry is evidence about a run;
    it is never a precondition for one.
    """
    def _warn(message):
        text = f"[telemetry] {message}"
        if logger is not None:
            try:
                logger.warning(text)
                return
            except Exception:
                pass
        print(text, flush=True)

    try:
        from telemetry import NullRecorder, recorder_for_rank
    except Exception as exc:
        _warn(f"package unimportable ({exc!r}); running WITHOUT telemetry")
        return _FallbackRecorder()
    if spec is None:
        _warn("no run spec could be resolved; running WITHOUT telemetry")
        return NullRecorder()
    try:
        inner = recorder_for_rank(rank, telemetry_root, spec,
                                  planned_steps=planned_steps, **kwargs)
    except Exception as exc:
        _warn(f"recorder construction failed ({exc!r}); running WITHOUT telemetry")
        return NullRecorder()
    return _GuardedRecorder(inner, _warn)


class _GuardedRecorder:
    """Proxy that makes every emission non-fatal.

    The recorder validates what it is handed (a payload key colliding with an
    envelope key, say, raises ``SchemaError``), which is right for a tool and
    wrong for a hot training loop: a diagnostic dict acquiring an awkward key
    must cost a warning, not a GPU-day.  ``BaseException`` deliberately passes
    through, because ``telemetry.lifecycle.Interrupted`` -- the preemption
    notice -- is one, and swallowing it would lose the terminal record it was
    raised to write.
    """

    def __init__(self, inner, warn):
        self._inner = inner
        self._warn = warn
        self._complained = False

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Delegated verbatim: __exit__ is what writes END, decides
        # COMPLETED/TIMEOUT/CRASHED, and reports whether it handled the
        # exception.  Guarding it would be guarding the one call that matters.
        return self._inner.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _guarded(self, name, *args, **kwargs):
        try:
            getattr(self._inner, name)(*args, **kwargs)
        except Exception as exc:
            if not self._complained:
                self._complained = True
                self._warn(f"{name}() failed ({exc!r}); further telemetry "
                           "errors on this run will be silent. Training continues.")

    def progress(self, *a, **k):
        self._guarded("progress", *a, **k)

    def evaluation(self, *a, **k):
        self._guarded("evaluation", *a, **k)

    def artifact(self, *a, **k):
        self._guarded("artifact", *a, **k)

    def notice(self, *a, **k):
        self._guarded("notice", *a, **k)

    def heartbeat(self, *a, **k):
        self._guarded("heartbeat", *a, **k)

    def set_last_step(self, *a, **k):
        self._guarded("set_last_step", *a, **k)


class _FallbackRecorder:
    """Stand-in used when the telemetry package itself is unimportable."""

    last_step = -1

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def progress(self, *_a, **_k): ...
    def evaluation(self, *_a, **_k): ...
    def artifact(self, *_a, **_k): ...
    def notice(self, *_a, **_k): ...
    def heartbeat(self, *_a, **_k): ...
    def set_last_step(self, *_a, **_k): ...
    def seal(self, *_a, **_k): ...


def peak_gpu_memory_bytes():
    """Peak allocated CUDA memory across visible devices, or ``None`` on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(max(torch.cuda.max_memory_allocated(d)
                           for d in range(torch.cuda.device_count())))
    except Exception:
        pass
    return None


def host_rss_bytes():
    """Peak host RSS (``VmHWM``) in bytes, or ``None`` where unavailable."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def is_nonfinite(value):
    """True for NaN/inf, tolerant of tensors, numpy scalars and ``None``."""
    import math
    if value is None:
        return False
    if hasattr(value, "item") and not isinstance(value, (int, float)):
        try:
            value = value.item()
        except Exception:
            return False
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)
    group.add_argument("--corruption-mode", type=str, default="gaussian",
        choices=["gaussian", "mask", "fourier", "blur", "downsample", "structured_mask", "mixture"],
        help="x0 start-state family: Gaussian (baseline), Bernoulli mask, Fourier low-pass, Gaussian blur, downsample/upsample, structured mask (blocks/patches/regions/minority-elementwise), or weighted mixture")
    group.add_argument("--mask-prob", type=float, default=0.5,
        help="fraction of latent elements replaced by noise under mask corruption")
    group.add_argument("--fourier-cutoff", type=float, default=0.25,
        help="fraction of radial frequency band kept under Fourier corruption")
    group.add_argument("--blur-sigma", type=float, default=1.0,
        help="Gaussian blur kernel sigma (latent-space pixels) under blur corruption")
    group.add_argument("--downsample-factor", type=float, default=4.0,
        help="spatial downsample/upsample factor under downsample corruption")
    group.add_argument("--gaussian-weight", type=float, default=1.0,
        help="mixture weight lambda_G (only used when --corruption-mode mixture)")
    group.add_argument("--mask-weight", type=float, default=0.0,
        help="mixture weight lambda_M (only used when --corruption-mode mixture)")
    group.add_argument("--fourier-weight", type=float, default=0.0,
        help="mixture weight lambda_F (only used when --corruption-mode mixture)")
    group.add_argument("--blur-weight", type=float, default=0.0,
        help="mixture weight lambda_B (only used when --corruption-mode mixture)")
    group.add_argument("--downsample-weight", type=float, default=0.0,
        help="mixture weight lambda_D (only used when --corruption-mode mixture)")
    group.add_argument("--structured-mask-weight", type=float, default=0.0,
        help="mixture weight lambda_SM for structured mask (only used when --corruption-mode mixture)")


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")