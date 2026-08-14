"""The lifecycle FSM and the terminal-record guarantee.

This module is the direct answer to the defect that started the audit: *start
emissions were not tied to finish emissions*.  In the previous design there were
no finish emissions at all -- ``train.py`` closed its metrics file without
writing anything, so a run killed at step 3,000 and a run that completed 20,000
steps left byte-identically *shaped* telemetry: a stream that simply stops.  Every
downstream comparison of "late training" was therefore comparing whatever each
run happened to reach, with no way to tell the difference.

The guarantee
-------------
Every execution that emits ``START`` also emits exactly one ``END``.  That is
enforced by a five-layer ladder, ordered by how violently the process dies:

1. **Normal return** -- the context manager's ``__exit__`` emits ``END``.
2. **Exception** -- same ``__exit__``, status ``crashed``, with the traceback.
3. **Signal** (``SIGTERM`` from preemption or ``scancel``; ``SIGUSR1`` from
   SLURM's ``--signal`` pre-timeout warning; ``SIGINT``) -- the handler raises
   :class:`Interrupted` in the main thread, which unwinds into ``__exit__``.
   Raising rather than writing from the handler keeps the write path
   non-reentrant-safe-by-luck instead of by hope.
4. **atexit** -- if some path bypasses ``__exit__`` (``os._exit`` aside), the
   registered hook still seals the stream.
5. **``SIGKILL`` / node death** -- unreachable from inside the process, by
   definition.  Covered outside it: the sbatch wrapper runs ``telemetry seal``
   with the observed exit code, and failing that the reconciler infers the
   terminal state from ``sacct`` and appends ``END`` with ``inferred: true``.

Layer 5 is why ``LOST`` exists as a distinct status.  "We know it died" and "we
never found out" are different epistemic states and the analysis treats them
differently; collapsing them is how a truncated run gets pooled with complete
ones.

Truncation is a first-class fact
--------------------------------
``END`` carries ``last_step`` and ``planned_steps``.  Their inequality is the
machine-checkable definition of a truncated run, which :mod:`telemetry.read`
turns into a hard gate on aggregation.
"""

from __future__ import annotations

import os
import signal
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional

from .emit import TelemetryWriter, open_writer
from .ids import RunSpec
from .schema import EventType, RunStatus, SchemaError

#: Signals SLURM uses to communicate impending death, mapped to the status that
#: best describes what happened.  SIGUSR1 is what `#SBATCH --signal=USR1@120`
#: delivers before a wall-clock kill; SIGTERM is preemption or scancel.
_SIGNAL_STATUS = {
    signal.SIGTERM: RunStatus.PREEMPTED,
    signal.SIGINT: RunStatus.CANCELLED,
}
if hasattr(signal, "SIGUSR1"):
    _SIGNAL_STATUS[signal.SIGUSR1] = RunStatus.TIMEOUT


class Interrupted(BaseException):
    """Raised in the main thread when a lifecycle signal arrives.

    Derives from ``BaseException``, not ``Exception``: training loops are full of
    broad ``except Exception`` handlers meant to tolerate a bad batch, and any
    one of them would otherwise swallow a preemption notice and let the run
    continue for the few seconds it has left -- losing the terminal record we
    interrupted it to write.
    """

    def __init__(self, signum: int, status: RunStatus) -> None:
        super().__init__(f"signal {signum} -> {status.value}")
        self.signum = signum
        self.status = status


def _peak_gpu_memory_bytes() -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return int(max(torch.cuda.max_memory_allocated(d)
                           for d in range(torch.cuda.device_count())))
    except Exception:
        pass
    return None


def _host_rss_bytes() -> Optional[int]:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _slurm_context() -> Dict[str, Any]:
    """Scheduler facts worth freezing into the record.

    Recorded at ``START`` because they are otherwise unrecoverable later: once
    ``sacct`` ages a job out of its retention window, the only surviving evidence
    of which node, partition and allocation produced a result is whatever the run
    itself wrote down.
    """
    keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_NNODES",
            "SLURM_NTASKS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_QOS",
            "SLURM_RESTART_COUNT", "SLURMD_NODENAME")
    return {k.lower(): os.environ[k] for k in keys if os.environ.get(k)}


def next_attempt(telemetry_root: str, spec: RunSpec, job_id: Any) -> int:
    """Pick an attempt number that does not collide with an existing stream.

    A SLURM **requeue reuses the job id**, so ``job_id`` alone does not identify
    an execution.  Two requeued attempts sharing one stream is precisely the
    interleaving bug this design exists to prevent, so the attempt counter is
    derived from what is already on disk rather than trusted from the
    environment.  ``SLURM_RESTART_COUNT`` is used as a floor when present.
    """
    from .ids import make_exec_id, split_exec_id

    events_dir = os.path.join(telemetry_root, spec.slug(), "events")
    floor = int(os.environ.get("SLURM_RESTART_COUNT") or 0)
    if not os.path.isdir(events_dir):
        return floor
    slug_job = make_exec_id(spec.run_uid, job_id, 0).split(":")[1]
    used = set()
    for name in os.listdir(events_dir):
        if not name.endswith(".jsonl"):
            continue
        try:
            _, existing_job, attempt = split_exec_id(name[: -len(".jsonl")])
        except SchemaError:
            continue
        if existing_job == slug_job:
            used.add(attempt)
    attempt = floor
    while attempt in used:
        attempt += 1
    return attempt


class RunRecorder:
    """Context manager owning one execution's lifecycle.

    Usage::

        spec = RunSpec.from_env()
        with RunRecorder(root, spec, planned_steps=20_000) as run:
            for step in ...:
                run.progress(step, kind="grad", grad_norm=..., clipped=...)
            run.set_last_step(step)

    Only rank 0 should construct one; other ranks get :class:`NullRecorder`, so
    call sites need no ``if rank == 0`` branching and cannot accidentally have
    one arm of that branch drift from the other.
    """

    def __init__(
        self,
        telemetry_root: str,
        spec: RunSpec,
        *,
        planned_steps: Optional[int] = None,
        job_id: Any = None,
        attempt: Optional[int] = None,
        wandb_module: Any = None,
        mirror_stderr: bool = True,
        install_signal_handlers: bool = True,
        extra_start: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.spec = spec
        self.telemetry_root = telemetry_root
        self.planned_steps = (planned_steps if planned_steps is not None
                              else spec.planned_steps)
        self.job_id = job_id if job_id is not None else os.environ.get("SLURM_JOB_ID")
        self.attempt = (attempt if attempt is not None
                        else next_attempt(telemetry_root, spec, self.job_id))
        self._wandb = wandb_module
        self._mirror = mirror_stderr
        self._install_handlers = install_signal_handlers
        self._extra_start = dict(extra_start or {})
        self.writer: Optional[TelemetryWriter] = None
        self.last_step: int = -1
        self._started_at: float = 0.0
        self._last_heartbeat: float = 0.0
        self._previous_handlers: List[Any] = []
        self._sealed = False
        self._nonfinite_events = 0
        # Completion is something the training loop KNOWS; inferring it from a
        # step counter is unreliable because `last_step` is the last *logged*
        # step, which trails the last *executed* step by the logging stride. An
        # early version of this class compared `last_step + 1 >= planned_steps`
        # and sealed every healthy run as `timeout` at a stride of 50.
        self._completion_asserted = False
        self._logged_steps: List[int] = []

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "RunRecorder":
        self.writer = open_writer(
            self.telemetry_root, self.spec, job_id=self.job_id,
            attempt=self.attempt, wandb_module=self._wandb,
            mirror_stderr=self._mirror)
        self._started_at = time.time()
        self._last_heartbeat = self._started_at
        payload: Dict[str, Any] = {
            "campaign": self.spec.campaign,
            "phase": self.spec.phase,
            "arm": self.spec.arm,
            "seed": self.spec.seed,
            "git_sha": self.spec.git_sha,
            "planned_steps": self.planned_steps,
            "config": dict(self.spec.params),
            "job_id": str(self.job_id) if self.job_id is not None else None,
            "attempt": self.attempt,
            "pid": os.getpid(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else None,
            "world_size": _world_size(),
            "slurm": _slurm_context(),
            "started_at_unix": self._started_at,
        }
        payload.update(self._extra_start)
        self.writer.emit(EventType.START, payload)
        if self._install_handlers:
            self._install()
        return self

    def _install(self) -> None:
        for signum, status in _SIGNAL_STATUS.items():
            try:
                previous = signal.signal(signum, self._handle_signal)
                self._previous_handlers.append((signum, previous))
            except (ValueError, OSError):
                # Not the main thread, or the signal is unavailable here.
                pass

    def _restore(self) -> None:
        for signum, previous in self._previous_handlers:
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):
                pass
        self._previous_handlers = []

    def _handle_signal(self, signum, _frame):  # pragma: no cover - signal path
        status = _SIGNAL_STATUS.get(signum, RunStatus.PREEMPTED)
        # Restore defaults first: if a second signal arrives while we are
        # unwinding, it should kill the process outright rather than re-entering
        # this handler and hanging the very shutdown that writes END.
        self._restore()
        raise Interrupted(signum, status)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._restore()
        if isinstance(exc, Interrupted):
            self.seal(exc.status, error=str(exc))
            return True  # a preemption is not a bug; do not spam a traceback
        if exc is not None:
            self.seal(RunStatus.CRASHED, error="".join(
                traceback.format_exception(exc_type, exc, tb))[-4000:])
            return False
        self.seal(RunStatus.COMPLETED if self._reached_plan()
                  else RunStatus.TIMEOUT)
        return False

    def _log_stride(self) -> int:
        """Typical gap between logged steps, used to bound the counter's lag.

        The recorder only sees the steps it was told about, so on a cadence of
        every 50 steps a run that executed all 20,000 reports ``last_step ==
        19,950``.  Measuring the stride from the observed steps -- rather than
        hard-coding a tolerance -- keeps the completion test correct for any
        cadence, including one that changes mid-run.
        """
        if len(self._logged_steps) < 2:
            return 1
        gaps = [b - a for a, b in zip(self._logged_steps, self._logged_steps[1:])
                if b > a]
        if not gaps:
            return 1
        gaps.sort()
        return max(1, gaps[len(gaps) // 2])

    def _reached_plan(self) -> bool:
        """Did this execution actually finish its planned work?

        Explicit assertion by the training loop wins: the loop is the only thing
        that knows it ran to completion.  Absent that -- an uninstrumented or
        legacy caller -- fall back to a stride-aware step comparison, which is
        approximate but no longer systematically wrong.
        """
        if self._completion_asserted:
            return True
        if self.planned_steps is None:
            return True
        return self.last_step + self._log_stride() >= self.planned_steps

    def mark_complete(self, last_step: Optional[int] = None) -> None:
        """Assert that the training loop ran to completion.

        Call this immediately after the loop exits normally.  It is the
        difference between a run that finished and a run that merely stopped
        without raising, which no amount of step arithmetic can distinguish
        reliably.
        """
        if last_step is not None:
            self.set_last_step(last_step)
        self._completion_asserted = True

    def seal(self, status: RunStatus, *, error: Optional[str] = None,
             **extra: Any) -> None:
        """Emit the single terminal record.  Idempotent."""
        if self._sealed or self.writer is None:
            return
        self._sealed = True
        payload: Dict[str, Any] = {
            "status": RunStatus(status).value,
            "last_step": self.last_step,
            "planned_steps": self.planned_steps,
            "truncated": bool(self.planned_steps is not None
                              and self.last_step + 1 < self.planned_steps),
            "wall_seconds": round(time.time() - self._started_at, 3),
            "records_emitted": self.writer.next_seq,
            "nonfinite_events": self._nonfinite_events,
            "peak_gpu_memory_bytes": _peak_gpu_memory_bytes(),
            "peak_host_rss_bytes": _host_rss_bytes(),
            "inferred": False,
        }
        if error:
            payload["error"] = error
        payload.update(extra)
        try:
            self.writer.emit(EventType.END, payload)
        finally:
            self.writer.close()

    # -- measurement ---------------------------------------------------------

    def progress(self, step: int, *, kind: str, **metrics: Any) -> None:
        """One per-step measurement record.

        ``kind`` is mandatory and is the discriminator that keeps structurally
        different measurements separable on a shared stream.  Its absence was a
        real defect: two record shapes were written to one file at the same step,
        so counts double-counted and one statistic was computed as a ratio of
        medians taken over two disjoint populations.
        """
        if self.writer is None:
            return
        if not kind:
            raise SchemaError("progress() requires a non-empty kind")
        self.last_step = max(self.last_step, int(step))
        self._count_nonfinite(metrics)
        self.writer.emit(EventType.PROGRESS, {"step": int(step), "kind": kind, **metrics})

    def evaluation(self, step: int, *, kind: str, **metrics: Any) -> None:
        """A periodic expensive evaluation, kept off the per-step stream."""
        if self.writer is None:
            return
        self.last_step = max(self.last_step, int(step))
        self._count_nonfinite(metrics)
        self.writer.emit(EventType.EVAL, {"step": int(step), "kind": kind, **metrics})

    def artifact(self, path: str, *, step: Optional[int] = None,
                 kind: str = "checkpoint", **extra: Any) -> None:
        """Bind a file on disk to the execution and step that produced it."""
        if self.writer is None:
            return
        payload: Dict[str, Any] = {"path": path, "kind": kind}
        if step is not None:
            payload["step"] = int(step)
        try:
            payload["bytes"] = os.path.getsize(path)
        except OSError:
            payload["bytes"] = None
        payload.update(extra)
        self.writer.emit(EventType.ARTIFACT, payload)

    def notice(self, message: str, *, level: str = "info", **extra: Any) -> None:
        if self.writer is None:
            return
        self.writer.emit(
            EventType.NOTICE, {"level": level, "message": message, **extra},
            sync=(level in ("error", "fatal")))

    def heartbeat(self, step: Optional[int] = None, *, min_interval_s: float = 300.0,
                  **extra: Any) -> None:
        """Periodic liveness beacon; rate-limited internally.

        Distinguishes "job is slow" from "job is wedged".  This repo has lost a
        night to exactly that ambiguity -- a training job deadlocked on a
        filesystem quota with its stdout pipe blocked, indistinguishable from a
        slow dataloader until someone looked.  A heartbeat whose *absence* is
        detectable turns that into a mechanical check.
        """
        if self.writer is None:
            return
        now = time.time()
        if now - self._last_heartbeat < min_interval_s:
            return
        self._last_heartbeat = now
        payload: Dict[str, Any] = {
            "level": "heartbeat",
            "message": "alive",
            "wall_seconds": round(now - self._started_at, 3),
            "last_step": self.last_step if step is None else int(step),
            "peak_gpu_memory_bytes": _peak_gpu_memory_bytes(),
            "peak_host_rss_bytes": _host_rss_bytes(),
        }
        payload.update(extra)
        self.writer.emit(EventType.NOTICE, payload, sync=True)

    def set_last_step(self, step: int) -> None:
        self.last_step = max(self.last_step, int(step))

    def _count_nonfinite(self, metrics: Mapping[str, Any]) -> None:
        import math
        for value in metrics.values():
            if isinstance(value, float) and not math.isfinite(value):
                self._nonfinite_events += 1
                return


class NullRecorder:
    """No-op recorder for non-zero ranks.

    Same surface as :class:`RunRecorder` so that call sites are rank-agnostic.
    The alternative -- guarding every emission with ``if rank == 0`` -- is how
    one arm of a branch drifts from the other.
    """

    last_step = -1

    def __enter__(self) -> "NullRecorder":
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def progress(self, *_a, **_k) -> None: ...
    def evaluation(self, *_a, **_k) -> None: ...
    def artifact(self, *_a, **_k) -> None: ...
    def notice(self, *_a, **_k) -> None: ...
    def heartbeat(self, *_a, **_k) -> None: ...
    def set_last_step(self, *_a, **_k) -> None: ...
    def seal(self, *_a, **_k) -> None: ...


def _world_size() -> int:
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def recorder_for_rank(rank: int, *args: Any, **kwargs: Any):
    """Return a real recorder on rank 0 and a null one elsewhere."""
    return RunRecorder(*args, **kwargs) if rank == 0 else NullRecorder()
