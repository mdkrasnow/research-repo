"""Layer 5 of the terminal-record ladder: seal a stream from *outside* the process.

Why this exists
---------------
:mod:`telemetry.lifecycle` guarantees that every execution which emits ``START``
also emits ``END`` -- but only for deaths the process can observe.  Four of the
five layers live inside python (normal return, exception, signal, ``atexit``).
The fifth cannot: ``SIGKILL``, an OOM kill, a node failure, or a wall-clock kill
that lands before the python signal handler is installed all destroy the process
without giving it a chance to write anything.

By definition that layer must be implemented by a *surviving* observer.  The
batch shell is the nearest one: it outlives the python process it launched, it
knows the exit status the kernel reported, and it is still running when SLURM
sends its pre-timeout warning.  So the shell prelude
(``slurm/lib/telemetry_env.sh``) installs a trap that invokes this module::

    python -m telemetry.seal --root "$EQM_TELEMETRY_ROOT" \
        --run-spec "$EQM_RUN_SPEC" --job-id "$SLURM_JOB_ID" \
        --exit-code 137 --status auto

Three properties this module must have, each closing a specific failure mode:

**Idempotence.**  In the common case python *did* seal its own stream and this
invocation must be a no-op.  A second ``END`` would break the "exactly one END
per exec_id" invariant that :mod:`telemetry.read` relies on to decide whether a
run is analyzable, so the check is "does an END already exist" and not "did we
already run".  Two racing sealers therefore also converge, which matters because
the EXIT trap and an explicit call in the sbatch body may both fire.

**Silence when there is nothing to seal.**  A job that died before python opened
its stream leaves no file.  Writing an ``END`` for it would require inventing a
``START``, i.e. fabricating the record whose absence is the very evidence that
the job never got going.  We exit 0 and say nothing; the reconciler will later
observe from ``sacct`` that a submitted job produced no stream at all, which is
a *different* and more honest fact.

**Never changing the job's exit status.**  This runs inside a shell ``trap`` on
EXIT.  If it raised, or exited nonzero, the observable status of the job would
become a function of the telemetry system's health -- i.e. telemetry could
manufacture the very failures it exists to report.  Every path here ends in
``sys.exit(0)``; unexpected exceptions are caught and printed to stderr.

Status inference
----------------
The status written here is always ``inferred: true``, because it is derived from
a wait-status rather than observed from inside the run.  The mapping is:

===================  =============  ==========================================
observed             status         reasoning
===================  =============  ==========================================
exit 0               completed      the command returned success
128+15 (SIGTERM)     preempted      scancel or preemption
128+9  (SIGKILL)     lost           OOM killer / node death: nothing observed it
128+10 (SIGUSR1)     timeout        SLURM's pre-wall-clock warning went unhandled
128+2  (SIGINT)      cancelled      a human interrupted it
other nonzero        crashed        the program failed on its own terms
===================  =============  ==========================================

``lost`` is deliberately distinct from ``crashed``: "we know it died" and "we
never found out how" are different epistemic states, and pooling them is how a
truncated run gets counted as a complete one.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Support both `python -m telemetry.seal` (package-relative) and a direct
# `python telemetry/seal.py` invocation, which a debugging human will try.
if __package__ in (None, ""):  # pragma: no cover - direct-script path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from telemetry.ids import RunSpec, make_exec_id, split_exec_id  # type: ignore
    from telemetry.schema import (  # type: ignore
        EventType, RunStatus, SchemaError, dumps, make_record,
    )
else:
    from .ids import RunSpec, make_exec_id, split_exec_id
    from .schema import EventType, RunStatus, SchemaError, dumps, make_record


#: Wait-status -> status.  Keys are the ``128+N`` encodings a POSIX shell reports
#: for a signal-terminated child, which is what ``$?`` gives the sbatch trap.
_SIGNAL_EXIT_STATUS: Dict[int, RunStatus] = {
    128 + int(signal.SIGTERM): RunStatus.PREEMPTED,
    128 + int(signal.SIGKILL): RunStatus.LOST,
    128 + int(signal.SIGINT): RunStatus.CANCELLED,
}
if hasattr(signal, "SIGUSR1"):
    _SIGNAL_EXIT_STATUS[128 + int(signal.SIGUSR1)] = RunStatus.TIMEOUT


def infer_status(exit_code: int, signal_name: Optional[str] = None) -> RunStatus:
    """Derive a terminal status from an exit code and an optional signal name.

    ``signal_name`` wins when present: the shell prelude knows which signal SLURM
    delivered to *it* (SIGUSR1 at the 120s warning, SIGTERM at preemption) and
    that is better evidence than the child's exit status, which in a pipeline may
    reflect a downstream ``tee`` rather than the trainer.
    """
    if signal_name:
        name = str(signal_name).upper()
        if not name.startswith("SIG"):
            name = "SIG" + name
        if name in ("SIGUSR1",):
            return RunStatus.TIMEOUT
        if name in ("SIGTERM",):
            return RunStatus.PREEMPTED
        if name in ("SIGINT",):
            return RunStatus.CANCELLED
        if name in ("SIGKILL",):
            return RunStatus.LOST
    if exit_code == 0:
        return RunStatus.COMPLETED
    if exit_code in _SIGNAL_EXIT_STATUS:
        return _SIGNAL_EXIT_STATUS[exit_code]
    return RunStatus.CRASHED


def _iter_records(path: str):
    """Yield parsed records, skipping unparseable lines.

    A truncated final line is expected here and is not an error: the whole point
    of this module is that it runs after an abrupt death, and an abrupt death is
    exactly when a partially-written line is on disk.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError:
        return


def _scan(path: str) -> Tuple[bool, bool, int, int, Optional[int]]:
    """``(has_start, has_end, last_step, max_seq, planned_steps)`` for one stream.

    ``planned_steps`` is read back off the stream's own ``START`` record rather
    than taken from the spec.  The spec's ``planned_steps`` is what the launcher
    intended; the START's is what the trainer actually resolved (a ``--max-steps``
    override, a resumed run's remaining budget).  Truncation is only a meaningful
    predicate against the latter, and this record is the sole surviving evidence
    of it once the process is gone.
    """
    has_start = has_end = False
    last_step = -1
    max_seq = -1
    planned: Optional[int] = None
    for record in _iter_records(path):
        event = record.get("event")
        if event == EventType.START.value:
            has_start = True
            value = record.get("planned_steps")
            if isinstance(value, int):
                planned = value
        elif event == EventType.END.value:
            has_end = True
        step = record.get("step")
        if isinstance(step, int):
            last_step = max(last_step, step)
        seq = record.get("seq")
        if isinstance(seq, int):
            max_seq = max(max_seq, seq)
    return has_start, has_end, last_step, max_seq, planned


def find_streams(root: str, spec: RunSpec, job_id: Any) -> List[str]:
    """Streams under ``root`` belonging to ``spec`` and ``job_id``, newest last.

    Matching is on the *slugified job id* embedded in the ``exec_id``, never on
    the filename as text: :func:`telemetry.ids.make_exec_id` slugifies array ids
    (``123_4`` -> ``123-4``), so a naive string compare against ``SLURM_JOB_ID``
    would miss every array task.  Ordering is by attempt number so the caller can
    take the last element and get the execution that is dying right now, rather
    than an earlier requeued attempt that was already sealed.
    """
    events_dir = os.path.join(root, spec.slug(), "events")
    if not os.path.isdir(events_dir):
        return []
    try:
        wanted_job = make_exec_id(spec.run_uid, job_id, 0).split(":")[1]
    except SchemaError:
        return []
    found: List[Tuple[int, str]] = []
    for name in sorted(os.listdir(events_dir)):
        if not name.endswith(".jsonl"):
            continue
        try:
            run_uid, existing_job, attempt = split_exec_id(name[: -len(".jsonl")])
        except SchemaError:
            continue
        if run_uid != spec.run_uid or existing_job != wanted_job:
            continue
        found.append((attempt, os.path.join(events_dir, name)))
    return [path for _, path in sorted(found)]


def seal_stream(path: str, status: RunStatus, *, exit_code: int,
                signal_name: Optional[str], reason: str) -> str:
    """Append an inferred ``END`` to ``path``.  Returns an outcome word.

    Written with a raw append rather than through :class:`TelemetryWriter`
    because the writer takes an exclusive advisory lock and asserts single-writer
    discipline -- correct for a live run, wrong for a post-mortem sealer whose
    entire job is to write into a stream whose owner is gone.  The envelope is
    still built by :func:`telemetry.schema.make_record`, so the record is
    validated identically to one the run would have written.

    ``seq`` continues the stream's own numbering.  A gap would be indistinguishable
    from record loss, which is the one thing ``seq`` exists to make provable.
    """
    has_start, has_end, last_step, max_seq, planned = _scan(path)
    if has_end:
        return "already-sealed"
    if not has_start:
        # A stream with neither START nor END carries no claim to close.
        return "no-start"

    exec_id = os.path.basename(path)[: -len(".jsonl")]
    run_uid, _job, _attempt = split_exec_id(exec_id)
    # Truncation is a fact about the step axis, not about the exit code.  A job
    # that exited 0 having reached step 3,000 of a planned 20,000 (a `--time`
    # kill that the launcher happened to swallow) is truncated, and pooling it
    # with a complete run is the exact defect the END record exists to prevent.
    truncated = (status is not RunStatus.COMPLETED
                 or (planned is not None and last_step + 1 < planned))
    payload: Dict[str, Any] = {
        "status": RunStatus(status).value,
        "last_step": last_step,
        "planned_steps": planned,
        "truncated": truncated,
        "inferred": True,
        "inferred_by": "telemetry.seal",
        "inferred_reason": reason,
        "observed_exit_code": exit_code,
        "observed_signal": signal_name,
        "sealed_at_unix": time.time(),
    }
    record = make_record(run_uid=run_uid, exec_id=exec_id, seq=max_seq + 1,
                         event=EventType.END, payload=payload)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(dumps(record) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            # Network filesystems occasionally refuse fsync on append handles.
            # A failed sync is not a reason to report failure from a sealer.
            pass
    return "sealed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m telemetry.seal",
        description="Append an inferred END to an execution stream whose process "
                    "died without writing one.  Idempotent; always exits 0.")
    parser.add_argument("--root", required=True,
                        help="EQM_TELEMETRY_ROOT: the campaign telemetry root.")
    parser.add_argument("--run-spec", required=True,
                        help="EQM_RUN_SPEC: the canonical spec JSON blob.")
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID") or "local")
    parser.add_argument("--exit-code", type=int, default=0,
                        help="Wait status observed by the shell (128+N for signals).")
    parser.add_argument("--signal", default=None, dest="signal_name",
                        help="Signal the shell itself received, e.g. USR1. Wins over "
                             "--exit-code when set, since the shell observed it directly.")
    parser.add_argument("--status", default="auto",
                        help="'auto' to infer, or an explicit telemetry.RunStatus value.")
    parser.add_argument("--all-attempts", action="store_true",
                        help="Seal every unsealed attempt of this job id, not just the "
                             "latest.  Used by the reconciler, not by the sbatch trap.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    def say(message: str) -> None:
        if not args.quiet:
            print(f"[telemetry.seal] {message}", file=sys.stderr, flush=True)

    try:
        spec = RunSpec.from_dict(json.loads(args.run_spec))
    except Exception as exc:
        # An unparseable spec means we cannot even name the stream.  Report and
        # succeed: refusing to exit 0 here would turn a telemetry misconfiguration
        # into an apparent training failure.
        say(f"unusable --run-spec ({exc!r}); nothing sealed")
        return 0

    if args.status and args.status != "auto":
        try:
            status = RunStatus(args.status)
        except ValueError:
            say(f"unknown --status {args.status!r}; falling back to inference")
            status = infer_status(args.exit_code, args.signal_name)
    else:
        status = infer_status(args.exit_code, args.signal_name)

    streams = find_streams(args.root, spec, args.job_id)
    if not streams:
        # The job died before python opened a stream (module load failed, git
        # checkout failed, OOM during import).  There is nothing to close.
        say(f"no stream for {spec.run_uid} job {args.job_id}; nothing to seal")
        return 0

    targets = streams if args.all_attempts else streams[-1:]
    reason = (f"sealed by sbatch trap: exit_code={args.exit_code}"
              + (f" signal={args.signal_name}" if args.signal_name else ""))
    for path in targets:
        try:
            outcome = seal_stream(path, status, exit_code=args.exit_code,
                                  signal_name=args.signal_name, reason=reason)
        except Exception as exc:  # never propagate: see module docstring
            say(f"failed to seal {path}: {exc!r}")
            continue
        say(f"{outcome}: {os.path.basename(path)}"
            + (f" -> {status.value}" if outcome == "sealed" else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - the whole point is total containment
        print(f"[telemetry.seal] unexpected failure, ignored: {exc!r}",
              file=sys.stderr, flush=True)
        sys.exit(0)
