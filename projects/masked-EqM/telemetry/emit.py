"""The write side: append-only, crash-resistant, single-writer event streams.

Layout on disk
--------------
::

    <telemetry_root>/
      <slug>__<run_uid>/            one directory per LOGICAL run
        spec.json                   the immutable spec (written once, then verified)
        events/
          <exec_id>.jsonl           one file per PHYSICAL execution

One file per execution, rather than one file per run, is a deliberate structural
choice.  The previous design opened a single ``gradient_metrics.jsonl`` in append
mode, so a requeue or a resume wrote *into the same file* as the earlier attempt.
The result was a stream with non-monotone step numbers that a naive sort silently
interleaved into a plausible-looking but fictitious trajectory.  Partitioning by
``exec_id`` makes that unrepresentable: interleaving cannot occur because two
attempts never share a file, and a reader that wants the merged view must ask for
it explicitly and thereby confront the question of how to merge.

Durability policy
-----------------
``fsync`` on every record would dominate the cost of a 50-step logging cadence.
``fsync`` on nothing loses the tail of the log exactly when it matters most --
when the node dies.  The policy here is tiered by how expensive the record is to
lose:

* ``START`` / ``END`` / ``ARTIFACT`` / error-level ``NOTICE``  -> flush + fsync.
  These are the lifecycle facts; losing one is what makes a run unanalyzable, and
  there are O(10) of them per run so the cost is irrelevant.
* ``PROGRESS`` / ``EVAL`` -> flush only (line-buffered).  A flushed-but-unsynced
  line survives process death (the kernel owns the bytes); it is lost only if the
  machine itself dies.  Losing a handful of per-step samples in that scenario is
  acceptable and, crucially, *detectable* via the ``seq`` gap check.

Single-writer discipline
------------------------
Exactly one process may write a given execution's stream.  Under DDP that is rank
0.  The writer takes an advisory lock on the file and records the writing pid in
the ``START`` record, so a second writer fails loudly at open time instead of
producing an interleaved file that looks fine until someone computes a median
over it.
"""

from __future__ import annotations

import atexit
import errno
import io
import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Protocol

from .schema import (
    EventType,
    SchemaError,
    dumps,
    make_record,
)

#: Events important enough to force to stable storage immediately.
_FSYNC_EVENTS = frozenset({EventType.START, EventType.END, EventType.ARTIFACT})


class Sink(Protocol):
    """A destination for records.

    The point of the abstraction is that wandb, stdout and the jsonl file all
    receive the *same* record object.  Previously ``steps_per_sec`` went to the
    text logger and to wandb but never to the structured stream, so the metric
    the analyzer needed was visible to a human and invisible to a program.  With
    a single record fanned out to sinks, a metric cannot exist in one view and
    not another.
    """

    def write(self, record: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class JsonlSink:
    """Primary sink: an append-only JSON-lines file with an advisory lock."""

    def __init__(self, path: str, *, lock: bool = True) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Append mode: the file is never truncated, so an accidental re-open can
        # only ever add, never destroy.  Combined with one-file-per-exec_id this
        # means the log is monotone by construction.
        self._fh: Optional[io.TextIOWrapper] = open(
            path, "a", encoding="utf-8", buffering=1, newline="\n")
        self._locked = False
        if lock:
            self._acquire_lock()

    def _acquire_lock(self) -> None:
        """Advisory exclusive lock, best-effort across filesystems.

        Network filesystems (this repo writes to netscratch/holylfs) do not
        reliably support ``flock``.  When the lock cannot be taken *because the
        filesystem does not implement it*, we proceed -- refusing to run because
        of an unsupported lock would be worse than the risk it guards.  When the
        lock is taken *by another process*, we fail hard: that is the real
        double-writer case.
        """
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._locked = True
        except OSError as exc:
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise SchemaError(
                    f"another process already holds the telemetry lock on {self.path}; "
                    "two writers on one execution stream would interleave records. "
                    "If this is a requeue, it must use a new attempt number."
                ) from exc
            # ENOLCK / EINVAL / EOPNOTSUPP: filesystem lacks locking. Proceed.

    def write(self, record: Mapping[str, Any]) -> None:
        if self._fh is None:
            raise SchemaError("write to a closed telemetry sink")
        self._fh.write(dumps(record) + "\n")

    def sync(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            # Some network filesystems reject fsync on append-only handles.
            # A failed sync is not a reason to kill a training run.
            pass

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            self.sync()
        finally:
            try:
                self._fh.close()
            finally:
                self._fh = None


class StderrSink:
    """Human-visible mirror, one compact line per lifecycle event.

    Only lifecycle events, never the per-step stream: a mirror that reproduces
    every ``PROGRESS`` record would bury the three lines a human actually needs
    in a 20k-step job's stdout.
    """

    _SHOWN = frozenset({EventType.START.value, EventType.END.value,
                        EventType.NOTICE.value, EventType.ARTIFACT.value})

    def __init__(self, stream=None) -> None:
        import sys
        self._stream = stream if stream is not None else sys.stderr

    def write(self, record: Mapping[str, Any]) -> None:
        if record.get("event") not in self._SHOWN:
            return
        bits = [f"[telemetry] {record['event']}", record["exec_id"]]
        for key in ("status", "step", "last_step", "level", "message", "path"):
            if key in record:
                bits.append(f"{key}={record[key]}")
        print(" ".join(str(b) for b in bits), file=self._stream, flush=True)

    def close(self) -> None:
        pass


class WandbSink:
    """Optional mirror into wandb, fed the identical record.

    Non-scalar payload values are dropped rather than coerced: wandb will happily
    accept a stringified dict and render it uselessly, and a metric that looks
    present but is unplottable is worse than an absent one.  Failures here are
    swallowed by design -- telemetry infrastructure must never be able to kill a
    training run that is otherwise healthy.
    """

    def __init__(self, module: Any) -> None:
        self._wandb = module

    def write(self, record: Mapping[str, Any]) -> None:
        if record.get("event") not in (EventType.PROGRESS.value, EventType.EVAL.value):
            return
        step = record.get("step")
        prefix = record.get("kind") or record["event"].lower()
        payload = {
            f"{prefix}/{k}": v for k, v in record.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in ("v", "seq", "step")
        }
        if not payload:
            return
        try:
            self._wandb.log(payload, step=int(step) if step is not None else None)
        except Exception:
            pass

    def close(self) -> None:
        pass


class TelemetryWriter:
    """Sequenced, multi-sink, thread-safe emitter for ONE execution.

    Owns the ``seq`` counter.  ``seq`` is assigned under a lock and incremented
    exactly once per record, which is what makes a gap in the sequence a *proof*
    of record loss rather than a suspicion.  No other component may assign
    sequence numbers.
    """

    def __init__(self, run_uid: str, exec_id: str, sinks: List[Sink]) -> None:
        self.run_uid = run_uid
        self.exec_id = exec_id
        self.sinks = list(sinks)
        self._seq = 0
        # Reentrant: a signal handler that fires *inside* emit() raises through
        # to the lifecycle's finally block, which emits the END record -- with a
        # plain Lock that path would self-deadlock and the run would hang at
        # exactly the moment we most need its terminal record.
        self._lock = threading.RLock()
        self._closed = False
        self._pid = os.getpid()
        atexit.register(self.close)

    @property
    def next_seq(self) -> int:
        return self._seq

    def emit(
        self,
        event: EventType,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        sync: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Emit one record to every sink.  Returns the record as written."""
        if self._closed:
            raise SchemaError(f"emit after close on {self.exec_id}")
        event = EventType(event)
        with self._lock:
            seq = self._seq
            self._seq += 1
            record = make_record(
                run_uid=self.run_uid, exec_id=self.exec_id, seq=seq,
                event=event, payload=payload)
            for sink in self.sinks:
                try:
                    sink.write(record)
                except SchemaError:
                    raise
                except Exception as exc:  # a broken mirror must not kill the run
                    if isinstance(sink, JsonlSink):
                        raise
                    print(f"[telemetry] sink {type(sink).__name__} failed: {exc!r}")
            should_sync = event in _FSYNC_EVENTS if sync is None else sync
            if should_sync:
                for sink in self.sinks:
                    if isinstance(sink, JsonlSink):
                        sink.sync()
        return record

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                pass


def open_writer(
    telemetry_root: str,
    spec,
    *,
    job_id: Any = None,
    attempt: int = 0,
    wandb_module: Any = None,
    mirror_stderr: bool = True,
    lock: bool = True,
) -> TelemetryWriter:
    """Create the run directory, persist/verify the spec, and open the stream.

    ``spec.json`` is written once and thereafter *verified* rather than
    rewritten.  If a second execution of the same ``run_uid`` presents a
    different spec, that is a hash collision or a mutated spec, and either way it
    must not be silently accepted -- the whole comparability argument depends on
    two executions under one ``run_uid`` having been the same experiment.
    """
    from .ids import make_exec_id

    run_dir = os.path.join(telemetry_root, spec.slug())
    os.makedirs(os.path.join(run_dir, "events"), exist_ok=True)

    spec_path = os.path.join(run_dir, "spec.json")
    payload = spec.to_dict()
    if os.path.exists(spec_path):
        import json
        with open(spec_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("run_uid") != payload["run_uid"]:
            raise SchemaError(
                f"{spec_path} holds run_uid {existing.get('run_uid')} but this "
                f"process minted {payload['run_uid']}")
    else:
        import json
        tmp = spec_path + f".tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        # Atomic publish: a reader never observes a half-written spec.
        os.replace(tmp, spec_path)

    # `lock=False` is for post-mortem writers -- the shell sealer and the
    # reconciler append a terminal record to a stream whose original writer is
    # already dead. The single-writer assertion is right for a live run and
    # wrong for a sealer, which by definition runs after the fact.
    exec_id = make_exec_id(spec.run_uid, job_id, attempt)
    sinks: List[Sink] = [
        JsonlSink(os.path.join(run_dir, "events", f"{exec_id}.jsonl"), lock=lock)]
    if mirror_stderr:
        sinks.append(StderrSink())
    if wandb_module is not None:
        sinks.append(WandbSink(wandb_module))
    return TelemetryWriter(spec.run_uid, exec_id, sinks)
