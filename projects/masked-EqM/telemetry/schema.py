"""Event schema for masked-EqM run telemetry.

The whole telemetry design rests on one idea: **the append-only event log is the
only source of truth, and every state document is a fold over it.**  A mutable
"status" field that some writer overwrites destroys history and can silently
disagree with reality (this is exactly what `results/btm/manifest.jsonl` did --
it froze `status="submitted"` forever).  An append-only log of typed events with
a total order per execution cannot lie about the past; the worst it can do is be
incomplete, and incompleteness is *detectable* (see `read.py`).

Three concepts, kept strictly separate:

``run_uid``
    Identity of a LOGICAL experiment: the content hash of the immutable
    specification (campaign, phase, arm, seed, git sha, and every parameter that
    changes what is computed).  Deterministic, so re-submitting the identical
    spec yields the identical ``run_uid`` and the two are joinable by
    construction.  This is the key you compare arms on.

``exec_id``
    Identity of a PHYSICAL execution of that spec: ``run_uid`` + scheduler job
    id + attempt counter.  A requeue, a resume-from-checkpoint, or a manual
    resubmission produces a NEW ``exec_id`` under the SAME ``run_uid``.  This is
    the key that owns a lifecycle.  Invalidation ("that run was contaminated")
    applies to executions, not to logical runs.

``seq``
    A monotone counter, starting at 0, per ``exec_id``.  Gives every event a
    total order independent of clock skew, and -- critically -- makes RECORD
    LOSS DETECTABLE: a gap in ``seq`` is proof that events were dropped, which a
    timestamp-ordered log can never establish.

Every emitted line carries the full envelope.  That is a few dozen redundant
bytes per record, and it buys the property that any single line is
self-identifying: no consumer ever has to recover identity by parsing a
filesystem path (the previous design's fatal flaw -- path parsing is a lossy,
non-injective decoding that silently collapsed distinct arms together).
"""

from __future__ import annotations

import datetime as _dt
import enum
import json
import math
import re
from typing import Any, Dict, Iterable, Mapping, Optional

# Bump ONLY on a breaking change to the envelope.  Consumers refuse to
# aggregate logs whose major version they do not understand, rather than
# silently misreading them.
SCHEMA_VERSION = 1

# Envelope keys.  Every record has exactly these, plus its payload.
ENVELOPE_KEYS = ("v", "ts", "run_uid", "exec_id", "seq", "event")

_RUN_UID_RE = re.compile(r"^r[0-9a-f]{16}$")
_EXEC_ID_RE = re.compile(r"^r[0-9a-f]{16}:[A-Za-z0-9_.-]+:a\d+$")


class EventType(str, enum.Enum):
    """The closed set of event types.

    Kept deliberately small.  A new *measurement* is a new payload key on
    ``PROGRESS`` or ``EVAL``, never a new event type; a new event type is only
    justified by a new position in the lifecycle FSM.
    """

    #: Execution has begun.  Carries the fully resolved config, the environment
    #: it actually resolved to (world size, device count, git sha as seen at
    #: runtime), and the plan (``planned_steps``).  Exactly one per exec_id.
    START = "START"

    #: A per-step measurement stream.  Discriminated by the ``kind`` payload key
    #: so that structurally different measurements sharing a stream remain
    #: separable -- the previous design multiplexed two record shapes onto one
    #: file with no discriminator, which inflated sample counts and made
    #: cross-population ratios look like within-population ones.
    PROGRESS = "PROGRESS"

    #: A periodic but expensive evaluation (held-out probes, target-match,
    #: sampling, FID).  Separated from PROGRESS because its cadence differs and
    #: because aggregations almost always want one or the other, never both
    #: pooled.
    EVAL = "EVAL"

    #: A checkpoint was written.  Ties an artifact on disk to the exact step and
    #: execution that produced it, which is otherwise only recoverable by
    #: guessing from a filename.
    ARTIFACT = "ARTIFACT"

    #: Something noteworthy but non-terminal: a caught exception that was
    #: retried, a nonfinite gradient that was skipped, a scheduler preemption
    #: warning, a config coercion.  Carries ``level`` and ``message``.
    NOTICE = "NOTICE"

    #: A scheduler-observed or externally-observed state transition, appended by
    #: the reconciler rather than by the run itself.
    OBSERVED = "OBSERVED"

    #: Execution has ended.  Exactly one per exec_id, and it is the record whose
    #: absence means "this run's telemetry is not trustworthy".  Carries
    #: ``status``, ``last_step``, ``planned_steps``, wall time, peak memory and
    #: -- when the run died -- the exception.
    END = "END"


class RunStatus(str, enum.Enum):
    """Terminal (and pre-terminal) states of one execution.

    The distinctions matter for analysis, not just for bookkeeping:
    ``COMPLETED`` runs may be compared to each other; ``TIMEOUT`` and
    ``PREEMPTED`` runs are *truncated* and their late-training windows are not
    comparable to a complete run's; ``CRASHED`` runs may additionally be
    corrupted before the crash.
    """

    RUNNING = "running"

    #: Reached the planned step count and exited cleanly.
    COMPLETED = "completed"

    #: Raised an exception out of the training loop.
    CRASHED = "crashed"

    #: Received SIGTERM from the scheduler (preemption or `scancel`).
    PREEMPTED = "preempted"

    #: Hit the wall-clock limit (SIGUSR1 / SIGTERM at the time limit).
    TIMEOUT = "timeout"

    #: Explicitly cancelled by a human.
    CANCELLED = "cancelled"

    #: The process vanished without emitting END (SIGKILL, node failure, OOM
    #: killer).  Never written by the run itself -- by definition it could not.
    #: Written by the shell-level sealer or the reconciler, always with
    #: ``inferred: true``.
    LOST = "lost"


#: Statuses after which no further events may be appended for that exec_id.
TERMINAL_STATUSES = frozenset({
    RunStatus.COMPLETED, RunStatus.CRASHED, RunStatus.PREEMPTED,
    RunStatus.TIMEOUT, RunStatus.CANCELLED, RunStatus.LOST,
})

#: Statuses that leave the step axis truncated: a run in one of these states
#: stopped early, so its "late" window is NOT the same region of training as a
#: completed run's.  Aggregators must not pool these with COMPLETED runs.
TRUNCATED_STATUSES = frozenset({
    RunStatus.CRASHED, RunStatus.PREEMPTED, RunStatus.TIMEOUT,
    RunStatus.CANCELLED, RunStatus.LOST,
})


class SchemaError(ValueError):
    """Raised when a record violates the envelope contract.

    Deliberately an error and not a warning.  The failure mode this whole
    module exists to eliminate is *silent* corruption of an aggregate; a loud
    exception at write time or load time is strictly preferable to a plausible
    wrong number in a results table.
    """


def utc_now_iso() -> str:
    """Timestamp in RFC3339 UTC with millisecond precision.

    Millisecond precision, not microsecond: it is enough to order events across
    processes and it keeps the envelope narrow.  Ordering WITHIN an execution
    never relies on this field -- that is what ``seq`` is for.
    """
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    """Coerce a payload value into something ``json.dumps`` accepts losslessly.

    Two hazards this closes:

    * NaN/Inf. ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens, which are
      invalid JSON and which ``json.loads`` accepts by default -- so a NaN
      written by one tool and read by a stricter parser elsewhere blows up far
      from its origin.  We encode them as the strings ``"NaN"``/``"Infinity"``
      and decode them back in ``read.py``, so nonfinite values survive the round
      trip *and* remain visible as anomalies rather than vanishing.
    * numpy / torch scalars.  These serialize inconsistently across versions;
      anything exposing ``.item()`` is reduced to a python scalar first.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        try:
            return _json_safe(value.item())
        except Exception:  # pragma: no cover - exotic tensor types
            return repr(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def decode_scalar(value: Any) -> Any:
    """Inverse of the nonfinite encoding in :func:`_json_safe`."""
    if value == "NaN":
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    return value


def validate_run_uid(run_uid: str) -> str:
    if not isinstance(run_uid, str) or not _RUN_UID_RE.match(run_uid):
        raise SchemaError(f"malformed run_uid: {run_uid!r} (expected r<16 hex>)")
    return run_uid


def validate_exec_id(exec_id: str) -> str:
    if not isinstance(exec_id, str) or not _EXEC_ID_RE.match(exec_id):
        raise SchemaError(
            f"malformed exec_id: {exec_id!r} (expected <run_uid>:<job>:a<attempt>)")
    return exec_id


def make_record(
    *,
    run_uid: str,
    exec_id: str,
    seq: int,
    event: EventType,
    payload: Optional[Mapping[str, Any]] = None,
    ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one validated envelope+payload record.

    Payload keys may not shadow envelope keys; that would make the record
    ambiguous to a consumer and is rejected rather than silently resolved in
    either direction.
    """
    if not isinstance(seq, int) or seq < 0:
        raise SchemaError(f"seq must be a non-negative int, got {seq!r}")
    event = EventType(event)
    record: Dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "ts": ts or utc_now_iso(),
        "run_uid": validate_run_uid(run_uid),
        "exec_id": validate_exec_id(exec_id),
        "seq": seq,
        "event": event.value,
    }
    for key, value in (payload or {}).items():
        key = str(key)
        if key in ENVELOPE_KEYS:
            raise SchemaError(
                f"payload key {key!r} collides with a reserved envelope key")
        record[key] = _json_safe(value)
    return record


def validate_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a record read back from disk.

    Returns a plain dict.  Raises :class:`SchemaError` with a specific reason --
    consumers surface that reason to the user rather than dropping the line,
    because a line that fails to parse is evidence about the run, not noise.
    """
    if not isinstance(record, Mapping):
        raise SchemaError(f"record is not an object: {type(record).__name__}")
    missing = [k for k in ENVELOPE_KEYS if k not in record]
    if missing:
        raise SchemaError(f"record missing envelope keys: {', '.join(missing)}")
    version = record["v"]
    if not isinstance(version, int):
        raise SchemaError(f"non-integer schema version: {version!r}")
    if version > SCHEMA_VERSION:
        raise SchemaError(
            f"record schema v{version} is newer than this reader (v{SCHEMA_VERSION}); "
            "upgrade the analysis code rather than reading it partially")
    validate_run_uid(record["run_uid"])
    validate_exec_id(record["exec_id"])
    if not isinstance(record["seq"], int) or record["seq"] < 0:
        raise SchemaError(f"bad seq: {record['seq']!r}")
    try:
        EventType(record["event"])
    except ValueError:
        raise SchemaError(f"unknown event type: {record['event']!r}") from None
    if record["exec_id"].split(":", 1)[0] != record["run_uid"]:
        raise SchemaError(
            f"exec_id {record['exec_id']!r} does not belong to run_uid "
            f"{record['run_uid']!r} -- a record cannot be attributed")
    return dict(record)


def dumps(record: Mapping[str, Any]) -> str:
    """Serialize one record to a single JSON line.

    ``sort_keys`` makes the on-disk bytes a deterministic function of the
    content, so identical records diff identically and a log can be compared
    across machines.  ``allow_nan=False`` is the enforcement half of the
    nonfinite encoding above: it turns "a NaN leaked through un-encoded" into an
    immediate error at the write site instead of invalid JSON on disk.
    """
    return json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))


def required_start_fields() -> Iterable[str]:
    """Payload keys a START record must carry to be analyzable.

    These are precisely the facts that a downstream analyst currently cannot
    recover from a finished run's artifacts: what commit ran, on how many ranks,
    with what arm/seed, and how many steps it was *supposed* to do.  The last
    one is what makes truncation detectable.
    """
    return (
        "campaign", "arm", "seed", "git_sha", "planned_steps",
        "world_size", "job_id", "attempt", "config",
    )
