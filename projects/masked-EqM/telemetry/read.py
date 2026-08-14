"""The read side: loading, validating, and *gating* event logs before analysis.

This module exists to make one class of error impossible: silently aggregating a
run that is not comparable to the runs beside it.

The failure it replaces
-----------------------
The previous analyzer defined its analysis windows relative to each run's own
observed extent::

    lo, hi = rows[0]["step"], rows[-1]["step"]        # analyze_image.py
    edges = [lo + (hi - lo) * i / n for i in range(n + 1)]

and then grouped by ``window == "late"`` *across runs*.  For a campaign whose
entire hypothesis is about late-training behaviour, that means a run killed at
step 3,000 contributed its steps 2k-3k to the "late" column while a complete run
contributed its steps 13k-20k, printed side by side, indistinguishable.  Since
nothing recorded whether a run finished, there was no way to notice.

Two fixes, both enforced here rather than left to the analyst:

1. **Absolute windows.** Window edges are computed from the *planned* step axis,
   which is shared by construction across arms of an experiment.  "Late" means
   the same interval of training for everyone or the comparison is void.
2. **A completeness gate.** :class:`CompletenessPolicy` is a total predicate with
   an explicit reason for every rejection.  Runs that fail it are not dropped
   silently -- they are moved to a quarantine list that the report is expected to
   print, because "three of your six runs died" is itself a finding.

Detecting loss, not just absence
--------------------------------
Because every record carries a per-execution ``seq``, a gap in the sequence is
*proof* that records were lost -- something a timestamp-ordered log can never
establish.  A log with gaps is quarantined even if it has a clean ``END``: its
statistics are computed over an unknown subsample.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import glob
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ids import RunSpec, split_exec_id
from .schema import (
    SCHEMA_VERSION,
    EventType,
    RunStatus,
    SchemaError,
    TRUNCATED_STATUSES,
    decode_scalar,
    validate_record,
)


@dataclasses.dataclass
class ParseIssue:
    """One malformed or unreadable line, retained rather than discarded.

    A line that fails to parse is evidence about the run -- most often a torn
    final write from a hard kill -- and silently skipping it is how a truncated
    log passes for a complete one.
    """

    path: str
    lineno: int
    reason: str
    raw: str = ""


@dataclasses.dataclass
class RunLog:
    """One physical execution: its events, its lifecycle, and its defects."""

    path: str
    exec_id: str
    run_uid: str
    job_id: str
    attempt: int
    events: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    issues: List[ParseIssue] = dataclasses.field(default_factory=list)
    spec: Optional[RunSpec] = None

    # -- lifecycle facts ----------------------------------------------------

    @property
    def start(self) -> Optional[Dict[str, Any]]:
        for event in self.events:
            if event["event"] == EventType.START.value:
                return event
        return None

    @property
    def end(self) -> Optional[Dict[str, Any]]:
        # Scan backwards: END is by contract the last record, and on a
        # pathological double-seal we want the one that actually terminated.
        for event in reversed(self.events):
            if event["event"] == EventType.END.value:
                return event
        return None

    @property
    def status(self) -> Optional[RunStatus]:
        end = self.end
        if end is None:
            return RunStatus.RUNNING if self.start is not None else None
        try:
            return RunStatus(end.get("status"))
        except ValueError:
            return None

    @property
    def inferred_end(self) -> bool:
        """True when the terminal record was written by the sealer/reconciler.

        An inferred END is weaker evidence than a self-reported one: it says the
        scheduler observed the job stop, not that the process shut down
        cleanly. Analyses that care about the distinction can see it.
        """
        end = self.end
        return bool(end and end.get("inferred"))

    @property
    def planned_steps(self) -> Optional[int]:
        for event in (self.end, self.start):
            if event and event.get("planned_steps") is not None:
                return int(event["planned_steps"])
        return None

    @property
    def last_step(self) -> int:
        end = self.end
        if end and end.get("last_step") is not None:
            return int(end["last_step"])
        steps = [e["step"] for e in self.events if isinstance(e.get("step"), int)]
        return max(steps) if steps else -1

    @property
    def git_sha(self) -> Optional[str]:
        start = self.start
        return start.get("git_sha") if start else None

    @property
    def world_size(self) -> Optional[int]:
        start = self.start
        return start.get("world_size") if start else None

    @property
    def last_ts(self) -> Optional[str]:
        """Timestamp of the most recent event, for liveness checks."""
        return self.events[-1]["ts"] if self.events else None

    @property
    def age_seconds(self) -> Optional[float]:
        """Seconds since the last event.

        Distinguishes *alive but slow* from *died without sealing* -- two states
        that ``status`` alone conflates, since both present as a START with no
        END. A wedged job (this repo has lost a night to one deadlocked on a
        filesystem quota) is exactly a run whose age grows without bound.
        """
        stamp = self.last_ts
        if not stamp:
            return None
        try:
            when = _dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (_dt.datetime.now(_dt.timezone.utc) - when).total_seconds()

    @property
    def seq_gaps(self) -> List[Tuple[int, int]]:
        """Missing ``seq`` intervals -- positive proof of record loss."""
        seen = sorted({e["seq"] for e in self.events})
        gaps: List[Tuple[int, int]] = []
        expected = 0
        for value in seen:
            if value > expected:
                gaps.append((expected, value - 1))
            expected = value + 1
        return gaps

    @property
    def duplicate_seqs(self) -> List[int]:
        """Repeated ``seq`` values -- proof that two writers shared a stream."""
        counts: Dict[int, int] = defaultdict(int)
        for event in self.events:
            counts[event["seq"]] += 1
        return sorted(s for s, n in counts.items() if n > 1)

    # -- measurement access -------------------------------------------------

    def records(self, event: EventType, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """All records of one event type, optionally narrowed to one ``kind``.

        ``kind`` is the discriminator that keeps structurally different
        measurements from being pooled.  Callers that omit it are asking for a
        heterogeneous population and should know it; the aggregation helpers
        below all require it.
        """
        out = [e for e in self.events if e["event"] == event.value]
        if kind is not None:
            out = [e for e in out if e.get("kind") == kind]
        return sorted(out, key=lambda e: (e.get("step", -1), e["seq"]))

    def series(self, key: str, *, event: EventType = EventType.PROGRESS,
               kind: Optional[str] = None) -> List[Tuple[int, float]]:
        """``(step, value)`` pairs for one metric, nonfinite values dropped.

        Nonfinite values are *counted* by the caller via :meth:`nonfinite_count`
        rather than being silently absent: a NaN in a gradient norm is a fact
        about the run, and a median computed over the surviving finite values
        while pretending the NaNs did not exist is exactly the kind of quiet
        misreport this module exists to prevent.
        """
        out: List[Tuple[int, float]] = []
        for record in self.records(event, kind):
            if key not in record:
                continue
            value = decode_scalar(record[key])
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)) and math.isfinite(value):
                out.append((int(record.get("step", -1)), float(value)))
        return out

    def nonfinite_count(self, key: str, *, event: EventType = EventType.PROGRESS,
                        kind: Optional[str] = None) -> int:
        total = 0
        for record in self.records(event, kind):
            if key not in record:
                continue
            value = decode_scalar(record[key])
            if isinstance(value, float) and not math.isfinite(value):
                total += 1
        return total


@dataclasses.dataclass
class Rejection:
    """Why a run may not be aggregated.  Always surfaced, never swallowed."""

    exec_id: str
    reason: str
    detail: str = ""


@dataclasses.dataclass
class CompletenessPolicy:
    """The gate between "we have telemetry" and "we may compute a statistic".

    Every criterion is defeasible by explicit configuration, because some
    analyses legitimately want truncated runs (e.g. "how far did each arm get
    before dying?").  What is *not* negotiable is that relaxing a criterion is a
    visible act at the call site rather than an accident of a missing check.
    """

    require_start: bool = True
    require_end: bool = True
    require_status: Sequence[RunStatus] = (RunStatus.COMPLETED,)
    forbid_truncated: bool = True
    forbid_seq_gaps: bool = True
    forbid_duplicate_seqs: bool = True
    forbid_parse_issues: bool = True
    min_last_step: Optional[int] = None
    require_schema_version: int = SCHEMA_VERSION

    def evaluate(self, run: RunLog) -> Optional[Rejection]:
        """Return ``None`` if the run may be aggregated, else the reason."""
        if self.require_start and run.start is None:
            return Rejection(run.exec_id, "no_start",
                             "stream has no START record; the run's identity and "
                             "plan are unknown")
        if self.require_end and run.end is None:
            return Rejection(run.exec_id, "no_end",
                             "stream has no terminal record: this run either is "
                             "still going or died without being sealed. Its step "
                             "range is not comparable to a completed run's.")
        status = run.status
        if self.require_status and status is not None and status not in self.require_status:
            allowed = ", ".join(s.value for s in self.require_status)
            return Rejection(run.exec_id, f"status_{status.value}",
                             f"terminal status {status.value!r} is not in {{{allowed}}}")
        if self.forbid_truncated and status in TRUNCATED_STATUSES:
            return Rejection(run.exec_id, "truncated",
                             f"stopped at step {run.last_step} of "
                             f"{run.planned_steps}")
        if self.forbid_truncated and run.planned_steps is not None \
                and run.last_step + 1 < run.planned_steps:
            return Rejection(run.exec_id, "short",
                             f"reached step {run.last_step}, planned "
                             f"{run.planned_steps}")
        if self.min_last_step is not None and run.last_step < self.min_last_step:
            return Rejection(run.exec_id, "too_short",
                             f"last step {run.last_step} < required {self.min_last_step}")
        if self.forbid_seq_gaps and run.seq_gaps:
            spans = ", ".join(f"{a}-{b}" for a, b in run.seq_gaps[:5])
            return Rejection(run.exec_id, "seq_gaps",
                             f"records lost (missing seq {spans}); every statistic "
                             "over this stream is an unknown subsample")
        if self.forbid_duplicate_seqs and run.duplicate_seqs:
            return Rejection(run.exec_id, "duplicate_seqs",
                             f"duplicate seq {run.duplicate_seqs[:5]}: two writers "
                             "shared this stream")
        if self.forbid_parse_issues and run.issues:
            first = run.issues[0]
            return Rejection(run.exec_id, "parse_issues",
                             f"{len(run.issues)} unreadable line(s), first at "
                             f"{first.path}:{first.lineno}: {first.reason}")
        return None


#: Convenience policy for "show me everything, I know what I'm doing".
PERMISSIVE = CompletenessPolicy(
    require_end=False, require_status=(), forbid_truncated=False,
    forbid_seq_gaps=False, forbid_duplicate_seqs=False, forbid_parse_issues=False,
)


def load_run_log(path: str, spec: Optional[RunSpec] = None) -> RunLog:
    """Parse one ``<exec_id>.jsonl`` stream, retaining every defect found."""
    exec_id = os.path.basename(path)[: -len(".jsonl")]
    run_uid, job_id, attempt = split_exec_id(exec_id)
    log = RunLog(path=path, exec_id=exec_id, run_uid=run_uid, job_id=job_id,
                 attempt=attempt, spec=spec)
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = validate_record(json.loads(line))
            except json.JSONDecodeError as exc:
                # Overwhelmingly this is a torn final write from a hard kill.
                log.issues.append(ParseIssue(path, lineno, f"invalid JSON: {exc}",
                                             line[:200]))
                continue
            except SchemaError as exc:
                log.issues.append(ParseIssue(path, lineno, str(exc), line[:200]))
                continue
            if record["exec_id"] != exec_id:
                log.issues.append(ParseIssue(
                    path, lineno,
                    f"record belongs to {record['exec_id']}, not to this file"))
                continue
            log.events.append(record)
    log.events.sort(key=lambda e: e["seq"])
    return log


@dataclasses.dataclass
class LogicalRun:
    """All executions of one logical run (one ``run_uid``).

    A logical run may have several executions: a preemption and its rerun, a
    requeue, a manual resubmission.  Keeping them grouped -- rather than letting
    them look like unrelated runs, as the old path-keyed scheme did -- is what
    makes "this arm was attempted four times and completed once" a statable fact.
    """

    run_uid: str
    spec: Optional[RunSpec]
    directory: str
    executions: List[RunLog] = dataclasses.field(default_factory=list)
    #: The uid as literally recorded in spec.json, NOT recomputed from the spec
    #: fields. Keeping the stored literal is what makes `spec_is_authentic`
    #: meaningful: if the uid were re-derived from the (possibly edited) fields,
    #: the hash check would compare a value against itself and could never fail.
    minted_run_uid: Optional[str] = None
    #: Raw spec.json contents, retained because `RunSpec.from_dict` drops keys it
    #: does not model (provenance, legacy_sources) that consumers still need.
    spec_raw: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def provenance(self) -> str:
        """``native`` for a run that instrumented itself, else what migrate set.

        A reconstructed run must never be silently indistinguishable from one
        that recorded its own telemetry.
        """
        return str(self.spec_raw.get("provenance") or "native")

    @property
    def spec_is_authentic(self) -> Optional[bool]:
        """Does the stored uid match the hash of the stored spec?

        ``None`` when unknowable (no stored uid). ``False`` is proof that the
        spec was edited after minting -- e.g. a results row hand-corrected to say
        ``fd_k: 4`` on a run that actually executed ``fd_k: 1``.
        """
        if not self.minted_run_uid or self.spec is None:
            return None
        return self.minted_run_uid == self.spec.run_uid

    @property
    def arm(self) -> str:
        if self.spec:
            return self.spec.arm
        start = self.canonical.start if self.canonical else None
        return (start or {}).get("arm", "unknown")

    @property
    def seed(self) -> Any:
        if self.spec:
            return self.spec.seed
        start = self.canonical.start if self.canonical else None
        return (start or {}).get("seed")

    @property
    def canonical(self) -> Optional[RunLog]:
        """The execution that represents this run.

        Preference order: a COMPLETED execution, else the one that reached the
        furthest step, tie-broken by the latest attempt.  Deliberately NOT
        "concatenate all attempts": merging a preempted attempt with its rerun
        produces a step sequence that never happened, which is precisely the
        artifact the old append-mode metrics file manufactured.
        """
        if not self.executions:
            return None
        completed = [e for e in self.executions if e.status == RunStatus.COMPLETED]
        pool = completed or self.executions
        return max(pool, key=lambda e: (e.last_step, e.attempt))

    def analyzable(self, policy: Optional[CompletenessPolicy] = None
                   ) -> Tuple[Optional[RunLog], Optional[Rejection]]:
        policy = policy or CompletenessPolicy()
        best: Optional[Rejection] = None
        for execution in sorted(self.executions,
                                key=lambda e: (e.last_step, e.attempt), reverse=True):
            rejection = policy.evaluate(execution)
            if rejection is None:
                return execution, None
            best = best or rejection
        return None, best or Rejection(self.run_uid, "no_executions",
                                       "run directory contains no event streams")


@dataclasses.dataclass
class Campaign:
    """Every logical run under a telemetry root."""

    root: str
    runs: List[LogicalRun] = dataclasses.field(default_factory=list)

    def by_uid(self, run_uid: str) -> Optional[LogicalRun]:
        for run in self.runs:
            if run.run_uid == run_uid:
                return run
        return None

    def select(self, *, campaign: Optional[str] = None, phase: Optional[str] = None,
               arm: Optional[str] = None, seed: Optional[int] = None,
               git_sha: Optional[str] = None) -> List[LogicalRun]:
        filters = (campaign, phase, arm, seed, git_sha)

        def matches(run: LogicalRun) -> bool:
            spec = run.spec
            if spec is None:
                # A run whose spec failed to load is exactly what an operator
                # needs to SEE during an inventory. Excluding it from an
                # unfiltered listing hides the broken runs from the tool whose
                # job is to find them. It is still excluded once any filter is
                # set, since an unreadable spec cannot satisfy a predicate.
                return all(f is None for f in filters)
            return all([
                campaign is None or spec.campaign == campaign,
                phase is None or spec.phase == phase,
                arm is None or spec.arm == arm,
                seed is None or int(spec.seed) == int(seed),
                git_sha is None or spec.git_sha == git_sha,
            ])
        return [r for r in self.runs if matches(r)]

    def partition(self, policy: Optional[CompletenessPolicy] = None
                  ) -> Tuple[List[Tuple[LogicalRun, RunLog]], List[Tuple[LogicalRun, Rejection]]]:
        """Split into (analyzable, quarantined).

        Both halves are returned because a report that prints only the first is
        lying by omission: how many runs died, and how, is part of the result.
        """
        good: List[Tuple[LogicalRun, RunLog]] = []
        bad: List[Tuple[LogicalRun, Rejection]] = []
        for run in self.runs:
            execution, rejection = run.analyzable(policy)
            if execution is not None:
                good.append((run, execution))
            else:
                bad.append((run, rejection))  # type: ignore[arg-type]
        return good, bad


def load_campaign(root: str) -> Campaign:
    """Load every run directory beneath ``root``."""
    campaign = Campaign(root=root)
    for spec_path in sorted(glob.glob(os.path.join(root, "*", "spec.json"))):
        directory = os.path.dirname(spec_path)
        try:
            with open(spec_path, encoding="utf-8") as handle:
                spec_data = json.load(handle)
            spec = RunSpec.from_dict(spec_data)
        except (json.JSONDecodeError, KeyError, OSError):
            spec, spec_data = None, {}
        # Prefer the STORED uid over the recomputed one. Recomputing would make
        # `spec_is_authentic` compare a value against itself, silently disabling
        # the only check that detects a spec edited after the fact.
        minted = spec_data.get("run_uid")
        run_uid = str(minted or (spec.run_uid if spec else
                                 os.path.basename(directory).rsplit("__", 1)[-1]))
        logical = LogicalRun(run_uid=run_uid, spec=spec, directory=directory,
                             minted_run_uid=str(minted) if minted else None,
                             spec_raw=dict(spec_data))
        for stream in sorted(glob.glob(os.path.join(directory, "events", "*.jsonl"))):
            try:
                logical.executions.append(load_run_log(stream, spec))
            except SchemaError:
                # A file whose NAME is not a valid exec_id is not one of ours.
                continue
        campaign.runs.append(logical)
    return campaign


# -- windows ----------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Window:
    name: str
    lo: int
    hi: int

    def contains(self, step: int) -> bool:
        return self.lo <= step <= self.hi


def absolute_windows(planned_steps: int, n: int = 3,
                     names: Sequence[str] = ("early", "mid", "late")) -> List[Window]:
    """Fixed step windows on the PLANNED axis, identical for every arm.

    This is the load-bearing difference from the previous implementation.  Edges
    depend only on ``planned_steps`` -- a property of the experiment design, not
    of how far any particular run happened to get -- so "late" denotes the same
    interval of training for every run being compared.  Two runs with different
    ``planned_steps`` therefore have different windows, which is correct and is
    exactly why :func:`shared_windows` refuses to compare them.
    """
    if planned_steps <= 0:
        raise ValueError(f"planned_steps must be positive, got {planned_steps}")
    if n < 1:
        raise ValueError("n must be >= 1")
    labels = list(names) if len(names) == n else [f"w{i}" for i in range(n)]
    edges = [round(planned_steps * i / n) for i in range(n + 1)]
    return [Window(labels[i], edges[i], max(edges[i], edges[i + 1] - 1))
            for i in range(n)]


def shared_windows(runs: Iterable[RunLog], n: int = 3) -> List[Window]:
    """Windows valid for a whole comparison group, or an error.

    Refuses to invent a common axis when the runs were not planned to the same
    length: silently reconciling that difference is how non-comparable runs end
    up in one table.  If you genuinely want to compare a 15-epoch run to an
    80-epoch one, truncate explicitly and say so.
    """
    planned = {r.planned_steps for r in runs if r.planned_steps is not None}
    if not planned:
        raise ValueError(
            "no run declares planned_steps, so no absolute window axis exists; "
            "these runs cannot be compared window-by-window")
    if len(planned) > 1:
        raise ValueError(
            f"runs were planned to different lengths ({sorted(planned)}); a shared "
            "'late' window does not exist. Truncate to a common horizon "
            "explicitly if that is what you intend.")
    return absolute_windows(planned.pop(), n=n)


# -- aggregation -------------------------------------------------------------

@dataclasses.dataclass
class Summary:
    """A statistic plus everything needed to judge whether to believe it."""

    n: int
    median: float
    p95: float
    mean: float
    minimum: float
    maximum: float
    nonfinite: int
    stdev: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def summarize(values: Sequence[float], nonfinite: int = 0) -> Optional[Summary]:
    """Summarize a sample, or ``None`` if it is empty.

    ``None`` rather than NaN: a NaN propagates into a printed table and reads as
    a measured value, while ``None`` forces the caller to decide what an absent
    sample means.  ``n`` and ``nonfinite`` travel with every statistic so a
    median over three points is never mistaken for a median over three hundred.
    """
    finite = [float(v) for v in values if math.isfinite(v)]
    if not finite:
        return None
    ordered = sorted(finite)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return Summary(
        n=len(ordered),
        median=statistics.median(ordered),
        p95=ordered[index],
        mean=statistics.fmean(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        nonfinite=nonfinite,
        stdev=statistics.stdev(ordered) if len(ordered) > 1 else None,
    )


def windowed(run: RunLog, key: str, windows: Sequence[Window], *,
             event: EventType = EventType.PROGRESS,
             kind: Optional[str] = None) -> Dict[str, Optional[Summary]]:
    """Summarize one metric per absolute window.

    ``kind`` is threaded through so a caller cannot accidentally pool two
    different measurement populations -- the defect that made the old
    ``update_over_param`` a ratio of medians taken over disjoint record types.
    """
    pairs = run.series(key, event=event, kind=kind)
    nonfinite = run.nonfinite_count(key, event=event, kind=kind)
    out: Dict[str, Optional[Summary]] = {}
    for window in windows:
        selected = [v for step, v in pairs if window.contains(step)]
        out[window.name] = summarize(selected, nonfinite if window is windows[-1] else 0)
    return out


def rate(run: RunLog, key: str, windows: Sequence[Window], *,
         kind: Optional[str] = None) -> Dict[str, Optional[float]]:
    """Fraction of records in each window for which a boolean flag is true.

    Separate from :func:`windowed` because a rate over booleans and a median over
    reals are different questions, and computing the former with the latter's
    machinery (as ``clip_rate_pct`` did, via ``sum(bool(c))/len(c)`` on a
    heterogeneous population) invites exactly the pooling error above.
    """
    out: Dict[str, Optional[float]] = {}
    for window in windows:
        flags = [bool(decode_scalar(r[key]))
                 for r in run.records(EventType.PROGRESS, kind)
                 if key in r and window.contains(int(r.get("step", -1)))]
        out[window.name] = (100.0 * sum(flags) / len(flags)) if flags else None
    return out


def controlled_comparison(a: LogicalRun, b: LogicalRun) -> Dict[str, Any]:
    """Check that two runs differ in exactly the field under test.

    An A/B claim is sound only if the two arms' specs are identical apart from
    the manipulated variable.  Previously that was asserted in prose in a
    launcher docstring; here it is computed from the specs the runs actually
    ran with, so a comparison silently confounded by a changed git sha or a
    different batch size is detectable rather than assumed away.
    """
    from .ids import differing_fields

    if a.spec is None or b.spec is None:
        return {"controlled": False,
                "reason": "one or both runs lack a recorded spec"}
    diff = differing_fields(a.spec, b.spec)
    return {
        "controlled": len(diff) <= 1,
        "differing_fields": diff,
        "reason": ("" if len(diff) <= 1 else
                   f"{len(diff)} fields differ; this comparison is confounded"),
    }
