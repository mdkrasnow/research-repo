"""Backfill: legacy provenance artifacts -> unified telemetry event logs.

What this tool is for
---------------------
The new telemetry system (:mod:`telemetry.schema`, :mod:`telemetry.ids`,
:mod:`telemetry.emit`, :mod:`telemetry.lifecycle`) can only describe runs that
were launched through it.  Everything this project has produced so far predates
it and survives only as six mutually inconsistent hand- and code-maintained
ledgers.  This module reads those ledgers and re-expresses what they assert in
the new schema, so that historical runs become queryable by the same code as
future ones -- **without ever claiming more about them than the ledgers actually
support.**

Three rules that shape every design decision below
--------------------------------------------------

1.  **Read-only with respect to history.**  No legacy file is opened for
    writing, moved, or deleted.  Output lands under a new root
    (``results/telemetry/`` by default) which the migrator owns entirely.

2.  **A reconstruction must never be mistakable for a measurement.**  Every run
    emitted here carries ``provenance: "reconstructed"``, a ``confidence``
    level, and an explicit ``unknown_fields`` list, in its ``spec.json``, in its
    ``START``-position event, and in the migration report.  A field the ledger
    does not contain is written as the sentinel
    :data:`telemetry.legacy.UNKNOWN`, never inferred from a similar run.

3.  **Ledger facts are ``OBSERVED``, not ``START``/``END``.**  ``START`` means
    "the process said it was starting"; ``END`` means "the process (or a sealer
    watching it) said it stopped".  Neither is available for a run that finished
    before this system existed.  What *is* available is a third-party assertion
    -- "a ledger says job 35436507 failed" -- and ``OBSERVED`` is exactly the
    event type for that.  A terminal ``END`` is synthesized only when the ledger
    states a terminal status, and always with ``inferred: true``.

Where the reconstruction deliberately stops
-------------------------------------------
* A job whose sources **disagree** about its terminal status gets **no ``END``
  at all**, plus an error-level ``NOTICE`` naming both claims.  Under the
  lifecycle contract a missing ``END`` means "this run's telemetry is not
  trustworthy" -- which is precisely, and correctly, the epistemic state of a
  job that one ledger calls ``failed`` and another calls ``completed``.
  Choosing a winner would manufacture certainty that does not exist.
* Legacy statuses that are not lifecycle states at all (``superseded``,
  ``"INVALID -- discarded"``) never become an ``END``.  They are administrative
  annotations about a record, not observations of a process.
* A metric stream whose path contains no job id is **not** attached to a job by
  name resemblance.  It migrates as its own run at ``confidence: "low"`` with
  ``job_id`` listed as unknown.

Idempotence
-----------
The migration is a pure function of the repository contents: every emitted
timestamp comes from the legacy record or from the fixed sentinel
:data:`telemetry.legacy.UNKNOWN_TS`, never from the clock.  The rendered event
sequence for each execution is therefore byte-stable, and a content digest per
execution is stored in ``_migration_state.json``.  A second run recomputes the
digests, finds them unchanged, and writes nothing.  If a legacy ledger has
changed since the last migration, the affected execution's stream is rewritten
wholesale (not appended to) so that the output remains a faithful function of
the current input -- this is safe only because the output root is migrator-owned,
which is asserted via an ``.migrator-owned`` marker before any rewrite.

CLI::

    python -m telemetry.migrate --project-root projects/masked-EqM \\
        --telemetry-root projects/masked-EqM/results/telemetry \\
        --report-dir projects/masked-EqM/results/telemetry_migration
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
import shutil
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .emit import JsonlSink
from .ids import RunSpec, make_exec_id
from .legacy import (
    BTM_SPEC_KEYS,
    LegacyFact,
    LegacyPaths,
    MetricStream,
    PIPELINE_EXPECTED_KEYS,
    PIPELINE_NEAR_DUPLICATE_KEYS,
    SRC_BTM_MANIFEST,
    SRC_DEC_EVENTS,
    SRC_DEC_JOBS,
    SRC_DEC_STATUS_EVENTS,
    SRC_DELT_EVENTS,
    SRC_DELT_JOBS,
    SRC_METRIC_STREAM,
    SRC_PIPELINE_ACTIVE,
    SRC_PIPELINE_COMPLETED,
    SRC_VARIANTS_TSV,
    UNKNOWN,
    UNKNOWN_SEED,
    UNKNOWN_TS,
    classify_metric_record,
    discover_metric_streams,
    iter_metric_records,
    load_all_facts,
    split_seed_suffix,
)
from .schema import EventType, RunStatus, dumps, make_record

#: Bumped when the reconstruction logic changes in a way that alters output.
#: Recorded in every run's spec so that a log can be traced to the code that
#: produced it -- a migration is itself an experiment and needs a version.
MIGRATOR_VERSION = 1

#: Marker file proving the output root belongs to this tool.  Checked before any
#: destructive rewrite.  Without it the migrator refuses to remove a file, which
#: makes "pointed the migrator at the wrong directory" a no-op instead of a
#: catastrophe.
OWNER_MARKER = ".migrator-owned"

STATE_FILE = "_migration_state.json"

#: Legacy status strings that denote a terminal lifecycle outcome, mapped to the
#: closest :class:`~telemetry.schema.RunStatus`.
#:
#: ``failed -> CRASHED`` rather than a dedicated "failed": the new enum
#: distinguishes *how* a run stopped, and a legacy ``failed`` covers a nonzero
#: exit for any reason.  ``CRASHED`` is the honest superset -- it asserts only
#: "did not complete", and it is in ``TRUNCATED_STATUSES`` so aggregators refuse
#: to pool it with completed runs, which is the behaviour we want under
#: uncertainty.
TERMINAL_STATUS_MAP: Dict[str, RunStatus] = {
    "completed": RunStatus.COMPLETED,
    "complete": RunStatus.COMPLETED,
    "success": RunStatus.COMPLETED,
    "succeeded": RunStatus.COMPLETED,
    "failed": RunStatus.CRASHED,
    "fail": RunStatus.CRASHED,
    "crashed": RunStatus.CRASHED,
    "cancelled": RunStatus.CANCELLED,
    "canceled": RunStatus.CANCELLED,
    "timeout": RunStatus.TIMEOUT,
    "timed_out": RunStatus.TIMEOUT,
    "preempted": RunStatus.PREEMPTED,
    "lost": RunStatus.LOST,
}

#: Terminal, but with a caveat that must travel with the record.  ``completed``
#: is the right lifecycle state (the process did reach its end) but the caveat
#: bears on whether its *outputs* are complete, which is a different question and
#: is preserved as a qualifier rather than collapsed into the status.
QUALIFIED_TERMINAL_MAP: Dict[str, Tuple[RunStatus, str]] = {
    "completed_with_write_error": (
        RunStatus.COMPLETED,
        "ledger status was 'completed_with_write_error': the process finished but "
        "at least one output write failed, so its artifacts may be incomplete"),
}

#: Legacy statuses that describe a run still in flight.  No ``END`` is
#: synthesized; the missing terminal record is the correct signal.
NON_TERMINAL_STATUSES = frozenset({
    "pending", "running", "submitted", "queued", "blocked", "retrying",
})

#: Legacy statuses that are annotations about a *record*, not observations of a
#: *process*.  These never produce an ``END``.
NON_LIFECYCLE_STATUSES = frozenset({
    "invalid -- discarded", "invalid", "superseded", "skipped_by_user",
    "not_promoted", "negative", "negative_no_full_launch", "pass", "retry",
})

#: Sources whose ``status`` field is a gate verdict or a stage marker rather than
#: a job lifecycle state, and which must therefore never drive an ``END``.
#: Campaign *event* logs record things like ``PASS`` and ``RUNNING`` about a
#: *stage*; a stage passing says nothing about whether the job exited cleanly.
NON_LIFECYCLE_SOURCES = frozenset({
    SRC_DEC_EVENTS, SRC_DEC_STATUS_EVENTS, SRC_DELT_EVENTS, SRC_VARIANTS_TSV,
})

#: How much a source can be trusted to describe a run's *specification*, highest
#: first.  The BTM manifest wins because it is the only ledger recording actual
#: scientific parameters; ``pipeline.json`` follows because it at least records a
#: git sha and a phase; the hand-edited campaign files come last.
SPEC_SOURCE_RANK: Dict[str, int] = {
    SRC_BTM_MANIFEST: 0,
    SRC_PIPELINE_ACTIVE: 1,
    SRC_PIPELINE_COMPLETED: 1,
    SRC_DEC_JOBS: 2,
    SRC_VARIANTS_TSV: 3,
    SRC_DELT_JOBS: 4,
    SRC_DEC_EVENTS: 5,
    SRC_DEC_STATUS_EVENTS: 5,
    SRC_DELT_EVENTS: 6,
    SRC_METRIC_STREAM: 7,
}

#: Campaign label assigned to each source when nothing better is available.
SOURCE_CAMPAIGN: Dict[str, str] = {
    SRC_BTM_MANIFEST: "btm",
    SRC_PIPELINE_ACTIVE: "legacy_pipeline",
    SRC_PIPELINE_COMPLETED: "legacy_pipeline",
    SRC_DEC_JOBS: "direct_energy_campaign",
    SRC_DEC_EVENTS: "direct_energy_campaign",
    SRC_DEC_STATUS_EVENTS: "direct_energy_campaign",
    SRC_DELT_JOBS: "direct_energy_longer_training",
    SRC_DELT_EVENTS: "direct_energy_longer_training",
    SRC_VARIANTS_TSV: "results_variants",
    SRC_METRIC_STREAM: "legacy_metric_stream",
}

CONFIDENCE_ORDER = ("high", "medium", "low", "none")


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Reconstruction:
    """One logical run reconstructed from one or more legacy facts."""

    spec: RunSpec
    exec_id: str
    job_id: str
    provenance: str
    confidence: str
    unknown_fields: List[str]
    primary_source: str
    facts: List[LegacyFact] = dataclasses.field(default_factory=list)
    notices: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    terminal: Optional[Dict[str, Any]] = None
    streams: List[MetricStream] = dataclasses.field(default_factory=list)


def _normalize_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip().lower()


def _fact_sort_key(fact: LegacyFact) -> Tuple[Any, ...]:
    """Deterministic total order over facts about one job.

    Ordered by timestamp when one exists, then by source and locator.  The
    secondary keys matter: 45 of the longer-training events use ``at`` and 74 use
    ``timestamp``, several share a timestamp to the second, and one is out of
    order relative to its neighbours.  Sorting by time alone would leave the
    output non-deterministic under those ties, and a non-deterministic output
    breaks the idempotence check.
    """
    return (fact.ts or "", fact.source, fact.locator)


def _reconstruct_spec(job_id: Optional[str], facts: Sequence[LegacyFact],
                      ) -> Tuple[RunSpec, str, List[str], str]:
    """Build the most faithful ``RunSpec`` the available facts support.

    Returns ``(spec, primary_source, unknown_fields, confidence)``.

    The arm is taken **verbatim** from whichever label the primary source states,
    with only a strict ``_s<N>`` / ``_seed<N>`` suffix removed.  It is not mapped
    onto a canonical arm vocabulary.  That refusal is deliberate: the historical
    bug this system replaces was a *decoder* that mapped ``btm_scalar_fd_directional4``
    onto ``btm_scalar_fd_directional`` by prefix match and averaged two arms
    together.  Carrying the label verbatim cannot collapse two distinct labels,
    at the cost of the same experiment under two different ledger names getting
    two ``run_uid``s -- an over-count, which the contradiction report makes
    visible and which is the safe direction to err in.
    """
    ordered = sorted(facts, key=lambda f: (SPEC_SOURCE_RANK.get(f.source, 99),
                                           _fact_sort_key(f)))
    primary = ordered[0]
    unknown: List[str] = []

    label = next((f.label for f in ordered if f.label), None)
    if not label:
        unknown.append("arm")
        arm = UNKNOWN
        seed: Optional[int] = None
    else:
        arm, seed = split_seed_suffix(str(label))

    # A declared seed always beats one decoded from a name.
    declared_seed = next((f.raw.get("seed") for f in ordered
                          if isinstance(f.raw.get("seed"), int)), None)
    if declared_seed is not None:
        seed = int(declared_seed)
    if seed is None:
        unknown.append("seed")
        seed = UNKNOWN_SEED

    git_sha = next((f.git_sha for f in ordered if f.git_sha), None)
    if not git_sha:
        unknown.append("git_sha")
        git_sha = UNKNOWN

    phase = next((f.phase for f in ordered if f.phase), None)
    if not phase:
        unknown.append("phase")
        phase = UNKNOWN

    campaign = SOURCE_CAMPAIGN.get(primary.source, "legacy")

    params: Dict[str, Any] = {
        # Identity-bearing on purpose: it is the only handle distinguishing two
        # ledger entries whose scientific parameters were never recorded.
        "legacy_label": str(label) if label else UNKNOWN,
        "legacy_source": primary.source,
        "migrator_version": MIGRATOR_VERSION,
    }
    planned_steps: Optional[int] = None

    if primary.source == SRC_BTM_MANIFEST:
        for key in BTM_SPEC_KEYS:
            if key in primary.raw:
                params[key] = primary.raw[key]
        max_steps = primary.raw.get("max_steps")
        if isinstance(max_steps, int):
            planned_steps = max_steps
        confidence = "high"
    elif primary.source in (SRC_PIPELINE_ACTIVE, SRC_PIPELINE_COMPLETED):
        # pipeline.json records no scientific parameters at all -- only prose in
        # `description`, which is not machine-recoverable into a spec.
        unknown.append("params")
        confidence = "medium"
    elif primary.source in (SRC_DEC_JOBS,):
        unknown.append("params")
        confidence = "medium"
    elif primary.source == SRC_METRIC_STREAM:
        unknown.append("params")
        confidence = "low"
    else:
        unknown.append("params")
        confidence = "low"

    if planned_steps is None:
        unknown.append("planned_steps")
    if job_id is None:
        unknown.append("job_id")

    # Confidence is capped by how much is unknown, independent of source rank:
    # a BTM row with no git sha is not a high-confidence reconstruction just
    # because its parameters are rich.
    for blocker in ("arm", "git_sha"):
        if blocker in unknown:
            confidence = "low"
    if "arm" in unknown and "job_id" in unknown:
        confidence = "none"

    spec = RunSpec(
        campaign=campaign, arm=arm, seed=int(seed), git_sha=git_sha,
        phase=str(phase), planned_steps=planned_steps, params=params)
    return spec, primary.source, sorted(set(unknown)), confidence


def _terminal_decision(facts: Sequence[LegacyFact]) -> Tuple[Optional[Dict[str, Any]],
                                                             List[Dict[str, Any]]]:
    """Decide whether -- and with what status -- to synthesize a terminal ``END``.

    Returns ``(terminal_or_None, notices)``.

    The rule, in order:

    * Only *lifecycle-bearing* sources are consulted.  A campaign event saying
      ``status: PASS`` is a gate verdict about a stage, not an observation that a
      process exited, and letting it seal a run would fabricate a lifecycle.
    * If the lifecycle-bearing sources name **more than one distinct terminal
      status**, no ``END`` is emitted and an error notice records every claim
      with its locator.  A disputed run is exactly a run whose telemetry is not
      trustworthy, and the schema already has a representation for that: no
      terminal record.
    * A single terminal status seals the run with ``inferred: true``.
    * A non-terminal status (``pending``, ``running``) seals nothing, but if the
      entry lives inside ``completed_runs`` that stranding is itself reported.
    """
    notices: List[Dict[str, Any]] = []
    claims: List[Tuple[RunStatus, LegacyFact, Optional[str]]] = []
    for fact in facts:
        if fact.source in NON_LIFECYCLE_SOURCES:
            continue
        status = _normalize_status(fact.status)
        if status is None:
            continue
        if status in TERMINAL_STATUS_MAP:
            claims.append((TERMINAL_STATUS_MAP[status], fact, None))
        elif status in QUALIFIED_TERMINAL_MAP:
            mapped, caveat = QUALIFIED_TERMINAL_MAP[status]
            claims.append((mapped, fact, caveat))
            notices.append({
                "level": "warning", "message": caveat,
                "legacy_status": fact.status, "source": fact.source,
                "locator": fact.locator, "code": "qualified_terminal_status",
            })
        elif status in NON_TERMINAL_STATUSES:
            if fact.source == SRC_PIPELINE_COMPLETED:
                notices.append({
                    "level": "error",
                    "message": (f"entry sits in completed_runs but its status is "
                                f"non-terminal ({fact.status!r}); the ledger never "
                                f"learned how this job ended"),
                    "source": fact.source, "locator": fact.locator,
                    "code": "stranded_non_terminal",
                })
        elif status in NON_LIFECYCLE_STATUSES:
            notices.append({
                "level": "error",
                "message": (f"status {fact.status!r} is an annotation about the "
                            f"record, not a lifecycle state; no terminal event "
                            f"can be inferred from it"),
                "source": fact.source, "locator": fact.locator,
                "code": "non_lifecycle_status",
            })
        else:
            notices.append({
                "level": "error",
                "message": f"unrecognized legacy status {fact.status!r}",
                "source": fact.source, "locator": fact.locator,
                "code": "unknown_status_value",
            })

    distinct = sorted({claim[0].value for claim in claims})
    if not claims:
        return None, notices
    if len(distinct) > 1:
        notices.append({
            "level": "error",
            "message": ("legacy sources disagree about this job's terminal status; "
                        "no END is synthesized, so the run reads as untrustworthy "
                        "rather than as one of two mutually exclusive outcomes"),
            "code": "disputed_terminal_status",
            "claims": [{"status": s.value, "legacy_status": f.status,
                        "source": f.source, "locator": f.locator}
                       for s, f, _ in claims],
        })
        return None, notices

    status = claims[0][0]
    caveats = sorted({c for _, _, c in claims if c})
    evidence = [{"source": f.source, "locator": f.locator,
                 "legacy_status": f.status} for _, f, _ in claims]
    last = max(claims, key=lambda c: _fact_sort_key(c[1]))[1]
    exit_code = next((f.raw.get("exit_code") for _, f, _ in claims
                      if f.raw.get("exit_code") is not None), None)
    return {
        "status": status.value,
        "ts": last.ts or UNKNOWN_TS,
        "ts_unknown": last.ts is None,
        "evidence": evidence,
        "caveats": caveats,
        "exit_code": exit_code,
        "duration": next((f.raw.get("duration") for _, f, _ in claims
                          if f.raw.get("duration") is not None), None),
        "final_metric": next((f.raw.get("final_metric") for _, f, _ in claims
                              if f.raw.get("final_metric") is not None), None),
    }, notices


def _schema_drift_notices(facts: Sequence[LegacyFact]) -> List[Dict[str, Any]]:
    """Report missing documented keys and ad-hoc near-duplicate key pairs."""
    notices: List[Dict[str, Any]] = []
    for fact in facts:
        if fact.source not in (SRC_PIPELINE_ACTIVE, SRC_PIPELINE_COMPLETED):
            continue
        missing = [k for k in PIPELINE_EXPECTED_KEYS if k not in fact.raw]
        if missing:
            notices.append({
                "level": "warning",
                "message": ("pipeline entry is missing documented keys: "
                            + ", ".join(missing)),
                "source": fact.source, "locator": fact.locator,
                "missing_keys": missing, "code": "schema_drift_missing_keys",
            })
        for left, right in PIPELINE_NEAR_DUPLICATE_KEYS:
            if left in fact.raw and right in fact.raw:
                notices.append({
                    "level": "warning",
                    "message": (f"entry carries both {left!r} and {right!r}, a "
                                f"near-duplicate key pair; they are preserved "
                                f"separately because no evidence says they are "
                                f"synonyms"),
                    "source": fact.source, "locator": fact.locator,
                    "code": "near_duplicate_keys",
                })
    return notices


def build_reconstructions(
    facts: Sequence[LegacyFact],
    streams: Sequence[MetricStream],
) -> Tuple[List[Reconstruction], List[LegacyFact], List[MetricStream]]:
    """Group facts into logical runs.  Returns ``(runs, unattributed, streams_kept)``.

    Grouping is by ``job_id``, because that is the only key every artifact
    shares.  Facts with no job id at all cannot be attributed to an execution and
    are returned separately rather than being folded into a plausible neighbour
    -- they are written to ``unattributed_facts.jsonl`` so that nothing is lost
    and nothing is invented.

    Content addressing does the rest: two job ids whose reconstructed specs are
    identical mint the same ``run_uid`` and land in the same run directory as two
    executions, without the migrator needing to know they were related.
    """
    by_job: Dict[str, List[LegacyFact]] = collections.defaultdict(list)
    unattributed: List[LegacyFact] = []
    for fact in facts:
        if fact.job_id:
            by_job[fact.job_id].append(fact)
        else:
            unattributed.append(fact)

    streams_by_job: Dict[str, List[MetricStream]] = collections.defaultdict(list)
    orphan_streams: List[MetricStream] = []
    for stream in streams:
        if stream.job_id:
            streams_by_job[stream.job_id].append(stream)
        else:
            orphan_streams.append(stream)

    runs: List[Reconstruction] = []
    # Attempt numbers are assigned per (run_uid, job_id).  In practice every job
    # id yields one execution; the counter exists because nothing in the legacy
    # data forbids two.
    attempts: Dict[Tuple[str, str], int] = collections.defaultdict(int)

    for job_id in sorted(by_job):
        job_facts = sorted(by_job[job_id], key=_fact_sort_key)
        spec, primary, unknown, confidence = _reconstruct_spec(job_id, job_facts)
        attempt = attempts[(spec.run_uid, job_id)]
        attempts[(spec.run_uid, job_id)] += 1
        terminal, notices = _terminal_decision(job_facts)
        notices = notices + _schema_drift_notices(job_facts)
        runs.append(Reconstruction(
            spec=spec, exec_id=make_exec_id(spec.run_uid, job_id, attempt),
            job_id=job_id, provenance="reconstructed", confidence=confidence,
            unknown_fields=unknown, primary_source=primary, facts=job_facts,
            notices=notices, terminal=terminal,
            streams=sorted(streams_by_job.get(job_id, []), key=lambda s: s.path)))

    # Streams with no job id in their path become their own runs.  Their pseudo
    # job id is derived from the path digest so it is stable and obviously not a
    # SLURM id.
    for stream in sorted(orphan_streams, key=lambda s: s.path):
        pseudo = "nojob-" + hashlib.blake2b(
            stream.path.encode("utf-8"), digest_size=4).hexdigest()
        fact = LegacyFact(
            source=SRC_METRIC_STREAM,
            locator=os.path.relpath(stream.path),
            path=stream.path,
            job_id=None,
            label=stream.run_tag,
            status=None,
            ts=None,
            raw={"path": stream.path, "n_records": stream.n_records,
                 "identity_source": "enclosing_path (LOSSY -- no job id present)"},
        )
        spec, primary, unknown, confidence = _reconstruct_spec(None, [fact])
        runs.append(Reconstruction(
            spec=spec, exec_id=make_exec_id(spec.run_uid, pseudo, 0),
            job_id=pseudo, provenance="reconstructed", confidence=confidence,
            unknown_fields=unknown, primary_source=primary, facts=[fact],
            notices=[{
                "level": "warning",
                "message": ("this run's identity was recovered from a filesystem "
                            "path containing no job id; it cannot be joined to any "
                            "ledger and may duplicate a run reconstructed elsewhere"),
                "code": "path_only_identity", "path": stream.path,
            }],
            terminal=None, streams=[stream]))

    return runs, unattributed, [s for s in streams if s.job_id] + orphan_streams


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_records(run: Reconstruction, *, max_stream_records: Optional[int] = None,
                   ) -> List[Dict[str, Any]]:
    """Render the full, deterministic event sequence for one execution.

    Order: a ``NOTICE`` announcing the reconstruction, one ``OBSERVED`` per
    legacy fact in timestamp order, every anomaly ``NOTICE``, the replayed
    metric-stream ``PROGRESS`` records, and finally the inferred ``END`` if one
    is warranted.  ``seq`` is assigned by position, so it remains the total order
    the schema promises even though none of these events was emitted live.
    """
    records: List[Dict[str, Any]] = []
    seq = 0

    def emit(event: EventType, payload: Mapping[str, Any], ts: Optional[str]) -> None:
        nonlocal seq
        records.append(make_record(
            run_uid=run.spec.run_uid, exec_id=run.exec_id, seq=seq, event=event,
            payload=payload, ts=ts or UNKNOWN_TS))
        seq += 1

    first_ts = next((f.ts for f in run.facts if f.ts), None)
    emit(EventType.NOTICE, {
        "level": "info",
        "message": ("reconstructed from legacy ledgers by telemetry.migrate; this "
                    "execution emitted no native telemetry and no START record "
                    "exists or can honestly be synthesized"),
        "code": "reconstruction_header",
        "provenance": run.provenance,
        "confidence": run.confidence,
        "unknown_fields": run.unknown_fields,
        "primary_source": run.primary_source,
        "migrator_version": MIGRATOR_VERSION,
        "sources": sorted({f.source for f in run.facts}),
        "ts_unknown": first_ts is None,
    }, first_ts)

    for fact in sorted(run.facts, key=_fact_sort_key):
        emit(EventType.OBSERVED, {
            "source": fact.source,
            "locator": fact.locator,
            "artifact_path": os.path.relpath(fact.path) if fact.path else None,
            "legacy_label": fact.label,
            "legacy_status": fact.status,
            "observed_at": fact.ts,
            "ts_unknown": fact.ts is None,
            # The whole original record travels with the observation.  A lossy
            # projection here would mean the migration destroyed information the
            # ledger still held, which defeats the point of migrating.
            "legacy_record": dict(fact.raw),
        }, fact.ts)

    for notice in run.notices:
        payload = {k: v for k, v in notice.items() if k != "level"}
        emit(EventType.NOTICE, {"level": notice.get("level", "warning"), **payload},
             first_ts)

    for stream in run.streams:
        records.extend(_render_stream(run, stream, seq, max_stream_records))
        seq = len(records)

    if run.terminal:
        emit(EventType.END, {
            "status": run.terminal["status"],
            # last_step is genuinely unknown for a reconstructed run: no ledger
            # records it.  -1 is the recorder's own "never saw a step" value.
            "last_step": -1,
            "planned_steps": run.spec.planned_steps,
            "truncated": None,
            "inferred": True,
            "inference_basis": run.terminal["evidence"],
            "caveats": run.terminal["caveats"],
            "legacy_exit_code": run.terminal["exit_code"],
            "legacy_duration": run.terminal["duration"],
            "legacy_final_metric": run.terminal["final_metric"],
            "ts_unknown": run.terminal["ts_unknown"],
            "unknown_fields": ["last_step", "wall_seconds", "peak_gpu_memory_bytes"],
        }, run.terminal["ts"])
    return records


def _render_stream(run: Reconstruction, stream: MetricStream, start_seq: int,
                   limit: Optional[int]) -> List[Dict[str, Any]]:
    """Replay one legacy metric file as ``PROGRESS`` events with identity stamped on.

    Each record keeps its original payload verbatim and gains: the envelope
    (which is what it never had), a ``kind`` from
    :func:`telemetry.legacy.classify_metric_record`, the reason that
    classification was reached, and the source line number.  The reason travels
    with the data on purpose -- a downstream analyst filtering on
    ``kind == "grad"`` can check *why* each record was called that without
    re-running the migration.
    """
    records: List[Dict[str, Any]] = []
    seq = start_seq
    for line_no, record in iter_metric_records(stream.path):
        if limit is not None and len(records) >= limit:
            records.append(make_record(
                run_uid=run.spec.run_uid, exec_id=run.exec_id, seq=seq,
                event=EventType.NOTICE, ts=UNKNOWN_TS, payload={
                    "level": "warning",
                    "message": (f"metric replay truncated at {limit} records by "
                                f"--max-stream-records; the migration is INCOMPLETE "
                                f"for this stream"),
                    "code": "stream_replay_truncated",
                    "stream_path": os.path.relpath(stream.path),
                    "total_records": stream.n_records,
                }))
            break
        if "_unparseable" in record:
            records.append(make_record(
                run_uid=run.spec.run_uid, exec_id=run.exec_id, seq=seq,
                event=EventType.NOTICE, ts=UNKNOWN_TS, payload={
                    "level": "error",
                    "message": "unparseable line in legacy metric stream",
                    "code": "unparseable_metric_line",
                    "stream_path": os.path.relpath(stream.path),
                    "line": line_no, **record}))
            seq += 1
            continue
        kind, reason = classify_metric_record(record)
        step = record.get("step")
        payload = {k: v for k, v in record.items() if k != "step"}
        records.append(make_record(
            run_uid=run.spec.run_uid, exec_id=run.exec_id, seq=seq,
            event=EventType.PROGRESS, ts=UNKNOWN_TS, payload={
                "step": int(step) if isinstance(step, (int, float)) else -1,
                "kind": kind,
                "kind_basis": reason,
                "replayed": True,
                "ts_unknown": True,
                "source_path": os.path.relpath(stream.path),
                "source_line": line_no,
                "step_unknown": not isinstance(step, (int, float)),
                **payload}))
        seq += 1
    return records


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Digest of the exact bytes :func:`write_run` would write.

    Defined over the serialized bytes rather than over the record objects so
    that :func:`_file_digest` can recompute the identical value from a file on
    disk.  That symmetry is what lets the idempotence check verify *content*
    rather than mere existence -- an interrupted previous migration leaves a
    truncated stream whose recorded digest is still "correct", and an
    existence-only check would let that truncation stand forever.
    """
    hasher = hashlib.blake2b(digest_size=16)
    for record in records:
        hasher.update(dumps(record).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _file_digest(path: str) -> Optional[str]:
    """Digest of a stream already on disk, comparable with :func:`_digest`."""
    try:
        hasher = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


def _load_state(telemetry_root: str) -> Dict[str, Any]:
    path = os.path.join(telemetry_root, STATE_FILE)
    if not os.path.exists(path):
        return {"migrator_version": MIGRATOR_VERSION, "executions": {}}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(telemetry_root: str, state: Mapping[str, Any]) -> None:
    path = os.path.join(telemetry_root, STATE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def write_run(telemetry_root: str, run: Reconstruction,
              records: Sequence[Mapping[str, Any]], state: Dict[str, Any],
              *, dry_run: bool = False) -> str:
    """Write one execution's stream.  Returns ``"written"``/``"unchanged"``/``"rewritten"``.

    Idempotence is by content digest rather than by "does the file exist": an
    interrupted previous migration can leave a half-written stream, and existence
    alone would let that half-file stand forever.  Comparing digests turns a
    partial write into a rewrite on the next run.
    """
    run_dir = os.path.join(telemetry_root, run.spec.slug())
    events_dir = os.path.join(run_dir, "events")
    stream_path = os.path.join(events_dir, f"{run.exec_id}.jsonl")
    digest = _digest(records)
    previous = state.setdefault("executions", {}).get(run.exec_id)

    if previous == digest and _file_digest(stream_path) == digest:
        return "unchanged"
    if dry_run:
        return "written" if previous is None else "rewritten"

    os.makedirs(events_dir, exist_ok=True)
    spec_payload = dict(run.spec.to_dict())
    spec_payload.update({
        "provenance": run.provenance,
        "confidence": run.confidence,
        "unknown_fields": run.unknown_fields,
        "primary_source": run.primary_source,
        "migrator_version": MIGRATOR_VERSION,
        "legacy_sources": sorted({f.source for f in run.facts}),
    })
    spec_path = os.path.join(run_dir, "spec.json")
    tmp = spec_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(spec_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, spec_path)

    outcome = "written"
    if os.path.exists(stream_path):
        _assert_owned(telemetry_root)
        os.remove(stream_path)
        outcome = "rewritten"
    sink = JsonlSink(stream_path)
    try:
        for record in records:
            sink.write(record)
    finally:
        sink.close()
    state["executions"][run.exec_id] = digest
    return outcome


def _prune(telemetry_root: str, stale_exec_ids: Sequence[str],
           state: Dict[str, Any]) -> List[str]:
    """Remove event streams (and now-empty run directories) for stale executions.

    Guarded by :func:`_assert_owned`, so pointing the migrator at a directory it
    did not create is a no-op rather than a deletion.
    """
    _assert_owned(telemetry_root)
    removed: List[str] = []
    for exec_id in stale_exec_ids:
        state["executions"].pop(exec_id, None)
        removed.append(exec_id)
    for entry in sorted(os.listdir(telemetry_root)):
        run_dir = os.path.join(telemetry_root, entry)
        events_dir = os.path.join(run_dir, "events")
        if not os.path.isdir(events_dir):
            continue
        for name in sorted(os.listdir(events_dir)):
            if not name.endswith(".jsonl"):
                continue
            if name[: -len(".jsonl")] in set(stale_exec_ids):
                os.remove(os.path.join(events_dir, name))
        if not os.listdir(events_dir):
            shutil.rmtree(run_dir)
    return removed


def _assert_owned(telemetry_root: str) -> None:
    marker = os.path.join(telemetry_root, OWNER_MARKER)
    if not os.path.exists(marker):
        raise RuntimeError(
            f"refusing to remove files under {telemetry_root}: no {OWNER_MARKER} "
            f"marker, so this directory was not created by telemetry.migrate")


def _ensure_root(telemetry_root: str) -> None:
    os.makedirs(telemetry_root, exist_ok=True)
    marker = os.path.join(telemetry_root, OWNER_MARKER)
    if not os.path.exists(marker):
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(
                "Created by telemetry.migrate. Every file under this directory is\n"
                "derived output and may be regenerated or replaced by the migrator.\n"
                "Do not hand-edit: hand-edited derived state is the defect this\n"
                "whole subsystem exists to remove.\n")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def migrate(project_root: str, telemetry_root: str, *, report_dir: Optional[str] = None,
            dry_run: bool = False, max_stream_records: Optional[int] = None,
            include_streams: bool = True) -> Dict[str, Any]:
    """Run the full migration.  Returns the machine-readable report."""
    paths = LegacyPaths(project_root)
    facts, missing = load_all_facts(paths)

    streams: List[MetricStream] = []
    if include_streams:
        # The telemetry output root is excluded from the walk so that a second
        # migration does not ingest the first one's output as legacy data.
        streams = discover_metric_streams(
            paths.results_dir, exclude=[telemetry_root])

    runs, unattributed, _ = build_reconstructions(facts, streams)

    if not dry_run:
        _ensure_root(telemetry_root)
    state = _load_state(telemetry_root) if os.path.isdir(telemetry_root) else \
        {"migrator_version": MIGRATOR_VERSION, "executions": {}}

    outcomes = collections.Counter()
    per_run: List[Dict[str, Any]] = []
    event_counts = collections.Counter()
    kind_counts = collections.Counter()

    for run in sorted(runs, key=lambda r: (r.spec.slug(), r.exec_id)):
        records = render_records(run, max_stream_records=max_stream_records)
        for record in records:
            event_counts[record["event"]] += 1
            if record["event"] == EventType.PROGRESS.value:
                kind_counts[record.get("kind", "?")] += 1
        outcome = write_run(telemetry_root, run, records, state, dry_run=dry_run)
        outcomes[outcome] += 1
        per_run.append({
            "run_uid": run.spec.run_uid,
            "slug": run.spec.slug(),
            "exec_id": run.exec_id,
            "job_id": run.job_id,
            "campaign": run.spec.campaign,
            "arm": run.spec.arm,
            "seed": run.spec.seed,
            "phase": run.spec.phase,
            "confidence": run.confidence,
            "unknown_fields": run.unknown_fields,
            "primary_source": run.primary_source,
            "sources": sorted({f.source for f in run.facts}),
            "n_facts": len(run.facts),
            "n_events": len(records),
            "terminal_status": run.terminal["status"] if run.terminal else None,
            "terminal_inferred": bool(run.terminal),
            "notice_codes": sorted({n.get("code", "?") for n in run.notices}),
            "streams": [os.path.relpath(s.path) for s in run.streams],
            "outcome": outcome,
        })

    # Executions the previous migration produced and this one does not.  These
    # arise whenever the reconstruction logic changes: a spec field that stops
    # being included moves a run to a new `run_uid`, and the old directory is
    # left behind describing a run under an identity no current code will ever
    # mint again.  Left alone, that residue accumulates and a consumer reading
    # the root finds *two* reconstructions of one job -- silently disagreeing,
    # which is the precise disease being cured.  The output root is entirely
    # derived, so removing the residue is safe and is the only way to keep the
    # invariant "this directory is a function of the current legacy data".
    produced = {run["exec_id"] for run in per_run}
    stale = sorted(set(state.get("executions", {})) - produced)
    pruned: List[str] = []
    if stale and not dry_run:
        pruned = _prune(telemetry_root, stale, state)

    if not dry_run:
        _save_state(telemetry_root, state)

    report: Dict[str, Any] = {
        "pruned_stale_executions": pruned,
        "migrator_version": MIGRATOR_VERSION,
        "project_root": os.path.relpath(project_root),
        "telemetry_root": os.path.relpath(telemetry_root),
        "dry_run": dry_run,
        "missing_artifacts": missing,
        "totals": {
            "legacy_facts": len(facts),
            "attributed_facts": len(facts) - len(unattributed),
            "unattributed_facts": len(unattributed),
            "reconstructed_runs": len({r.spec.run_uid for r in runs}),
            "reconstructed_executions": len(runs),
            "metric_streams": len(streams),
            "metric_streams_with_job_id": sum(1 for s in streams if s.job_id),
            "events_emitted": sum(event_counts.values()),
        },
        "events_by_type": dict(event_counts),
        "progress_by_kind": dict(kind_counts),
        "confidence_histogram": dict(collections.Counter(
            r.confidence for r in runs)),
        "unknown_field_histogram": dict(collections.Counter(
            field for r in runs for field in r.unknown_fields)),
        "terminal_status_histogram": dict(collections.Counter(
            (r.terminal or {}).get("status", "<none: unsealed>") for r in runs)),
        "notice_code_histogram": dict(collections.Counter(
            n.get("code", "?") for r in runs for n in r.notices)),
        "outcomes": dict(outcomes),
        "runs": per_run,
    }

    if report_dir and not dry_run:
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "migration_report.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(report_dir, "unattributed_facts.jsonl"), "w",
                  encoding="utf-8") as handle:
            for fact in sorted(unattributed, key=_fact_sort_key):
                handle.write(json.dumps(fact.to_dict(), sort_keys=True) + "\n")
        with open(os.path.join(report_dir, "migration_summary.md"), "w",
                  encoding="utf-8") as handle:
            handle.write(render_markdown(report))
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    """Human-readable summary of a migration report."""
    lines: List[str] = []
    add = lines.append
    add("# Telemetry migration summary\n")
    add(f"Migrator version {report['migrator_version']}; "
        f"source `{report['project_root']}`; output `{report['telemetry_root']}`.\n")
    add("## Totals\n")
    add("| quantity | value |")
    add("| --- | ---: |")
    for key, value in report["totals"].items():
        add(f"| {key} | {value} |")
    add("")
    for title, key in (("Events by type", "events_by_type"),
                       ("PROGRESS records by kind", "progress_by_kind"),
                       ("Runs by confidence", "confidence_histogram"),
                       ("Unknown spec fields", "unknown_field_histogram"),
                       ("Inferred terminal status", "terminal_status_histogram"),
                       ("Anomaly notices", "notice_code_histogram")):
        table = report.get(key) or {}
        if not table:
            continue
        add(f"## {title}\n")
        add("| key | count |")
        add("| --- | ---: |")
        for name, count in sorted(table.items(), key=lambda kv: (-kv[1], kv[0])):
            add(f"| `{name}` | {count} |")
        add("")
    if report.get("missing_artifacts"):
        add("## Artifacts not found\n")
        add("| artifact | path |")
        add("| --- | --- |")
        for item in report["missing_artifacts"]:
            add(f"| {item['artifact']} | `{item['path']}` |")
        add("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill legacy provenance artifacts into telemetry event logs.")
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--project-root", default=default_root)
    parser.add_argument("--telemetry-root", default=None,
                        help="output root (default: <project-root>/results/telemetry)")
    parser.add_argument("--report-dir", default=None,
                        help="where to write migration_report.json / .md")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute everything, write nothing")
    parser.add_argument("--max-stream-records", type=int, default=None,
                        help="cap replayed metric records per stream (marks the "
                             "stream INCOMPLETE in the log when it bites)")
    parser.add_argument("--no-streams", action="store_true",
                        help="skip legacy metric-stream replay")
    args = parser.parse_args(argv)

    telemetry_root = args.telemetry_root or os.path.join(
        args.project_root, "results", "telemetry")
    report = migrate(
        args.project_root, telemetry_root, report_dir=args.report_dir,
        dry_run=args.dry_run, max_stream_records=args.max_stream_records,
        include_streams=not args.no_streams)
    json.dump(report["totals"], sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
