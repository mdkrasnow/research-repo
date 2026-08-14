"""Read-only parsers for every legacy provenance artifact in this project.

Why this is its own module
--------------------------
Both :mod:`telemetry.migrate` (which *rewrites* history into the new schema) and
:mod:`telemetry.contradictions` (which *audits* history without rewriting it)
need to read exactly the same bytes and reach exactly the same conclusions about
them.  If each had its own parser, the migration could be built on a reading of
the ledgers that the audit never checked -- which is the failure mode the whole
telemetry effort exists to remove.  One parser, two consumers.

Hard invariant: **nothing in this module opens a legacy file for writing.**  The
legacy ledgers are the only surviving evidence of what actually ran; a migrator
that "cleans them up" destroys the very record it was written to preserve.  Every
output of this codebase lands in a new directory.

The unit of parsing is the :class:`LegacyFact`: one assertion, made by one
artifact, at one location inside it, about one job.  Deliberately *not* "one
run" -- the legacy ledgers have no primary key, so "run" is not a thing they can
be decomposed into without an inference step.  Keeping the inference out of the
parser is what lets the audit report disagreements that the migration had to
resolve.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: The sentinel written into a spec field whose true value the legacy record does
#: not contain.  A distinguished string rather than ``None``: ``None`` is dropped
#: by :func:`telemetry.ids.canonicalize`, so two runs whose seeds are unknown for
#: *different* reasons would hash identically to a run with no seed field at all.
#: A visible sentinel keeps "unknown" a value, greppable in every downstream
#: artifact, and impossible to mistake for a measured one.
UNKNOWN = "__unknown__"

#: Seed used when the legacy record does not state one.  Negative so that it can
#: never collide with a real seed, and always accompanied by ``"seed"`` in the
#: run's ``unknown_fields``.
UNKNOWN_SEED = -1

#: Deterministic stand-in for a timestamp the legacy record does not carry.
#: Deliberately *not* the current time and *not* the file's mtime: the migration
#: must be a pure function of the repository contents, or its idempotence check
#: degenerates into "rewrite everything on every run".
UNKNOWN_TS = "1970-01-01T00:00:00.000Z"

#: Artifact identifiers.  Stable strings -- they appear in every emitted record's
#: ``source`` field and in the contradiction report, so renaming one is a
#: breaking change to the output.
SRC_PIPELINE_ACTIVE = "pipeline.active_runs"
SRC_PIPELINE_COMPLETED = "pipeline.completed_runs"
SRC_BTM_MANIFEST = "btm.manifest"
SRC_DEC_JOBS = "direct_energy_campaign.status.jobs"
SRC_DEC_EVENTS = "direct_energy_campaign.events"
SRC_DEC_STATUS_EVENTS = "direct_energy_campaign.status.events"
SRC_DELT_JOBS = "direct_energy_longer_training.status.jobs"
SRC_DELT_EVENTS = "direct_energy_longer_training.events"
SRC_VARIANTS_TSV = "results_variants.tsv"
SRC_METRIC_STREAM = "metric_stream"

ALL_SOURCES = (
    SRC_PIPELINE_ACTIVE, SRC_PIPELINE_COMPLETED, SRC_BTM_MANIFEST,
    SRC_DEC_JOBS, SRC_DEC_EVENTS, SRC_DEC_STATUS_EVENTS,
    SRC_DELT_JOBS, SRC_DELT_EVENTS, SRC_VARIANTS_TSV, SRC_METRIC_STREAM,
)

#: A SLURM job id as it appears in this repo: a bare decimal, optionally with an
#: array-task suffix.  Anchored on both ends -- an unanchored scan over a
#: hand-edited ledger matches the mantissa of a float (this is not hypothetical:
#: ``direct_energy_longer_training/status.json`` stores
#: ``"epoch08_fid_none_value": 129.12085939165866`` inside its ``jobs`` dict, and
#: an unanchored ``\d{6,}`` finds a phantom job ``12085939`` inside it).
_JOB_ID_RE = re.compile(r"^\d{4,}(?:_\d+)?$")

#: Strict seed suffix.  Only these two spellings are decoded; anything else
#: leaves the seed unknown rather than guessing.  Under-decoding costs a marked
#: ``unknown``; over-decoding silently mislabels an arm, which is unrecoverable.
_SEED_SUFFIX_RE = re.compile(r"_(?:s|seed)(\d+)$")


def is_job_id(value: Any) -> bool:
    """True iff ``value`` is a plausible SLURM job id *in its entirety*."""
    if isinstance(value, bool) or isinstance(value, float):
        return False
    return bool(_JOB_ID_RE.match(str(value).strip()))


def split_seed_suffix(label: str) -> Tuple[str, Optional[int]]:
    """``("btm_IIA_G", 0)`` from ``"btm_IIA_G_s0"``; ``(label, None)`` otherwise.

    Note what this does *not* do: it does not map the residual stem onto a
    canonical arm name.  The prefix collision that motivated the whole identity
    rewrite (``btm_scalar_fd_directional`` being a prefix of
    ``btm_scalar_fd_directional4``) arose from exactly that kind of decoding.
    Here the stem is carried verbatim, so two distinct legacy labels remain two
    distinct identities no matter how similar they look.
    """
    match = _SEED_SUFFIX_RE.search(label or "")
    if not match:
        return label, None
    return label[: match.start()], int(match.group(1))


@dataclasses.dataclass(frozen=True)
class LegacyFact:
    """One assertion by one artifact at one location about one job.

    ``locator`` is a path into the artifact (``completed_runs[417]``,
    ``jobs['35335934']``) and is what makes a finding in the contradiction report
    actionable: a human can open the file and see the exact entry, which a
    summary count can never support.
    """

    source: str
    locator: str
    path: str
    job_id: Optional[str] = None
    label: Optional[str] = None
    status: Optional[str] = None
    ts: Optional[str] = None
    git_sha: Optional[str] = None
    phase: Optional[str] = None
    raw: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["raw"] = dict(self.raw)
        return out


# ---------------------------------------------------------------------------
# pipeline.json
# ---------------------------------------------------------------------------

def parse_pipeline(path: str) -> List[LegacyFact]:
    """``.state/pipeline.json`` -> one fact per ``active_runs``/``completed_runs``
    entry.

    The index in the locator is load-bearing.  This file has no primary key: 923
    completed entries carry only 853 distinct ``run_id``, so ``run_id`` cannot
    address an entry and neither can ``job_id`` (four job ids appear twice, with
    disagreeing statuses).  The ordinal position is the only address that exists.
    """
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    facts: List[LegacyFact] = []
    for key, source in ((("active_runs"), SRC_PIPELINE_ACTIVE),
                        (("completed_runs"), SRC_PIPELINE_COMPLETED)):
        for index, entry in enumerate(doc.get(key) or []):
            if not isinstance(entry, Mapping):
                continue
            job = entry.get("job_id")
            facts.append(LegacyFact(
                source=source,
                locator=f"{key}[{index}]",
                path=path,
                job_id=str(job) if job not in (None, "") else None,
                label=entry.get("run_id"),
                status=entry.get("status"),
                ts=(entry.get("completed_at") or entry.get("submitted_at")
                    or entry.get("started_at")),
                git_sha=entry.get("git_sha"),
                phase=str(entry.get("phase")) if entry.get("phase") is not None else None,
                raw=dict(entry),
            ))
    return facts


#: Keys that *should* be on every pipeline entry.  Used only to report drift --
#: nothing is invented to fill a gap.
PIPELINE_EXPECTED_KEYS = (
    "run_id", "job_id", "partition", "status", "description", "submitted_at",
    "git_sha", "sbatch_path", "expected_runtime", "phase", "gate",
)

#: Ad-hoc keys that accumulated outside the documented schema, grouped by the
#: concept they redundantly encode.  Recorded, never merged: ``ckpt`` and
#: ``checkpoint`` *probably* mean the same thing, and acting on "probably" is how
#: a provenance record becomes fiction.
PIPELINE_NEAR_DUPLICATE_KEYS = (
    ("ckpt", "checkpoint"),
    ("note", "analysis_note"),
    ("superseded_job_id", "supersedes"),
    ("started_at", "submitted_at"),
    ("checkpoint_direct", "checkpoint_none"),
)


# ---------------------------------------------------------------------------
# results/btm/manifest.jsonl
# ---------------------------------------------------------------------------

def parse_btm_manifest(path: str) -> List[LegacyFact]:
    """The BTM campaign manifest: the richest legacy source, and the most wrong.

    Richest, because it is the only ledger that records the actual scientific
    parameters (``btm_mode``, ``fd_k``, ``fd_eps``, ``tc``, ``max_steps``,
    ``global_batch``) rather than a prose description of them.  Most wrong,
    because its ``status`` field is a mutable cell written once at submission and
    never again: every row still says ``submitted``.  That is the exact defect
    the append-only design removes, so the migration keeps the parameters and
    treats the status as an assertion about the past ("this was submitted"),
    never as a current state.

    Its key for the run label is ``run_tag``, not ``run_id``, and the strings do
    not agree with the pipeline's (``btm_IIA_btm_scalar_exact_s0`` vs
    ``btm_IIA_G_s0``).  Only ``job_id`` joins the two.
    """
    facts: List[LegacyFact] = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            job = entry.get("job_id")
            facts.append(LegacyFact(
                source=SRC_BTM_MANIFEST,
                locator=f"line[{index}]",
                path=path,
                job_id=str(job) if job not in (None, "") else None,
                label=entry.get("run_tag"),
                status=entry.get("status"),
                ts=entry.get("submitted_at"),
                git_sha=entry.get("git_sha"),
                phase=entry.get("phase"),
                raw=dict(entry),
            ))
    return facts


#: Identity-bearing keys of a BTM manifest row.  Curated rather than "everything
#: in the row" because ``command`` and ``partition`` describe *how* a run was
#: launched, not what it computed, and folding them into the hash would make two
#: reruns of one experiment look like two experiments.
BTM_SPEC_KEYS = (
    "btm_mode", "ebm", "epochs", "fd_eps", "fd_k", "global_batch",
    "max_steps", "tc",
)


# ---------------------------------------------------------------------------
# results/direct_energy_campaign/{status.json,events.jsonl}
# ---------------------------------------------------------------------------

def parse_direct_energy_campaign(status_path: str, events_path: str) -> List[LegacyFact]:
    """The one campaign with a real code writer (``experiments/direct_energy/campaign.py``).

    Its ``status.json`` holds a ``jobs`` dict keyed by job id -- the only legacy
    structure that has a primary key at all.  Its ``events.jsonl`` is *nearly* a
    fold of the same information, except that five lines were hand-inserted
    directly into the file and never made it into ``status.json['events']``, and
    one of those lines is out of time order.  Both files are parsed and the
    discrepancy is reported rather than reconciled, because there is no evidence
    saying which of the two the author intended to be authoritative.
    """
    facts: List[LegacyFact] = []
    with open(status_path, encoding="utf-8") as handle:
        status = json.load(handle)
    for job_id, obj in (status.get("jobs") or {}).items():
        if not isinstance(obj, Mapping):
            continue
        facts.append(LegacyFact(
            source=SRC_DEC_JOBS,
            locator=f"jobs[{job_id!r}]",
            path=status_path,
            job_id=str(job_id),
            label=obj.get("stage"),
            status=obj.get("status"),
            ts=obj.get("completed_at") or obj.get("updated_at") or obj.get("recorded_at"),
            git_sha=obj.get("git_sha"),
            phase=obj.get("stage"),
            raw=dict(obj),
        ))
    for index, entry in enumerate(status.get("events") or []):
        facts.extend(_campaign_event_facts(
            entry, SRC_DEC_STATUS_EVENTS, f"events[{index}]", status_path))
    with open(events_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            facts.extend(_campaign_event_facts(
                json.loads(line), SRC_DEC_EVENTS, f"line[{index}]", events_path))
    return facts


def parse_longer_training(status_path: str, events_path: str) -> List[LegacyFact]:
    """The hand-maintained campaign: no code writer, and an inverted index.

    Two structural hazards, both handled explicitly:

    1.  Its ``jobs`` dict is inverted relative to the other campaign's -- it maps
        ``name -> job_id`` instead of ``job_id -> object``.  A reader that
        assumes one shape silently reads keys as job ids and gets 113 phantom
        "jobs" named ``none_seed0``.
    2.  That same dict is polluted with non-job entries whose values are floats
        (``"epoch08_fid_none_value": 129.12085939165866``).  This is why
        :data:`_JOB_ID_RE` is anchored: an unanchored scan mines job ids out of
        float mantissas.  Non-job entries are separated out and preserved as
        metrics rather than dropped.
    """
    facts: List[LegacyFact] = []
    with open(status_path, encoding="utf-8") as handle:
        status = json.load(handle)
    jobs = status.get("jobs") or {}
    for name, value in jobs.items():
        if not is_job_id(value):
            # Not a job at all: a metric, a state word ("complete"), a note.
            facts.append(LegacyFact(
                source=SRC_DELT_JOBS,
                locator=f"jobs[{name!r}]",
                path=status_path,
                job_id=None,
                label=str(name),
                status=None,
                ts=status.get("started_at"),
                raw={"key": name, "value": value, "_not_a_job": True},
            ))
            continue
        facts.append(LegacyFact(
            source=SRC_DELT_JOBS,
            locator=f"jobs[{name!r}]",
            path=status_path,
            job_id=str(value),
            label=str(name),
            status=None,  # the inverted dict records no per-job status at all
            # No per-job timestamp and no per-job commit exist in this structure.
            # The file has a top-level ``commit`` and ``started_at``, but those
            # describe the CAMPAIGN, and this dict spans jobs submitted across
            # many commits over several days.  Broadcasting a campaign-level fact
            # onto each of its 106 jobs would assert 106 things the ledger never
            # said -- and, worse, would then be reported as 103 git-sha
            # "disagreements" against pipeline.json, burying the real ones.  The
            # campaign-level values are preserved in ``raw`` as context.
            ts=None,
            git_sha=None,
            phase=None,
            raw={"key": name, "value": str(value),
                 "_campaign_commit": status.get("commit"),
                 "_campaign_started_at": status.get("started_at")},
        ))
    with open(events_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            facts.extend(_campaign_event_facts(
                json.loads(line), SRC_DELT_EVENTS, f"line[{index}]", events_path))
    return facts


def _campaign_event_facts(entry: Mapping[str, Any], source: str, locator: str,
                          path: str) -> List[LegacyFact]:
    """Explode one campaign event into one fact per job it mentions.

    Campaign events are shaped by whoever typed them: 76 distinct key-shapes over
    119 lines in the longer-training log, two competing timestamp keys (``at``
    and ``timestamp``) and two job keys (``job`` and ``jobs``, the latter
    sometimes a list).  Rather than normalize into a schema nobody wrote to, the
    parser extracts only what is structurally unambiguous -- when, which jobs,
    what stage, what status word -- and carries the entire original object along
    as ``raw`` so that nothing is lost by the extraction.

    An event that names N jobs becomes N facts, because a single event asserting
    ``status: RETRYING`` over three job ids is three assertions and the
    contradiction checker must be able to disagree with each one separately.
    """
    ts = entry.get("at") or entry.get("timestamp")
    stage = entry.get("stage") or entry.get("event")
    status = entry.get("status")
    jobs: List[str] = []
    for key in ("job", "jobs", "job_id"):
        value = entry.get(key)
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple)) else [value]
        jobs.extend(str(v) for v in candidates if is_job_id(v))
    jobs = sorted(set(jobs))
    if not jobs:
        return [LegacyFact(source=source, locator=locator, path=path, job_id=None,
                           label=stage, status=status, ts=ts,
                           git_sha=entry.get("git_sha") or entry.get("commit"),
                           phase=stage, raw=dict(entry))]
    return [LegacyFact(source=source, locator=locator, path=path, job_id=job,
                       label=stage, status=status, ts=ts,
                       git_sha=entry.get("git_sha") or entry.get("commit"),
                       phase=stage, raw=dict(entry))
            for job in jobs]


# ---------------------------------------------------------------------------
# results_variants.tsv
# ---------------------------------------------------------------------------

def parse_results_variants(path: str) -> List[LegacyFact]:
    """The hand-maintained results table: no code reader, no code writer.

    Because nothing writes it, it is stale by construction -- it stops before the
    entire current BTM campaign.  Because nothing reads it, that staleness has
    never surfaced as an error.  It is migrated anyway: its rows carry gate
    verdicts and checkpoint paths that exist nowhere else.
    """
    facts: List[LegacyFact] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            job = (row.get("job_id") or "").strip()
            facts.append(LegacyFact(
                source=SRC_VARIANTS_TSV,
                locator=f"row[{index}]",
                path=path,
                job_id=job if is_job_id(job) else None,
                label=row.get("run_id"),
                status=None,  # the table records a gate verdict, not a lifecycle state
                ts=row.get("date"),
                phase=row.get("phase"),
                raw={k: v for k, v in row.items() if k is not None},
            ))
    return facts


# ---------------------------------------------------------------------------
# per-run metric streams
# ---------------------------------------------------------------------------

#: ``<RUN_TAG>_job<JOBID>`` -- the only place a legacy metric stream's job id
#: exists.  Anchored at the end of a path component.
_RUN_DIR_RE = re.compile(r"^(?P<tag>.+)_job(?P<job>\d{4,}(?:_\d+)?)$")

#: Witness keys for a full gradient record (``train.py`` ~line 624).
_GRAD_WITNESS = frozenset({
    "grad_norm", "head_grad_norm", "backbone_grad_norm", "max_grad_norm",
    "clipped", "learning_rate", "adaptive_clip",
})

#: Witness keys for the Stage-2 checklist probe record (``train.py`` ~line 732).
_PROBE_WITNESS = frozenset({
    "delta_theta_norm", "probe_loss_pre", "probe_loss_post", "probe_delta_L",
    "probe_cos_field_update_vs_neg_r", "field_delta_norm", "P_t", "eta_func",
})

#: Witness keys for the WFB Stage-3 optimizer trace (a third writer entirely,
#: ``experiments/direct_energy``).
#:
#: Narrower than the Stage-3 record's full key set on purpose.  When ``train.py``
#: runs with ``--wfb-backward`` it *adds* ``lambda_max``, ``lam``, ``m_lanczos``,
#: ``r_norm`` and friends to its ordinary gradient record (train.py ~line 659), so
#: those keys are shared between the two writers and cannot witness either one.
#: Only keys that appear exclusively in the Stage-3 line-search trace -- the
#: step-acceptance machinery, which the training loop has no analogue of -- are
#: admitted here.  Including a shared key would turn every WFB-enabled gradient
#: record into an ``ambiguous`` classification, i.e. would silently discard the
#: exact runs the WFB investigation is about.
_WFB_WITNESS = frozenset({
    "eta_star", "eta_used", "n_backtracks", "predicted_delta_L",
    "actual_delta_L", "r_dot_q", "g_alpha_norm", "q_alpha_norm", "L_before",
})


def classify_metric_record(record: Mapping[str, Any]) -> Tuple[str, str]:
    """Assign a ``kind`` to one line of a legacy metric stream.

    The problem
    -----------
    ``gradient_metrics.jsonl`` is written by two call sites in ``train.py`` that
    share one file handle and emit **no discriminator field**:

    * ~line 624 writes the full gradient record: ``step``, ``loss``,
      ``grad_norm``, ``head_grad_norm``, ``backbone_grad_norm``,
      ``max_grad_norm``, ``clipped``, ``learning_rate``, ``adaptive_clip``,
      ``weight_decay`` (plus optional ``exact_fwrev`` / ``wfb`` / ``btm``
      blocks);
    * ~line 732 writes the Stage-2 checklist probe: ``step``,
      ``delta_theta_norm``, ``probe_loss_pre``, ``probe_loss_post``,
      ``probe_delta_L``, ``probe_cos_field_update_vs_neg_r``,
      ``field_delta_norm``, ``P_t``, ``eta_func``.

    Both fire on the same ``grad_log_every`` cadence, so both appear at the same
    ``step``.  A consumer that counts lines double-counts; a consumer that takes
    a ratio of two medians over "the file" takes it across two disjoint
    populations.

    The heuristic
    -------------
    The two shapes share exactly one key, ``step``.  Their remaining key sets are
    *disjoint*, so membership in either witness set is a sound discriminator:

    * keys meet :data:`_GRAD_WITNESS` and not :data:`_PROBE_WITNESS` -> ``grad``
    * keys meet :data:`_PROBE_WITNESS` and not :data:`_GRAD_WITNESS` -> ``probe``
    * keys meet :data:`_WFB_WITNESS` only -> ``wfb`` (a third writer)
    * anything else -> ``unknown``

    "Anything else" is not folded into the most likely bucket.  A record matching
    two witness sets would mean the two call sites had merged since this was
    written, and a record matching none would mean a writer this heuristic has
    never seen -- in both cases a guess would produce a plausible number from a
    misread population, which is the exact class of error being migrated away
    from.  ``unknown`` records are emitted with their key signature attached so
    the misclassification is inspectable rather than silent.

    Returns ``(kind, reason)``; the reason is stamped onto the emitted event.
    """
    keys = frozenset(record.keys())
    hits = {
        "grad": bool(keys & _GRAD_WITNESS),
        "probe": bool(keys & _PROBE_WITNESS),
        "wfb": bool(keys & _WFB_WITNESS),
    }
    matched = [name for name, hit in hits.items() if hit]
    if len(matched) == 1:
        kind = matched[0]
        witness = sorted(keys & {"grad": _GRAD_WITNESS, "probe": _PROBE_WITNESS,
                                 "wfb": _WFB_WITNESS}[kind])
        return kind, f"witness_keys={','.join(witness)}"
    if not matched:
        return "unknown", f"no_witness_key; signature={','.join(sorted(keys))}"
    return "unknown", f"ambiguous_matches={','.join(matched)}"


@dataclasses.dataclass(frozen=True)
class MetricStream:
    """A legacy per-step metric file whose identity lives only in its path.

    ``job_id`` is populated only when the path actually contains a
    ``_job<ID>`` component.  When it does not, the field stays ``None`` and the
    stream migrates as its own low-confidence run rather than being attached to
    a job by resemblance -- path-based attribution is the non-injective decoding
    the identity module was written to abolish, and doing it here would
    reintroduce it at the one place nobody would think to look.
    """

    path: str
    run_tag: Optional[str]
    job_id: Optional[str]
    experiment_dir: Optional[str]
    n_records: int


#: Filenames that are known per-run metric streams.
METRIC_STREAM_NAMES = ("gradient_metrics.jsonl", "fb_direct_metrics.jsonl",
                       "metrics.jsonl")

#: Suffix form used by the WFB stage-3 runs (``<name>_metrics.jsonl``).
METRIC_STREAM_SUFFIX = "_metrics.jsonl"


def discover_metric_streams(root: str, *, exclude: Sequence[str] = ()) -> List[MetricStream]:
    """Walk ``root`` for legacy metric streams, recording only what the path says."""
    excluded = tuple(os.path.abspath(e) for e in exclude)
    found: List[MetricStream] = []
    for dirpath, dirnames, filenames in os.walk(root):
        absdir = os.path.abspath(dirpath)
        if any(absdir == e or absdir.startswith(e + os.sep) for e in excluded):
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            if name not in METRIC_STREAM_NAMES and not name.endswith(METRIC_STREAM_SUFFIX):
                continue
            path = os.path.join(dirpath, name)
            tag, job, expdir = _identity_from_path(path)
            found.append(MetricStream(
                path=path, run_tag=tag, job_id=job, experiment_dir=expdir,
                n_records=_count_lines(path)))
    return found


def _identity_from_path(path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract ``(run_tag, job_id, experiment_dir)`` -- only when unambiguous.

    Walks upward from the file looking for a component matching
    ``<TAG>_job<JOBID>``.  If none matches, returns ``(parent_dir_name, None,
    parent_dir_name)``: the tag is kept as a *label* (it is the only human handle
    that exists) but the job id stays ``None``, which is what forces the
    low-confidence path in the migrator.
    """
    parts = os.path.normpath(path).split(os.sep)[:-1]
    for index in range(len(parts) - 1, -1, -1):
        match = _RUN_DIR_RE.match(parts[index])
        if match:
            return match.group("tag"), match.group("job"), parts[-1]
    return (parts[-1] if parts else None), None, (parts[-1] if parts else None)


def _count_lines(path: str) -> int:
    count = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def iter_metric_records(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Yield ``(line_number, record)`` for a metric stream, skipping blank lines.

    A line that fails to parse is yielded as ``{"_unparseable": <text>}`` rather
    than skipped: a corrupt line is evidence about the run (a crash mid-write, an
    interleaved second writer) and dropping it would erase that evidence.
    """
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                yield index, {"_unparseable": text[:500], "_error": str(exc)}
                continue
            if isinstance(record, Mapping):
                yield index, dict(record)
            else:
                yield index, {"_unparseable": text[:500], "_error": "not an object"}


# ---------------------------------------------------------------------------
# default artifact locations
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LegacyPaths:
    """Where the legacy artifacts live, relative to the project root."""

    root: str

    @property
    def pipeline(self) -> str:
        return os.path.join(self.root, ".state", "pipeline.json")

    @property
    def btm_manifest(self) -> str:
        return os.path.join(self.root, "results", "btm", "manifest.jsonl")

    @property
    def dec_status(self) -> str:
        return os.path.join(self.root, "results", "direct_energy_campaign", "status.json")

    @property
    def dec_events(self) -> str:
        return os.path.join(self.root, "results", "direct_energy_campaign", "events.jsonl")

    @property
    def delt_status(self) -> str:
        return os.path.join(self.root, "results", "direct_energy_longer_training", "status.json")

    @property
    def delt_events(self) -> str:
        return os.path.join(self.root, "results", "direct_energy_longer_training", "events.jsonl")

    @property
    def variants_tsv(self) -> str:
        return os.path.join(self.root, "results_variants.tsv")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.root, "results")


def load_all_facts(paths: LegacyPaths) -> Tuple[List[LegacyFact], List[Dict[str, str]]]:
    """Parse every artifact that exists.  Returns ``(facts, missing)``.

    A missing artifact is reported, not tolerated silently: "the BTM manifest
    contributed zero facts" and "the BTM manifest was not found" are different
    statements about the world and a report that conflates them is misleading.
    """
    facts: List[LegacyFact] = []
    missing: List[Dict[str, str]] = []

    def _try(name: str, path: str, fn) -> None:
        if not os.path.exists(path):
            missing.append({"artifact": name, "path": path, "reason": "not found"})
            return
        facts.extend(fn())

    _try("pipeline.json", paths.pipeline, lambda: parse_pipeline(paths.pipeline))
    _try("btm/manifest.jsonl", paths.btm_manifest,
         lambda: parse_btm_manifest(paths.btm_manifest))
    if os.path.exists(paths.dec_status) and os.path.exists(paths.dec_events):
        facts.extend(parse_direct_energy_campaign(paths.dec_status, paths.dec_events))
    else:
        missing.append({"artifact": "direct_energy_campaign",
                        "path": paths.dec_status, "reason": "not found"})
    if os.path.exists(paths.delt_status) and os.path.exists(paths.delt_events):
        facts.extend(parse_longer_training(paths.delt_status, paths.delt_events))
    else:
        missing.append({"artifact": "direct_energy_longer_training",
                        "path": paths.delt_status, "reason": "not found"})
    _try("results_variants.tsv", paths.variants_tsv,
         lambda: parse_results_variants(paths.variants_tsv))
    return facts, missing
