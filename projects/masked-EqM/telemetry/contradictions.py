"""Cross-artifact contradiction checker for the legacy provenance ledgers.

Why this exists separately from the migration
---------------------------------------------
The migration has to *resolve* things: it picks one spec source per job, it
decides whether to synthesize a terminal record.  Every such decision is a place
where a real disagreement in the source data can be quietly absorbed into a
plausible-looking output.  This module makes no decisions.  It reads the same
facts through the same parsers (:mod:`telemetry.legacy`) and reports, without
resolution, three classes of defect:

``disagreement``
    Two or more artifacts assert incompatible values for the same job id --
    different terminal statuses, different git shas, different run labels.  These
    are the findings that invalidate a results table, because a table built by
    joining on ``run_id`` gets a different answer depending on which ledger the
    author happened to open.

``orphan``
    A job id that exists in exactly one artifact.  Not an error by itself, but a
    coverage measurement: an orphan in the BTM manifest and nowhere else means
    ``pipeline.json`` -- the file the project's own rules call "the single source
    of truth for every submitted SLURM job" -- never learned about that job.

``collision``
    A key that was used as if it were unique and is not: 923 pipeline entries
    over 853 distinct ``run_id``, four job ids appearing twice, a manifest whose
    ``run_tag`` disagrees with the pipeline's ``run_id`` for the same job.

Output is both JSON (re-checkable in CI; a diff against a stored baseline shows
whether the ledgers got worse) and markdown (readable in a report).  The check is
non-destructive and can be run at any time.

CLI::

    python -m telemetry.contradictions --project-root projects/masked-EqM \\
        --out-dir projects/masked-EqM/results/telemetry_migration
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .legacy import (
    LegacyFact,
    LegacyPaths,
    PIPELINE_NEAR_DUPLICATE_KEYS,
    SRC_BTM_MANIFEST,
    SRC_DEC_EVENTS,
    SRC_DEC_JOBS,
    SRC_DEC_STATUS_EVENTS,
    SRC_DELT_EVENTS,
    SRC_DELT_JOBS,
    SRC_PIPELINE_ACTIVE,
    SRC_PIPELINE_COMPLETED,
    SRC_VARIANTS_TSV,
    is_job_id,
    load_all_facts,
    parse_pipeline,
)
from .migrate import (
    NON_LIFECYCLE_SOURCES,
    NON_LIFECYCLE_STATUSES,
    NON_TERMINAL_STATUSES,
    QUALIFIED_TERMINAL_MAP,
    TERMINAL_STATUS_MAP,
    _normalize_status,
)

SEVERITY_ORDER = ("critical", "major", "minor", "info")


def _finding(kind: str, severity: str, job_id: Optional[str], summary: str,
             **extra: Any) -> Dict[str, Any]:
    return {"kind": kind, "severity": severity, "job_id": job_id,
            "summary": summary, **extra}


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_status_disagreements(by_job: Mapping[str, Sequence[LegacyFact]]
                               ) -> List[Dict[str, Any]]:
    """Jobs whose lifecycle-bearing sources claim different terminal outcomes.

    Only lifecycle-bearing sources participate.  Campaign *event* logs record
    stage verdicts (``PASS``, ``RETRYING``, ``NEGATIVE``) which are not claims
    about how a process exited; treating them as such would generate hundreds of
    fake contradictions and bury the real ones.
    """
    findings: List[Dict[str, Any]] = []
    for job_id in sorted(by_job):
        claims: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
        for fact in by_job[job_id]:
            if fact.source in NON_LIFECYCLE_SOURCES:
                continue
            status = _normalize_status(fact.status)
            if status is None:
                continue
            if status in TERMINAL_STATUS_MAP:
                mapped = TERMINAL_STATUS_MAP[status].value
            elif status in QUALIFIED_TERMINAL_MAP:
                mapped = QUALIFIED_TERMINAL_MAP[status][0].value
            else:
                continue
            claims[mapped].append({"legacy_status": str(fact.status),
                                   "source": fact.source,
                                   "locator": fact.locator})
        if len(claims) > 1:
            findings.append(_finding(
                "disagreement", "critical", job_id,
                "conflicting terminal statuses for one job id: "
                + " vs ".join(sorted(claims)),
                field="status",
                values={k: v for k, v in sorted(claims.items())}))
    return findings


def check_field_disagreements(by_job: Mapping[str, Sequence[LegacyFact]],
                              ) -> List[Dict[str, Any]]:
    """Jobs on which artifacts disagree about git sha, label, or phase.

    A ``git_sha`` disagreement is critical: it means two ledgers believe the same
    job ran different code, so any claim about what that job tested rests on
    which ledger the reader consulted.  A ``label`` disagreement is major and
    expected here by construction -- the BTM manifest names a job
    ``btm_IIA_btm_scalar_exact_s0`` while ``pipeline.json`` names it
    ``btm_IIA_G_s0`` -- and is precisely why ``job_id`` is the only usable join
    key across these files.

    Short git shas are compared by prefix: ``92dc605`` and
    ``92dc605f...`` are the same commit, and reporting them as a disagreement
    would be noise.
    """
    findings: List[Dict[str, Any]] = []
    checks = (("git_sha", "critical", lambda f: f.git_sha),
              ("label", "major", lambda f: f.label),
              ("phase", "minor", lambda f: f.phase))
    for job_id in sorted(by_job):
        facts = by_job[job_id]
        for field, severity, getter in checks:
            values: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
            for fact in facts:
                value = getter(fact)
                if value in (None, ""):
                    continue
                values[str(value)].append({"source": fact.source,
                                           "locator": fact.locator})
            if field == "git_sha":
                values = _collapse_sha_prefixes(values)
            if len(values) > 1:
                findings.append(_finding(
                    "disagreement", severity, job_id,
                    f"artifacts disagree on {field}: "
                    + " vs ".join(sorted(values)),
                    field=field,
                    values={k: v for k, v in sorted(values.items())}))
    return findings


def _collapse_sha_prefixes(values: Mapping[str, List[Dict[str, str]]]
                           ) -> Dict[str, List[Dict[str, str]]]:
    """Merge git shas where one is a prefix of another (short vs full form)."""
    keys = sorted(values, key=len, reverse=True)
    merged: Dict[str, List[Dict[str, str]]] = {}
    for key in keys:
        for existing in merged:
            if existing.startswith(key) or key.startswith(existing):
                merged[existing].extend(values[key])
                break
        else:
            merged[key] = list(values[key])
    return merged


def check_orphans(by_job: Mapping[str, Sequence[LegacyFact]]) -> List[Dict[str, Any]]:
    """Job ids known to exactly one artifact.

    Severity is graded by *which* artifact is the sole witness.  A job the BTM
    manifest knows and ``pipeline.json`` does not is a violation of the project's
    own tracking rule and is major; a job only a campaign event log mentions is
    minor, because those logs routinely name jobs from other people's campaigns.
    """
    findings: List[Dict[str, Any]] = []
    pipeline_sources = {SRC_PIPELINE_ACTIVE, SRC_PIPELINE_COMPLETED}
    for job_id in sorted(by_job):
        sources = sorted({f.source for f in by_job[job_id]})
        if len(sources) > 1:
            continue
        only = sources[0]
        if only in pipeline_sources:
            severity, note = "minor", ("tracked only by pipeline.json; no campaign "
                                       "artifact corroborates it")
        elif only == SRC_BTM_MANIFEST:
            severity, note = "major", ("in the BTM manifest but absent from "
                                       "pipeline.json, which the project rules call "
                                       "the single source of truth for submissions")
        else:
            severity, note = "minor", f"mentioned only by {only}"
        findings.append(_finding(
            "orphan", severity, job_id,
            f"job id appears in exactly one artifact ({only})",
            sole_source=only, note=note,
            locators=[f.locator for f in by_job[job_id]]))
    return findings


def check_pipeline_collisions(pipeline_path: str) -> List[Dict[str, Any]]:
    """Duplicate-key collisions inside ``pipeline.json``.

    Two independent collisions, both consequences of the same root cause (the
    file has no primary key and nothing enforces one):

    * a ``run_id`` used by more than one entry -- so a lookup by ``run_id``
      returns an arbitrary one of them;
    * a ``job_id`` used by more than one entry -- so even the scheduler's own
      identifier does not address a single record.
    """
    findings: List[Dict[str, Any]] = []
    if not os.path.exists(pipeline_path):
        return findings
    facts = parse_pipeline(pipeline_path)
    for key, getter, severity in (("run_id", lambda f: f.label, "major"),
                                  ("job_id", lambda f: f.job_id, "critical")):
        groups: Dict[str, List[LegacyFact]] = collections.defaultdict(list)
        for fact in facts:
            value = getter(fact)
            if value:
                groups[str(value)].append(fact)
        for value, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            statuses = sorted({str(m.status) for m in members})
            findings.append(_finding(
                "collision", severity, value if key == "job_id" else None,
                f"{key}={value!r} addresses {len(members)} distinct entries",
                key=key, value=value, multiplicity=len(members),
                distinct_statuses=statuses,
                status_conflict=len(statuses) > 1,
                locators=[m.locator for m in members],
                job_ids=sorted({str(m.job_id) for m in members if m.job_id})))
    return findings


def check_status_enum(facts: Sequence[LegacyFact]) -> List[Dict[str, Any]]:
    """Status strings that are not lifecycle states, and runs stranded mid-flight."""
    findings: List[Dict[str, Any]] = []
    stranded: List[LegacyFact] = []
    unknown_values: Dict[str, List[LegacyFact]] = collections.defaultdict(list)
    for fact in facts:
        if fact.source in NON_LIFECYCLE_SOURCES:
            continue
        status = _normalize_status(fact.status)
        if status is None:
            continue
        if status in TERMINAL_STATUS_MAP or status in QUALIFIED_TERMINAL_MAP:
            if status in QUALIFIED_TERMINAL_MAP:
                unknown_values[str(fact.status)].append(fact)
            continue
        if status in NON_TERMINAL_STATUSES:
            if fact.source == SRC_PIPELINE_COMPLETED:
                stranded.append(fact)
            continue
        unknown_values[str(fact.status)].append(fact)
    for value, members in sorted(unknown_values.items()):
        findings.append(_finding(
            "collision", "major", None,
            f"status value {value!r} is outside the lifecycle enum "
            f"({len(members)} entries)",
            key="status_enum", value=value, multiplicity=len(members),
            locators=[m.locator for m in members][:20]))
    if stranded:
        by_status = collections.Counter(str(f.status) for f in stranded)
        findings.append(_finding(
            "collision", "critical", None,
            f"{len(stranded)} entries sit in completed_runs at a non-terminal "
            f"status ({dict(by_status)}); the ledger never learned their outcome",
            key="stranded_non_terminal", value=None, multiplicity=len(stranded),
            locators=[f.locator for f in stranded][:80]))
    return findings


def check_manifest_freeze(facts: Sequence[LegacyFact]) -> List[Dict[str, Any]]:
    """The BTM manifest's mutable ``status`` cell, frozen at submission.

    Reported as one finding rather than twelve: it is a single structural defect
    (a status field written once and never updated), and twelve identical rows
    would drown the per-job findings that need individual attention.
    """
    rows = [f for f in facts if f.source == SRC_BTM_MANIFEST]
    if not rows:
        return []
    statuses = collections.Counter(_normalize_status(f.status) for f in rows)
    if set(statuses) - {"submitted"}:
        return []
    return [_finding(
        "collision", "critical", None,
        f"all {len(rows)} BTM manifest rows are frozen at status 'submitted'; the "
        f"field is a mutable cell that was written once at submission and never "
        f"updated, so it carries no information about any run's outcome",
        key="frozen_status", value="submitted", multiplicity=len(rows),
        locators=[f.locator for f in rows])]


def check_event_log_drift(paths: LegacyPaths) -> List[Dict[str, Any]]:
    """``events.jsonl`` vs ``status.json['events']`` for the direct-energy campaign.

    The campaign writer appends to both.  Lines present in one and not the other
    were inserted by hand, which means the two views of "what happened" are not
    the same view, and a reader has no way to know which one the author meant.
    Ordering is checked too: an append-only log whose timestamps go backwards
    cannot be replayed by time.
    """
    findings: List[Dict[str, Any]] = []
    if not (os.path.exists(paths.dec_status) and os.path.exists(paths.dec_events)):
        return findings
    with open(paths.dec_status, encoding="utf-8") as handle:
        status = json.load(handle)
    embedded = [json.dumps(e, sort_keys=True) for e in (status.get("events") or [])]
    embedded_set = set(embedded)
    file_events: List[Dict[str, Any]] = []
    with open(paths.dec_events, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                file_events.append(json.loads(line))
    file_set = {json.dumps(e, sort_keys=True) for e in file_events}
    only_file = sorted(file_set - embedded_set)
    only_embedded = sorted(embedded_set - file_set)
    if only_file or only_embedded:
        findings.append(_finding(
            "disagreement", "major", None,
            f"direct_energy_campaign: events.jsonl has {len(file_events)} records "
            f"and status.json['events'] has {len(embedded)}; "
            f"{len(only_file)} only in the file, {len(only_embedded)} only in "
            f"status.json -- the two views of the campaign are not the same view",
            field="event_log_membership",
            only_in_events_jsonl=[json.loads(s) for s in only_file],
            only_in_status_json=[json.loads(s) for s in only_embedded]))
    timestamps = [e.get("at") or e.get("timestamp") for e in file_events]
    out_of_order = [
        {"index": i, "previous": timestamps[i - 1], "this": timestamps[i]}
        for i in range(1, len(timestamps))
        if timestamps[i] and timestamps[i - 1] and timestamps[i] < timestamps[i - 1]]
    if out_of_order:
        findings.append(_finding(
            "disagreement", "major", None,
            f"direct_energy_campaign/events.jsonl has {len(out_of_order)} "
            f"backwards timestamp transitions; an append-only log whose clock "
            f"goes backwards cannot be ordered by time",
            field="event_order", transitions=out_of_order))
    return findings


def check_inverted_jobs_dict(paths: LegacyPaths) -> List[Dict[str, Any]]:
    """The longer-training ``jobs`` dict: inverted orientation and float pollution.

    Two findings, kept separate because they have different consequences.  The
    inversion breaks any reader written against the other campaign's shape.  The
    pollution -- float metrics living inside a dict named ``jobs`` -- is what
    makes an unanchored job-id regex mine phantom ids out of mantissas, and the
    finding lists exactly which keys do it.
    """
    findings: List[Dict[str, Any]] = []
    if not os.path.exists(paths.delt_status):
        return findings
    with open(paths.delt_status, encoding="utf-8") as handle:
        status = json.load(handle)
    jobs = status.get("jobs") or {}
    if not jobs:
        return findings
    key_like_job = sum(1 for k in jobs if is_job_id(k))
    value_like_job = sum(1 for v in jobs.values() if is_job_id(v))
    if value_like_job > key_like_job:
        findings.append(_finding(
            "collision", "major", None,
            f"direct_energy_longer_training/status.json 'jobs' is INVERTED "
            f"relative to direct_energy_campaign's: it maps name -> job_id "
            f"({value_like_job} of {len(jobs)} values are job ids, "
            f"{key_like_job} of {len(jobs)} keys are), so a reader written for "
            f"the other shape reads {len(jobs)} phantom jobs named after arms",
            key="jobs_dict_orientation", value="name->job_id",
            multiplicity=len(jobs)))
    polluted = {k: v for k, v in jobs.items() if not is_job_id(v)}
    if polluted:
        findings.append(_finding(
            "collision", "major", None,
            f"{len(polluted)} entries in that same 'jobs' dict are not jobs at "
            f"all (metrics and state words); an unanchored job-id regex over this "
            f"file yields phantom ids from float mantissas",
            key="jobs_dict_pollution", value=None, multiplicity=len(polluted),
            entries={k: v for k, v in sorted(polluted.items())}))
    return findings


def check_timestamp_key_drift(facts: Sequence[LegacyFact], paths: LegacyPaths
                              ) -> List[Dict[str, Any]]:
    """Competing key names for the same concept inside one event log."""
    findings: List[Dict[str, Any]] = []
    if not os.path.exists(paths.delt_events):
        return findings
    with open(paths.delt_events, encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    for label, keys in (("timestamp", ("at", "timestamp")),
                        ("job", ("job", "jobs"))):
        counts = {k: sum(1 for e in events if k in e) for k in keys}
        if all(counts.values()):
            findings.append(_finding(
                "collision", "minor", None,
                f"direct_energy_longer_training/events.jsonl uses two competing "
                f"{label} keys: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
                key=f"competing_{label}_keys", value=None,
                multiplicity=sum(counts.values()), counts=counts))
    shapes = collections.Counter(tuple(sorted(e.keys())) for e in events)
    findings.append(_finding(
        "collision", "minor", None,
        f"direct_energy_longer_training/events.jsonl has {len(shapes)} distinct "
        f"key-shapes over {len(events)} records: it has no schema",
        key="event_shape_entropy", value=None, multiplicity=len(shapes)))
    return findings


def check_pipeline_schema_drift(paths: LegacyPaths) -> List[Dict[str, Any]]:
    """Missing documented keys and ad-hoc near-duplicate key pairs in pipeline.json."""
    findings: List[Dict[str, Any]] = []
    if not os.path.exists(paths.pipeline):
        return findings
    facts = parse_pipeline(paths.pipeline)
    for key in ("expected_runtime", "final_metric", "exit_code"):
        missing = [f for f in facts if key not in f.raw]
        if missing:
            findings.append(_finding(
                "collision", "minor", None,
                f"pipeline.json: {len(missing)} of {len(facts)} entries lack "
                f"{key!r}",
                key="schema_drift", value=key, multiplicity=len(missing)))
    documented = set()
    for fact in facts:
        documented.update(fact.raw.keys())
    for left, right in PIPELINE_NEAR_DUPLICATE_KEYS:
        if left in documented and right in documented:
            n_left = sum(1 for f in facts if left in f.raw)
            n_right = sum(1 for f in facts if right in f.raw)
            findings.append(_finding(
                "collision", "minor", None,
                f"pipeline.json carries the near-duplicate key pair "
                f"{left!r} ({n_left}) / {right!r} ({n_right}); no evidence says "
                f"they are synonyms, so neither can be safely merged",
                key="near_duplicate_keys", value=f"{left}|{right}",
                multiplicity=n_left + n_right))
    return findings


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def analyze(project_root: str) -> Dict[str, Any]:
    """Run every check.  Returns the machine-readable report."""
    paths = LegacyPaths(project_root)
    facts, missing = load_all_facts(paths)

    by_job: Dict[str, List[LegacyFact]] = collections.defaultdict(list)
    for fact in facts:
        if fact.job_id:
            by_job[fact.job_id].append(fact)

    findings: List[Dict[str, Any]] = []
    findings += check_status_disagreements(by_job)
    findings += check_field_disagreements(by_job)
    findings += check_orphans(by_job)
    findings += check_pipeline_collisions(paths.pipeline)
    findings += check_status_enum(facts)
    findings += check_manifest_freeze(facts)
    findings += check_event_log_drift(paths)
    findings += check_inverted_jobs_dict(paths)
    findings += check_timestamp_key_drift(facts, paths)
    findings += check_pipeline_schema_drift(paths)

    findings.sort(key=lambda f: (SEVERITY_ORDER.index(f["severity"]), f["kind"],
                                 str(f.get("job_id") or ""), f["summary"]))

    coverage: Dict[str, Dict[str, int]] = {}
    for source in sorted({f.source for f in facts}):
        jobs = {f.job_id for f in facts if f.source == source and f.job_id}
        coverage[source] = {
            "facts": sum(1 for f in facts if f.source == source),
            "distinct_job_ids": len(jobs),
        }

    return {
        "project_root": os.path.relpath(project_root),
        "missing_artifacts": missing,
        "totals": {
            "legacy_facts": len(facts),
            "facts_without_job_id": sum(1 for f in facts if not f.job_id),
            "distinct_job_ids": len(by_job),
            "job_ids_in_multiple_artifacts": sum(
                1 for j in by_job if len({f.source for f in by_job[j]}) > 1),
            "findings": len(findings),
        },
        "findings_by_kind": dict(collections.Counter(f["kind"] for f in findings)),
        "findings_by_severity": dict(collections.Counter(
            f["severity"] for f in findings)),
        "coverage_by_source": coverage,
        "findings": findings,
    }


def render_markdown(report: Mapping[str, Any], *, max_rows: int = 400) -> str:
    """Human-readable contradiction tables."""
    lines: List[str] = []
    add = lines.append
    add("# Legacy provenance contradiction report\n")
    add(f"Source: `{report['project_root']}`.  Generated by "
        "`python -m telemetry.contradictions`; re-runnable at any time, "
        "read-only with respect to every legacy file.\n")

    add("## Totals\n")
    add("| quantity | value |")
    add("| --- | ---: |")
    for key, value in report["totals"].items():
        add(f"| {key} | {value} |")
    add("")

    add("## Findings by kind and severity\n")
    add("| bucket | count |")
    add("| --- | ---: |")
    for key, value in sorted(report["findings_by_kind"].items()):
        add(f"| kind: {key} | {value} |")
    for key in SEVERITY_ORDER:
        if key in report["findings_by_severity"]:
            add(f"| severity: {key} | {report['findings_by_severity'][key]} |")
    add("")

    add("## Artifact coverage\n")
    add("| artifact | facts | distinct job ids |")
    add("| --- | ---: | ---: |")
    for source, stats in sorted(report["coverage_by_source"].items()):
        add(f"| `{source}` | {stats['facts']} | {stats['distinct_job_ids']} |")
    add("")

    for kind, title in (("disagreement", "Disagreements"),
                        ("collision", "Duplicate-key and schema collisions"),
                        ("orphan", "Orphans (job id known to one artifact only)")):
        rows = [f for f in report["findings"] if f["kind"] == kind]
        add(f"## {title} ({len(rows)})\n")
        if not rows:
            add("_none_\n")
            continue
        add("| severity | job id | detail |")
        add("| --- | --- | --- |")
        for row in rows[:max_rows]:
            job = f"`{row['job_id']}`" if row.get("job_id") else "—"
            detail = str(row["summary"]).replace("|", "\\|")
            add(f"| {row['severity']} | {job} | {detail} |")
        if len(rows) > max_rows:
            add(f"\n_{len(rows) - max_rows} further {kind} rows omitted from this "
                f"table; all of them are in the JSON report._\n")
        add("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-reference every job id across all legacy artifacts.")
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--project-root", default=default_root)
    parser.add_argument("--out-dir", default=None,
                        help="write contradictions.json and contradictions.md here")
    parser.add_argument("--max-rows", type=int, default=400,
                        help="cap rows per markdown table (JSON is never capped)")
    args = parser.parse_args(argv)

    report = analyze(args.project_root)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "contradictions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(args.out_dir, "contradictions.md"), "w",
                  encoding="utf-8") as handle:
            handle.write(render_markdown(report, max_rows=args.max_rows))
    json.dump({**report["totals"],
               "by_kind": report["findings_by_kind"],
               "by_severity": report["findings_by_severity"]},
              sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
