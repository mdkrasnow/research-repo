"""Legacy-shaped projections of the event log.  Outputs only, never inputs.

``AGENTS.md`` requires two documents: ``.state/pipeline.json``'s
``active_runs``/``completed_runs`` ledger, and ``results_variants.tsv``.  Both
were hand-maintained, and both drifted -- which is the *predicted* behaviour of
any document that is simultaneously the record of a fact and the only evidence
for it.  A ``status`` field a human types is a claim; a ``status`` field folded
out of an append-only log is a derivation.

So these formats are kept, and the protocol around them is kept, but their
epistemic status is inverted:

* They are **views**: pure functions of the event log, regenerable at any time.
* They are **never read back** by any part of the telemetry system.  Nothing in
  this package parses ``pipeline.json`` to learn anything; it exists to be
  written and to be read by humans and by the repo's older scripts.
* Drift is **detectable**: ``--check`` re-derives the view and diffs it against
  the committed file without writing.  A nonzero exit means the committed
  document no longer follows from the log, which is exactly the condition that
  previously went unnoticed for months.

Why keep the legacy shape at all rather than replacing it?  Because the
protocol, the sbatch scripts and the operator's habits are all built on it, and
a migration that requires everything to change at once does not happen.  A view
lets the new source of truth take over underneath a stable surface.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .ledger import LedgerView, build_ledger
from .read import Campaign, LogicalRun, RunLog, load_campaign
from .schema import EventType, RunStatus

#: Fields ``AGENTS.md`` requires on every ``active_runs`` entry.  Kept explicit
#: so a view that cannot supply one emits ``None`` loudly rather than omitting
#: the key and letting a consumer's ``.get()`` invent a default.
PIPELINE_FIELDS = ("run_id", "job_id", "partition", "status", "description",
                   "submitted_at", "git_sha", "sbatch_path", "expected_runtime",
                   "phase", "gate")

#: ``results_variants.tsv`` header, in order.
RESULTS_COLUMNS = ("run_id", "job_id", "date", "phase", "gate", "checkpoint",
                   "metric_name", "metric_value", "pass", "notes")


def _param(log: RunLog, key: str) -> Any:
    """A non-identifying annotation carried through START.

    Description, gate, partition and the rest are deliberately *not* part of the
    identity hash (:data:`telemetry.ids.NON_IDENTIFYING_KEYS`) -- two runs that
    differ only in their prose description are the same experiment.  They still
    have to reach this view, so they ride along in the START payload.
    """
    start = log.start or {}
    if start.get(key) is not None:
        return start[key]
    config = start.get("config")
    if isinstance(config, Mapping) and config.get(key) is not None:
        return config[key]
    return None


def _slurm(log: RunLog, key: str) -> Optional[str]:
    slurm = (log.start or {}).get("slurm") or {}
    value = slurm.get(key)
    return str(value) if value is not None else None


def _date(stamp: Optional[str]) -> Optional[str]:
    return stamp.split("T")[0] if stamp else None


def _observed(log: RunLog) -> List[Dict[str, Any]]:
    return [e for e in log.events if e["event"] == EventType.OBSERVED.value]


def _entry(run: LogicalRun, log: RunLog) -> Dict[str, Any]:
    """One pipeline-shaped entry, folded from one execution."""
    spec = run.spec
    status = log.status
    observations = _observed(log)
    latest = observations[-1] if observations else {}
    entry: Dict[str, Any] = {
        "run_id": (spec.slug().rsplit("__", 1)[0] if spec
                   else os.path.basename(run.directory).rsplit("__", 1)[0]),
        "job_id": log.job_id.replace("-", "_") if log.job_id[:1].isdigit() else log.job_id,
        "partition": _slurm(log, "slurm_job_partition") or _param(log, "partition"),
        "status": status.value if status else "unknown",
        "description": _param(log, "description"),
        "submitted_at": _date((log.start or {}).get("ts")),
        "git_sha": log.git_sha,
        "sbatch_path": _param(log, "sbatch_path"),
        "expected_runtime": _param(log, "expected_runtime"),
        "phase": (spec.phase if spec else _param(log, "phase")),
        "gate": _param(log, "gate"),
        # -- derived facts the hand-maintained file could never keep current --
        "run_uid": log.run_uid,
        "exec_id": log.exec_id,
        "attempt": log.attempt,
        "last_step": log.last_step,
        "planned_steps": log.planned_steps,
        "inferred_end": log.inferred_end,
    }
    if latest:
        # Surface the reconciler's verdict rather than silently overwriting the
        # run's own status with the scheduler's.  Both are visible; the reader
        # decides.
        entry["scheduler_state"] = latest.get("sacct_state")
        entry["disagreement"] = latest.get("disagreement")
        entry["completed_at"] = latest.get("sacct_end")
        entry["duration_seconds"] = latest.get("elapsed_seconds")
        entry["exit_code"] = latest.get("exit_code")
        entry["exit_signal"] = latest.get("exit_signal")
        entry["trusted_exit_code"] = latest.get("trusted_exit_code")
    end = log.end
    if end is not None:
        entry["final_metric"] = _final_metric_text(log)
        entry["wall_seconds"] = end.get("wall_seconds")
        if end.get("error"):
            entry["error"] = str(end["error"])[-500:]
    return entry


def _final_metric_text(log: RunLog) -> Optional[str]:
    """A one-line summary of the last EVAL, for the human-facing column."""
    evals = log.records(EventType.EVAL)
    if not evals:
        return None
    last = evals[-1]
    bits = [f"{k}={v}" for k, v in sorted(last.items())
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in ("v", "seq", "step")]
    kind = last.get("kind", "eval")
    return f"step {last.get('step')} {kind}: " + ", ".join(bits) if bits else None


def pipeline_view(campaign: Optional[Campaign] = None, *, root: Optional[str] = None
                  ) -> Dict[str, List[Dict[str, Any]]]:
    """``{"active_runs": [...], "completed_runs": [...]}`` folded from the log.

    Partition rule: an execution is *active* iff its own stream has no terminal
    record.  Deliberately not "iff the scheduler says it is running" -- the
    scheduler's opinion is recorded as an OBSERVED event and shown in the
    ``disagreement`` field, but it does not get to silently rewrite what the run
    reported about itself.  A run the scheduler thinks finished while its stream
    never sealed stays visible as ``lost_telemetry``, which is the finding.
    """
    if campaign is None:
        if root is None:
            raise ValueError("pipeline_view needs a campaign or a root")
        campaign = load_campaign(root)
    active: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []
    for run in sorted(campaign.runs, key=lambda r: r.run_uid):
        for log in sorted(run.executions, key=lambda l: (l.attempt, l.exec_id)):
            entry = _entry(run, log)
            (completed if log.end is not None else active).append(entry)
    key = lambda e: (str(e.get("submitted_at") or ""), str(e.get("job_id") or ""),
                     e["exec_id"])
    return {"active_runs": sorted(active, key=key),
            "completed_runs": sorted(completed, key=key)}


# -- results_variants.tsv ----------------------------------------------------

def _checkpoint_for(log: RunLog, step: Optional[int]) -> Optional[str]:
    """The newest checkpoint written at or before ``step``.

    Binds a reported metric to the artifact it was computed from, which the
    hand-maintained TSV could only do by someone remembering to paste a path.
    """
    best: Optional[str] = None
    best_step = -1
    for record in log.records(EventType.ARTIFACT):
        artifact_step = record.get("step")
        if not isinstance(artifact_step, int):
            continue
        if step is not None and artifact_step > step:
            continue
        if artifact_step >= best_step:
            best_step, best = artifact_step, str(record.get("path") or "")
    return best


def _reportable_evals(log: RunLog) -> List[Dict[str, Any]]:
    """Which EVAL records become TSV rows.

    Opt-in first: a record carrying ``report=True`` is one the producer meant as
    a headline result.  Falling back to "the last EVAL of each kind" when
    nothing opted in keeps the view useful for runs written before the flag
    existed, without ever emitting a row per eval step (which would turn a
    results table into a metrics dump).
    """
    opted = [r for r in log.records(EventType.EVAL) if r.get("report")]
    if opted:
        return opted
    last_by_kind: Dict[str, Dict[str, Any]] = {}
    for record in log.records(EventType.EVAL):
        last_by_kind[str(record.get("kind", "eval"))] = record
    return [last_by_kind[k] for k in sorted(last_by_kind)]


def results_rows(campaign: Optional[Campaign] = None, *, root: Optional[str] = None
                 ) -> List[Dict[str, Any]]:
    """``results_variants.tsv``-shaped rows, one per reported metric.

    ``pass`` is emitted only when the producer recorded a gate verdict
    (``passed`` on the EVAL record).  It is never inferred here: deciding
    whether a number clears a pre-registered gate is a research judgement that
    belongs at the point where the gate is defined, and a view that guessed it
    would be manufacturing exactly the kind of post-hoc reinterpretation the
    project's stop conditions forbid.
    """
    if campaign is None:
        if root is None:
            raise ValueError("results_rows needs a campaign or a root")
        campaign = load_campaign(root)
    rows: List[Dict[str, Any]] = []
    for run in sorted(campaign.runs, key=lambda r: r.run_uid):
        spec = run.spec
        for log in sorted(run.executions, key=lambda l: (l.attempt, l.exec_id)):
            run_id = (spec.slug().rsplit("__", 1)[0] if spec
                      else os.path.basename(run.directory).rsplit("__", 1)[0])
            for record in _reportable_evals(log):
                step = record.get("step")
                checkpoint = _checkpoint_for(log, step if isinstance(step, int) else None)
                metrics = sorted(
                    (k, v) for k, v in record.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and k not in ("v", "seq", "step"))
                for name, value in metrics:
                    rows.append({
                        "run_id": run_id,
                        "job_id": log.job_id.replace("-", "_")
                                  if log.job_id[:1].isdigit() else log.job_id,
                        "date": _date(record.get("ts")),
                        "phase": (spec.phase if spec else "") or "",
                        "gate": _param(log, "gate") or "",
                        "checkpoint": checkpoint or "",
                        "metric_name": f"{record.get('kind', 'eval')}.{name}",
                        "metric_value": value,
                        "pass": ("" if record.get("passed") is None
                                 else str(bool(record["passed"])).lower()),
                        "notes": str(record.get("notes") or ""),
                    })
    rows.sort(key=lambda r: (r["run_id"], str(r["job_id"]), r["metric_name"],
                            str(r["date"] or "")))
    return rows


def render_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render rows in the committed TSV's exact column order.

    Tabs and newlines inside a cell would silently create a column, so they are
    escaped rather than written through -- a corrupted row in a results table is
    worse than an ugly one.
    """
    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("\t", "\\t").replace("\n", " ").replace("\r", " ")

    lines = ["\t".join(RESULTS_COLUMNS)]
    for row in rows:
        lines.append("\t".join(cell(row.get(column)) for column in RESULTS_COLUMNS))
    return "\n".join(lines) + "\n"


def render_pipeline(view: Mapping[str, Any]) -> str:
    """Canonical JSON for the pipeline view; key-sorted, so it diffs cleanly."""
    return json.dumps(view, indent=2, sort_keys=True, allow_nan=False) + "\n"


# -- drift checking ----------------------------------------------------------

@dataclasses.dataclass
class Drift:
    """The difference between a committed document and its regeneration."""

    path: str
    drifted: bool
    reason: str
    diff: List[str] = dataclasses.field(default_factory=list)

    def render(self) -> str:
        head = ("DRIFT" if self.drifted else "OK") + f": {self.path} -- {self.reason}"
        return "\n".join([head, *self.diff[:200]])


def check_text(path: str, generated: str, *, context: int = 2) -> Drift:
    try:
        with open(path, encoding="utf-8") as handle:
            committed = handle.read()
    except OSError:
        return Drift(path, True, "committed file is missing or unreadable; the "
                                 "view has never been generated from this log")
    if committed == generated:
        return Drift(path, False, "matches the regenerated view")
    diff = list(difflib.unified_diff(
        committed.splitlines(), generated.splitlines(),
        fromfile=path, tofile="<regenerated>", lineterm="", n=context))
    return Drift(path, True, "committed content does not follow from the event "
                             "log", diff)


def check_pipeline(path: str, view: Mapping[str, Any]) -> Drift:
    """Compare only the two ledger keys, leaving hand-owned keys alone.

    ``pipeline.json`` also carries ``phase``, ``gates``, ``next_action`` and
    other genuinely human-authored fields.  Those are decisions, not
    observations, and are none of this module's business; comparing the whole
    document would report permanent, meaningless drift and train the reader to
    ignore the check.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            committed = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return Drift(path, True, f"unreadable: {exc}")
    subset = {key: committed.get(key, []) for key in ("active_runs", "completed_runs")}
    generated = {key: view.get(key, []) for key in ("active_runs", "completed_runs")}
    if subset == generated:
        return Drift(path, False, "active_runs/completed_runs match the log")
    diff = list(difflib.unified_diff(
        json.dumps(subset, indent=2, sort_keys=True).splitlines(),
        json.dumps(generated, indent=2, sort_keys=True).splitlines(),
        fromfile=f"{path}:committed", tofile="<regenerated>", lineterm="", n=2))
    return Drift(path, True, "active_runs/completed_runs do not follow from the "
                             "event log", diff)


def write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def update_pipeline_file(path: str, view: Mapping[str, Any]) -> None:
    """Replace ONLY the two view-owned keys, transactionally.

    The human-authored keys are preserved byte-for-byte in content; only
    ``active_runs`` and ``completed_runs`` are regenerated.  The update runs
    inside :func:`telemetry.reconcile.file_transaction`, so a concurrent editor
    is serialized against rather than clobbered.
    """
    from .reconcile import file_transaction

    with file_transaction(path) as state:
        state["active_runs"] = list(view.get("active_runs", []))
        state["completed_runs"] = list(view.get("completed_runs", []))


# -- CLI ---------------------------------------------------------------------

def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m telemetry.views",
        description="Project the event log into the legacy pipeline.json and "
                    "results_variants.tsv shapes. These are OUTPUTS: they are "
                    "regenerated from the log and never read back into it.")
    parser.add_argument("--root", required=True, help="telemetry root")
    parser.add_argument("--pipeline", default=None, help="pipeline.json path")
    parser.add_argument("--results", default=None, help="results_variants.tsv path")
    parser.add_argument("--ledger", default=None,
                        help="also write the markdown ledger here")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; report drift and exit 1 if any "
                             "committed file no longer follows from the log")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = load_campaign(args.root)
    view = pipeline_view(campaign)
    rows = results_rows(campaign)

    if args.check:
        drifts: List[Drift] = []
        if args.pipeline:
            drifts.append(check_pipeline(args.pipeline, view))
        if args.results:
            drifts.append(check_text(args.results, render_tsv(rows)))
        if args.ledger:
            drifts.append(check_text(args.ledger,
                                     build_ledger(args.root).render_markdown()))
        if not drifts:
            print("nothing to check: pass --pipeline / --results / --ledger")
            return 2
        for drift in drifts:
            print(drift.render())
        return 1 if any(d.drifted for d in drifts) else 0

    if args.pipeline:
        update_pipeline_file(args.pipeline, view)
        print(f"pipeline: {len(view['active_runs'])} active, "
              f"{len(view['completed_runs'])} completed -> {args.pipeline}")
    if args.results:
        write_atomic(args.results, render_tsv(rows))
        print(f"results: {len(rows)} row(s) -> {args.results}")
    if args.ledger:
        write_atomic(args.ledger, build_ledger(args.root).render_markdown())
        print(f"ledger -> {args.ledger}")
    if not (args.pipeline or args.results or args.ledger):
        print(render_pipeline(view), end="")
        print(render_tsv(rows), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
