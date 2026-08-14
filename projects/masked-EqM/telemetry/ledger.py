"""The ledger: a pure fold from event logs to a materialized state view.

Why a fold and not a document
-----------------------------
Every state document this repo has kept by hand -- ``pipeline.json``'s
``active_runs``, ``results/btm/manifest.jsonl``, ``results_variants.tsv`` -- has
the same defect: it is *written* by one process and *believed* by every other
one, with no way to check it against what actually happened.  ``manifest.jsonl``
froze ``status="submitted"`` on runs that had finished months earlier, and
nothing in the system could notice, because the manifest was the only thing that
claimed to know.

This module inverts that.  The ledger is a **fold over the append-only event
log** and holds no state of its own.  Concretely it is a function

    ledger : EventLog -> LedgerView

that is total, deterministic, and side-effect free.  Three consequences follow,
and they are the entire reason the module exists:

* **Regenerability.**  Deleting the ledger loses nothing.  Any disagreement
  between a committed ledger and a freshly folded one is, by construction,
  evidence that the *log* changed -- never that the ledger drifted.
* **Diffability.**  Because the fold is deterministic and the serialization is
  key-sorted, the rendered ledger is a pure function of the log bytes.  It can
  be committed, and ``git diff`` on it reads as "what the cluster did since last
  time" rather than "what someone remembered to type".
* **No write path to corrupt.**  There is no mutation API here at all.  A caller
  that wants to change what the ledger says must append an event, which is the
  only operation the design permits.

Determinism, specifically
-------------------------
"Deterministic" is load-bearing and is easy to lose by accident, so it is
enforced rather than hoped for:

* No wall-clock reads.  Nothing in the output comes from ``now()``; every
  timestamp is copied out of a record.  (``LedgerView.to_json`` takes an explicit
  ``generated_at`` only if a caller insists, and it defaults to ``None``.)
* No absolute paths in the payload by default.  The telemetry root differs
  between the cluster and a laptop; embedding it would make the same log render
  to different bytes on two machines and destroy the diff property.  Paths are
  emitted relative to the root.
* Total orders everywhere.  Runs sort by ``(campaign, phase, arm, seed,
  run_uid)`` and executions by ``(attempt, job_id, exec_id)``; both tuples end in
  a unique key so no tie is ever broken by dict or filesystem order.
* Key-sorted JSON with a fixed separator, matching :func:`telemetry.schema.dumps`.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .read import Campaign, LogicalRun, RunLog, load_campaign
from .schema import EventType, RunStatus

#: Bumped when the ledger's *rendered shape* changes in a way that would make a
#: committed ledger diff spuriously.  Consumers can then tell "the cluster did
#: something" apart from "the renderer was upgraded".
LEDGER_VERSION = 1


def _relpath(path: str, root: str) -> str:
    """Path relative to the telemetry root, with forward slashes.

    Absolute paths are machine-specific; a ledger containing them renders
    differently on the cluster and on a laptop from the *same* log, which breaks
    the byte-identity property the whole module is built on.
    """
    try:
        rel = os.path.relpath(path, root)
    except ValueError:  # different drive on Windows
        rel = path
    return rel.replace(os.sep, "/")


@dataclasses.dataclass(frozen=True)
class Transition:
    """One observed change of lifecycle state, with its provenance.

    ``source`` is the epistemically important field.  ``"run"`` means the process
    said this about itself; ``"observer"`` means an external party (the
    scheduler, via :mod:`telemetry.reconcile`) said it.  Collapsing the two is
    how "sacct thinks it completed" silently becomes "the run completed" -- the
    exact confusion :mod:`telemetry.reconcile` exists to surface.
    """

    at: str
    seq: int
    event: str
    status: Optional[str]
    source: str
    detail: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"at": self.at, "seq": self.seq, "event": self.event,
                "status": self.status, "source": self.source,
                "detail": self.detail}


@dataclasses.dataclass(frozen=True)
class ArtifactRef:
    """A file bound to the exact step and execution that produced it."""

    path: str
    kind: str
    step: Optional[int]
    bytes: Optional[int]

    def to_json(self) -> Dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "step": self.step,
                "bytes": self.bytes}


@dataclasses.dataclass
class ExecutionEntry:
    """The folded state of one physical execution."""

    exec_id: str
    run_uid: str
    job_id: str
    attempt: int
    status: str
    inferred_end: bool
    started_at: Optional[str]
    ended_at: Optional[str]
    wall_seconds: Optional[float]
    last_step: int
    planned_steps: Optional[int]
    truncated: Optional[bool]
    git_sha: Optional[str]
    world_size: Optional[int]
    partition: Optional[str]
    nodelist: Optional[str]
    stream: str
    timeline: List[Transition] = dataclasses.field(default_factory=list)
    artifacts: List[ArtifactRef] = dataclasses.field(default_factory=list)
    #: Reconciler findings copied verbatim out of OBSERVED records.  The ledger
    #: never computes a disagreement itself -- it reports the ones that were
    #: recorded, so the ledger stays a fold and the classification stays
    #: attributable to the reconcile run that made it.
    observations: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    defects: List[str] = dataclasses.field(default_factory=list)

    @property
    def open(self) -> bool:
        """True when this execution has no terminal record of its own."""
        return self.status in (RunStatus.RUNNING.value, "unknown")

    def to_json(self) -> Dict[str, Any]:
        return {
            "exec_id": self.exec_id,
            "run_uid": self.run_uid,
            "job_id": self.job_id,
            "attempt": self.attempt,
            "status": self.status,
            "inferred_end": self.inferred_end,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "wall_seconds": self.wall_seconds,
            "last_step": self.last_step,
            "planned_steps": self.planned_steps,
            "truncated": self.truncated,
            "git_sha": self.git_sha,
            "world_size": self.world_size,
            "partition": self.partition,
            "nodelist": self.nodelist,
            "stream": self.stream,
            "timeline": [t.to_json() for t in self.timeline],
            "artifacts": [a.to_json() for a in self.artifacts],
            "observations": self.observations,
            "defects": self.defects,
        }


@dataclasses.dataclass
class RunEntry:
    """The folded state of one logical run and all its attempts."""

    run_uid: str
    campaign: str
    phase: str
    arm: str
    seed: Any
    git_sha: Optional[str]
    planned_steps: Optional[int]
    slug: str
    executions: List[ExecutionEntry] = dataclasses.field(default_factory=list)

    @property
    def status(self) -> str:
        """The run's status: the best outcome any of its attempts reached.

        "Best" and not "latest": a logical run that completed once and was then
        re-attempted (a common accident when a launcher is re-run) has in fact
        completed, and reporting the newest attempt's status would hide that.
        Deliberately NOT a merge of attempts -- see
        :attr:`telemetry.read.LogicalRun.canonical` for why concatenating
        attempts manufactures trajectories that never happened.
        """
        if not self.executions:
            return "no_executions"
        order = [RunStatus.COMPLETED.value, RunStatus.RUNNING.value,
                 RunStatus.TIMEOUT.value, RunStatus.PREEMPTED.value,
                 RunStatus.CRASHED.value, RunStatus.CANCELLED.value,
                 RunStatus.LOST.value, "unknown"]
        rank = {name: i for i, name in enumerate(order)}
        return min((e.status for e in self.executions),
                   key=lambda s: (rank.get(s, len(order)), s))

    @property
    def attempts(self) -> int:
        return len(self.executions)

    @property
    def last_step(self) -> int:
        return max((e.last_step for e in self.executions), default=-1)

    def to_json(self) -> Dict[str, Any]:
        return {
            "run_uid": self.run_uid,
            "campaign": self.campaign,
            "phase": self.phase,
            "arm": self.arm,
            "seed": self.seed,
            "git_sha": self.git_sha,
            "planned_steps": self.planned_steps,
            "slug": self.slug,
            "status": self.status,
            "attempts": self.attempts,
            "last_step": self.last_step,
            "executions": [e.to_json() for e in self.executions],
        }


@dataclasses.dataclass
class LedgerView:
    """Every logical run under a telemetry root, folded.

    Holds no reference to the root: see the determinism note in the module
    docstring.  ``root`` is kept only so callers can re-fold.
    """

    runs: List[RunEntry] = dataclasses.field(default_factory=list)
    version: int = LEDGER_VERSION

    # -- lookups -------------------------------------------------------------

    def by_uid(self, run_uid: str) -> Optional[RunEntry]:
        for run in self.runs:
            if run.run_uid == run_uid:
                return run
        return None

    def executions(self) -> List[ExecutionEntry]:
        return [e for run in self.runs for e in run.executions]

    def open_executions(self) -> List[ExecutionEntry]:
        """Executions with no terminal record -- the reconciler's work list."""
        return [e for e in self.executions() if e.open]

    # -- serialization -------------------------------------------------------

    def to_json(self, *, generated_at: Optional[str] = None) -> Dict[str, Any]:
        """The ledger as a plain dict.

        ``generated_at`` defaults to ``None`` and is omitted entirely when unset.
        A timestamp would make every regeneration differ from the last, which
        would defeat the point of committing the file: the diff must show what
        the *cluster* did, not what time it was when someone looked.
        """
        payload: Dict[str, Any] = {
            "ledger_version": self.version,
            "runs": [run.to_json() for run in self.runs],
        }
        if generated_at is not None:
            payload["generated_at"] = generated_at
        return payload

    def dumps(self, *, generated_at: Optional[str] = None) -> str:
        """Canonical JSON text, newline-terminated.

        ``sort_keys`` + fixed separators means the bytes are a pure function of
        the content, exactly as in :func:`telemetry.schema.dumps`.
        """
        return json.dumps(self.to_json(generated_at=generated_at),
                          sort_keys=True, indent=2, allow_nan=False) + "\n"

    # -- rendering -----------------------------------------------------------

    def render_markdown(self, *, title: str = "Run ledger") -> str:
        """A human-readable ledger, deterministic to the byte.

        Two tables, because they answer different questions: a per-run roll-up
        ("what state is each experiment in") and a per-execution detail table
        ("what did each attempt actually do").  Disagreements recorded by the
        reconciler get their own section rather than a footnote -- a scheduler
        that contradicts a run is a finding, not a formatting concern.
        """
        lines: List[str] = [f"# {title}", ""]
        lines.append(f"_Generated by `telemetry.ledger` v{self.version}. "
                     "This file is a FOLD over the event log and is fully "
                     "regenerable; do not hand-edit._")
        lines.append("")

        if not self.runs:
            lines.append("_No runs under this telemetry root._")
            return "\n".join(lines) + "\n"

        lines.append("## Logical runs")
        lines.append("")
        lines.append("| run_uid | campaign | phase | arm | seed | status | "
                     "attempts | last_step | planned | git_sha |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for run in self.runs:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                run.run_uid, run.campaign or "-", run.phase or "-", run.arm or "-",
                _fmt(run.seed), run.status, run.attempts, run.last_step,
                _fmt(run.planned_steps), (run.git_sha or "-")[:12]))
        lines.append("")

        lines.append("## Executions")
        lines.append("")
        lines.append("| exec_id | job_id | attempt | status | src | started | "
                     "wall_s | last_step | world | artifacts |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for run in self.runs:
            for execution in run.executions:
                lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    execution.exec_id, execution.job_id, execution.attempt,
                    execution.status,
                    "inferred" if execution.inferred_end else "self",
                    execution.started_at or "-", _fmt(execution.wall_seconds),
                    execution.last_step, _fmt(execution.world_size),
                    len(execution.artifacts)))
        lines.append("")

        findings = [(run, execution, obs)
                    for run in self.runs for execution in run.executions
                    for obs in execution.observations
                    if obs.get("disagreement") not in (None, "", "agree")]
        lines.append("## Reconciliation findings")
        lines.append("")
        if not findings:
            lines.append("_No recorded disagreement between the scheduler and "
                         "any run's own terminal record._")
        else:
            lines.append("| exec_id | disagreement | scheduler | self-reported | detail |")
            lines.append("|---|---|---|---|---|")
            for _run, execution, obs in findings:
                lines.append("| {} | {} | {} | {} | {} |".format(
                    execution.exec_id, obs.get("disagreement"),
                    obs.get("sacct_state", "-"),
                    obs.get("self_status", "-"),
                    str(obs.get("detail", "")).replace("|", "\\|")))
        lines.append("")

        defective = [(execution.exec_id, defect)
                     for run in self.runs for execution in run.executions
                     for defect in execution.defects]
        if defective:
            lines.append("## Stream defects")
            lines.append("")
            for exec_id, defect in defective:
                lines.append(f"- `{exec_id}`: {defect}")
            lines.append("")
        return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


# -- the fold ---------------------------------------------------------------

def _fold_execution(log: RunLog, root: str) -> ExecutionEntry:
    """Fold one execution's event stream into its materialized state."""
    start = log.start
    end = log.end
    slurm = (start or {}).get("slurm") or {}

    timeline: List[Transition] = []
    artifacts: List[ArtifactRef] = []
    observations: List[Dict[str, Any]] = []

    for record in sorted(log.events, key=lambda e: e["seq"]):
        event = record["event"]
        if event == EventType.START.value:
            timeline.append(Transition(
                at=record["ts"], seq=record["seq"], event=event,
                status=RunStatus.RUNNING.value, source="run",
                detail=f"job {record.get('job_id')} attempt {record.get('attempt')}"))
        elif event == EventType.OBSERVED.value:
            timeline.append(Transition(
                at=record["ts"], seq=record["seq"], event=event,
                status=record.get("status"), source="observer",
                detail=str(record.get("disagreement") or record.get("detail") or "")))
            observations.append({
                key: value for key, value in sorted(record.items())
                if key not in ("v", "run_uid", "exec_id", "event")
            })
        elif event == EventType.END.value:
            timeline.append(Transition(
                at=record["ts"], seq=record["seq"], event=event,
                status=record.get("status"), source="run",
                detail=("inferred by " + str(record.get("inferred_by") or "sealer"))
                if record.get("inferred") else "self-reported"))
        elif event == EventType.ARTIFACT.value:
            artifacts.append(ArtifactRef(
                path=str(record.get("path", "")), kind=str(record.get("kind", "")),
                step=record.get("step"), bytes=record.get("bytes")))

    defects: List[str] = []
    if log.seq_gaps:
        spans = ", ".join(f"{a}-{b}" for a, b in log.seq_gaps[:5])
        defects.append(f"record loss: missing seq {spans}")
    if log.duplicate_seqs:
        defects.append(f"duplicate seq {log.duplicate_seqs[:5]}: two writers")
    if log.issues:
        first = log.issues[0]
        defects.append(f"{len(log.issues)} unreadable line(s), first at "
                       f"line {first.lineno}: {first.reason}")
    if start is None:
        defects.append("no START record: this stream's plan and identity are unknown")

    status = log.status
    wall = (end or {}).get("wall_seconds")

    return ExecutionEntry(
        exec_id=log.exec_id,
        run_uid=log.run_uid,
        job_id=log.job_id,
        attempt=log.attempt,
        status=status.value if status is not None else "unknown",
        inferred_end=log.inferred_end,
        started_at=(start or {}).get("ts"),
        ended_at=(end or {}).get("ts"),
        wall_seconds=float(wall) if isinstance(wall, (int, float)) else None,
        last_step=log.last_step,
        planned_steps=log.planned_steps,
        truncated=(end or {}).get("truncated"),
        git_sha=log.git_sha,
        world_size=log.world_size,
        partition=slurm.get("slurm_job_partition"),
        nodelist=slurm.get("slurm_job_nodelist"),
        stream=_relpath(log.path, root),
        timeline=timeline,
        artifacts=sorted(artifacts, key=lambda a: (a.step if a.step is not None else -1,
                                                   a.kind, a.path)),
        observations=observations,
        defects=defects,
    )


def _fold_run(run: LogicalRun, root: str) -> RunEntry:
    spec = run.spec
    executions = [_fold_execution(log, root) for log in run.executions]
    # Total order with a unique final key: no tie is left to filesystem order.
    executions.sort(key=lambda e: (e.attempt, e.job_id, e.exec_id))
    start = run.canonical.start if run.canonical else None
    return RunEntry(
        run_uid=run.run_uid,
        campaign=(spec.campaign if spec else (start or {}).get("campaign", "")) or "",
        phase=(spec.phase if spec else (start or {}).get("phase", "")) or "",
        arm=run.arm or "",
        seed=run.seed,
        git_sha=(spec.git_sha if spec else (start or {}).get("git_sha")),
        planned_steps=(spec.planned_steps if spec and spec.planned_steps is not None
                       else next((e.planned_steps for e in executions
                                  if e.planned_steps is not None), None)),
        slug=os.path.basename(run.directory),
        executions=executions,
    )


def fold(campaign: Campaign) -> LedgerView:
    """Fold a loaded campaign into a ledger.  Pure; no I/O, no clock.

    Kept separate from :func:`build_ledger` so tests (and any caller that
    already holds a :class:`~telemetry.read.Campaign`) can exercise the fold
    without touching the filesystem, and so that the "deterministic and side
    effect free" claim is checkable on a single function.
    """
    entries = [_fold_run(run, campaign.root) for run in campaign.runs]
    entries.sort(key=lambda r: (r.campaign, r.phase, r.arm, str(r.seed), r.run_uid))
    return LedgerView(runs=entries)


def build_ledger(root: str) -> LedgerView:
    """Load every event stream under ``root`` and fold it."""
    return fold(load_campaign(root))


# -- CLI ---------------------------------------------------------------------

def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m telemetry.ledger",
        description="Fold a telemetry root into a materialized ledger. "
                    "Deterministic: the same event logs always render to the "
                    "same bytes, so the output can be committed and diffed.")
    parser.add_argument("--root", required=True, help="telemetry root")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--out", default=None,
                        help="write here instead of stdout (atomic replace)")
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if --out differs from the "
                             "freshly folded ledger, printing the drift")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    view = build_ledger(args.root)
    text = view.dumps() if args.format == "json" else view.render_markdown()

    if args.check:
        if not args.out:
            print("--check requires --out (there is nothing to compare against)")
            return 2
        try:
            with open(args.out, encoding="utf-8") as handle:
                committed = handle.read()
        except OSError:
            print(f"DRIFT: {args.out} does not exist; the ledger has never been "
                  "generated from the current log")
            return 1
        if committed == text:
            print(f"OK: {args.out} matches the folded ledger "
                  f"({len(view.runs)} runs)")
            return 0
        import difflib
        print(f"DRIFT: {args.out} differs from the folded ledger")
        for line in difflib.unified_diff(
                committed.splitlines(), text.splitlines(),
                fromfile=args.out, tofile="<folded>", lineterm="", n=2):
            print(line)
        return 1

    if args.out:
        tmp = args.out + f".tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, args.out)
        print(f"wrote {args.out} ({len(view.runs)} runs)")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
