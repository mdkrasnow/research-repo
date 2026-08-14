"""Command line interface: the layer that makes the event log *usable*.

The read side (:mod:`telemetry.read`) already refuses to aggregate a run it
cannot vouch for.  That is necessary but not sufficient: a gate nobody invokes
protects nothing.  This module exists so that the checks are one command away
and so that their output is something a human will actually read before
believing a number.

Three commands carry the weight:

``doctor``
    A prioritized health report over the whole telemetry root.  It answers the
    question "is anything about to lie to me?" and exits nonzero when the answer
    is yes, so it composes into CI, a pre-analysis hook, or a cron.  Every
    finding names the run, states the evidence, and says what to DO -- a finding
    a reader cannot act on is a finding they will learn to ignore.

``gate``
    Prints exactly which runs the default :class:`~telemetry.read.CompletenessPolicy`
    would admit to an aggregate and which it would quarantine, with the reason.
    This is the command to run *before* believing a results table: it turns
    "three of your six runs died" from an invisible fact into a printed one.

``ls`` / ``show``
    Inventory and detail.  Deliberately keyed on ``run_uid``/``exec_id`` rather
    than on directory names, because every historical identity bug in this repo
    came from a human-readable name being pressed into service as a join key.

Design constraints held throughout:

* **Pure stdlib.**  This tool has to run on a login node with whatever python is
  loaded; a dependency it cannot import is a tool that is not there when the
  question is urgent.
* **Degrade, never crash.**  An empty root, a half-written spec, a directory
  with no streams -- each is a *finding*, not a traceback.  A diagnostic tool
  that dies on malformed input is useless precisely on the inputs that matter.
* **Sibling modules are optional.**  ``ledger``/``reconcile``/``migrate``/
  ``contradictions`` are imported lazily inside their subcommand, so this file
  works before they exist and keeps working if one of them is broken.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import read as _read
from .ids import split_exec_id, verify_spec
from .schema import (
    EventType,
    RunStatus,
    SchemaError,
    required_start_fields,
    validate_record,
)

DEFAULT_ROOT = os.environ.get("EQM_TELEMETRY_ROOT") or os.path.join(
    "results", "telemetry")

#: How long a stream may go without any record before a still-RUNNING execution
#: is treated as wedged rather than slow.  One hour is far longer than any
#: legitimate logging cadence in this repo (heartbeats default to 300s), so a
#: breach is evidence, not noise.
DEFAULT_STALE_HOURS = 1.0

_SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


# ---------------------------------------------------------------------------
# terminal helpers
# ---------------------------------------------------------------------------

class _Style:
    """ANSI only when explicitly asked for AND writing to a terminal.

    Escape codes in a redirected stream corrupt the very artifact someone is
    trying to paste into a bug report, so the default is plain text.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled) and sys.stdout.isatty()

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self(text, "2")

    def bold(self, text: str) -> str:
        return self(text, "1")


def _table(rows: Sequence[Sequence[Any]], headers: Sequence[str],
           aligns: Optional[Sequence[str]] = None) -> str:
    """Fixed-width column layout.

    Written out rather than pulled from a dependency because the whole tool is
    stdlib-only, and because alignment is the difference between a table a human
    scans and a wall a human skips.
    """
    cells = [[("" if c is None else str(c)) for c in row] for row in rows]
    head = [str(h) for h in headers]
    widths = [len(h) for h in head]
    for row in cells:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    aligns = list(aligns or ["<"] * len(head))
    aligns += ["<"] * (len(head) - len(aligns))

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(
            f"{row[i]:{aligns[i]}{widths[i]}}" for i in range(len(head))).rstrip()

    lines = [fmt(head), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in cells)
    return "\n".join(lines)


def _short(sha: Optional[str], n: int = 8) -> str:
    return (sha or "")[:n] or "-"


def _parse_ts(ts: Any) -> Optional[_dt.datetime]:
    """RFC3339 -> aware datetime, or ``None`` if unparseable.

    ``None`` rather than an exception: a malformed timestamp is one more defect
    to report, and it must not take the whole health report down with it.
    """
    if not isinstance(ts, str):
        return None
    try:
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _age_seconds(ts: Any, now: Optional[_dt.datetime] = None) -> Optional[float]:
    parsed = _parse_ts(ts)
    if parsed is None:
        return None
    now = now or _dt.datetime.now(_dt.timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return (now - parsed).total_seconds()


def _human_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(max(0, seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    if seconds < 172800:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load(root: str) -> "_read.Campaign":
    """Load a campaign, tolerating a root that does not exist yet.

    An absent root is the normal state before the first run; it is not an error
    and must not print a traceback at someone who is checking whether anything
    has been recorded.
    """
    if not os.path.isdir(root):
        return _read.Campaign(root=root)
    return _read.load_campaign(root)


def _filter(campaign: "_read.Campaign", *, campaign_name: Optional[str] = None,
            phase: Optional[str] = None, arm: Optional[str] = None,
            seed: Optional[int] = None) -> List["_read.LogicalRun"]:
    """Filter logical runs, keeping spec-less runs visible when unfiltered.

    ``Campaign.select`` drops runs whose spec failed to load, which is correct
    for analysis (you cannot compare what you cannot identify) but wrong for an
    inventory: a run directory with an unreadable spec is exactly the thing an
    operator needs to see.  So with no filters we return everything.
    """
    if campaign_name is None and phase is None and arm is None and seed is None:
        return list(campaign.runs)
    out = []
    for run in campaign.runs:
        spec = run.spec
        if spec is None:
            continue
        if campaign_name is not None and spec.campaign != campaign_name:
            continue
        if phase is not None and spec.phase != phase:
            continue
        if arm is not None and spec.arm != arm:
            continue
        if seed is not None and int(spec.seed) != int(seed):
            continue
        out.append(run)
    return out


def _run_sort_key(run: "_read.LogicalRun") -> Tuple:
    spec = run.spec
    if spec is None:
        return ("~", "~", "~", 0, run.run_uid)
    try:
        seed = int(spec.seed)
    except (TypeError, ValueError):
        seed = 0
    return (spec.campaign, spec.phase, spec.arm, seed, run.run_uid)


def _status_str(execution: Optional["_read.RunLog"]) -> str:
    if execution is None:
        return "no-events"
    status = execution.status
    if status is None:
        return "unknown"
    if status is RunStatus.RUNNING:
        return "running"
    return status.value + ("*" if execution.inferred_end else "")


# ---------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Finding:
    """One health problem, with the evidence and the remedy attached.

    ``action`` is not decoration.  A report of defects with no remedies trains
    its reader to scroll past it; a report where every line ends in a command or
    a decision keeps being read.
    """

    severity: str          # ERROR | WARN | INFO
    code: str              # machine-stable identifier
    subject: str           # run_uid or exec_id the finding is about
    message: str           # what is wrong, with evidence
    action: str            # what to do about it

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _job_key(execution: "_read.RunLog") -> str:
    return execution.job_id


def _is_reconstructed(execution: "_read.RunLog") -> bool:
    """True when the stream declares itself a migration of a legacy record.

    :mod:`telemetry.migrate` writes a ``reconstruction_header`` NOTICE and marks
    its records ``provenance="reconstructed"`` precisely because a legacy run
    left no START and none can be honestly synthesized.  A reconstructed stream
    is therefore *expected* to be missing exactly the records whose absence is an
    ERROR on a native stream.

    Reporting those absences as errors would bury the ~1300 real findings in
    ~1000 restatements of "this run predates the telemetry system", and a report
    with that signal-to-noise ratio stops being read -- which costs more than the
    check was worth.  So the provenance is honoured: the missing lifecycle
    records are reported once, in aggregate, as a note.
    """
    for event in execution.events[:4]:
        if event.get("code") == "reconstruction_header":
            return True
        if event.get("provenance") == "reconstructed":
            return True
    return False


def _minted_run_uid(run: "_read.LogicalRun") -> Optional[str]:
    """The ``run_uid`` that was minted when the run actually launched.

    This has to be read from evidence *outside* the reconstructed
    :class:`~telemetry.ids.RunSpec`, and the reason is subtle enough to be worth
    stating: ``read.load_campaign`` builds the spec with ``RunSpec.from_dict``,
    which drops the stored ``run_uid`` field, and ``LogicalRun.run_uid`` is then
    the hash *recomputed from the loaded fields*.  Calling ``verify_spec`` on
    that pair is a tautology -- it can never fail, however badly ``spec.json``
    was edited.

    The two pieces of independent evidence are the ``run_uid`` literal that the
    launcher wrote into ``spec.json``, and the stream filenames, which encode the
    ``exec_id`` (hence the ``run_uid``) that the process emitted under.  Either
    disagreeing with the hash of the spec on disk is proof the spec was mutated
    after minting.
    """
    path = os.path.join(run.directory, "spec.json")
    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle).get("run_uid")
    except (OSError, json.JSONDecodeError, AttributeError):
        stored = None
    if isinstance(stored, str) and stored:
        return stored
    for execution in run.executions:
        if execution.run_uid:
            return execution.run_uid
    return None


def collect_findings(campaign: "_read.Campaign", *,
                     stale_seconds: float = DEFAULT_STALE_HOURS * 3600,
                     now: Optional[_dt.datetime] = None) -> List[Finding]:
    """Every health check, in one pass over the loaded campaign.

    The checks are grouped by the *kind of lie* they prevent:

    * lifecycle integrity (unsealed / inferred / stale) -- prevents a truncated
      run passing for a complete one;
    * stream integrity (seq gaps, duplicate seq, parse issues) -- prevents a
      statistic over an unknown subsample being reported as a statistic over the
      whole;
    * identity integrity (spec hash, shared job ids, multi-attempt) -- prevents
      two different experiments being averaged as one;
    * comparison integrity (confounded groups) -- prevents an A/B claim whose
      arms differ in more than the manipulated variable.
    """
    now = now or _dt.datetime.now(_dt.timezone.utc)
    findings: List[Finding] = []
    add = findings.append

    by_job: Dict[str, List["_read.RunLog"]] = defaultdict(list)
    reconstructed = 0

    for run in sorted(campaign.runs, key=_run_sort_key):
        label = run.run_uid

        # -- identity: does the recorded spec still hash to its own uid? ------
        if run.spec is None:
            add(Finding(
                "WARN", "spec_unreadable", label,
                f"{run.directory}/spec.json is missing or unparseable, so this "
                "run's identity and plan cannot be read from disk",
                "restore spec.json from the launcher, or treat this directory as "
                "unidentified and exclude it from every aggregate"))
        else:
            minted = _minted_run_uid(run)
            if minted:
                label = minted
                try:
                    verify_spec(minted, run.spec.identity())
                except SchemaError as exc:
                    add(Finding(
                        "ERROR", "spec_hash_mismatch", label,
                        f"recorded spec does not hash to the run_uid it was "
                        f"minted under: {exc}",
                        "the spec was edited after the run was minted. Do NOT cite "
                        "this run: recover the real parameters from the START "
                        "record (`telemetry show " + label + "`) before using it"))

        if not run.executions:
            add(Finding(
                "WARN", "no_executions", label,
                "run directory exists but contains no event streams: a spec was "
                "minted and nothing ever ran (or the stream was deleted)",
                "either launch it or remove the empty directory so the inventory "
                "reflects reality"))
            continue

        # -- multi-attempt is a fact worth stating, not a defect -------------
        if len(run.executions) > 1:
            attempts = ", ".join(
                f"{e.job_id}:a{e.attempt}({_status_str(e)})"
                for e in sorted(run.executions, key=lambda e: (e.job_id, e.attempt)))
            add(Finding(
                "INFO", "multiple_attempts", label,
                f"{len(run.executions)} executions of this logical run: {attempts}",
                "confirm the analysis uses ONE execution (read.LogicalRun.canonical "
                "picks the completed one); never concatenate attempts -- the merged "
                "step sequence never happened"))

        for execution in sorted(run.executions, key=lambda e: (e.job_id, e.attempt)):
            by_job[_job_key(execution)].append(execution)
            eid = execution.exec_id
            last_ts = execution.events[-1]["ts"] if execution.events else None
            age = _age_seconds(last_ts, now)
            stale = age is not None and age > stale_seconds
            legacy = _is_reconstructed(execution)
            reconstructed += int(legacy)

            if execution.start is None:
                if not legacy:
                    add(Finding(
                        "ERROR", "no_start", eid,
                        "stream has no START record, so the run's plan, arm, seed "
                        "and git sha are unknown",
                        "exclude from every aggregate; if the file was truncated "
                        "from the front, recover it from the scheduler's stdout"))
            else:
                missing = [k for k in required_start_fields()
                           if execution.start.get(k) is None]
                if missing:
                    add(Finding(
                        "WARN", "start_fields_missing", eid,
                        "START record omits " + ", ".join(missing),
                        "patch the launcher to pass these through RunSpec; without "
                        "planned_steps in particular, truncation is undetectable"))

            if execution.end is None:
                # A reconstructed stream that never got a terminal record is a
                # hole in the LEGACY bookkeeping, not a wedged process: there is
                # no live job to check and nothing to wait for.
                severity = "ERROR" if (stale and not legacy) else "WARN"
                seen = _human_duration(age)
                add(Finding(
                    severity, "unsealed", eid,
                    f"START with no END (last record {seen} ago, step "
                    f"{execution.last_step} of {execution.planned_steps}): this "
                    "execution's step range is not comparable to a completed run's",
                    "if the job is still in `squeue`, wait; otherwise seal it -- "
                    "`python -m telemetry reconcile` infers the terminal state from "
                    "sacct and appends an END with inferred=true"))
                if stale and not legacy:
                    add(Finding(
                        "ERROR", "stale_heartbeat", eid,
                        f"still RUNNING but no event for {_human_duration(age)} "
                        f"(threshold {_human_duration(stale_seconds)}); a wedged job "
                        "and a slow one are indistinguishable without this check",
                        "check `squeue -j " + execution.job_id + "` and the node's "
                        "filesystem quota -- a blocked stdout pipe on a full quota "
                        "presents exactly like this"))
            elif execution.inferred_end and not legacy:
                add(Finding(
                    "WARN", "inferred_end", eid,
                    f"terminal record was inferred (status "
                    f"{(execution.status.value if execution.status else '?')}), not "
                    "written by the process: the run died without sealing itself",
                    "treat wall-time and last_step as lower bounds; check whether the "
                    "final checkpoint was actually flushed before citing it"))

            if execution.seq_gaps:
                spans = ", ".join(f"{a}-{b}" for a, b in execution.seq_gaps[:6])
                extra = "" if len(execution.seq_gaps) <= 6 else \
                    f" (+{len(execution.seq_gaps) - 6} more)"
                add(Finding(
                    "ERROR", "seq_gap", eid,
                    f"missing seq {spans}{extra}: records were LOST, so every "
                    "statistic over this stream is an unknown subsample",
                    "do not aggregate this execution. If the gap is at the tail it "
                    "is a torn write; if in the middle, a second writer or a "
                    "filesystem failure -- investigate before rerunning"))

            if execution.duplicate_seqs:
                add(Finding(
                    "ERROR", "duplicate_seq", eid,
                    f"duplicate seq {execution.duplicate_seqs[:6]}: two writers "
                    "appended to one execution stream, so its records interleave",
                    "a requeue must use a NEW attempt number (see "
                    "lifecycle.next_attempt). Discard this stream and rerun"))

            for issue in execution.issues[:5]:
                add(Finding(
                    "ERROR", "parse_issue", eid,
                    f"{issue.path}:{issue.lineno}: {issue.reason}",
                    "usually a torn final write from a hard kill. Confirm the last "
                    "intact record, then exclude the execution or repair the tail"))
            if len(execution.issues) > 5:
                add(Finding(
                    "ERROR", "parse_issue", eid,
                    f"{len(execution.issues) - 5} further unreadable line(s) "
                    "suppressed",
                    "run `python -m telemetry validate " + execution.path + "` for "
                    "the full list"))

    # -- one job id, several executions --------------------------------------
    for job_id, executions in sorted(by_job.items()):
        if job_id == "local" or len(executions) < 2:
            continue
        uids = {e.run_uid for e in executions}
        ids = ", ".join(sorted(e.exec_id for e in executions))
        if len(uids) > 1:
            add(Finding(
                "ERROR", "job_id_collision", job_id,
                f"scheduler job {job_id} is claimed by {len(uids)} DIFFERENT "
                f"logical runs: {ids}",
                "two distinct experiments recorded the same job id -- one of them "
                "has the wrong spec. Resolve before citing either; a job id is the "
                "only handle back to sacct"))
        else:
            add(Finding(
                "WARN", "job_id_reused", job_id,
                f"job {job_id} has {len(executions)} attempts: {ids}",
                "expected for a SLURM requeue. Confirm each attempt got a distinct "
                "attempt number (it did, or they would share a file)"))

    if reconstructed:
        add(Finding(
            "INFO", "reconstructed_executions", campaign.root,
            f"{reconstructed} execution(s) were reconstructed from legacy ledgers "
            "by telemetry.migrate and carry no START by construction; their "
            "lifecycle records are second-hand",
            "treat their step counts and statuses as claims from the old ledgers, "
            "not as measurements. Any NEW result must come from a natively "
            "recorded run"))

    findings.extend(_comparison_findings(campaign))
    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.code,
                                 f.subject))
    return findings


def _brief(items, limit: int = 6) -> str:
    """Join a detail list, truncating so one finding cannot become a wall.

    A finding whose evidence runs to sixty lines is a finding a reader scrolls
    past, which defeats the purpose of reporting it at all.  The truncated tail
    is always recoverable from ``--json``.
    """
    items = list(items)
    head = "; ".join(items[:limit])
    return head if len(items) <= limit else f"{head}; +{len(items) - limit} more"


def _comparison_findings(campaign: "_read.Campaign") -> List[Finding]:
    """Confounded comparison groups.

    A group is (campaign, phase): the set of runs a results table will put in
    one row band.  Within a group, arms are supposed to differ *only* in the arm
    label and whatever the arm label denotes.  If two arms were planned to
    different step counts, "late training" is a different interval of training
    for each of them and the comparison is void (this is the exact defect
    :func:`read.shared_windows` refuses to paper over).  If two arms ran
    different git shas, the manipulated variable is not the only thing that
    changed.

    Both are reported at ERROR, because unlike a dead run -- which is visible as
    a missing row -- a confounded comparison produces a *complete-looking* table
    with a wrong conclusion in it.
    """
    out: List[Finding] = []
    groups: Dict[Tuple[str, str], List["_read.LogicalRun"]] = defaultdict(list)
    for run in campaign.runs:
        if run.spec is not None:
            groups[(run.spec.campaign, run.spec.phase)].append(run)

    for (name, phase), runs in sorted(groups.items()):
        arms = {r.spec.arm for r in runs if r.spec}
        if len(arms) < 2:
            continue
        label = f"{name}/{phase or '-'}"

        # Distinguish a DESIGNED comparison from a migrated inventory.  The
        # legacy ledgers assigned one "arm" per historical job, so a phase can
        # contain 183 arms with one run each -- nobody is A/B-ing those, and
        # calling their heterogeneous git shas an ERROR would make the headline
        # error count meaningless.  A designed group is small, or has several
        # seeds per arm; anything else is reported at WARN with the caveat named.
        designed = len(arms) <= 12 or len(runs) >= 2 * len(arms)
        severity = "ERROR" if designed else "WARN"
        caveat = ("" if designed else
                  " -- note this group has one run per arm, so it may be a "
                  "migrated inventory rather than a designed comparison")

        planned: Dict[str, set] = defaultdict(set)
        shas: Dict[str, set] = defaultdict(set)
        for run in runs:
            spec = run.spec
            assert spec is not None
            declared = spec.planned_steps
            if declared is None:
                execution = run.canonical
                declared = execution.planned_steps if execution else None
            planned[spec.arm].add(declared)
            shas[spec.arm].add(spec.git_sha)

        distinct_planned = {p for values in planned.values() for p in values}
        if len(distinct_planned) > 1:
            detail = _brief(
                f"{arm}={sorted(v, key=lambda x: (x is None, x))}"
                for arm, v in sorted(planned.items()))
            out.append(Finding(
                severity, "confounded_planned_steps", label,
                f"arms in this comparison group were planned to different step "
                f"counts ({detail}): a shared 'late' window does not exist, so any "
                "window-by-window table over these arms compares different "
                "intervals of training" + caveat,
                "requeue the short arm to the common horizon, or truncate every arm "
                "explicitly to the smallest planned_steps and say so in the caption"))

        distinct_shas = {s for values in shas.values() for s in values}
        if len(distinct_shas) > 1:
            detail = _brief(f"{arm}={sorted(_short(s) for s in v)}"
                            for arm, v in sorted(shas.items()))
            out.append(Finding(
                severity, "confounded_git_sha", label,
                f"{len(arms)} arms in this comparison group ran "
                f"{len(distinct_shas)} different commits ({detail}): the arm label "
                "is not the only thing that differs between them" + caveat,
                "rerun the arms on one commit, or demonstrate the diff between the "
                "shas cannot affect the measured quantity -- and cite that argument "
                "next to the number"))

        # A pairwise spec diff catches confounders the two checks above do not
        # name explicitly (a changed lr, a different batch size).
        by_arm: Dict[str, "_read.LogicalRun"] = {}
        for run in sorted(runs, key=_run_sort_key):
            if run.spec and run.spec.arm not in by_arm:
                by_arm[run.spec.arm] = run
        # One finding per GROUP, not per arm pair.  A group with k arms has k-1
        # adjacent pairs, and emitting one finding each restates the same fact
        # k-1 times -- on the migrated corpus that alone produced 330 lines
        # saying the same thing about the same handful of parameters.
        ordered = sorted(by_arm.items())
        varying: Dict[str, int] = defaultdict(int)
        for i in range(len(ordered) - 1):
            (_arm_a, run_a), (_arm_b, run_b) = ordered[i], ordered[i + 1]
            verdict = _read.controlled_comparison(run_a, run_b)
            diff = {k for k in (verdict.get("differing_fields") or {})
                    if k not in ("arm", "seed")}
            if len(diff) > 1:
                for field in diff:
                    varying[field] += 1
        if varying:
            fields = ", ".join(sorted(varying, key=lambda k: (-varying[k], k))[:12])
            more = "" if len(varying) <= 12 else f" (+{len(varying) - 12} more)"
            out.append(Finding(
                "WARN", "confounded_spec", label,
                f"across the {len(ordered)} arms of this group, specs differ in "
                f"more than the arm label: {fields}{more}",
                "an A/B claim is sound only if the arms differ in the manipulated "
                "variable alone -- either equalize these fields or state which of "
                "them the claim is conditional on. `telemetry show <uid>` prints "
                "each arm's params"))
    return out


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_ls(args: argparse.Namespace) -> int:
    campaign = _load(args.root)
    runs = sorted(_filter(campaign, campaign_name=args.campaign, phase=args.phase,
                          arm=args.arm), key=_run_sort_key)

    rows = []
    for run in runs:
        execution = run.canonical
        spec = run.spec
        planned = (spec.planned_steps if spec and spec.planned_steps is not None
                   else (execution.planned_steps if execution else None))
        rows.append({
            "run_uid": run.run_uid,
            "campaign": spec.campaign if spec else "?",
            "phase": (spec.phase if spec else "") or "-",
            "arm": run.arm,
            "seed": run.seed,
            "status": _status_str(execution),
            "last_step": execution.last_step if execution else None,
            "planned_steps": planned,
            "attempts": len(run.executions),
            "git_sha": (spec.git_sha if spec else
                        (execution.git_sha if execution else None)),
            "directory": run.directory,
        })

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    if not rows:
        print(f"no runs under {args.root}"
              + ("" if os.path.isdir(args.root) else " (root does not exist yet)"))
        return 0

    table = [[
        r["run_uid"], r["arm"], r["seed"], r["status"],
        f"{r['last_step']}/{r['planned_steps'] if r['planned_steps'] is not None else '?'}",
        r["attempts"], _short(r["git_sha"]),
    ] for r in rows]
    print(_table(table,
                 ["RUN_UID", "ARM", "SEED", "STATUS", "STEP/PLAN", "ATT", "GIT"],
                 ["<", "<", ">", "<", ">", ">", "<"]))
    print(f"\n{len(rows)} run(s) under {args.root}")
    return 0


def _resolve(campaign: "_read.Campaign", token: str
             ) -> Tuple[Optional["_read.LogicalRun"], Optional["_read.RunLog"]]:
    """Resolve a run_uid, an exec_id, or an unambiguous prefix of either.

    Prefix resolution is offered because a 17-character id is not something a
    human retypes, but an *ambiguous* prefix resolves to nothing rather than to
    the first match: silently picking one of two candidates is the same class of
    error as the prefix-collision bug that motivated content addressing.
    """
    try:
        uid, _job, _attempt = split_exec_id(token)
        run = campaign.by_uid(uid)
        if run is not None:
            for execution in run.executions:
                if execution.exec_id == token:
                    return run, execution
            return run, None
    except SchemaError:
        pass

    exact = campaign.by_uid(token)
    if exact is not None:
        return exact, None

    matches = [r for r in campaign.runs if r.run_uid.startswith(token)
               or (_minted_run_uid(r) or "").startswith(token)]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, None
    for run in campaign.runs:
        for execution in run.executions:
            if execution.exec_id.startswith(token):
                return run, execution
    return None, None


def cmd_show(args: argparse.Namespace) -> int:
    campaign = _load(args.root)
    run, chosen = _resolve(campaign, args.target)
    if run is None:
        candidates = [r.run_uid for r in campaign.runs
                      if r.run_uid.startswith(args.target)]
        if len(candidates) > 1:
            print(f"ambiguous: {args.target!r} matches {', '.join(candidates)}",
                  file=sys.stderr)
        else:
            print(f"no run matching {args.target!r} under {args.root}",
                  file=sys.stderr)
        return 2

    style = _Style(args.color)
    print(style.bold(f"run_uid   {run.run_uid}"))
    print(f"directory {run.directory}")

    spec = run.spec
    if spec is None:
        print("spec      <unreadable>  -- identity cannot be verified")
    else:
        print(f"campaign  {spec.campaign}   phase={spec.phase or '-'}   "
              f"arm={spec.arm}   seed={spec.seed}")
        print(f"git_sha   {spec.git_sha}")
        print(f"planned   {spec.planned_steps}")
        minted = _minted_run_uid(run)
        try:
            verify_spec(minted or run.run_uid, spec.identity())
            print("spec hash OK (run_uid is the hash of this spec)")
        except SchemaError as exc:
            print(style("spec hash MISMATCH: " + str(exc), "31;1"))
            print(style(f"  minted as {minted}, spec on disk hashes to "
                        f"{run.run_uid} -- the spec was edited after the fact",
                        "31;1"))
        if spec.params:
            print("params")
            for key in sorted(spec.params):
                print(f"    {key} = {spec.params[key]}")

    if not run.executions:
        print("\nno executions recorded")
        return 0

    for execution in sorted(run.executions, key=lambda e: (e.job_id, e.attempt)):
        if chosen is not None and execution is not chosen:
            continue
        marker = " (canonical)" if execution is run.canonical else ""
        print("\n" + style.bold(f"exec {execution.exec_id}{marker}"))
        print(f"    job_id      {execution.job_id}   attempt={execution.attempt}")
        print(f"    stream      {execution.path}")
        start, end = execution.start, execution.end
        print(f"    started     {start['ts'] if start else '<no START>'}")
        if start:
            print(f"    host        {start.get('hostname')}  pid={start.get('pid')}"
                  f"  world_size={start.get('world_size')}")
            slurm = start.get("slurm") or {}
            if slurm:
                print("    slurm       " + "  ".join(
                    f"{k}={v}" for k, v in sorted(slurm.items())))
        if end:
            print(f"    ended       {end['ts']}  status={end.get('status')}"
                  f"  inferred={bool(end.get('inferred'))}")
            print(f"    last_step   {end.get('last_step')} / "
                  f"{end.get('planned_steps')}  truncated={end.get('truncated')}")
            print(f"    wall        {end.get('wall_seconds')}s  "
                  f"records={end.get('records_emitted')}  "
                  f"nonfinite={end.get('nonfinite_events')}")
            if end.get("error"):
                print("    error       " + str(end["error"]).strip().splitlines()[-1])
        else:
            age = _age_seconds(execution.events[-1]["ts"]) if execution.events else None
            print(f"    ended       <no END>  (last record {_human_duration(age)} ago)")

        counts = defaultdict(int)
        for event in execution.events:
            counts[event["event"]] += 1
        print("    events      " + (", ".join(
            f"{k}={counts[k]}" for k in sorted(counts)) or "none"))
        gaps = execution.seq_gaps
        print("    seq         "
              + (f"{len(execution.events)} records, "
                 f"{'CONTIGUOUS' if not gaps else 'GAPS ' + str(gaps[:6])}"
                 + (f", DUPLICATES {execution.duplicate_seqs[:6]}"
                    if execution.duplicate_seqs else "")))

        artifacts = execution.records(EventType.ARTIFACT)
        if artifacts:
            print("    artifacts")
            for record in artifacts:
                print(f"        step {record.get('step')}  {record.get('kind')}  "
                      f"{record.get('path')}  {record.get('bytes')}B")

        notices = [e for e in execution.records(EventType.NOTICE)
                   if e.get("level") != "heartbeat"]
        if notices:
            print("    notices")
            for record in notices[: args.limit]:
                print(f"        [{record.get('level')}] {record.get('message')}")
            if len(notices) > args.limit:
                print(f"        ... {len(notices) - args.limit} more")

        if execution.issues:
            print("    parse issues")
            for issue in execution.issues[: args.limit]:
                print(f"        line {issue.lineno}: {issue.reason}")

        head = execution.events[: args.limit]
        tail = execution.events[-args.limit:] if len(execution.events) > 2 * args.limit else []
        print("    first events")
        for event in head:
            print("        " + _one_line_event(event))
        if tail:
            print(f"    ... {len(execution.events) - len(head) - len(tail)} events ...")
            print("    last events")
            for event in tail:
                print("        " + _one_line_event(event))
    return 0


def _one_line_event(event: Dict[str, Any], value_width: int = 72) -> str:
    """Render one record as a single scannable line.

    Values are elided at a fixed width rather than printed in full.  Records in
    this system legitimately carry whole configs and whole migrated legacy rows;
    printing them verbatim turns a "last 20 events" view into thousands of
    columns of wrapped JSON, in which the fields a reader came for (step, status,
    seq) are unfindable.  The full record is always one `validate`/`jq` away.
    """
    fixed = {"v", "ts", "run_uid", "exec_id", "seq", "event"}
    parts = []
    for key in sorted(k for k in event if k not in fixed):
        text = str(event[key])
        if len(text) > value_width:
            text = text[: value_width - 3] + "..."
        parts.append(f"{key}={text}")
    return f"#{event['seq']:<5} {event['ts']} {event['event']:<8} {' '.join(parts)}"


def cmd_doctor(args: argparse.Namespace) -> int:
    campaign = _load(args.root)
    findings = collect_findings(
        campaign, stale_seconds=args.stale_hours * 3600.0)
    counts = {level: sum(1 for f in findings if f.severity == level)
              for level in ("ERROR", "WARN", "INFO")}
    exit_code = 1 if counts["ERROR"] else 0

    executions = sum(len(r.executions) for r in campaign.runs)
    if args.json:
        print(json.dumps({
            "root": args.root,
            "root_exists": os.path.isdir(args.root),
            "runs": len(campaign.runs),
            "executions": executions,
            "summary": counts,
            "findings": [f.as_dict() for f in findings],
            "exit_code": exit_code,
        }, indent=2, sort_keys=True))
        return exit_code

    style = _Style(args.color)
    print(style.bold(f"telemetry doctor: {args.root}"))
    if not os.path.isdir(args.root):
        print("  root does not exist yet -- nothing has been recorded.")
        print("  0 errors, 0 warnings, 0 notes")
        return 0
    print(f"  {len(campaign.runs)} logical run(s), {executions} execution(s)")
    print(f"  {counts['ERROR']} error(s), {counts['WARN']} warning(s), "
          f"{counts['INFO']} note(s)")

    if not findings:
        print("\nno defects found: every stream is sealed, contiguous, "
              "identity-verified, and every comparison group is controlled.")
        return 0

    by_code = defaultdict(int)
    for finding in findings:
        by_code[(finding.severity, finding.code)] += 1
    print("\n" + style.bold("BY CHECK"))
    print(_table([[sev, code, n] for (sev, code), n in
                  sorted(by_code.items(), key=lambda kv: (
                      _SEVERITY_ORDER[kv[0][0]], -kv[1], kv[0][1]))],
                 ["SEV", "CHECK", "N"], ["<", "<", ">"]))

    # A health report is only useful if it is read.  On a large root a single
    # check can produce hundreds of instances of the same defect; printing all of
    # them buries the checks that fired once.  So each check shows its first
    # `--per-check` instances and states how many it withheld.
    for level in ("ERROR", "WARN", "INFO"):
        subset = [f for f in findings if f.severity == level]
        if not subset:
            continue
        heading = {"ERROR": "ERRORS -- do not publish numbers over these runs",
                   "WARN": "WARNINGS -- read before citing these runs",
                   "INFO": "NOTES"}[level]
        print("\n" + style.bold(heading))
        shown: Dict[str, int] = defaultdict(int)
        for finding in subset:
            shown[finding.code] += 1
            if args.per_check and shown[finding.code] > args.per_check:
                continue
            print(f"  [{finding.code}] {finding.subject}")
            for line in _wrap(finding.message, 4):
                print(line)
            for line in _wrap("-> " + finding.action, 4):
                print(style.dim(line))
        for code, total in sorted(shown.items()):
            if args.per_check and total > args.per_check:
                print(style.dim(
                    f"  ... {total - args.per_check} further [{code}] finding(s) "
                    f"withheld; use --per-check 0 or --json for all"))
    return exit_code


def _wrap(text: str, indent: int, width: int = 96) -> List[str]:
    import textwrap
    pad = " " * indent
    return textwrap.wrap(text, width=width, initial_indent=pad,
                         subsequent_indent=pad + "   ") or [pad + text]


def cmd_gate(args: argparse.Namespace) -> int:
    """Show the admit/quarantine split under the DEFAULT policy.

    Deliberately the default policy and not a configurable one: the point of the
    command is to show what a naive aggregate would and would not be entitled
    to include.  An analysis that legitimately wants truncated runs relaxes the
    policy at its own call site, where the relaxation is visible in the code
    that publishes the number.
    """
    campaign = _load(args.root)
    selected = _filter(campaign, campaign_name=args.campaign, phase=args.phase)
    subset = _read.Campaign(root=campaign.root, runs=selected)
    policy = _read.CompletenessPolicy()
    good, bad = subset.partition(policy)

    admitted = [{
        "run_uid": run.run_uid, "arm": run.arm, "seed": run.seed,
        "exec_id": execution.exec_id, "status": _status_str(execution),
        "last_step": execution.last_step, "planned_steps": execution.planned_steps,
        "git_sha": execution.git_sha,
    } for run, execution in sorted(good, key=lambda p: _run_sort_key(p[0]))]

    quarantined = [{
        "run_uid": run.run_uid, "arm": run.arm, "seed": run.seed,
        "reason": rejection.reason, "detail": rejection.detail,
        "exec_id": rejection.exec_id,
    } for run, rejection in sorted(bad, key=lambda p: _run_sort_key(p[0]))]

    if args.json:
        print(json.dumps({
            "root": args.root, "policy": dataclasses.asdict(_policy_view(policy)),
            "admitted": admitted, "quarantined": quarantined,
        }, indent=2, sort_keys=True, default=str))
        return 0

    style = _Style(args.color)
    print(style.bold(f"aggregation gate: {args.root}"))
    print("policy: START and END required, terminal status must be 'completed', "
          "no truncation, no seq gaps, no duplicate seq, no parse issues.")
    if not selected:
        print("\nno runs match the selection"
              + ("" if os.path.isdir(args.root) else " (root does not exist yet)"))
        return 0

    print(f"\n{style.bold('ADMITTED')} ({len(admitted)})")
    if admitted:
        print(_table([[a["run_uid"], a["arm"], a["seed"],
                       f"{a['last_step']}/{a['planned_steps']}",
                       _short(a["git_sha"]), a["exec_id"]] for a in admitted],
                     ["RUN_UID", "ARM", "SEED", "STEP/PLAN", "GIT", "EXEC_ID"],
                     ["<", "<", ">", ">", "<", "<"]))
    else:
        print("  none -- any aggregate over this selection would be empty.")

    print(f"\n{style.bold('QUARANTINED')} ({len(quarantined)})")
    if quarantined:
        print(_table([[q["run_uid"], q["arm"], q["seed"], q["reason"], q["detail"]]
                      for q in quarantined],
                     ["RUN_UID", "ARM", "SEED", "REASON", "DETAIL"],
                     ["<", "<", ">", "<", "<"]))
        print("\nquarantined runs are NOT a rounding error: how many runs died, and "
              "how, belongs in the results table beside the surviving ones.")
    else:
        print("  none.")

    per_arm: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for a in admitted:
        per_arm[a["arm"]][0] += 1
    for q in quarantined:
        per_arm[q["arm"]][1] += 1
    if per_arm:
        print("\n" + _table([[arm, counts[0], counts[1]]
                             for arm, counts in sorted(per_arm.items())],
                            ["ARM", "ADMITTED", "QUARANTINED"], ["<", ">", ">"]))
    return 0


@dataclasses.dataclass
class _PolicyView:
    require_start: bool
    require_end: bool
    require_status: List[str]
    forbid_truncated: bool
    forbid_seq_gaps: bool
    forbid_duplicate_seqs: bool
    forbid_parse_issues: bool


def _policy_view(policy: "_read.CompletenessPolicy") -> _PolicyView:
    return _PolicyView(
        require_start=policy.require_start, require_end=policy.require_end,
        require_status=[s.value for s in policy.require_status],
        forbid_truncated=policy.forbid_truncated,
        forbid_seq_gaps=policy.forbid_seq_gaps,
        forbid_duplicate_seqs=policy.forbid_duplicate_seqs,
        forbid_parse_issues=policy.forbid_parse_issues)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate raw streams against the schema, line by line.

    Reports *every* bad line with its number and reason rather than stopping at
    the first: a truncated tail and a systematically malformed field need
    different fixes, and you cannot tell them apart from one error.
    """
    target = args.path
    if os.path.isdir(target):
        paths = sorted(glob.glob(os.path.join(target, "**", "*.jsonl"),
                                 recursive=True))
    elif os.path.isfile(target):
        paths = [target]
    else:
        print(f"no such file or directory: {target}", file=sys.stderr)
        return 2

    if not paths:
        print(f"no .jsonl streams under {target}")
        return 0

    total_lines = total_bad = 0
    for path in paths:
        bad: List[Tuple[int, str]] = []
        lines = 0
        expected_exec = os.path.basename(path)[: -len(".jsonl")]
        try:
            handle = open(path, encoding="utf-8")
        except OSError as exc:
            print(f"{path}: cannot open: {exc}")
            total_bad += 1
            continue
        with handle:
            for lineno, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                lines += 1
                try:
                    record = validate_record(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    bad.append((lineno, f"invalid JSON: {exc}"))
                    continue
                except SchemaError as exc:
                    bad.append((lineno, str(exc)))
                    continue
                if record["exec_id"] != expected_exec:
                    bad.append((lineno, f"record belongs to {record['exec_id']}, "
                                        f"but the file is named {expected_exec}"))
        total_lines += lines
        total_bad += len(bad)
        status = "OK" if not bad else f"{len(bad)} BAD"
        print(f"{path}: {lines} record(s), {status}")
        for lineno, reason in bad:
            print(f"    line {lineno}: {reason}")

    print(f"\n{len(paths)} file(s), {total_lines} record(s), {total_bad} bad line(s)")
    if total_bad:
        print("a line that does not parse is evidence about the run (usually a torn "
              "final write), not noise -- decide explicitly whether to exclude the "
              "execution before aggregating it.")
    return 1 if total_bad else 0


def cmd_tail(args: argparse.Namespace) -> int:
    campaign = _load(args.root)
    run, chosen = _resolve(campaign, args.target)
    if run is None:
        print(f"no run matching {args.target!r} under {args.root}", file=sys.stderr)
        return 2
    execution = chosen
    if execution is None:
        if not run.executions:
            print(f"{run.run_uid}: no executions recorded")
            return 0
        # Newest execution = the one whose last record is most recent, falling
        # back to the highest attempt when a stream is empty.
        execution = max(run.executions,
                        key=lambda e: ((e.events[-1]["ts"] if e.events else ""),
                                       e.attempt))
    print(f"{execution.exec_id}  ({_status_str(execution)})  {execution.path}")
    events = execution.events[-args.n:]
    if not events:
        print("  <no records>")
    for event in events:
        print("  " + _one_line_event(event))
    if execution.issues:
        print(f"  ... plus {len(execution.issues)} unreadable line(s); "
              f"run `python -m telemetry validate {execution.path}`")
    return 0


# -- thin dispatchers to sibling modules -------------------------------------

def _dispatch(module_name: str, args: argparse.Namespace,
              extra: Sequence[str]) -> int:
    """Hand off to a sibling module, degrading clearly when it is absent.

    Imported lazily and by name so that this CLI is usable before those modules
    land and stays usable if one of them fails to import -- a health-check tool
    that cannot start because an unrelated module is broken is a health-check
    tool that is missing when it is needed.
    """
    import importlib

    try:
        module = importlib.import_module(f".{module_name}", __package__)
    except ImportError as exc:
        print(f"`telemetry {module_name}` is not available yet: "
              f"telemetry/{module_name}.py could not be imported ({exc}).",
              file=sys.stderr)
        return 3

    # Forward the global --root unless the caller already supplied one after the
    # subcommand.  Every module in this package takes --root; one that does not
    # will reject it with its own usage message, which is a clearer failure than
    # silently running against the wrong tree.
    argv = list(extra)
    if "--root" not in argv and not any(a.startswith("--root=") for a in argv):
        argv = ["--root", args.root] + argv

    for entry in ("main", "cli", "run"):
        fn: Optional[Callable[..., Any]] = getattr(module, entry, None)
        if callable(fn):
            try:
                result = fn(argv)
            except TypeError:
                result = fn()
            except SystemExit as exc:
                # Sibling modules parse their own args; argparse exits rather
                # than returning.  Translating that into our exit code keeps the
                # dispatcher transparent instead of aborting the process from
                # inside a library call.
                return int(exc.code or 0)
            return int(result) if isinstance(result, int) else 0

    print(f"telemetry/{module_name}.py exists but exposes no main(argv) entry "
          f"point; nothing to dispatch to. Available: "
          f"{', '.join(sorted(n for n in vars(module) if not n.startswith('_')))}",
          file=sys.stderr)
    return 3


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m telemetry",
        description="Inspect, health-check and gate masked-EqM run telemetry.")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help=f"telemetry root (default: {DEFAULT_ROOT})")
    parser.add_argument("--color", action="store_true",
                        help="ANSI colour when stdout is a terminal")
    sub = parser.add_subparsers(dest="command")

    p_ls = sub.add_parser("ls", help="one line per logical run")
    p_ls.add_argument("--campaign")
    p_ls.add_argument("--phase")
    p_ls.add_argument("--arm")
    p_ls.add_argument("--json", action="store_true")
    p_ls.set_defaults(func=cmd_ls)

    p_show = sub.add_parser("show", help="full detail for one run or execution")
    p_show.add_argument("target", help="run_uid, exec_id, or unambiguous prefix")
    p_show.add_argument("--limit", type=int, default=5,
                        help="events shown at each end (default 5)")
    p_show.set_defaults(func=cmd_show)

    p_doc = sub.add_parser("doctor", help="prioritized health report (exit 1 on ERROR)")
    p_doc.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_HOURS,
                       help="a RUNNING execution with no record for this long is "
                            f"reported as wedged (default {DEFAULT_STALE_HOURS})")
    p_doc.add_argument("--per-check", type=int, default=8,
                       help="max instances printed per check before the rest are "
                            "summarized (0 = print everything; default 8)")
    p_doc.add_argument("--json", action="store_true")
    p_doc.set_defaults(func=cmd_doctor)

    p_gate = sub.add_parser("gate", help="what would be admitted to an aggregate")
    p_gate.add_argument("--campaign")
    p_gate.add_argument("--phase")
    p_gate.add_argument("--json", action="store_true")
    p_gate.set_defaults(func=cmd_gate)

    p_val = sub.add_parser("validate", help="schema-check a stream or a whole root")
    p_val.add_argument("path")
    p_val.set_defaults(func=cmd_validate)

    p_tail = sub.add_parser("tail", help="last N events of the newest execution")
    p_tail.add_argument("target")
    p_tail.add_argument("-n", type=int, default=20)
    p_tail.set_defaults(func=cmd_tail)

    for name, helptext in (
        ("reconcile", "reconcile against sacct (telemetry/reconcile.py)"),
        ("migrate", "migrate legacy logs (telemetry/migrate.py)"),
        ("contradictions", "cross-source contradictions (telemetry/contradictions.py)"),
        ("ledger", "fold the event log into a ledger view (telemetry/ledger.py)"),
    ):
        p = sub.add_parser(name, help=helptext,
                           add_help=False)  # forward -h to the sibling module
        p.set_defaults(func=None, dispatch=name)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    if getattr(args, "dispatch", None):
        return _dispatch(args.dispatch, args, extra)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:  # `| head`
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
