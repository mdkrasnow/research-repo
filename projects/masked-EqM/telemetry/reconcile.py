"""The reconciler: append what the scheduler observed, and name the disagreements.

What this replaces
------------------
``scripts/cluster/reconcile_pipeline.py`` read ``pipeline.json``, asked ``sacct``
about every ``active_runs`` job id, and rewrote the file.  An audit found eight
defects; every one of them is addressed here and each fix carries a comment
naming the defect, so the mapping stays checkable:

(a) **All-or-nothing.**  The old code raised ``RuntimeError`` the moment sacct
    returned no row for *any* job id, so one job aged out of the accounting
    retention window blocked reconciliation of every other job forever.  Here a
    missing row is a *classification* (:attr:`Disagreement.MISSING_FROM_SACCT`),
    not an exception, and reconciliation of the remaining jobs proceeds.  See
    :func:`reconcile`.
(b) **No ``--starttime``.**  ``sacct`` defaults to midnight *today*; a job that
    ended at 23:50 yesterday returns no row and would then have been reported as
    an error (defect a) or as still active.  :func:`build_sacct_command` always
    passes an explicit window, derived from the earliest START in the telemetry
    or from an explicit flag.
(c) **Array task ids.**  The ledger writes ``"34719790_2"`` while ``JobIDRaw``
    returns a distinct numeric id, so the old dict lookup could never match and
    array runs were unreconcilable in principle.  :func:`parse_sacct` requests
    *both* ``JobID`` and ``JobIDRaw`` and indexes each row under every
    normalization of both, so a lookup by either form hits.
(d) **Incomplete terminal set.**  ``boot_fail``, ``revoked``, ``special_exit``
    were absent, so a BOOT_FAIL job stayed "active" forever.  They are terminal
    here.  ``requeued`` and ``suspended`` are *not* terminal in SLURM -- adding
    them to the terminal set would close a run that is about to produce more
    telemetry -- so they are given an explicit non-terminal classification
    instead of falling through a default; the bug was that they were
    unrecognized, not that they were non-terminal.  See :data:`SACCT_STATES`.
(e) **No subprocess timeout.**  This repo's SSH ControlMaster drops regularly and
    a hung ``ssh`` would hang the reconciler indefinitely.  Every invocation goes
    through :func:`run_command` with a mandatory timeout, and a timeout degrades
    to "no rows", i.e. defect (a)'s partial-progress path.
(f) **Unparsed exit code.**  ``"0:0"`` was stored as an opaque string.
    :func:`parse_exit_code` splits it into ``(code, signal)`` so that "killed by
    SIGKILL" is distinguishable from "exited 9".
(g) **Read-then-write was not a transaction.**  The old code read the pipeline
    file, made an SSH round trip taking seconds to minutes, then atomically
    replaced the file -- destroying any edit made in that window.  Here the SSH
    happens *before* the critical section, and the merge happens inside
    :func:`file_transaction`, which holds an exclusive lock and re-reads the file
    after acquiring it.
(h) **Empty ``active_runs``.**  ``",".join([])`` produced ``sacct -X -j
    --format=...``, in which ``--format`` is consumed as the argument to ``-j``.
    :func:`query_sacct` short-circuits on an empty id set and never builds a
    command.

The point of the whole thing
----------------------------
The reconciler does **not** decide what happened.  It records what the scheduler
says, next to what the run said about itself, and *names their disagreement*.
Three disagreements matter and each means something different:

``sacct: COMPLETED`` + no ``END`` in the stream
    Lost telemetry.  The job exited 0 but the run never sealed its stream, so
    every per-step statistic over it is an unknown subsample.  The scheduler's
    "completed" is about the *process*, not about the *training*.

``END: completed`` + ``sacct: TIMEOUT``
    A run misreporting itself.  Almost always a launcher whose planned-step
    count is wrong, so the loop "finished" a plan shorter than intended, or a
    seal written on the way out of a wall-clock kill.  Its ``COMPLETED`` status
    must not be trusted for aggregation.

``sacct: COMPLETED`` on a job whose sbatch pipes to ``tee`` without ``PIPESTATUS``
    An untrustworthy exit code.  In ``python train.py | tee log``, ``$?`` is
    ``tee``'s status, so a crashed trainer is recorded as a clean exit.  Detected
    statically by :func:`sbatch_exit_code_trustworthy`.

Append-only, always
-------------------
Every finding is appended as an :attr:`~telemetry.schema.EventType.OBSERVED`
record on the execution's own stream.  Nothing is edited, nothing is deleted; a
later reconcile that learns more appends again.  The ledger
(:mod:`telemetry.ledger`) folds these records into its view, so "what we now
believe" is always derivable from "what was observed, and when".
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import enum
import json
import os
import re
import subprocess
import tempfile
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Set, Tuple)

from .ids import split_exec_id
from .read import Campaign, RunLog, load_campaign
from .schema import (EventType, RunStatus, SchemaError, dumps, make_record,
                     utc_now_iso)

#: sacct fields requested, in order.  ``JobID`` *and* ``JobIDRaw`` both appear
#: because neither alone is sufficient: ``JobID`` carries the array form
#: (``123_4``) that the ledger uses, ``JobIDRaw`` carries the flattened numeric
#: id that some tooling records.  Requesting both is defect (c)'s fix.
SACCT_FIELDS = ("JobID", "JobIDRaw", "State", "ExitCode", "Elapsed",
                "Start", "End", "NodeList", "Partition")

#: Every sacct state this reconciler recognizes, mapped to
#: ``(is_terminal, RunStatus | None)``.  Explicit rather than "anything not in a
#: small set is still running": an unrecognized state used to mean "leave it
#: active forever" (defect d).
SACCT_STATES: Dict[str, Tuple[bool, Optional[RunStatus]]] = {
    # -- terminal -----------------------------------------------------------
    "completed": (True, RunStatus.COMPLETED),
    "failed": (True, RunStatus.CRASHED),
    "cancelled": (True, RunStatus.CANCELLED),
    "timeout": (True, RunStatus.TIMEOUT),
    "preempted": (True, RunStatus.PREEMPTED),
    "node_fail": (True, RunStatus.LOST),
    "out_of_memory": (True, RunStatus.CRASHED),
    "deadline": (True, RunStatus.TIMEOUT),
    # defect (d): these five were absent from the old terminal set.
    "boot_fail": (True, RunStatus.LOST),
    "revoked": (True, RunStatus.CANCELLED),
    "special_exit": (True, RunStatus.CRASHED),
    # -- recognized but NOT terminal ---------------------------------------
    # A requeued or suspended job will run again and will emit more telemetry;
    # sealing it would be a lie.  What the old code got wrong was leaving them
    # unrecognized, so they were indistinguishable from "still running".
    "requeued": (False, None),
    "requeue_hold": (False, None),
    "resizing": (False, None),
    "suspended": (False, None),
    "pending": (False, None),
    "running": (False, RunStatus.RUNNING),
    "configuring": (False, None),
    "completing": (False, None),
    "signaling": (False, None),
    "stage_out": (False, None),
}

#: The terminal subset, derived so the two can never drift apart.
TERMINAL_SACCT_STATES = frozenset(
    name for name, (terminal, _s) in SACCT_STATES.items() if terminal)

_ARRAY_RE = re.compile(r"^(\d+)_(\d+|\[[^\]]*\])$")
_STEP_SUFFIX_RE = re.compile(r"\.(batch|extern|interactive|\d+)$")
#: A pipeline whose exit status is silently ``tee``'s, not the trainer's.
_TEE_PIPE_RE = re.compile(r"\|\s*tee\b")
_PIPEFAIL_RE = re.compile(r"set\s+-[a-zA-Z]*o\s+pipefail|PIPESTATUS")


class Disagreement(str, enum.Enum):
    """How the scheduler's account and the run's own account relate.

    Naming these is the entire product of the module.  A reconciler that folded
    them all into "status updated" would destroy exactly the information a human
    needs: whether a number in a results table may be believed.
    """

    #: Scheduler and run agree, or the run is legitimately still going.
    AGREE = "agree"

    #: sacct says the job is still alive.  Nothing to reconcile yet.
    STILL_ACTIVE = "still_active"

    #: sacct reports a terminal state but the stream has no END: the process
    #: died (or was killed) without sealing.  Statistics over it are a subsample
    #: of unknown size.
    LOST_TELEMETRY = "lost_telemetry"

    #: The run sealed itself with one status; the scheduler reports another.
    #: The run's self-report is the one to distrust -- it was written by a
    #: process that was, by hypothesis, in the middle of being killed.
    SELF_MISREPORT = "self_misreport"

    #: sacct says COMPLETED/exit 0, but the submitting script pipes the trainer
    #: through ``tee`` without ``pipefail``/``PIPESTATUS``, so the recorded exit
    #: status is ``tee``'s and carries no information about the trainer.
    UNTRUSTWORTHY_EXIT = "untrustworthy_exit"

    #: sacct returned no row: outside the queried window, or aged out of the
    #: accounting retention.  NOT an error, and explicitly not a reason to stop
    #: reconciling the other jobs (defect a).
    MISSING_FROM_SACCT = "missing_from_sacct"

    #: The scheduler knows this job but no telemetry stream exists for it: the
    #: job died before python opened its log, or was never telemetry-aware.
    NO_STREAM = "no_stream"


@dataclasses.dataclass(frozen=True)
class ExitStatus:
    """A parsed sacct ``ExitCode`` -- defect (f).

    ``"0:0"`` is ``code=0, signal=0``; ``"0:9"`` is "killed by SIGKILL", which
    is a completely different event from ``"9:0"`` ("exited with status 9") and
    was indistinguishable while both were stored as strings.
    """

    code: Optional[int]
    signal: Optional[int]
    raw: str

    @property
    def killed(self) -> bool:
        return bool(self.signal)

    def to_json(self) -> Dict[str, Any]:
        return {"exit_code": self.code, "exit_signal": self.signal,
                "exit_code_raw": self.raw}


def parse_exit_code(raw: Any) -> ExitStatus:
    """Split sacct's ``code:signal`` pair.  Total: never raises."""
    text = "" if raw is None else str(raw).strip()
    code: Optional[int] = None
    sig: Optional[int] = None
    if text:
        parts = text.split(":")
        try:
            code = int(parts[0])
        except (ValueError, IndexError):
            code = None
        if len(parts) > 1:
            try:
                sig = int(parts[1])
            except ValueError:
                sig = None
    return ExitStatus(code=code, signal=sig, raw=text)


def parse_elapsed(raw: Any) -> Optional[int]:
    """``[DD-]HH:MM:SS`` -> seconds.  ``None`` when unparseable."""
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        head, text = text.split("-", 1)
        try:
            days = int(head)
        except ValueError:
            return None
    bits = text.split(":")
    try:
        values = [int(b) for b in bits]
    except ValueError:
        return None
    while len(values) < 3:
        values.insert(0, 0)
    hours, minutes, seconds = values[-3:]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def normalize_job_id(raw: Any) -> str:
    """Canonical form of a SLURM job id -- defect (c).

    Strips step suffixes (``123.batch``, ``123.extern``, ``123.0``), which sacct
    emits as separate rows and which must fold onto their parent job, and
    normalizes the array separator so ``34719790-2`` (the *slugified* form that
    :func:`telemetry.ids.make_exec_id` embeds in an ``exec_id``) and
    ``34719790_2`` (the ledger/sacct form) are the same key.  Without that fold,
    an array task's ledger entry and its sacct row can never meet.
    """
    text = str(raw or "").strip()
    text = _STEP_SUFFIX_RE.sub("", text)
    match = re.match(r"^(\d+)[-_](\d+)$", text)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return text


def job_id_aliases(raw: Any) -> List[str]:
    """Every key a row or a query might legitimately be looked up under.

    For an array task ``123_4`` that is ``{"123_4", "123"}``: some tooling in
    this repo records the array *parent* id, and a query for the parent should
    still find the task rather than silently reporting it missing.
    """
    canonical = normalize_job_id(raw)
    out = [canonical]
    match = _ARRAY_RE.match(canonical)
    if match:
        out.append(match.group(1))
    return out


@dataclasses.dataclass
class SacctRow:
    """One accounting record, normalized."""

    job_id: str
    raw_job_id: str
    state: str
    exit_status: ExitStatus
    elapsed_seconds: Optional[int]
    started_at: Optional[str]
    ended_at: Optional[str]
    nodelist: Optional[str]
    partition: Optional[str]

    @property
    def terminal(self) -> bool:
        return SACCT_STATES.get(self.state, (False, None))[0]

    @property
    def status(self) -> Optional[RunStatus]:
        return SACCT_STATES.get(self.state, (False, None))[1]

    @property
    def recognized(self) -> bool:
        return self.state in SACCT_STATES

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "sacct_job_id": self.job_id,
            "sacct_state": self.state,
            "sacct_terminal": self.terminal,
            "sacct_recognized": self.recognized,
            "elapsed_seconds": self.elapsed_seconds,
            "sacct_start": self.started_at,
            "sacct_end": self.ended_at,
            "nodelist": self.nodelist,
            "partition": self.partition,
        }
        payload.update(self.exit_status.to_json())
        return payload


def _clean(value: str) -> Optional[str]:
    value = (value or "").strip()
    return None if value in ("", "Unknown", "None", "N/A") else value


def parse_sacct(text: str) -> Dict[str, SacctRow]:
    """Parse ``-P -n`` sacct output into a lookup keyed by every alias.

    Rows are indexed under the normalization of ``JobID`` *and* of ``JobIDRaw``
    and under the array parent id, so a caller holding any of those forms finds
    the row (defect c).  Malformed lines are skipped rather than fatal: a
    truncated SSH response must not be able to abort reconciliation of the rows
    that did arrive (defect a, again).
    """
    rows: Dict[str, SacctRow] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) < len(SACCT_FIELDS):
            fields = fields + [""] * (len(SACCT_FIELDS) - len(fields))
        (job_id, raw_job_id, state, exit_code, elapsed,
         started, ended, nodelist, partition) = fields[:len(SACCT_FIELDS)]
        if not job_id.strip() and not raw_job_id.strip():
            continue
        # "CANCELLED by 123456" -> "cancelled"; "COMPLETED" -> "completed".
        state_word = (state or "").strip().split()[0].lower() if state.strip() else ""
        row = SacctRow(
            job_id=normalize_job_id(job_id or raw_job_id),
            raw_job_id=normalize_job_id(raw_job_id or job_id),
            state=state_word,
            exit_status=parse_exit_code(exit_code),
            elapsed_seconds=parse_elapsed(elapsed),
            started_at=_clean(started),
            ended_at=_clean(ended),
            nodelist=_clean(nodelist),
            partition=_clean(partition),
        )
        keys: List[str] = []
        for candidate in (job_id, raw_job_id):
            if str(candidate or "").strip():
                keys.extend(job_id_aliases(candidate))
        for key in keys:
            if not key:
                continue
            # A step row (".batch") folds onto its parent id; the parent's own
            # row must win, so never let a later step row overwrite it.
            if key in rows and rows[key].state and not _STEP_SUFFIX_RE.search(str(job_id)):
                continue
            rows.setdefault(key, row)
            if not _STEP_SUFFIX_RE.search(str(job_id)):
                rows[key] = row
    return rows


# -- talking to the cluster --------------------------------------------------

#: Signature of a command runner: ``(argv, timeout) -> stdout``.  Injectable so
#: every code path in this module is testable without a cluster.
Runner = Callable[[Sequence[str], float], str]


class SacctUnavailable(RuntimeError):
    """The scheduler could not be reached.  Never fatal to a reconcile."""


def run_command(argv: Sequence[str], timeout: float) -> str:
    """Run a command with a hard timeout -- defect (e).

    The timeout is a required positional, not a defaulted keyword, so a call
    site cannot omit it by accident.  This repo's SSH ControlMaster drops
    regularly (2FA re-auth), and ``subprocess.check_output`` with no timeout
    turns that into a reconciler that hangs until someone notices.
    """
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise SacctUnavailable(
            f"command timed out after {timeout}s: {' '.join(map(str, argv))}"
        ) from exc
    except OSError as exc:
        raise SacctUnavailable(f"could not execute {argv[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        raise SacctUnavailable(
            f"exit {completed.returncode}: {(completed.stderr or '').strip()[:400]}")
    return completed.stdout


def build_sacct_command(job_ids: Sequence[str], *, starttime: str,
                        endtime: str = "now") -> str:
    """The sacct invocation, with an explicit time window -- defect (b).

    ``sacct`` with no ``--starttime`` reports only jobs that started since
    midnight *today*.  A job that ended at 23:50 yesterday therefore returns no
    row at all, which the old reconciler could only interpret as an error.  The
    window is always explicit here, and :func:`default_starttime` derives it
    from the telemetry rather than guessing.
    """
    if not job_ids:
        raise ValueError("build_sacct_command called with no job ids; "
                         "callers must short-circuit (defect h)")
    return (
        "sacct -X -n -P"
        f" -j {','.join(job_ids)}"
        f" --starttime={starttime} --endtime={endtime}"
        f" --format={','.join(SACCT_FIELDS)}"
    )


def default_starttime(campaign: Optional[Campaign] = None, *,
                      fallback_days: int = 30) -> str:
    """A query window that provably covers every open execution.

    Derived from the earliest START timestamp among executions with no terminal
    record, minus a day of slack, so the window is as narrow as it can be while
    still being *provably* wide enough.  Falls back to a fixed lookback when the
    telemetry says nothing (e.g. reconciling ids read from ``pipeline.json``
    alone).
    """
    earliest: Optional[_dt.datetime] = None
    for log in _iter_logs(campaign):
        if log.end is not None:
            continue
        start = log.start
        stamp = (start or {}).get("ts")
        if not stamp:
            continue
        try:
            parsed = _dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        except ValueError:
            continue
        if earliest is None or parsed < earliest:
            earliest = parsed
    if earliest is None:
        return f"now-{fallback_days}days"
    return (earliest - _dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")


def _iter_logs(campaign: Optional[Campaign]) -> Iterable[RunLog]:
    if campaign is None:
        return []
    return [log for run in campaign.runs for log in run.executions]


def query_sacct(job_ids: Sequence[str], *, ssh: Optional[str],
                starttime: str, endtime: str = "now", timeout: float = 120.0,
                runner: Optional[Runner] = None,
                chunk_size: int = 200) -> Tuple[Dict[str, SacctRow], List[str]]:
    """Ask sacct about ``job_ids``.  Returns ``(rows, errors)``.

    Two properties, both direct fixes:

    * **Empty input is not an error and produces no command** (defect h).  The
      old code built ``sacct -X -j --format=...``, in which ``--format`` is
      swallowed as the argument to ``-j``.
    * **A failed chunk does not lose the others** (defect a).  Ids are queried in
      chunks and a chunk that times out or errors contributes its message to
      ``errors`` while every other chunk's rows are kept.  Partial knowledge is
      strictly better than none, and the jobs whose rows are missing are
      classified :attr:`Disagreement.MISSING_FROM_SACCT` rather than aborting.
    """
    unique = sorted({normalize_job_id(j) for j in job_ids if str(j or "").strip()})
    if not unique:
        return {}, []
    runner = runner or run_command
    rows: Dict[str, SacctRow] = {}
    errors: List[str] = []
    for index in range(0, len(unique), chunk_size):
        chunk = unique[index:index + chunk_size]
        command = build_sacct_command(chunk, starttime=starttime, endtime=endtime)
        argv = [ssh, command] if ssh else ["bash", "-lc", command]
        try:
            output = runner(argv, timeout)
        except SacctUnavailable as exc:
            errors.append(f"{chunk[0]}..{chunk[-1]}: {exc}")
            continue
        rows.update(parse_sacct(output))
    return rows, errors


# -- static trust analysis of the submitting script ---------------------------

def sbatch_exit_code_trustworthy(text: str) -> Tuple[bool, str]:
    """Does this script's exit status actually reflect the trainer's?

    ``python train.py 2>&1 | tee log`` exits with ``tee``'s status, so a trainer
    that segfaulted is recorded by SLURM as ``COMPLETED``, exit ``0:0``.  The
    only fixes are ``set -o pipefail`` or an explicit ``${PIPESTATUS[0]}`` check;
    in their absence the scheduler's verdict on this job carries no information
    about the trainer and must not be allowed to overrule the run's own stream.

    Returns ``(trustworthy, reason)``.  A conservative static check: it can only
    ever *downgrade* trust, never manufacture it.
    """
    if not _TEE_PIPE_RE.search(text):
        return True, ""
    if _PIPEFAIL_RE.search(text):
        return True, ""
    return False, ("script pipes through `tee` without `set -o pipefail` or a "
                   "${PIPESTATUS[0]} check: the recorded exit status is tee's, "
                   "not the trainer's")


def load_sbatch_trust(paths: Iterable[str]) -> Dict[str, Tuple[bool, str]]:
    """Evaluate :func:`sbatch_exit_code_trustworthy` for each readable path."""
    out: Dict[str, Tuple[bool, str]] = {}
    for path in sorted(set(p for p in paths if p)):
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                out[path] = sbatch_exit_code_trustworthy(handle.read())
        except OSError:
            # Unreadable is not untrustworthy; say nothing rather than guess.
            continue
    return out


# -- the classification ------------------------------------------------------

@dataclasses.dataclass
class Finding:
    """One reconciled execution: both accounts, and their relationship."""

    exec_id: str
    run_uid: str
    job_id: str
    disagreement: Disagreement
    self_status: Optional[str]
    sacct_state: Optional[str]
    observed_status: Optional[str]
    detail: str
    row: Optional[SacctRow] = None
    trusted_exit_code: bool = True
    appended: bool = False

    def payload(self) -> Dict[str, Any]:
        """The OBSERVED record body."""
        payload: Dict[str, Any] = {
            "observer": "sacct",
            "disagreement": self.disagreement.value,
            "self_status": self.self_status,
            "status": self.observed_status,
            "detail": self.detail,
            "trusted_exit_code": self.trusted_exit_code,
            "inferred": True,
        }
        if self.row is not None:
            payload.update(self.row.to_json())
        else:
            payload["sacct_state"] = self.sacct_state
        return payload

    def fingerprint(self) -> str:
        """Content identity of the finding, ignoring when it was made.

        Used for idempotence: reconciling twice with no change on the cluster
        must not append a second, identical OBSERVED record.  Otherwise the log
        (and every ledger diff folded from it) grows without any new
        information, which is how a signal-carrying diff becomes noise nobody
        reads.
        """
        body = self.payload()
        return json.dumps({k: v for k, v in body.items() if k not in ("detail",)},
                          sort_keys=True, separators=(",", ":"))


def classify(log: Optional[RunLog], row: Optional[SacctRow], *,
             trusted_exit: Tuple[bool, str] = (True, "")) -> Finding:
    """Relate one execution's own record to the scheduler's.

    The order of the checks is the argument.  Trust is evaluated *first*,
    because an untrustworthy exit code makes every subsequent comparison against
    ``COMPLETED`` meaningless; then absence of a row (we learned nothing); then
    the run's own silence (lost telemetry); then, only when both parties spoke,
    whether they agree.
    """
    exec_id = log.exec_id if log else ""
    run_uid = log.run_uid if log else ""
    job_id = normalize_job_id(log.job_id if log else (row.job_id if row else ""))
    self_status = log.status.value if (log and log.status) else None
    trustworthy, trust_reason = trusted_exit

    if log is None:
        return Finding(
            exec_id=exec_id, run_uid=run_uid, job_id=job_id,
            disagreement=Disagreement.NO_STREAM, self_status=None,
            sacct_state=row.state if row else None,
            observed_status=(row.status.value if row and row.status else None),
            detail=("the scheduler knows this job but no telemetry stream exists: "
                    "the job died before the run opened its log, or was launched "
                    "without telemetry"),
            row=row, trusted_exit_code=trustworthy)

    if row is None:
        return Finding(
            exec_id=exec_id, run_uid=run_uid, job_id=job_id,
            disagreement=Disagreement.MISSING_FROM_SACCT, self_status=self_status,
            sacct_state=None, observed_status=None,
            detail=("sacct returned no row for this job: it is outside the queried "
                    "window or has aged out of accounting retention. The run's own "
                    "record stands unchallenged; this is not evidence of anything."),
            row=None, trusted_exit_code=trustworthy)

    if not row.terminal:
        return Finding(
            exec_id=exec_id, run_uid=run_uid, job_id=job_id,
            disagreement=Disagreement.STILL_ACTIVE, self_status=self_status,
            sacct_state=row.state,
            observed_status=(row.status.value if row.status else None),
            detail=f"scheduler reports {row.state!r}; nothing terminal to reconcile",
            row=row, trusted_exit_code=trustworthy)

    scheduler_status = row.status.value if row.status else None

    if not trustworthy and row.state == "completed":
        # Evaluated before the agreement checks: a COMPLETED that came out of an
        # unguarded `| tee` is not evidence, so it must not be allowed to
        # "confirm" the run's self-report OR to contradict it.
        return Finding(
            exec_id=exec_id, run_uid=run_uid, job_id=job_id,
            disagreement=Disagreement.UNTRUSTWORTHY_EXIT, self_status=self_status,
            sacct_state=row.state, observed_status=None,
            detail=("scheduler reports COMPLETED but that verdict is not usable: "
                    + trust_reason),
            row=row, trusted_exit_code=False)

    if log.end is None:
        return Finding(
            exec_id=exec_id, run_uid=run_uid, job_id=job_id,
            disagreement=Disagreement.LOST_TELEMETRY, self_status=self_status,
            sacct_state=row.state, observed_status=RunStatus.LOST.value,
            detail=(f"scheduler reports {row.state!r} but the stream has no END "
                    f"record: the run never sealed itself, so its telemetry stops "
                    f"at step {log.last_step} for unknown reasons and every "
                    f"statistic over it is a subsample of unknown size"),
            row=row, trusted_exit_code=trustworthy)

    if scheduler_status is not None and self_status is not None \
            and scheduler_status != self_status:
        return Finding(
            exec_id=exec_id, run_uid=run_uid, job_id=job_id,
            disagreement=Disagreement.SELF_MISREPORT, self_status=self_status,
            sacct_state=row.state, observed_status=scheduler_status,
            detail=(f"the run sealed itself {self_status!r} but the scheduler "
                    f"reports {row.state!r}: the self-report was written by a "
                    f"process that was being killed and must not be trusted for "
                    f"aggregation"),
            row=row, trusted_exit_code=trustworthy)

    return Finding(
        exec_id=exec_id, run_uid=run_uid, job_id=job_id,
        disagreement=Disagreement.AGREE, self_status=self_status,
        sacct_state=row.state, observed_status=scheduler_status,
        detail="scheduler and run agree", row=row,
        trusted_exit_code=trustworthy)


# -- appending ---------------------------------------------------------------

def _max_seq(path: str) -> int:
    highest = -1
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
                if isinstance(record, dict) and isinstance(record.get("seq"), int):
                    highest = max(highest, record["seq"])
    except OSError:
        return -1
    return highest


def _last_observation_fingerprint(log: RunLog) -> Optional[str]:
    for record in reversed(log.events):
        if record["event"] == EventType.OBSERVED.value:
            body = {k: v for k, v in record.items()
                    if k not in ("v", "ts", "seq", "run_uid", "exec_id", "event",
                                 "detail")}
            return json.dumps(body, sort_keys=True, separators=(",", ":"))
    return None


def append_observed(log: RunLog, finding: Finding, *, ts: Optional[str] = None
                    ) -> bool:
    """Append one OBSERVED record.  Returns whether anything was written.

    A raw append, deliberately not through :class:`~telemetry.emit.TelemetryWriter`:
    the writer asserts single-writer discipline with an exclusive lock, which is
    correct for a live run and wrong for an external observer whose whole job is
    to write into a stream it does not own.  The envelope is still built by
    :func:`telemetry.schema.make_record`, so the record is validated exactly as
    the run's own would be, and ``seq`` continues the stream's numbering -- a gap
    would be indistinguishable from record loss, the one thing ``seq`` exists to
    prove.

    Idempotent by content: an unchanged finding is not re-appended.
    """
    payload = finding.payload()
    if _last_observation_fingerprint(log) == json.dumps(
            {k: v for k, v in payload.items() if k != "detail"},
            sort_keys=True, separators=(",", ":")):
        return False
    record = make_record(run_uid=log.run_uid, exec_id=log.exec_id,
                         seq=_max_seq(log.path) + 1, event=EventType.OBSERVED,
                         payload=payload, ts=ts)
    with open(log.path, "a", encoding="utf-8") as handle:
        handle.write(dumps(record) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    log.events.append(record)
    return True


# -- transactional state-file update (defect g) -------------------------------

@contextlib.contextmanager
def file_transaction(path: str, *, timeout: float = 30.0):
    """Exclusive, re-reading, atomic update of a JSON state file -- defect (g).

    The old reconciler read ``pipeline.json``, then made an SSH round trip taking
    seconds to minutes, then replaced the file with a value derived from the
    *stale* read.  Any edit made in that window -- by a human, by a concurrent
    submit script -- was silently destroyed.

    The invariant restored here is the standard one for optimistic-free
    concurrent update: **no observation used to compute the new value may
    predate the acquisition of the lock**.  Concretely, this context manager
    takes an exclusive ``flock`` on a sidecar lock file, *then* reads the
    document, yields it for mutation, and atomically replaces it before
    releasing.  The expensive, slow, network-bound part (the sacct query) is
    performed by the caller *outside* this block, so the critical section is
    bounded by local file I/O.

    The sidecar lock (rather than locking the file itself) is what makes the
    atomic ``os.replace`` compatible with locking: replacing the file would
    otherwise swap out the very inode the lock is held on.
    """
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    acquired = False
    try:
        try:
            import fcntl
            import time as _time
            deadline = _time.time() + timeout
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError:
                    if _time.time() >= deadline:
                        raise TimeoutError(
                            f"could not acquire {lock_path} within {timeout}s; "
                            "another reconcile or submit is holding it")
                    _time.sleep(0.05)
        except ImportError:  # pragma: no cover - non-POSIX
            pass

        # Re-read AFTER acquiring: this is the whole point.
        try:
            with open(path, encoding="utf-8") as source:
                state = json.load(source)
        except FileNotFoundError:
            state = {}
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path} is not valid JSON: {exc}") from exc

        yield state

        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tx-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as sink:
                json.dump(state, sink, indent=2, sort_keys=True)
                sink.write("\n")
                sink.flush()
                try:
                    os.fsync(sink.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    finally:
        if acquired:
            with contextlib.suppress(Exception):
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


# -- the reconcile itself ----------------------------------------------------

@dataclasses.dataclass
class ReconcileResult:
    findings: List[Finding] = dataclasses.field(default_factory=list)
    errors: List[str] = dataclasses.field(default_factory=list)
    queried: List[str] = dataclasses.field(default_factory=list)
    appended: int = 0
    sealed: List[str] = dataclasses.field(default_factory=list)

    def by_disagreement(self) -> Dict[str, List[Finding]]:
        out: Dict[str, List[Finding]] = {}
        for finding in self.findings:
            out.setdefault(finding.disagreement.value, []).append(finding)
        return out

    def render(self) -> str:
        lines = [f"queried {len(self.queried)} job id(s)"]
        for name, group in sorted(self.by_disagreement().items()):
            lines.append(f"  {name}: {len(group)}")
            for finding in sorted(group, key=lambda f: f.exec_id)[:20]:
                lines.append(f"    - {finding.exec_id or finding.job_id}: "
                             f"{finding.detail}")
        lines.append(f"appended {self.appended} OBSERVED record(s)")
        if self.sealed:
            lines.append(f"sealed {len(self.sealed)} lost stream(s): "
                         + ", ".join(sorted(self.sealed)[:10]))
        if self.errors:
            lines.append("scheduler errors (partial progress was still made):")
            lines.extend(f"  ! {e}" for e in self.errors)
        return "\n".join(lines)


def open_executions(campaign: Campaign) -> List[RunLog]:
    """Executions with no terminal record: the set worth asking sacct about."""
    return [log for run in campaign.runs for log in run.executions
            if log.end is None]


def _sbatch_path_for(log: RunLog) -> Optional[str]:
    start = log.start or {}
    for source in (start, start.get("config") or {}):
        if isinstance(source, Mapping) and source.get("sbatch_path"):
            return str(source["sbatch_path"])
    return None


def reconcile(
    root: str,
    *,
    ssh: Optional[str] = None,
    extra_job_ids: Sequence[str] = (),
    starttime: Optional[str] = None,
    endtime: str = "now",
    timeout: float = 120.0,
    runner: Optional[Runner] = None,
    dry_run: bool = False,
    seal_lost: bool = False,
    repo_root: Optional[str] = None,
    ts: Optional[str] = None,
) -> ReconcileResult:
    """Ask the scheduler about every open execution and record the answers.

    Never raises on a scheduler problem and never aborts a partially completed
    reconcile: an unreachable cluster, a timed-out chunk, or a job that aged out
    of accounting all degrade to findings, not exceptions (defect a/e).
    """
    campaign = load_campaign(root)

    # Work list = every execution with no terminal record, PLUS any execution
    # whose job id was named explicitly.  The second half matters: re-checking a
    # run that already sealed itself is exactly how a SELF_MISREPORT is found,
    # and treating a named-but-sealed job as "no stream" would report the
    # opposite of the truth.
    requested = {normalize_job_id(j) for j in extra_job_ids if str(j or "").strip()}
    by_job: Dict[str, List[RunLog]] = {}
    for run in campaign.runs:
        for log in run.executions:
            key = normalize_job_id(log.job_id)
            if log.end is None or key in requested:
                by_job.setdefault(key, []).append(log)

    job_ids = sorted(set(by_job) | requested)
    job_ids = [j for j in job_ids if j and j != "local"]

    result = ReconcileResult(queried=job_ids)
    # Defect (h): an empty work list is a normal outcome, not a malformed
    # command.  query_sacct short-circuits, and we still return a valid result.
    rows, errors = query_sacct(
        job_ids, ssh=ssh,
        starttime=starttime or default_starttime(campaign),
        endtime=endtime, timeout=timeout, runner=runner)
    result.errors = errors

    trust_cache: Dict[str, Tuple[bool, str]] = {}

    def trust_for(log: RunLog) -> Tuple[bool, str]:
        path = _sbatch_path_for(log)
        if not path:
            return (True, "")
        candidates = [path]
        if repo_root and not os.path.isabs(path):
            candidates.insert(0, os.path.join(repo_root, path))
        for candidate in candidates:
            if candidate in trust_cache:
                return trust_cache[candidate]
            try:
                with open(candidate, encoding="utf-8", errors="replace") as handle:
                    verdict = sbatch_exit_code_trustworthy(handle.read())
            except OSError:
                continue
            trust_cache[candidate] = verdict
            return verdict
        return (True, "")

    seen_jobs: Set[str] = set()
    for job_id in sorted(by_job):
        seen_jobs.add(job_id)
        row = rows.get(job_id)
        for log in sorted(by_job[job_id], key=lambda l: (l.attempt, l.exec_id)):
            finding = classify(log, row, trusted_exit=trust_for(log))
            if not dry_run and finding.disagreement is not Disagreement.STILL_ACTIVE:
                finding.appended = append_observed(log, finding, ts=ts)
                if finding.appended:
                    result.appended += 1
            result.findings.append(finding)

            if (seal_lost and not dry_run
                    and finding.disagreement is Disagreement.LOST_TELEMETRY):
                from .seal import seal_stream
                outcome = seal_stream(
                    log.path, RunStatus.LOST, exit_code=-1, signal_name=None,
                    reason=(f"reconciler: sacct reports {finding.sacct_state!r} "
                            "but the stream was never sealed"))
                if outcome == "sealed":
                    result.sealed.append(log.exec_id)

    # Job ids we were told about that have no stream at all.
    for job_id in job_ids:
        if job_id in seen_jobs:
            continue
        result.findings.append(classify(None, rows.get(job_id)))

    return result


def apply_to_pipeline(pipeline_path: str, result: ReconcileResult) -> Dict[str, int]:
    """Fold findings into a legacy ``pipeline.json`` inside one transaction.

    Provided for the transition period only: ``pipeline.json`` is a *view*
    (see :mod:`telemetry.views`), and the long-term answer is to regenerate it
    rather than to patch it.  What matters here is that the patch obeys defect
    (g)'s invariant -- the sacct query already happened, the read happens under
    the lock, and the merge is by ``job_id`` so a concurrently added entry
    survives instead of being clobbered by a stale snapshot.
    """
    terminal_by_job = {
        f.job_id: f for f in result.findings
        if f.disagreement in (Disagreement.LOST_TELEMETRY,
                              Disagreement.SELF_MISREPORT,
                              Disagreement.AGREE)
        and f.row is not None and f.row.terminal
    }
    counts = {"moved": 0, "updated": 0, "kept": 0}
    with file_transaction(pipeline_path) as state:
        active = state.get("active_runs") or []
        completed = state.setdefault("completed_runs", [])
        remaining = []
        for entry in active:
            finding = terminal_by_job.get(normalize_job_id(entry.get("job_id")))
            if finding is None:
                counts["kept"] += 1
                remaining.append(entry)
                continue
            updated = dict(entry)
            updated["status"] = finding.observed_status or finding.self_status or "unknown"
            updated["reconciled_from"] = "sacct"
            updated["disagreement"] = finding.disagreement.value
            if finding.row is not None:
                updated["completed_at"] = finding.row.ended_at
                updated["duration_seconds"] = finding.row.elapsed_seconds
                updated.update(finding.row.exit_status.to_json())
            completed.append(updated)
            counts["moved"] += 1
        state["active_runs"] = remaining
    return counts


# -- CLI ---------------------------------------------------------------------

def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m telemetry.reconcile",
        description="Cross-check every open execution against sacct and append "
                    "OBSERVED events recording what the scheduler said and how it "
                    "disagrees with the run's own record. Never mutates history.")
    parser.add_argument("--root", required=True, help="telemetry root")
    parser.add_argument("--ssh", default=None,
                        help="path to scripts/cluster/ssh.sh; omitted runs sacct "
                             "locally (only useful on a login node)")
    parser.add_argument("--job-id", action="append", default=[],
                        dest="job_ids", help="extra job id to ask about")
    parser.add_argument("--starttime", default=None,
                        help="sacct --starttime; default derives it from the "
                             "earliest unsealed START (defect b)")
    parser.add_argument("--endtime", default="now")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="hard timeout per sacct invocation (defect e)")
    parser.add_argument("--dry-run", action="store_true",
                        help="classify and print; append nothing")
    parser.add_argument("--seal-lost", action="store_true",
                        help="also append an inferred END(lost) to streams the "
                             "scheduler says are finished but that never sealed")
    parser.add_argument("--pipeline", default=None,
                        help="legacy pipeline.json to fold findings into, "
                             "transactionally")
    parser.add_argument("--repo-root", default=None,
                        help="root for resolving relative sbatch_path values "
                             "when checking exit-code trustworthiness")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = reconcile(
        args.root, ssh=args.ssh, extra_job_ids=args.job_ids,
        starttime=args.starttime, endtime=args.endtime, timeout=args.timeout,
        dry_run=args.dry_run, seal_lost=args.seal_lost,
        repo_root=args.repo_root)
    print(result.render())
    if args.pipeline and not args.dry_run:
        counts = apply_to_pipeline(args.pipeline, result)
        print(f"pipeline: moved={counts['moved']} kept={counts['kept']}")
    # Errors are reported but are NOT a failure exit: partial progress was made
    # and the caller's other work should not be blocked (defect a).
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
