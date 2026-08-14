#!/usr/bin/env python3
"""Reconcile a project's active SLURM ledger against sacct/squeue.

Identity note (the defect this replaced): AGENTS.md mandates storing `job_id`
"including `_N` for array tasks", but this script queried
sacct with the JobID-Raw format, which prints the EXPANDED numeric id of an array task
(e.g. `39090543`) rather than the ledger's `39090540_3`.  The dict lookup then
missed and the script raised, aborting reconciliation for the ENTIRE project --
so the mandated "reconcile before planning" step could not run at all for any
project with an array job in its ledger.  We now ask for `JobID` (which prints
the `<jobid>_<taskid>` form) and index every record under both its raw form and
a normalized form, so ledger ids in either notation resolve.

A job that sacct does not know about is classified `unknown` and left in
`active_runs` with a note; it no longer takes the whole reconciliation down.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# SLURM terminal states, lowercased with spaces collapsed to underscores.
TERMINAL = {
    "completed", "failed", "cancelled", "timeout", "out_of_memory",
    "node_fail", "preempted", "deadline",
    # previously missing:
    "boot_fail", "revoked", "special_exit",
}


def normalize(job_id: str) -> str:
    """Canonical key for a SLURM id.

    `123_4` (array task), `123.batch` (step), and `123` all reduce to a form
    that can be matched against a ledger entry written in any of them.
    """
    job_id = str(job_id).strip()
    job_id = job_id.split(".", 1)[0]  # drop step suffixes (.batch/.extern/.0)
    return job_id


def keys_for(job_id: str) -> set[str]:
    """All lookup keys a record/ledger entry should be findable under."""
    base = normalize(job_id)
    keys = {base}
    # `123_4` also answers to its bare array-parent id when the ledger tracks
    # the parent, and `123_[5-9]` (pending array range) answers to `123`.
    match = re.match(r"^(\d+)_", base)
    if match:
        keys.add(match.group(1))
    return keys


def query_sacct(ssh: Path, ids: list[str], timeout: int) -> dict:
    command = (
        "sacct -X -j " + ",".join(ids) +
        " --format=JobID,State,ExitCode,Elapsed,End -P -n"
    )
    output = subprocess.check_output([str(ssh), command], text=True,
                                     timeout=timeout)
    accounting: dict[str, dict] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, status, exit_code, elapsed, ended = parts[:5]
        observed = {
            "status": status.split()[0].lower().replace(" ", "_")
                      if status.strip() else "unknown",
            "exit_code": exit_code,
            "duration": elapsed,
            "completed_at": None if ended in {"", "Unknown", "None"} else ended,
        }
        for key in keys_for(job_id):
            accounting.setdefault(key, observed)
    return accounting


def lookup(accounting: dict, job_id: str):
    for key in keys_for(job_id):
        if key in accounting:
            return accounting[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--ssh", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180,
                        help="seconds to wait for the sacct query")
    args = parser.parse_args()
    state = json.loads(args.pipeline.read_text())
    runs = state.get("active_runs") or []
    if not runs:
        # No ledger entries is a legitimate state (nothing in flight), not an
        # error: `sacct -j ` with an empty id list would query EVERY job.
        state.setdefault("completed_runs", [])
        print("active=0 moved_to_completed=0 unknown=0 (empty active_runs)")
        return

    ids = sorted({normalize(run["job_id"]) for run in runs})
    accounting = query_sacct(args.ssh, ids, args.timeout)

    active, completed = [], state.setdefault("completed_runs", [])
    moved, unknown = 0, []
    for run in runs:
        observed = lookup(accounting, run["job_id"])
        if observed is None:
            # One job sacct has forgotten (purged, or a bad id) must not abort
            # reconciliation for every other job in the project.
            updated = dict(run)
            updated["status"] = "unknown"
            updated["reconcile_note"] = (
                "sacct returned no record for this job_id; left active for "
                "manual review")
            updated["reconciled_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds")
            active.append(updated)
            unknown.append(str(run["job_id"]))
            continue
        if observed["status"] in TERMINAL:
            updated = dict(run)
            updated.update({key: value for key, value in observed.items()
                            if value is not None})
            updated["reconciled_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds")
            completed.append(updated)
            moved += 1
        else:
            updated = dict(run)
            updated["status"] = observed["status"]
            active.append(updated)
    state["active_runs"] = active
    with tempfile.NamedTemporaryFile("w", dir=args.pipeline.parent,
                                     delete=False) as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.pipeline)
    print(f"active={len(active)} moved_to_completed={moved} "
          f"unknown={len(unknown)}")
    if unknown:
        print("unknown job_ids (no sacct record): " + ", ".join(unknown),
              file=sys.stderr)


if __name__ == "__main__":
    main()
