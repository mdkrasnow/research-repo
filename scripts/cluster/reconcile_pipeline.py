#!/usr/bin/env python3
"""Reconcile a project's active SLURM ledger against sacct/squeue."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=Path)
    parser.add_argument("--ssh", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.pipeline.read_text())
    ids = sorted({str(run["job_id"]) for run in state["active_runs"]})
    command = (
        "sacct -X -j " + ",".join(ids) +
        " --format=JobIDRaw,State,ExitCode,Elapsed,End -P -n"
    )
    output = subprocess.check_output([str(args.ssh), command], text=True)
    accounting = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, status, exit_code, elapsed, ended = line.split("|")
        accounting[job_id] = {
            "status": status.split()[0].lower(),
            "exit_code": exit_code,
            "duration": elapsed,
            "completed_at": None if ended in {"", "Unknown", "None"} else ended,
        }
    active, completed = [], state.setdefault("completed_runs", [])
    moved = 0
    terminal = {"completed", "failed", "cancelled", "timeout", "out_of_memory",
                "node_fail", "preempted", "deadline"}
    for run in state["active_runs"]:
        observed = accounting.get(str(run["job_id"]))
        if observed is None:
            raise RuntimeError(f"sacct returned no record for {run['job_id']}")
        if observed["status"] in terminal:
            updated = dict(run)
            updated.update({key: value for key, value in observed.items() if value is not None})
            updated["reconciled_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            completed.append(updated)
            moved += 1
        else:
            updated = dict(run)
            updated["status"] = observed["status"]
            active.append(updated)
    state["active_runs"] = active
    with tempfile.NamedTemporaryFile("w", dir=args.pipeline.parent, delete=False) as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.pipeline)
    print(f"active={len(active)} moved_to_completed={moved}")


if __name__ == "__main__":
    main()
