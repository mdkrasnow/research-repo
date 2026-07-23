"""Resumable direct-energy campaign state controller.

This controller intentionally owns only orchestration state and Slurm status;
individual scientific stages write their metrics beside the state file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/direct_energy_campaign"
STATUS = RESULTS / "status.json"
EVENTS = RESULTS / "events.jsonl"
CONFIG = Path(__file__).with_name("config.json")
SSH = ROOT.parents[1] / "scripts/cluster/ssh.sh"


def now():
    return datetime.now(timezone.utc).isoformat()


def event(state, stage, status, **extra):
    record = {"at": now(), "stage": stage, "status": status, **extra}
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    state.setdefault("events", []).append(record)


def load():
    if STATUS.exists():
        return json.loads(STATUS.read_text())
    return {"campaign": "direct_energy_campaign", "stages": {}, "jobs": {}, "created_at": now()}


def save(state):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    summary = RESULTS / "summary.md"
    lines = ["# Direct-energy campaign", "", f"Updated: {now()}", "", "| Stage | Status | Job |", "|---|---|---|"]
    for name, item in state.get("stages", {}).items():
        lines.append(f"| {name} | {item.get('status', 'PENDING')} | {item.get('job_id', '')} |")
    lines += ["", "| Job | Stage | Status | Commit |", "|---|---|---|---|"]
    for job_id, item in sorted(state.get("jobs", {}).items()):
        lines.append(f"| {job_id} | {item.get('stage', '')} | {item.get('status', '')} | {item.get('git_sha', '')} |")
    summary.write_text("\n".join(lines) + "\n")


def graph():
    return [
        ("0_audit", []), ("1_infrastructure", ["0_audit"]),
        ("2_fixed_batch", ["1_infrastructure"]), ("3_pilot", ["2_fixed_batch"]),
        ("4_sample_probe", ["3_pilot"]), ("5_scale_audit", ["3_pilot"]),
        ("6_stepsize", ["4_sample_probe"]), ("7_nfe", ["4_sample_probe"]),
        ("8_off_trajectory", ["4_sample_probe"]), ("9_generalization", ["4_sample_probe"]),
        ("10_decision", ["5_scale_audit", "6_stepsize", "7_nfe", "8_off_trajectory", "9_generalization"]),
        ("11_full", ["10_decision"]), ("12_report", ["10_decision"]),
    ]


def cluster_states(job_ids):
    """Return scheduler states without assuming the controller runs on Slurm."""
    if not job_ids:
        return {}
    command = [str(SSH), "sacct -n -P -j " + ",".join(job_ids) + " --format=JobIDRaw,State,ExitCode"]
    output = subprocess.check_output(command, text=True)
    states = {}
    for line in output.splitlines():
        columns = line.split("|")
        if len(columns) < 3 or not columns[0] or "." in columns[0]:
            continue
        state = columns[1].split()[0].upper()
        if state:
            states[columns[0]] = {"status": state.lower(), "exit_code": columns[2]}
    return states


def poll(state):
    states = cluster_states(list(state.get("jobs", {})))
    terminal = {"completed", "failed", "cancelled", "timeout", "out_of_memory"}
    for job_id, update in states.items():
        item = state["jobs"][job_id]
        old = item.get("status")
        new = update["status"]
        item.update(update)
        if old != new:
            item["updated_at"] = now()
            event(state, item.get("stage", "unknown"), new.upper(), job_id=job_id,
                  exit_code=update["exit_code"])
        if new in terminal:
            item["completed_at"] = now()
    return state


def main(args):
    state = load()
    if args.dry_run:
        config = json.loads(CONFIG.read_text())
        print(json.dumps(config, indent=2, sort_keys=True))
        for stage, dependencies in graph():
            print(f"{stage}: after {', '.join(dependencies) or 'start'}")
        return
    if args.record_job:
        stage, job_id, git_sha = args.record_job
        state.setdefault("jobs", {})[job_id] = {
            "stage": stage, "status": "pending", "git_sha": git_sha,
            "recorded_at": now(),
        }
        state.setdefault("stages", {}).setdefault(stage, {})["job_id"] = job_id
        state["stages"][stage]["status"] = "RUNNING"
        state["stages"][stage]["updated_at"] = now()
        event(state, stage, "RUNNING", job_id=job_id, git_sha=git_sha)
        save(state)
        return
    if args.poll:
        poll(state)
        save(state)
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    if args.mark:
        stage, status = args.mark
        state["stages"][stage] = {"status": status, "updated_at": now()}
        event(state, stage, status)
        save(state)
        return
    save(state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mark", nargs=2, metavar=("STAGE", "STATUS"))
    parser.add_argument("--record-job", nargs=3, metavar=("STAGE", "JOB_ID", "GIT_SHA"))
    parser.add_argument("--poll", action="store_true")
    main(parser.parse_args())
