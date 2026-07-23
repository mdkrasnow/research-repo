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


def main(args):
    state = load()
    if args.dry_run:
        for stage, dependencies in graph():
            print(f"{stage}: after {', '.join(dependencies) or 'start'}")
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
    main(parser.parse_args())
