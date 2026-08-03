"""Idempotent local submit/monitor runner for the immutable experiment manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
PROJECT = REPO / "projects" / "masked-EqM"
TRANSIENT_STATES = {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "REQUEUED"}
SUCCESS_STATES = {"COMPLETED"}
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}


def run(command, *, env=None) -> str:
    result = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout.strip()


def remote(command: str) -> str:
    return run([str(REPO / "scripts/cluster/ssh.sh"), command])


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def scheduler_state(job_id: str) -> str:
    output = remote(
        f"sacct -j {job_id} -X -o JobIDRaw,State -n -P | awk -F'|' '$1==\"{job_id}\" {{print $2; exit}}'"
    )
    return output.split()[0].split("+")[0] if output else "UNKNOWN"


def completion_exists(path: str) -> bool:
    return remote(f"test -f {path} && echo yes || echo no").strip() == "yes"


def latest_checkpoint(directory: str) -> str | None:
    output = remote(f"ls -1t {directory}/checkpoints/*.pt 2>/dev/null | head -1")
    return output.strip() or None


def submit(task, resume=None) -> str:
    exports = [f"CONFIG_REL={task['config_rel']}"]
    if task.get("task"):
        exports.append(f"TASK={task['task']}")
    if resume:
        exports.append(f"RESUME_CKPT={resume}")
    env = os.environ.copy()
    env["SBATCH_EXPORTS"] = ",".join(exports)
    return run(
        [
            str(REPO / "scripts/cluster/remote_submit.sh"),
            task["sbatch"],
            "masked-EqM",
        ],
        env=env,
    ).splitlines()[-1]


def tail(task, job_id):
    pattern = task.get("log_pattern")
    if not pattern:
        return ""
    return remote(f"tail -20 {pattern.replace('{job_id}', job_id)} 2>/dev/null")


def reconcile(manifest, state):
    tasks = {task["id"]: task for task in manifest["workflow"]}
    for task_id, task_state in state["tasks"].items():
        task = tasks[task_id]
        if completion_exists(task["completion_marker"]):
            task_state.update(status="completed", scheduler_state="COMPLETED")
            continue
        job_id = task_state.get("job_id")
        if not job_id:
            continue
        current = scheduler_state(job_id)
        task_state["scheduler_state"] = current
        task_state["last_polled_at"] = datetime.now(timezone.utc).isoformat()
        if current in ACTIVE_STATES:
            task_state["status"] = "active"
            log_tail = tail(task, job_id)
            if log_tail:
                task_state["last_log_tail"] = log_tail[-4000:]
        elif current in SUCCESS_STATES:
            task_state["status"] = "awaiting_artifact"
        elif current in TRANSIENT_STATES and task_state.get("retries", 0) < 2:
            resume = latest_checkpoint(task["output_dir"]) if task["kind"] == "training" else None
            new_id = submit(task, resume=resume)
            task_state.update(
                job_id=new_id,
                status="active",
                scheduler_state="PENDING",
                retries=task_state.get("retries", 0) + 1,
                resumed_from=resume,
            )
            task_state.setdefault("history", []).append(
                {"job_id": job_id, "state": current, "retry_job_id": new_id, "resume": resume}
            )
        else:
            task_state["status"] = "failed"
            task_state["failure_state"] = current

    for task in manifest["workflow"]:
        task_state = state["tasks"][task["id"]]
        if task_state["status"] != "missing":
            continue
        if not all(state["tasks"][dep]["status"] == "completed" for dep in task.get("depends_on", [])):
            continue
        active_training = sum(
            state["tasks"][entry["id"]]["status"] in {"active", "awaiting_artifact"}
            for entry in manifest["workflow"]
            if entry["kind"] == "training"
        )
        active_evaluation = sum(
            state["tasks"][entry["id"]]["status"] in {"active", "awaiting_artifact"}
            for entry in manifest["workflow"]
            if entry["kind"] in {"recovery", "generation"}
        )
        if task["kind"] == "training" and active_training >= 2:
            continue
        if task["kind"] in {"recovery", "generation"} and active_evaluation >= 2:
            continue
        if completion_exists(task["completion_marker"]):
            task_state.update(status="completed", scheduler_state="COMPLETED")
            continue
        job_id = submit(task)
        task_state.update(
            job_id=job_id,
            status="active",
            scheduler_state="PENDING",
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )


def main(manifest_path: str, once: bool, poll_seconds: int) -> int:
    manifest = json.loads(Path(manifest_path).read_text())
    state_path = PROJECT / "experiments/masked_eqm_field_shaping/job_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["manifest_sha256"] != manifest["manifest_sha256"]:
            raise RuntimeError("job_state belongs to a different immutable manifest")
    else:
        state = {
            "goal": manifest["experiment_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "status": "running",
            "tasks": {
                task["id"]: {"status": "missing", "job_id": None, "retries": 0, "history": []}
                for task in manifest["workflow"]
            },
        }
    while True:
        reconcile(manifest, state)
        statuses = {entry["status"] for entry in state["tasks"].values()}
        if statuses == {"completed"}:
            state["status"] = "complete"
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
        elif "failed" in statuses:
            state["status"] = "failed"
        else:
            state["status"] = "running"
        atomic_json(state_path, state)
        print(json.dumps({key: value["status"] for key, value in state["tasks"].items()}, sort_keys=True), flush=True)
        if once or state["status"] in {"complete", "failed"}:
            return 0 if state["status"] == "complete" else 1
        time.sleep(max(10, min(poll_seconds, 60)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    raise SystemExit(main(args.manifest, args.once, args.poll_seconds))
