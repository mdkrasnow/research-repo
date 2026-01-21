---
description: Submit the next READY experiment to SLURM and record job tracking information.
allowed-tools: Read, Write, Edit, Bash, Glob
argument-hint: --project <slug> [--queue-id Q-###]
---
# /run-experiments

Submit one READY experiment from the project's queue to the SLURM cluster.

## Usage

```bash
/run-experiments --project my-experiment          # Submit next READY item
/run-experiments --project my-experiment --queue-id Q-001  # Submit specific item
```

## What It Does

1. **Validates** the project exists and has READY items in queue
2. **Selects** the next READY experiment (or specified by `--queue-id`)
3. **Submits** the SLURM sbatch script to cluster (or remote cluster if needed)
4. **Records** job ID in `runs/<run_id>/submit.json`
5. **Updates** `pipeline.json` with active run tracking
6. **Sets** polling schedule (`next_poll_after` at now + 10-15 minutes)

## Remote Submission (No Local SLURM)

If SLURM is not available locally (laptop, non-cluster machine):

- Ensures SSH session is active via `scripts/cluster/ensure_session.sh`
- Submits remotely via `scripts/cluster/submit.sh`
- Syncs logs via `scripts/cluster/remote_fetch.sh <project_slug>`

If SSH session missing:
- Sets `pipeline.needs_user_input.value=true`
- Prompts: `scripts/cluster/ssh_bootstrap.sh`
- Returns without marking job as submitted

## Polling Strategy

- **Initial poll**: ~10-15 minutes after submission (catches early failures)
- **Adaptive polling**: Increases interval each check (cap at 60 minutes)
- Logs automatically synced after each submission and poll

## Output

```json
{
  "run_id": "run-20250121-001",
  "job_id": "12345",
  "submitted_at": "2025-01-21T14:30:00Z",
  "status": "submitted",
  "next_poll_at": "2025-01-21T14:45:00Z"
}
```

## Error Handling

- **SSH session missing**: Sets `needs_user_input=true`, prompts bootstrap
- **Queue empty**: Returns message "No READY experiments in queue"
- **Invalid queue-id**: Returns error with available items
- **SLURM submission failure**: Logs error to `debugging.md`, updates pipeline status

## Related Skills

- **`/dispatch`**: Main orchestrator (calls this for RUN phase)
- **`/check-results`**: Poll results of submitted jobs
- **`/check-status`**: Monitor active runs
