---
description: Poll SLURM job status and write results when experiments complete.
allowed-tools: Read, Write, Edit, Bash, Glob
argument-hint: --project <slug>
---
# /check-results

Poll the status of active SLURM jobs and process results when experiments complete.

## Usage

```bash
/check-results --project my-experiment
```

## What It Does

1. **Reads** active runs from `pipeline.json.active_runs`
2. **Queries** each job's status remotely via `scripts/cluster/status.sh`
3. **Fetches** SLURM logs via `scripts/cluster/remote_fetch.sh`
4. **Parses** completed job logs
5. **Writes** `runs/<run_id>/results.md` with parsed results
6. **Updates** `results/summary.md` with new findings
7. **Updates** `documentation/queue.md` (moves completed items to DONE)
8. **Updates** `pipeline.json` (removes from active_runs, transitions phase)

## Remote Polling (CRITICAL)

SLURM is **never available locally**. All polling uses remote execution:

- **SSH session**: `scripts/cluster/ensure_session.sh` (checks/sets up)
- **Status check**: `scripts/cluster/status.sh <job_id>` (remote)
- **Log fetch**: `scripts/cluster/remote_fetch.sh <project_slug>` (remote)

## Job States

| State | Action |
|-------|--------|
| RUNNING | Next poll in 10-60 minutes |
| COMPLETED | Parse logs, write results.md |
| FAILED | Log error to debugging.md, set phase=DEBUG |
| UNKNOWN | Fetch logs to diagnose |

## Completion Handling

When job completes:

1. **Parse** SLURM log output
2. **Write** `runs/<run_id>/results.md` with metrics and artifacts
3. **Move** queue.md item to DONE section
4. **Remove** from `pipeline.active_runs`
5. **Transition** phase based on remaining queue items

## Output

```markdown
# Results — run-20250121-001

## Job Status
- Job ID: 12345
- Status: COMPLETED
- Duration: 45 minutes
- Exit Code: 0

## Metrics
- loss: 0.234
- accuracy: 0.967
- ...

## Artifacts
- model checkpoint saved to: ...
- logs saved to: ...
```

## Error Handling

- **SSH session missing**: Prompts to run `scripts/cluster/ssh_bootstrap.sh`
- **Job not found**: Checks logs, adds to `debugging.md`
- **Parse error**: Saves raw logs and flags for manual review
- **Multiple jobs active**: Polls each independently, batches updates

## Related Skills

- **`/dispatch`**: Main orchestrator (calls this for CHECK phase)
- **`/run-experiments`**: Submit jobs (this polls their status)
- **`/check-status`**: See overall project status
