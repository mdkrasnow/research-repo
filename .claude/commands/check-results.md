---
description: Poll active SLURM runs for a project; write results.md and update summary/queue.
allowed-tools: Read, Write, Edit, Bash, Glob
argument-hint: --project <slug>
---
# /check-results
Check `pipeline.json.active_runs`, query job state, and if completed, write `runs/<run_id>/results.md` and update `results/summary.md`.

## Remote polling on laptops
If `squeue/sacct` are not available locally:
1) Ensure SSH session:
   - `scripts/cluster/ensure_session.sh`
2) If missing, set `needs_user_input` with prompt:
   - `scripts/cluster/ssh_bootstrap.sh`
3) Poll remotely:
   - `scripts/cluster/status.sh <job_id>`
4) Fetch logs:
   - `scripts/cluster/remote_fetch.sh <project_slug>`

## Completion handling
When job completes:
- parse the newest SLURM log for the run and write `runs/<run_id>/results.md`
- move queue item to DONE
- remove from `pipeline.active_runs`
- set `phase=CHECK` or `phase=RUN` depending on remaining queue items
