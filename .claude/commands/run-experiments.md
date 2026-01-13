---
description: Submit the next READY experiment for a project and record job IDs.
allowed-tools: Read, Write, Edit, Bash, Glob
argument-hint: --project <slug> [--queue-id Q-###]
---
# /run-experiments
Submit one READY queue item for the project. Create a run ledger and update pipeline + queue.

Hybrid waiting policy:
- next_poll_after starts at now + 10-15 minutes
- increase poll interval each time you check and job is still running (cap at 60 min)

## Required behavior on laptops (no local SLURM)
If `sbatch` is not present locally:
1) Ensure SSH session exists:
   - `scripts/cluster/ensure_session.sh`
2) If session missing:
   - set `pipeline.needs_user_input.value=true`
   - prompt: `scripts/cluster/ssh_bootstrap.sh`
   - return without marking failure
3) Submit remotely:
   - `scripts/cluster/submit.sh <path-to-sbatch> <project_slug>`
   - record `job_id` into `runs/<run_id>/submit.json`
   - append to `pipeline.active_runs`
   - set `phase=WAIT_SLURM` and `next_poll_after=now+poll_interval`

## Log / artifact sync
After submission (and on every poll cycle), sync logs:
- `scripts/cluster/remote_fetch.sh <project_slug>`
