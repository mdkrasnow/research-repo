---
description: Main orchestrator: advance up to batch_size projects by one step each and update pipeline/todo/debugging/queue.
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
argument-hint: [--project <slug>]
---
# /dispatch

Advance up to `batch_size` projects (from `.claude/ralph/config.json`) by **one step each**.

Cluster-aware rule:
- **CRITICAL**: SLURM is NEVER available locally. ALL SLURM operations MUST use remote execution via `scripts/cluster/*`.
- If `phase` implies SLURM actions (RUN / WAIT_SLURM / CHECK), use remote execution via `scripts/cluster/*`.
- If no SSH session exists (checked via `scripts/cluster/ensure_session.sh`), set `needs_user_input=true` with prompt:
  "Run scripts/cluster/ssh_bootstrap.sh (interactive SSH+2FA) then rerun /dispatch"
  and DO NOT spin.

## Job Verification (MUST DO FIRST)
Before processing any projects, verify all outstanding SLURM jobs:
1. For each project with `active_runs` in pipeline.json:
   - Check if jobs are actually running via `scripts/cluster/status.sh <job_id>` (remote SLURM only)
   - If job is NOT running but status says "running":
     - Fetch logs via `scripts/cluster/remote_fetch.sh <project_slug>`
     - Check the SLURM logs in `slurm/logs/<job_name>_<job_id>.{out,err}`
     - Determine failure reason from logs
     - Add detailed entry to `documentation/debugging.md`:
       - Error message from logs
       - Root cause analysis
       - Run ID and job ID
       - Log file paths
       - Timestamp
     - Update `pipeline.json`:
       - Move from `active_runs` to `completed_runs` with `status: "failed"`
       - Add `failed_at` timestamp
       - Add `error` field with failure summary
     - Set `phase=DEBUG` if blocking, otherwise continue
2. Only after verification is complete, proceed with normal dispatch

For each touched project:
- Acquire lock (`scripts/ralph/lock.sh acquire projects/<slug>`)
- Execute one step based on pipeline phase / derived next action:
  - DEBUG, CHECK/POLL, TEST, RUN, IMPLEMENT, DEBATE
- Update:
  - `.state/pipeline.json`
  - `documentation/implementation-todo.md`
  - `documentation/debugging.md`
  - `documentation/queue.md`
  - run ledger under `runs/<run_id>/` for experiments
- Release lock

Hybrid SLURM waiting:
- If a job is RUNNING and `next_poll_after` is in the future, skip it and work on other projects.
- If nothing actionable, exit (Stop hook will then allow stopping).
