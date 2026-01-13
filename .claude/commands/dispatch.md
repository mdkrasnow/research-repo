---
description: Main orchestrator: advance up to batch_size projects by one step each and update pipeline/todo/debugging/queue.
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
argument-hint: [--project <slug>]
---
# /dispatch

Advance up to `batch_size` projects (from `.claude/ralph/config.json`) by **one step each**.

Cluster-aware rule:
- If `phase` implies SLURM actions (RUN / WAIT_SLURM / CHECK) and `sbatch` is missing locally,
  attempt remote execution via `scripts/cluster/*`.
- If no SSH session exists, set `needs_user_input=true` with prompt:
  "Run scripts/cluster/ssh_bootstrap.sh (interactive SSH+2FA) then rerun /dispatch"
  and DO NOT spin.

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
