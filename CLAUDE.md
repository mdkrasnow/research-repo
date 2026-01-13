# Claude Code Rules for this repo

## Scope & Isolation
- Only modify files inside the target project directory: `projects/<slug>/...`
- Do NOT mix outputs between projects. Never write to another project's `runs/`, `results/`, `slurm/`, or `.state/`.

## Authoritative state
- `projects/<slug>/.state/pipeline.json` is the source of truth for what happens next.
- Every command must:
  1) acquire the project lock (see `scripts/ralph/lock.sh`)
  2) read pipeline.json
  3) do exactly one step (or one batch if explicitly specified)
  4) update pipeline.json + relevant docs
  5) release the lock

## Required structured outputs
After each step, update (as applicable):
- `projects/<slug>/.state/pipeline.json`
- `projects/<slug>/documentation/implementation-todo.md`
- `projects/<slug>/documentation/debugging.md`
- `projects/<slug>/documentation/queue.md`
- `projects/<slug>/runs/<run_id>/...` for experiments

## SLURM discipline
- Never submit a job without creating a run folder and `submit.json`.
- Always record job IDs and how to reproduce (command, git SHA if available, config snapshot).

## Stopping
- The repo includes a Stop hook that may block stopping while work is actionable.
- If a project requires user input, set `needs_user_input.value=true` and include `prompt`.
