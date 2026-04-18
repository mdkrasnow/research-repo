# Claude Code Rules for this repo

## Alignment-first protocol (READ BEFORE ACTING)
Before doing anything non-trivial on a project (submitting jobs, writing configs, kicking off autoresearch, changing code, deciding what to try next), you **MUST** first read these files in order:
1. `projects/<slug>/.state/pipeline.json` — authoritative next action, current phase, and (if present) `publication_goal` block with target venues, current stage, and exit gate
2. `projects/<slug>/documentation/publishability-plan.md` (if present) — the north-star strategic plan toward publication
3. `projects/<slug>/program.md` (if autoresearch) — governance, constraints, baseline
4. `projects/<slug>/documentation/queue.md` — top-of-queue actions
5. `projects/<slug>/results.tsv` (if autoresearch) — what's already been tried

Every proposed action must answer: **"does this move us closer to a credible NeurIPS/ICML/ICLR submission per the publishability-plan?"** If no, de-prioritize or escalate. Proxy-scale gains are not publishable — they are filters for paper-scale confirmation runs. Never scale up a result that has not passed its stage exit gate (e.g., a 3-seed repeatability check).

If the state files disagree with what the user is asking for, surface the conflict rather than silently overriding. If a stage exit gate has not passed, say so before launching the next stage's compute.

## PI update protocol
If the project has `documentation/pi-updates.md`, check its trigger list after any experimental outcome, stage transition, or blocker. When a trigger fires:
1. Append a draft update entry to `pi-updates.md` using the template there.
2. Set `pipeline.json:needs_user_input.value=true` with a prompt pointing to the draft.
3. Do not send anything yourself — the user reviews and sends.
Also prepare a weekly digest draft on the day specified in `pipeline.json:publication_goal.pi_update_cadence.weekly_digest_day`, even if no trigger fired.

## Scope & Isolation
- Only modify files inside the target project directory: `projects/<slug>/...`
- Do NOT mix outputs between projects. Never write to another project's `runs/`, `results/`, `slurm/`, or `.state/`.

## Commit message discipline
- **Always include key metric results in commit messages** when committing after observing experimental outcomes. Example: "Resubmit q202-q204 with fixed sign (CD MSE 3.42 @ 100K steps, baseline 0.55)". This makes `git log` a searchable record of what worked and what didn't, without needing to re-fetch cluster logs.
- When cancelling and resubmitting jobs, note in the commit message what the last observed MSE was and why.

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
- **CRITICAL**: SLURM is NEVER available locally. ALL SLURM operations (sbatch, squeue, sacct) MUST use remote execution via `scripts/cluster/*`. Never check for or attempt to use local SLURM commands.
- Never submit a job without creating a run folder and `submit.json`.
- Always record job IDs and how to reproduce (command, git SHA if available, config snapshot).
- **CPU vs GPU usage rule**:
  - **Local execution**: ALWAYS use CPU only. Never use GPU resources for local development or testing.
  - **Cluster execution**: ALWAYS use GPU resources (typically A100s). Configure SLURM scripts with:
    - `--gres=gpu:1` (or more if needed)
    - `module load cuda/11.8.0-fasrc01` (or appropriate CUDA version)
    - GPU verification check using `nvidia-smi`
  - All SLURM batch scripts MUST be GPU-configured; never submit CPU-only jobs to the cluster
- **Automated git workflow**:
  - Each SLURM job automatically clones the repository fresh to `/tmp/project-job-$SLURM_JOB_ID`
  - The current git commit SHA is captured at submission time and passed via `GIT_SHA` environment variable
  - Jobs checkout the exact commit, run the experiment, and cleanup automatically
  - NO manual repository maintenance needed on cluster (no git pull, no rsync of code)
  - Only `slurm/` directory is synced to remote (for sbatch scripts and log structure)
- **Partition selection rule**:
  - For experiments expected to take < 24 hours: Use `--partition=gpu_test` for better priority queue positioning
  - For experiments expected to take ≥ 24 hours: Use `--partition=gpu`
  - gpu_test has higher priority but 24-hour time limit; gpu has lower priority but longer time limits
- **QOS management & partition diversification rule**:
  - When submitting multiple jobs simultaneously (e.g., 6 evaluation experiments), the cluster applies per-partition QOS limits that prevent submitting too many jobs to the same partition at once
  - **Solution**: Distribute jobs across both `gpu_test` AND `gpu` partitions to avoid hitting per-partition submission limits
  - **Strategy**: For a batch of N jobs, split them roughly evenly:
    - Submit ~N/2 jobs to `gpu_test` (higher priority, faster queue)
    - Submit remaining ~N/2 jobs to `gpu` (standard queue)
    - This allows all jobs to submit successfully without waiting for QOS limit resets
  - **When to apply**: Whenever submitting 4+ jobs in the same dispatch operation
  - **Example**: For 6 evaluation jobs, submit 3 to gpu_test and 3 to gpu to avoid "QOSMaxSubmitJobPerUserLimit" errors
  - Jobs on different partitions run in parallel without interference
- **Early polling rule**: After submitting a job, set `next_poll_after` to ~60 seconds after submission (not the default 15-minute interval). This catches initialization errors (missing modules, wrong Python version, etc.) quickly. After the early poll succeeds, resume normal polling intervals.
- **Job verification rule** (`/dispatch` MUST DO FIRST): Before processing any projects, verify all outstanding jobs in `active_runs`:
  1. Check if jobs marked "running" are actually running via `scripts/cluster/status.sh <job_id>` (remote SLURM)
  2. If a job is NOT running but status says "running":
     - Fetch logs via `scripts/cluster/remote_fetch.sh <project_slug>`
     - Check SLURM logs (`slurm/logs/<job_name>_<job_id>.{out,err}`)
     - Determine failure reason
     - Add entry to `documentation/debugging.md` (error, root cause, run_id, job_id, logs, timestamp)
     - Update `pipeline.json`: move to `completed_runs` with `status: "failed"`, add `failed_at` and `error`
     - Set `phase=DEBUG` if blocking
  3. Only after verification complete, proceed with normal dispatch

## Stopping
- The repo includes a Stop hook that may block stopping while work is actionable.
- If a project requires user input, set `needs_user_input.value=true` and include `prompt`.

## Autoresearch Mode

Projects can run in **autoresearch mode** — a continuous, autonomous experiment loop inspired by Karpathy's autoresearch paradigm. Instead of manual dispatch cycles, the agent forms hypotheses, implements changes, runs experiments, and keeps/reverts based on a single objective metric.

### Key concepts
- **`program.md`**: Governance file at `projects/<slug>/program.md`. Defines the objective metric, constraints, allowed files, ratchet rules, and termination conditions. The human writes this file; the agent writes code.
- **Ratchet**: Every experiment either improves the metric (KEEP) or doesn't (REVERT). The codebase monotonically improves. No human judgment needed for keep/revert decisions.
- **`results.tsv`**: Tab-separated experiment log at `projects/<slug>/results.tsv`. Tracks iteration, metric, status (KEEP/REVERT/CRASH), description, git SHA, timestamp. The agent reads this to decide what to try next.
- **Pilot runs**: Autoresearch uses short pilot runs (`pilot_steps` in program.md) for rapid iteration. Full training runs are done manually after autoresearch identifies the best configuration.

### Rules
- **ONE change per iteration**: Isolate variables. Never modify multiple dimensions simultaneously.
- **Metric decides**: No DEBATE phase. If the metric improved, keep it. If it regressed, revert it.
- **Respect file constraints**: Only modify files listed in `program.md:files_allowed`. Never modify `program.md` or the evaluation script.
- **Use `git revert`**: When reverting, use `git revert HEAD --no-edit` (preserves history) not `git reset`.
- **Auto-terminate**: Stop on plateau, max iterations, target achieved, or max wall hours.
- **Phase = AUTORESEARCH**: When active, pipeline.json has `phase: "AUTORESEARCH"` and an `autoresearch` object tracking progress.

### Entering autoresearch mode
- Run `/autoresearch --project <slug>` (requires `program.md` to exist)
- Or create a project with `/make-project` and answer "yes" to autoresearch eligibility
- The agent transitions pipeline.json to `phase: "AUTORESEARCH"` and begins the loop

### Resuming
- If a session ends mid-autoresearch, running `/autoresearch --project <slug>` again resumes from where it left off by reading `results.tsv` and `pipeline.json:autoresearch.iteration`.
