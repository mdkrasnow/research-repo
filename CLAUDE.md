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
