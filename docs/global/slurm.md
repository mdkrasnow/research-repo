# SLURM (Harvard FAS RC + general)

Harvard FAS RC documentation:
https://docs.rc.fas.harvard.edu/kb/category/cluster-usage/

## Remote SLURM from a laptop (recommended)
If you're running Claude Code on macOS/Windows, `sbatch` will not exist locally.
This repo supports a **hybrid policy**:

1) Establish an SSH session to the FASRC login node once (includes 2FA)
2) Reuse that session via SSH ControlMaster for ~30 minutes
3) Submit + poll jobs remotely, then sync logs back locally

### Setup
1. Copy config:
   - `.claude/cluster.example.json` → `.claude/cluster.local.json`
2. Fill in:
   - `ssh.user`
   - `remote.repo_root` (path where this repo exists on the cluster)
3. Bootstrap SSH session (interactive, includes 2FA):
   - `scripts/cluster/ssh_bootstrap.sh`

After that, `/dispatch` and `/run-experiments` can submit to SLURM remotely.

### Security notes
- Never store passwords or 2FA codes in repo files.
- Prefer SSH keys + agent forwarding where possible.

## Core commands
- `sinfo`
- `squeue -u $USER`
- `sacct -j <jobid> --format=...`
- `sbatch <script.sbatch>`
- `scancel <jobid>`

## Automated Git Workflow

**IMPORTANT**: SLURM jobs automatically clone the repository fresh for each run. No manual repository maintenance needed on the cluster!

**How it works:**
1. When submitting a job, the current git commit SHA is captured locally
2. SLURM script clones the repo to `/tmp/ired-job-$SLURM_JOB_ID`
3. Checks out the exact commit SHA used for submission
4. Runs the experiment from that isolated clone
5. Cleans up the work directory after completion

**Benefits:**
- ✓ Always runs with exact code version from submission
- ✓ No manual `git pull` needed on cluster
- ✓ Each job is isolated (different jobs don't interfere)
- ✓ Automatic cleanup after job completes
- ✓ Reproducible - commit SHA recorded in submit.json

**Environment Variables Passed to Jobs:**
- `GIT_URL`: Repository URL (from `.claude/cluster.local.json`)
- `GIT_SHA`: Commit SHA to checkout (auto-captured at submission)

## Partition Selection Rule

**For experiments with runtime < 24 hours:**
- Use `--partition=gpu_test`
- Benefits: Better queue priority, faster start times
- Time limit: 24 hours maximum

**For experiments with runtime ≥ 24 hours:**
- Use `--partition=gpu`
- Benefits: Longer time limits available
- Trade-off: Lower priority, longer queue waits

This rule optimizes queue positioning while respecting partition time limits.

## Sbatch skeleton
```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=projects/<slug>/slurm/logs/%x_%j.out
#SBATCH --error=projects/<slug>/slurm/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_test          # Use gpu_test for jobs < 24h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Configuration (passed via environment variables from submit script)
GIT_URL="${GIT_URL:-https://github.com/USER/repo.git}"
GIT_SHA="${GIT_SHA:-main}"

# Create work directory for this job
WORK_DIR="${TMPDIR:-/tmp}/project-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

echo "Git URL: $GIT_URL"
echo "Git SHA: $GIT_SHA"
echo "Work directory: $WORK_DIR"

# Load modules (Harvard FAS RC specific versions)
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Clone repository
echo "Cloning repository..."
git clone "$GIT_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout "$GIT_SHA"

echo "Repository cloned and checked out to: $(git rev-parse --short HEAD)"
echo "Repository info:"
git log -1 --oneline

# Run experiment
echo "Starting experiment..."
python projects/<slug>/experiments/run.py --config projects/<slug>/configs/exp.yaml

echo "Experiment completed at $(date)"

# Cleanup
echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"
```

## Harvard FAS RC Module Versions

**Available Python versions** (as of 2026-01):
- python/3.10.9-fasrc01, python/3.10.12-fasrc01, python/3.10.13-fasrc01
- python/3.12.5-fasrc01, python/3.12.8-fasrc01, python/3.12.11-fasrc01, python/3.12.11-fasrc02

**Available CUDA versions** (as of 2026-01):
- cuda/11.3.1-fasrc01, cuda/11.8.0-fasrc01
- cuda/12.0.1-fasrc01, cuda/12.2.0-fasrc01, cuda/12.4.1-fasrc01, cuda/12.9.1-fasrc01

**Recommended for PyTorch**:
- python/3.10.13-fasrc01 + cuda/11.8.0-fasrc01 (stable compatibility)

To check available modules: `module spider <module_name>`

## Repo contract
- Generated scripts live in `projects/<slug>/slurm/jobs/`
- Logs go to `projects/<slug>/slurm/logs/`
- Every submission gets a run ledger under `projects/<slug>/runs/<run_id>/`

## Helper scripts
- Local/remote submit: `scripts/cluster/submit.sh <sbatch> <project_slug>`
- Local/remote status: `scripts/cluster/status.sh <jobid>`
- Remote log sync: `scripts/cluster/remote_fetch.sh <project_slug>`
