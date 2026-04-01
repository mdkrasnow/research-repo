#!/bin/bash
#SBATCH --job-name=diff-eqm-fid
#SBATCH --output=slurm/logs/diff-eqm-fid_%A.out
#SBATCH --error=slurm/logs/diff-eqm-fid_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# FID evaluation needs more memory (Inception-v3 + 10K generated samples)
# and more time (~30min generation + ~10min FID computation)

# Load required modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU
nvidia-smi || { echo "No GPU available"; exit 1; }

# Set working directory
WORK_DIR="/tmp/diff-eqm-fid-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository at specific git SHA
GIT_URL="${GIT_URL:-https://github.com/mkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || { echo "SHA not in last 50 commits, fetching full..."; git fetch --unshallow origin 2>&1 | tail -5; git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1; }

# Install dependencies
pip install timm diffusers accelerate scipy 2>&1 | tail -5

# Sync model checkpoint from persistent storage (training results aren't in git)
REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "=== Syncing model checkpoint ===" >&2
rsync -av "$REPO_ROOT/projects/diff-EqM/results/" "$WORK_DIR/projects/diff-EqM/results/" 2>&1 | tail -10

# Also sync precomputed reference stats if they exist
if [ -f "$REPO_ROOT/projects/diff-EqM/results/cifar10_inception_stats.npz" ]; then
    echo "=== Found precomputed reference stats ===" >&2
fi

# FID config path
FID_CONFIG="${FID_CONFIG:-projects/diff-EqM/configs/eval_fid.json}"

echo "=== Starting FID evaluation with config: ${FID_CONFIG} ===" >&2
python -u projects/diff-EqM/experiments/evaluate_fid.py \
    --config "${FID_CONFIG}" \
    2>&1 | tee fid_eval.log

FID_EXIT=$?
echo "=== FID evaluation exit code: $FID_EXIT ===" >&2

# Sync results back (reference stats, sample grid, logs)
echo "=== Syncing results ===" >&2
rsync -av "$WORK_DIR/projects/diff-EqM/results/" "$REPO_ROOT/projects/diff-EqM/results/" 2>&1 | tail -10
cp fid_eval.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/fid_eval_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo "=== Done ===" >&2
exit $FID_EXIT
