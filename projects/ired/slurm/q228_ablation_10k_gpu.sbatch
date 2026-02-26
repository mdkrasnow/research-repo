#!/bin/bash
#SBATCH --job-name=ired_q228_10k_b
#SBATCH --output=projects/ired/slurm/logs/%x_%A_%a.out
#SBATCH --error=projects/ired/slurm/logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=2-3

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

# Map array index to config
CONFIGS=(
    "projects/ired/configs/q228_implied_noise.json"
    "projects/ired/configs/q229_rank_recovery.json"
    "projects/ired/configs/q230_energy_ranking.json"
    "projects/ired/configs/q231_recovery_mining.json"
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Config: $CONFIG"

# Configuration (passed via environment variables)
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

# Create work directory for this job
WORK_DIR="${TMPDIR:-/tmp}/ired-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

echo "Git URL: $GIT_URL"
echo "Git SHA: $GIT_SHA"
echo "Work directory: $WORK_DIR"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU info:"
    nvidia-smi
fi

# Clone repository
echo "Cloning repository..."
git clone "$GIT_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout "$GIT_SHA"

echo "Repository cloned and checked out to: $(git rev-parse --short HEAD)"
echo "Repository info:"
git log -1 --oneline

echo "Starting 10K-step run with config: $CONFIG"
python projects/ired/experiments/matrix_inversion_mining.py \
    --config "$CONFIG" \
    --seed 0 \
    --train-steps 10000

echo "Experiment completed at $(date)"

# Copy results to persistent storage before cleanup
REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying results to persistent storage..."
if [ -d "$WORK_DIR/projects/ired/results" ]; then
    mkdir -p "$REPO_ROOT/projects/ired/results"
    rsync -av "$WORK_DIR/projects/ired/results/" "$REPO_ROOT/projects/ired/results/" --include='*/' --include='*.pt' --include='*.json' --include='*.md' --exclude='*'
    echo "Results copied to $REPO_ROOT/projects/ired/results/"
else
    echo "Warning: No results directory found at $WORK_DIR/projects/ired/results"
fi

# Cleanup
echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"
