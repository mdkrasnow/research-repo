#!/bin/bash
#SBATCH --job-name=ired_q225_ood_eval_gpu
#SBATCH --output=projects/ired/slurm/logs/%x_%j_%a.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=4-7

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

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

# Copy checkpoint from persistent storage
REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying q225 checkpoint to work directory..."
mkdir -p "$WORK_DIR/projects/ired/results/ds_inverse/q225_tam_ctl_full_100k"
if [ -f "$REPO_ROOT/projects/ired/results/ds_inverse/q225_tam_ctl_full_100k/final_model.pt" ]; then
    cp "$REPO_ROOT/projects/ired/results/ds_inverse/q225_tam_ctl_full_100k/final_model.pt" "$WORK_DIR/projects/ired/results/ds_inverse/q225_tam_ctl_full_100k/"
    echo "✓ q225 checkpoint copied"
else
    echo "ERROR: q225 checkpoint not found at $REPO_ROOT/projects/ired/results/ds_inverse/q225_tam_ctl_full_100k/final_model.pt"
    exit 1
fi

# Run evaluation with seed
echo "Starting q225 OOD evaluation with seed $SLURM_ARRAY_TASK_ID..."
python projects/ired/experiments/matrix_inversion_mining.py \
    --config projects/ired/configs/q225_ood_evaluation.json \
    --seed "$SLURM_ARRAY_TASK_ID"

echo "Evaluation completed at $(date)"

# Copy results to persistent storage before cleanup
echo "Copying evaluation results to persistent storage..."
if [ -d "$WORK_DIR/projects/ired/results/evaluation" ]; then
    mkdir -p "$REPO_ROOT/projects/ired/results/evaluation"
    rsync -av "$WORK_DIR/projects/ired/results/evaluation/" "$REPO_ROOT/projects/ired/results/evaluation/" --include='*/' --include='*.json' --include='*.md' --exclude='*'
    echo "Evaluation results copied to $REPO_ROOT/projects/ired/results/evaluation/"
else
    echo "Warning: No evaluation results directory found at $WORK_DIR/projects/ired/results/evaluation"
fi

# Cleanup
echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"

