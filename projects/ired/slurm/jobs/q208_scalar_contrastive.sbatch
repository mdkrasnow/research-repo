#!/bin/bash
#SBATCH --job-name=ired_q208_scalar_cl
#SBATCH --output=projects/ired/slurm/logs/%x_%A_%a.out
#SBATCH --error=projects/ired/slurm/logs/%x_%A_%a.err
#SBATCH --array=0-9%2
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

WORK_DIR="${TMPDIR:-/tmp}/ired-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

echo "Git URL: $GIT_URL"
echo "Git SHA: $GIT_SHA"
echo "Work directory: $WORK_DIR"

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

if command -v nvidia-smi &> /dev/null; then
    echo "GPU info:"
    nvidia-smi
fi

echo "Cloning repository..."
git clone "$GIT_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout "$GIT_SHA"

echo "Repository cloned and checked out to: $(git rev-parse --short HEAD)"
echo "Repository info:"
git log -1 --oneline

echo "Starting q208: scalar energy + IRED contrastive loss..."
python projects/ired/experiments/matrix_inversion_mining.py \
    --config projects/ired/configs/q208_scalar_contrastive.json \
    --seed $SLURM_ARRAY_TASK_ID

echo "Experiment completed at $(date)"

REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying results to persistent storage..."
if [ -d "$WORK_DIR/results" ]; then
    mkdir -p "$REPO_ROOT/projects/ired/results"
    rsync -av "$WORK_DIR/results/" "$REPO_ROOT/projects/ired/results/"
    echo "Results copied to $REPO_ROOT/projects/ired/results/"
else
    echo "Warning: No results directory found at $WORK_DIR/results"
fi

echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"
