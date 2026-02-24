#!/bin/bash
#SBATCH --job-name=ired_q223_test_minimal
#SBATCH --output=projects/ired/slurm/logs/%x_%j.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j.err
#SBATCH --time=00:10:00
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

echo "Starting MINIMAL TEST: q223 with 100 steps only..."
python projects/ired/experiments/matrix_inversion_mining.py \
    --config projects/ired/configs/q223_test_minimal.json \
    --seed 0

echo "Experiment completed at $(date)"

REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying results to persistent storage..."
if [ -d "$WORK_DIR/projects/ired/results" ]; then
    mkdir -p "$REPO_ROOT/projects/ired/results"
    # Use recursive glob patterns to match files at any depth
    rsync -av --include='*/' --include='*.pt' --include='*.json' --include='*.md' --exclude='*' "$WORK_DIR/projects/ired/results/" "$REPO_ROOT/projects/ired/results/"
    echo "Results copied to $REPO_ROOT/projects/ired/results/"
    # Verify checkpoint was copied
    if [ -f "$REPO_ROOT/projects/ired/results/ds_inverse/q223_test_minimal/final_model.pt" ]; then
        echo "✓✓✓ SUCCESS! Checkpoint synced: $(ls -lh $REPO_ROOT/projects/ired/results/ds_inverse/q223_test_minimal/final_model.pt | awk '{print $5, $9}')"
    else
        echo "✗ FAILED: Checkpoint NOT found after rsync"
        echo "Debugging: Contents of results directory:"
        find "$REPO_ROOT/projects/ired/results" -type f -name "*.pt" 2>/dev/null | head -20
    fi
else
    echo "Warning: No results directory found at $WORK_DIR/projects/ired/results"
fi

echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"

echo "Job completed at $(date)"
