#!/bin/bash
#SBATCH --job-name=ired_q222_tam_delta_sweep
#SBATCH --output=projects/ired/slurm/logs/%x_%j_%a.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-15

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Configuration (passed via environment variables)
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

# Decode config and seed from array task ID
# Array 0-15: 4 configs × 4 seeds
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 4 ))
CONFIGS=("q222_tam_delta0.5" "q222_tam_delta1.0" "q222_tam_delta2.0" "q222_tam_delta3.0")
CONFIG_NAME="${CONFIGS[$CONFIG_IDX]}"

echo "Config Index: $CONFIG_IDX"
echo "Seed: $SEED"
echo "Config Name: $CONFIG_NAME"

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

# Run experiment with config and seed
echo "Starting TAM pgd_delta sweep with config=$CONFIG_NAME, seed=$SEED..."
python projects/ired/experiments/matrix_inversion_mining.py \
    --config "projects/ired/configs/${CONFIG_NAME}.json" \
    --seed "$SEED"

echo "Experiment completed at $(date)"

# Copy results to persistent storage before cleanup
REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying results to persistent storage..."
if [ -d "$WORK_DIR/projects/ired/results" ]; then
    mkdir -p "$REPO_ROOT/projects/ired/results"
    # Use recursive glob patterns to match files at any depth
    rsync -av --include='*/' --include='*.pt' --include='*.json' --include='*.md' --exclude='*' "$WORK_DIR/projects/ired/results/" "$REPO_ROOT/projects/ired/results/"
    echo "Results copied to $REPO_ROOT/projects/ired/results/"
    # Verify checkpoint was copied
    if [ -f "$REPO_ROOT/projects/ired/results/ds_inverse/q222_tam_delta/final_model.pt" ]; then
        echo "✓ Checkpoint synced: $(ls -lh $REPO_ROOT/projects/ired/results/ds_inverse/q222_tam_delta/final_model.pt | awk '{print $5, $9}')"
    fi
else
    echo "Warning: No results directory found at $WORK_DIR/projects/ired/results"
fi

# Cleanup
echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"
