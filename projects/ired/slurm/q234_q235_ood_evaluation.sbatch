#!/bin/bash
#SBATCH --job-name=ired_q234_q235_ood
#SBATCH --output=projects/ired/slurm/logs/%x_%j_%a.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-1

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

CONFIGS=(
    "projects/ired/configs/q234_ood_evaluation.json"
    "projects/ired/configs/q235_ood_evaluation.json"
)
CHECKPOINT_DIRS=(
    "projects/ired/results/ds_inverse/q234_recovery_2step"
    "projects/ired/results/ds_inverse/q235_timestep_weight"
)
OUTPUT_DIRS=(
    "projects/ired/results/evaluation/q234_recovery_2step_ood"
    "projects/ired/results/evaluation/q235_timestep_weight_ood"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
CKPT_DIR="${CHECKPOINT_DIRS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="${OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}"
echo "Config: $CONFIG"
echo "Checkpoint dir: $CKPT_DIR"

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
git log -1 --oneline

REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying checkpoint to work directory..."
mkdir -p "$WORK_DIR/$CKPT_DIR"
if [ -f "$REPO_ROOT/$CKPT_DIR/model-100.pt" ]; then
    cp "$REPO_ROOT/$CKPT_DIR/model-100.pt" "$WORK_DIR/$CKPT_DIR/"
    echo "✓ Checkpoint copied ($(ls -lh $WORK_DIR/$CKPT_DIR/model-100.pt | awk '{print $5}'))"
else
    echo "ERROR: Checkpoint not found at $REPO_ROOT/$CKPT_DIR/model-100.pt"
    exit 1
fi

echo "Starting OOD evaluation with seed 0..."
PYTHONPATH="$WORK_DIR/projects/ired:${PYTHONPATH:-}" \
python projects/ired/experiments/evaluate_robustness.py \
    --config "$CONFIG" \
    --output-dir "$OUT_DIR/seed_0" \
    --seed 0

echo "Evaluation completed at $(date)"

if [ -d "$WORK_DIR/projects/ired/results/evaluation" ]; then
    mkdir -p "$REPO_ROOT/projects/ired/results/evaluation"
    rsync -av "$WORK_DIR/projects/ired/results/evaluation/" "$REPO_ROOT/projects/ired/results/evaluation/" --include='*/' --include='*.json' --include='*.md' --exclude='*'
    echo "Evaluation results copied"
else
    echo "Warning: No evaluation results directory found"
fi

cd /
rm -rf "$WORK_DIR"
