#!/bin/bash
#SBATCH --job-name=ired_q245_steps
#SBATCH --output=projects/ired/slurm/logs/%x_%A_%a.out
#SBATCH --error=projects/ired/slurm/logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-4
set -euo pipefail

# Map array task ID to inner-loop opt steps
# Total opt steps = inner_steps * 10 landscapes
STEP_COUNTS=(1 2 5 10 20)
INNER_STEPS=${STEP_COUNTS[$SLURM_ARRAY_TASK_ID]}
TOTAL_STEPS=$((INNER_STEPS * 10))
echo "=== q245 Step-Semantics Check ==="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Inner-loop opt steps: $INNER_STEPS"
echo "Total opt steps (inner x 10 landscapes): $TOTAL_STEPS"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"

WORK_DIR="${TMPDIR:-/tmp}/ired-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

nvidia-smi

for attempt in 1 2 3; do
    if timeout 120 git clone "$GIT_URL" "$WORK_DIR" 2>/dev/null; then
        break
    fi
    echo "git clone attempt $attempt failed, retrying..."
    rm -rf "$WORK_DIR" && mkdir -p "$WORK_DIR"
    if [ "$attempt" -eq 3 ]; then
        echo "git clone failed, copying from local repo"
        cp -r "$REPO_ROOT/." "$WORK_DIR/"
    fi
done

cd "$WORK_DIR"
git checkout "$GIT_SHA" 2>/dev/null || true

# Copy baseline checkpoint
CKPT_DIR="projects/ired/results/ds_inverse/q230_baseline_no_mining"
echo "Copying baseline checkpoint to work directory..."
mkdir -p "$WORK_DIR/$CKPT_DIR"
if [ -f "$REPO_ROOT/$CKPT_DIR/model-100.pt" ]; then
    cp "$REPO_ROOT/$CKPT_DIR/model-100.pt" "$WORK_DIR/$CKPT_DIR/"
    echo "checkpoint copied"
else
    echo "ERROR: Checkpoint not found at $REPO_ROOT/$CKPT_DIR/model-100.pt"
    exit 1
fi

# Run OOD evaluation
echo ""
echo "=== OOD Evaluation (inner_steps=$INNER_STEPS, total=$TOTAL_STEPS) ==="
PYTHONPATH="$WORK_DIR/projects/ired:${PYTHONPATH:-}" \
python -u projects/ired/experiments/evaluate_robustness.py \
    --config projects/ired/configs/q245_step_semantics_ood.json \
    --output-dir "projects/ired/results/evaluation/q245_step_semantics/ood_steps_${INNER_STEPS}" \
    --inference-opt-steps "$INNER_STEPS" \
    --seed 0

# Run ID evaluation
echo ""
echo "=== ID Evaluation (inner_steps=$INNER_STEPS, total=$TOTAL_STEPS) ==="
PYTHONPATH="$WORK_DIR/projects/ired:${PYTHONPATH:-}" \
python -u projects/ired/experiments/evaluate_robustness.py \
    --config projects/ired/configs/q245_step_semantics_id.json \
    --output-dir "projects/ired/results/evaluation/q245_step_semantics/id_steps_${INNER_STEPS}" \
    --inference-opt-steps "$INNER_STEPS" \
    --seed 0

# Sync results back
if [ -d "$WORK_DIR/projects/ired/results/evaluation/q245_step_semantics" ]; then
    mkdir -p "$REPO_ROOT/projects/ired/results/evaluation/q245_step_semantics"
    rsync -av "$WORK_DIR/projects/ired/results/evaluation/q245_step_semantics/" \
        "$REPO_ROOT/projects/ired/results/evaluation/q245_step_semantics/" \
        --include='*/' --include='*.json' --include='*.md' --exclude='*'
    echo "Results synced to persistent storage"
fi

cd /
rm -rf "$WORK_DIR"

echo ""
echo "=== q245 Complete ==="
echo "Inner steps: $INNER_STEPS, Total steps: $TOTAL_STEPS"
echo "Expected IRED Table 2 (harder): 10→0.2110, 20→0.2100, 30→0.2090, 40→0.2063"
