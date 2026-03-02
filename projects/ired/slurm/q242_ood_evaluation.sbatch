#!/bin/bash
#SBATCH --job-name=ired_q242_ood_eval
#SBATCH --output=projects/ired/slurm/logs/%x_%j.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
set -euo pipefail

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

CKPT_DIR="projects/ired/results/ds_inverse/q242_hard_state_replay_unc_100k"
echo "Copying q242 checkpoint to work directory..."
mkdir -p "$WORK_DIR/$CKPT_DIR"
if [ -f "$REPO_ROOT/$CKPT_DIR/model-100.pt" ]; then
    cp "$REPO_ROOT/$CKPT_DIR/model-100.pt" "$WORK_DIR/$CKPT_DIR/"
    echo "checkpoint copied"
else
    echo "ERROR: Checkpoint not found at $REPO_ROOT/$CKPT_DIR/model-100.pt"
    exit 1
fi

PYTHONPATH="$WORK_DIR/projects/ired:${PYTHONPATH:-}" \
python projects/ired/experiments/evaluate_robustness.py \
    --config projects/ired/configs/q242_ood_evaluation.json \
    --output-dir projects/ired/results/evaluation/q242_replay_unc_ood/seed_0 \
    --seed 0

if [ -d "$WORK_DIR/projects/ired/results/evaluation" ]; then
    mkdir -p "$REPO_ROOT/projects/ired/results/evaluation"
    rsync -av "$WORK_DIR/projects/ired/results/evaluation/" "$REPO_ROOT/projects/ired/results/evaluation/" --include='*/' --include='*.json' --include='*.md' --exclude='*'
fi

cd /
rm -rf "$WORK_DIR"
