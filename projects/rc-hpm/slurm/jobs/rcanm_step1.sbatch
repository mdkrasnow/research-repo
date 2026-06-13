#!/bin/bash
#SBATCH --job-name=rcanm-step1
#SBATCH --output=projects/rc-hpm/slurm/logs/rcanm-step1_%A.out
#SBATCH --error=projects/rc-hpm/slurm/logs/rcanm-step1_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# RC-ANM Step 1: premise probe on a trained EqM-B/2 v10 checkpoint.
# Inference-only (no training, no FID). Mines ANM endpoints at an eps sweep,
# scores safety + gradient impact, writes a verdict JSON synced back to repo.
# preregistration-rcanm-step1.md.
set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
echo "=== GPU ===" >&2; nvidia-smi || { echo "no GPU"; exit 1; }
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/imagenet1k_80ep_v10_b2_lambda10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
BATCHES="${BATCHES:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPS_GRID="${EPS_GRID:-0.25,0.5,1.0,1.5}"

WORK_DIR="/tmp/rcanm-step1-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?set GIT_SHA to the commit with the probe}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== clone @ ${GIT_SHA} ===" >&2
git init . >&2
git remote add origin "$GIT_URL" >&2
git fetch --depth=50 origin main 2>&1 | tail -3 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -2 || { git fetch --unshallow origin 2>&1 | tail -2; git checkout "${GIT_SHA}" 2>&1 | tail -2 || exit 1; }

echo "=== deps ===" >&2
pip install timm diffusers accelerate scipy torchvision 2>&1 | tail -3

[ -f "$CKPT" ] || { echo "ERROR ckpt not found: $CKPT" >&2; exit 1; }
[ -d "$IMAGENET_PATH" ] || { echo "ERROR imagenet not found: $IMAGENET_PATH" >&2; exit 1; }

OUT="$WORK_DIR/rcanm_step1_verdict.json"
SYNC="$REPO_ROOT/projects/rc-hpm/results/rcanm_step1_verdict.json"
mkdir -p "$(dirname "$SYNC")"

cd "$WORK_DIR/projects/rc-hpm"
python3 experiments/rcanm_step1_probe.py \
    --ckpt "$CKPT" --data-path "$IMAGENET_PATH" --out "$OUT" \
    --batches "$BATCHES" --batch-size "$BATCH_SIZE" --eps-grid "$EPS_GRID" \
    2>&1 | tee "$WORK_DIR/probe.log"

cp "$OUT" "$SYNC" && echo "=== synced verdict -> $SYNC ===" >&2
cat "$OUT" >&2
