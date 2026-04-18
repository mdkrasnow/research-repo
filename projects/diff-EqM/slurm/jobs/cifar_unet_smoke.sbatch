#!/bin/bash
#SBATCH --job-name=cifar-unet-smoke
#SBATCH --output=slurm/logs/cifar-unet-smoke_%A.out
#SBATCH --error=slurm/logs/cifar-unet-smoke_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1

# Stage A.5 Step B — CIFAR-10 UNet smoke test.
#
# Purpose: confirm our data pipeline + FID eval are correct by training a
# diffusers UNet2DModel on CIFAR-10 with plain flow-matching loss. Target
# FID<15. If hit, the Stage A.5 bug is isolated to architecture (transformer
# vs UNet), unblocking Step A (port FM UNet + graft EqM loss).
#
# Emits "cifar10_smoke_fid: <value>" on the final line for grep-ability.

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/cifar-unet-smoke-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || {
    echo "SHA not in last 50 commits, fetching full..." >&2
    git fetch --unshallow origin 2>&1 | tail -5
    git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1
}

echo "=== Installing dependencies ===" >&2
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -5

# Use the persistent repo checkout's reference stats so we reuse the same
# CIFAR-10 Inception reference used by the broken 80ep runs. This is the
# controlled variable: if FID is good here, reference stats are fine.
REPO_ROOT="/n/home03/mkrasnow/research-repo"
REF_STATS="$REPO_ROOT/projects/diff-EqM/results/cifar10_inception_stats.npz"
OUTPUT_DIR="$WORK_DIR/projects/diff-EqM/results/cifar_unet_smoke"
mkdir -p "$OUTPUT_DIR"

# If the reference stats file exists in the persistent repo, copy into the
# work tree so the script finds it at the default relative path.
if [ -f "$REF_STATS" ]; then
    mkdir -p "$WORK_DIR/projects/diff-EqM/results"
    cp "$REF_STATS" "$WORK_DIR/projects/diff-EqM/results/cifar10_inception_stats.npz"
    echo "Using reference stats from $REF_STATS" >&2
else
    echo "No precomputed reference stats; will compute on first run (slower)" >&2
fi

# Default hyperparams; can be overridden via env vars before sbatch.
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-2e-4}"
SAMPLE_STEPS="${SAMPLE_STEPS:-100}"
FINAL_FID_SAMPLES="${FINAL_FID_SAMPLES:-10000}"
EVAL_EVERY="${EVAL_EVERY_EPOCHS:-25}"

echo "=== Training ===" >&2
echo "epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR" >&2
echo "sample_steps=$SAMPLE_STEPS final_fid_samples=$FINAL_FID_SAMPLES" >&2

python projects/diff-EqM/experiments/train_cifar_unet_smoke.py \
    --output-dir "$OUTPUT_DIR" \
    --data-dir ./data \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --sample-steps "$SAMPLE_STEPS" \
    --final-fid-samples "$FINAL_FID_SAMPLES" \
    --eval-every-epochs "$EVAL_EVERY" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

# Persist final checkpoint + log to the home repo results dir.
PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/cifar_unet_smoke_${SLURM_JOB_ID}"
mkdir -p "$PERSIST_DIR"
cp -v "$OUTPUT_DIR/final.pt" "$PERSIST_DIR/" 2>/dev/null || echo "WARN: no final.pt"
cp -v "$OUTPUT_DIR/train_log.tsv" "$PERSIST_DIR/" 2>/dev/null || echo "WARN: no train_log.tsv"
cp -v "$OUTPUT_DIR/train.log" "$PERSIST_DIR/" 2>/dev/null || true

echo "=== Done ===" >&2
