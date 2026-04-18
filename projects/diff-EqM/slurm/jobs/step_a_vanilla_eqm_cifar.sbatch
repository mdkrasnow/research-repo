#!/bin/bash
#SBATCH --job-name=stepa-van-cifar
#SBATCH --output=slurm/logs/stepa-van-cifar_%A.out
#SBATCH --error=slurm/logs/stepa-van-cifar_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1

# Stage A.5 Step A — Vanilla EqM on CIFAR-10 using Flow Matching's UNet.
#
# Combines FM's CIFAR-10 UNet (128 channels, 4 res blocks, attn at res=2)
# with EqM's c(t)-weighted velocity target (a=0.8, gain=4). Target FID <= 3.66
# on 50K samples (paper reports 3.36 for EqM with their tuned UNet + sampler).
#
# Emits "cifar10_eqm_fid: <value>" on the final line.

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/stepa-van-cifar-${SLURM_JOB_ID}"
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

# Reuse CIFAR-10 Inception reference stats from persistent repo.
REPO_ROOT="/n/home03/mkrasnow/research-repo"
REF_STATS="$REPO_ROOT/projects/diff-EqM/results/cifar10_inception_stats.npz"
OUTPUT_DIR="$WORK_DIR/projects/diff-EqM/results/step_a_vanilla_eqm_cifar_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
if [ -f "$REF_STATS" ]; then
    mkdir -p "$WORK_DIR/projects/diff-EqM/results"
    cp "$REF_STATS" "$WORK_DIR/projects/diff-EqM/results/cifar10_inception_stats.npz"
fi

EPOCHS="${EPOCHS:-1800}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-2e-4}"
SEED="${SEED:-0}"
SAMPLER="${SAMPLER:-euler}"
EULER_NUM_STEPS="${EULER_NUM_STEPS:-50}"
EVAL_EVERY="${EVAL_EVERY_EPOCHS:-100}"

echo "=== Training vanilla EqM on CIFAR-10 via FM UNet ===" >&2
echo "epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR seed=$SEED sampler=$SAMPLER" >&2

python projects/diff-EqM/experiments/train_cifar_eqm_unet.py \
    --output-dir "$OUTPUT_DIR" \
    --data-dir ./data \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --sampler "$SAMPLER" \
    --euler-num-steps "$EULER_NUM_STEPS" \
    --eval-every-epochs "$EVAL_EVERY" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

# Persist.
PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/step_a_vanilla_eqm_cifar_${SLURM_JOB_ID}"
mkdir -p "$PERSIST_DIR"
cp -v "$OUTPUT_DIR/final.pt" "$PERSIST_DIR/" 2>/dev/null || echo "WARN: no final.pt"
cp -v "$OUTPUT_DIR/checkpoint.pt" "$PERSIST_DIR/" 2>/dev/null || true
cp -v "$OUTPUT_DIR/train_log.tsv" "$PERSIST_DIR/" 2>/dev/null || true
cp -v "$OUTPUT_DIR/train.log" "$PERSIST_DIR/" 2>/dev/null || true

echo "=== Done ===" >&2
