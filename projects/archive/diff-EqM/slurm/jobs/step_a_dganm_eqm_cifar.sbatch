#!/bin/bash
#SBATCH --job-name=stepa-dganm-cifar
#SBATCH --output=slurm/logs/stepa-dganm-cifar_%A.out
#SBATCH --error=slurm/logs/stepa-dganm-cifar_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Stage A.5 Step A — DG-ANM on CIFAR-10 via FM UNet (parallel to vanilla).
#
# Current-best DG-ANM hyperparams from IN-100 autoresearch proxy:
#   gamma=6.0, mining_epsilon=0.8, mining_steps=3, mine_every=5,
#   neg_margin=5.0, lr=2e-4.
# These may not transfer from 2ep IN-100 to 1800ep CIFAR; that's part of
# what this run tells us.
#
# Emits "cifar10_dganm_fid: <value>".

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/stepa-dganm-cifar-${SLURM_JOB_ID}"
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

REPO_ROOT="/n/home03/mkrasnow/research-repo"
REF_STATS="$REPO_ROOT/projects/diff-EqM/results/cifar10_inception_stats.npz"
OUTPUT_DIR="$WORK_DIR/projects/diff-EqM/results/step_a_dganm_cifar_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
if [ -f "$REF_STATS" ]; then
    mkdir -p "$WORK_DIR/projects/diff-EqM/results"
    cp "$REF_STATS" "$WORK_DIR/projects/diff-EqM/results/cifar10_inception_stats.npz"
fi

EPOCHS="${EPOCHS:-1800}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-2e-4}"
SEED="${SEED:-0}"
GAMMA="${GAMMA:-6.0}"
MINING_EPSILON="${MINING_EPSILON:-0.8}"
MINING_STEPS="${MINING_STEPS:-3}"
MINE_EVERY="${MINE_EVERY:-5}"
NEG_MARGIN="${NEG_MARGIN:-5.0}"
EULER_NUM_STEPS="${EULER_NUM_STEPS:-50}"
EVAL_EVERY="${EVAL_EVERY_EPOCHS:-100}"

echo "=== Training DG-ANM EqM on CIFAR-10 via FM UNet ===" >&2
echo "epochs=$EPOCHS bs=$BATCH_SIZE lr=$LR seed=$SEED gamma=$GAMMA " \
     "mining_eps=$MINING_EPSILON steps=$MINING_STEPS mine_every=$MINE_EVERY " \
     "neg_margin=$NEG_MARGIN" >&2

python projects/diff-EqM/experiments/train_cifar_dganm_unet.py \
    --output-dir "$OUTPUT_DIR" \
    --data-dir ./data \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --gamma "$GAMMA" \
    --mining-epsilon "$MINING_EPSILON" \
    --mining-steps "$MINING_STEPS" \
    --mine-every "$MINE_EVERY" \
    --neg-margin "$NEG_MARGIN" \
    --euler-num-steps "$EULER_NUM_STEPS" \
    --eval-every-epochs "$EVAL_EVERY" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/step_a_dganm_cifar_${SLURM_JOB_ID}"
mkdir -p "$PERSIST_DIR"
cp -v "$OUTPUT_DIR/final.pt" "$PERSIST_DIR/" 2>/dev/null || echo "WARN: no final.pt"
cp -v "$OUTPUT_DIR/checkpoint.pt" "$PERSIST_DIR/" 2>/dev/null || true
cp -v "$OUTPUT_DIR/train_log.tsv" "$PERSIST_DIR/" 2>/dev/null || true
cp -v "$OUTPUT_DIR/train.log" "$PERSIST_DIR/" 2>/dev/null || true

echo "=== Done ===" >&2
