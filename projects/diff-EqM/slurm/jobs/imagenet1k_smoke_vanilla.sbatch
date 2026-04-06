#!/bin/bash
#SBATCH --job-name=eqm-1k-smoke-van
#SBATCH --output=slurm/logs/eqm-1k-smoke-van_%A.out
#SBATCH --error=slurm/logs/eqm-1k-smoke-van_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Smoke test: 1-epoch vanilla EqM-B/2 on full ImageNet-1K 256x256 (single GPU)
# Validates training pipeline with 1000 classes before full 80-epoch run

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/eqm-1k-smoke-van-${SLURM_JOB_ID}"
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
pip install timm diffusers accelerate scipy torchvision 2>&1 | tail -10

# ImageNet-1K (pre-resized 256x256)
IMAGENET_PATH="/n/netscratch/kempner_alvarez_lab/datasets/imagenet1k-256/train"
RESULTS_DIR="projects/diff-EqM/results/imagenet1k_smoke_vanilla"
REPO_ROOT="/n/home03/mkrasnow/research-repo"

if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet-1K data not found at $IMAGENET_PATH" >&2
    exit 1
fi
FOUND_CLASSES=$(ls -d "$IMAGENET_PATH"/*/ 2>/dev/null | wc -l)
echo "=== Found $FOUND_CLASSES class directories in $IMAGENET_PATH ===" >&2

mkdir -p "$WORK_DIR/$RESULTS_DIR"

MASTER_PORT=$((29500 + RANDOM % 1000))

TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path $IMAGENET_PATH \
    --num-classes 1000 \
    --global-batch-size 16 \
    --epochs 1 \
    --image-size 256 \
    --ckpt-every 999999 \
    --results-dir $RESULTS_DIR"

echo "=== Mining DISABLED (vanilla EqM) ===" >&2
echo "=== Starting 1-epoch smoke test on ImageNet-1K ===" >&2
echo "=== Command: $TRAIN_CMD ===" >&2
echo "=== Start time: $(date) ===" >&2
echo "=== Job ID: $SLURM_JOB_ID ===" >&2
echo "=== Node: $(hostname) ===" >&2

eval $TRAIN_CMD 2>&1 | tee train.log

TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit code: $TRAIN_EXIT ===" >&2
echo "=== Training end time: $(date) ===" >&2

if [ $TRAIN_EXIT -eq 0 ]; then
    CKPT_COUNT=$(find "$WORK_DIR/$RESULTS_DIR" -name "*.pt" -type f 2>/dev/null | wc -l)
    echo "=== Checkpoints found: $CKPT_COUNT ===" >&2
    if [ "$CKPT_COUNT" -eq 0 ]; then
        echo "WARNING: Training succeeded but no .pt checkpoints found" >&2
        ls -laR "$WORK_DIR/$RESULTS_DIR/" >&2
    else
        find "$WORK_DIR/$RESULTS_DIR" -name "*.pt" -exec ls -lh {} \; >&2
    fi
fi

echo "=== Syncing results ===" >&2
mkdir -p "$REPO_ROOT/projects/diff-EqM/results/"
rsync -av "$WORK_DIR/$RESULTS_DIR/" "$REPO_ROOT/projects/diff-EqM/results/imagenet1k_smoke_vanilla/" 2>&1 | tail -10
cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet1k_smoke_vanilla_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo "=== Cleaning up work directory ===" >&2
rm -rf "$WORK_DIR"

echo "=== Smoke test done (exit=$TRAIN_EXIT) ===" >&2
exit $TRAIN_EXIT
