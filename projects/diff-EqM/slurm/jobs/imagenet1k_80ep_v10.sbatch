#!/bin/bash
#SBATCH --job-name=eqm-1k-80ep-v10
#SBATCH --output=slurm/logs/eqm-1k-80ep-v10_%A.out
#SBATCH --error=slurm/logs/eqm-1k-80ep-v10_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# v10-only 80-epoch EqM-B/2 on ImageNet-1K 256x256, 4x GPU DDP.
# v10 = PGD hard-example mining on EqM regression target. K=1 FGSM,
# eps_radius=0.3, mining_lr=0.05, lambda=0.1 (per CIFAR Phase 0.3 PASS:
# FID 13.40 vs vanilla 14.17, mining ratio stable 1.047-1.049 across 150ep).
#
# Cost estimate: vanilla 80ep ~24h on 4xA100; v10 K=1 adds ~50% per step
# (one extra forward+backward) -> ~36h. 72h gpu limit accommodates.
#
# Per CLAUDE.md "Smoke-time sample probe" rule: --ckpt-every 5000 saves
# early checkpoints; sample FID one early (~5K step ckpt) before letting
# the full 80ep run complete to catch v10 mechanism failures fast.

set -uo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/eqm-1k-80ep-v10-${SLURM_JOB_ID}"
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
    git fetch --unshallow origin 2>&1 | tail -5
    git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1
}

echo "=== Installing dependencies ===" >&2
pip install timm diffusers accelerate scipy torchvision 2>&1 | tail -10

IMAGENET_PATH="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"
SEED="${SEED:-0}"
RESULTS_DIR="projects/diff-EqM/results/imagenet1k_80ep_v10_seed${SEED}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"

if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet-1K data not found at $IMAGENET_PATH" >&2
    exit 1
fi

PERSISTENT_RESULTS="$REPO_ROOT/$RESULTS_DIR"
mkdir -p "$WORK_DIR/$RESULTS_DIR"
mkdir -p "$PERSISTENT_RESULTS"

sync_checkpoints() {
    if [ -d "$WORK_DIR/$RESULTS_DIR" ]; then
        rsync -a --include='*/' --include='*.pt' --exclude='*' \
            "$WORK_DIR/$RESULTS_DIR/" "$PERSISTENT_RESULTS/" 2>/dev/null || true
        echo "=== Synced checkpoints to persistent storage at $(date) ===" >&2
    fi
}
(
    while true; do
        sleep 1800
        sync_checkpoints
    done
) &
SYNC_PID=$!

CLEANED_UP=0
cleanup() {
    [ $CLEANED_UP -eq 1 ] && return
    CLEANED_UP=1
    echo "=== Caught signal, syncing checkpoints before exit ===" >&2
    sync_checkpoints
    cp "$WORK_DIR/train.log" "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet1k_80ep_v10_${SLURM_JOB_ID}.log" 2>/dev/null || true
    kill $SYNC_PID 2>/dev/null || true
    echo "=== Cleanup done at $(date) ===" >&2
}
trap cleanup SIGTERM SIGINT EXIT

MASTER_PORT=$((29500 + RANDOM % 1000))

TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path $IMAGENET_PATH \
    --num-classes 1000 \
    --global-batch-size 256 \
    --epochs 80 \
    --image-size 256 \
    --results-dir $RESULTS_DIR \
    --ckpt-every 5000 \
    --global-seed $SEED \
    --use-mining \
    --mining-flavor v10 \
    --gamma 0.1 \
    --v10-K 1 \
    --v10-eps-radius 0.3 \
    --mining-lr 0.05 \
    --mine-every 1"

if [ -n "${RESUME_CKPT:-}" ] && [ -f "$RESUME_CKPT" ]; then
    echo "=== Resuming from checkpoint: $RESUME_CKPT ===" >&2
    LOCAL_CKPT="$WORK_DIR/resume_ckpt.pt"
    cp "$RESUME_CKPT" "$LOCAL_CKPT"
    TRAIN_CMD="$TRAIN_CMD --ckpt $LOCAL_CKPT"
fi

echo "=== v10 mining ENABLED (gamma=0.1, eps=0.3, K=1, lr=0.05, mine-every=1) ===" >&2
echo "=== Starting v10 80-epoch training on ImageNet-1K, seed=$SEED ===" >&2
echo "=== Job ID: $SLURM_JOB_ID, Node: $(hostname), Start: $(date) ===" >&2

set +e
eval $TRAIN_CMD 2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
echo "=== Training exit code: $TRAIN_EXIT, end: $(date) ===" >&2

echo "=== Final sync to persistent storage ===" >&2
rsync -av "$WORK_DIR/$RESULTS_DIR/" "$PERSISTENT_RESULTS/" 2>&1 | tail -20
cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet1k_80ep_v10_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo "=== Cleaning up work directory ===" >&2
rm -rf "$WORK_DIR"

exit $TRAIN_EXIT
