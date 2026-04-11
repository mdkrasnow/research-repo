#!/bin/bash
#SBATCH --job-name=eqm-1k-80ep-dganm
#SBATCH --output=slurm/logs/eqm-1k-80ep-dganm_%A.out
#SBATCH --error=slurm/logs/eqm-1k-80ep-dganm_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# Full 80-epoch DG-ANM EqM-B/2 on ImageNet-1K 256x256 (4x GPU DDP)
# DG-ANM params from autoresearch best: gamma=4.0, epsilon=0.3, steps=3
# ~94h total at 1.18 steps/sec. 72h limit means we need checkpoint resumption.
# Checkpoints sync to persistent storage every 30min + on SIGTERM.

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }
echo "=== Visible GPUs: ${CUDA_VISIBLE_DEVICES:-all} ===" >&2

WORK_DIR="/tmp/eqm-1k-80ep-dganm-${SLURM_JOB_ID}"
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
IMAGENET_PATH="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"
RESULTS_DIR="projects/diff-EqM/results/imagenet1k_80ep_dganm"
REPO_ROOT="/n/home03/mkrasnow/research-repo"

if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet-1K data not found at $IMAGENET_PATH" >&2
    exit 1
fi
FOUND_CLASSES=$(ls -d "$IMAGENET_PATH"/*/ 2>/dev/null | wc -l)
echo "=== Found $FOUND_CLASSES class directories in $IMAGENET_PATH ===" >&2

PERSISTENT_RESULTS="$REPO_ROOT/projects/diff-EqM/results/imagenet1k_80ep_dganm"
mkdir -p "$WORK_DIR/$RESULTS_DIR"
mkdir -p "$PERSISTENT_RESULTS"

# --- Periodic checkpoint sync to persistent storage ---
sync_checkpoints() {
    if [ -d "$WORK_DIR/$RESULTS_DIR" ]; then
        rsync -a --include='*/' --include='*.pt' --exclude='*' \
            "$WORK_DIR/$RESULTS_DIR/" "$PERSISTENT_RESULTS/" 2>/dev/null || true
        echo "=== Synced checkpoints to persistent storage at $(date) ===" >&2
    fi
}

# Background sync loop: every 30 minutes
(
    while true; do
        sleep 1800
        sync_checkpoints
    done
) &
SYNC_PID=$!

# Trap SIGTERM (sent by SLURM before kill) to save checkpoints
CLEANED_UP=0
cleanup() {
    [ $CLEANED_UP -eq 1 ] && return
    CLEANED_UP=1
    echo "=== Caught signal, syncing checkpoints before exit ===" >&2
    sync_checkpoints
    cp "$WORK_DIR/train.log" "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet1k_80ep_dganm_${SLURM_JOB_ID}.log" 2>/dev/null || true
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
    --ckpt-every 25000 \
    --use-mining \
    --gamma 4.0 \
    --mining-epsilon 0.3 \
    --mining-steps 3 \
    --neg-margin 5.0 \
    --mine-every 5"

# Resume from checkpoint if provided
if [ -n "${RESUME_CKPT:-}" ] && [ -f "$RESUME_CKPT" ]; then
    echo "=== Resuming from checkpoint: $RESUME_CKPT ===" >&2
    LOCAL_CKPT="$WORK_DIR/resume_ckpt.pt"
    cp "$RESUME_CKPT" "$LOCAL_CKPT"
    TRAIN_CMD="$TRAIN_CMD --ckpt $LOCAL_CKPT"
fi

echo "=== Mining ENABLED (gamma=4.0, epsilon=0.3, steps=3, margin=5.0, mine-every=5) ===" >&2
echo "=== Starting 80-epoch training on ImageNet-1K ===" >&2
echo "=== Command: $TRAIN_CMD ===" >&2
echo "=== Start time: $(date) ===" >&2
echo "=== Job ID: $SLURM_JOB_ID ===" >&2
echo "=== Node: $(hostname) ===" >&2

set +e
eval $TRAIN_CMD 2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
echo "=== Training exit code: $TRAIN_EXIT ===" >&2
echo "=== Training end time: $(date) ===" >&2

# Final sync (full results, not just checkpoints)
echo "=== Final sync to persistent storage ===" >&2
rsync -av "$WORK_DIR/$RESULTS_DIR/" "$PERSISTENT_RESULTS/" 2>&1 | tail -20
cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet1k_80ep_dganm_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo "=== Cleaning up work directory ===" >&2
rm -rf "$WORK_DIR"

echo "=== Done (exit=$TRAIN_EXIT) ===" >&2
exit $TRAIN_EXIT
