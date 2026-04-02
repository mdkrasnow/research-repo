#!/bin/bash
#SBATCH --job-name=eqm-imgnet-smoke
#SBATCH --output=slurm/logs/eqm-imgnet-smoke_%A.out
#SBATCH --error=slurm/logs/eqm-imgnet-smoke_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Smoke test: 1-epoch EqM-B/2 on ImageNet 256x256 (single A100)
# Validates training script, data loading, and checkpoint saving before full run
# Expected: ~2-3h for 1 epoch with batch size 64 on single GPU

set -euo pipefail

# Load required modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU
echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

# Set working directory
WORK_DIR="/tmp/eqm-imgnet-smoke-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository at specific git SHA
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

# Install dependencies
echo "=== Installing dependencies ===" >&2
pip install timm diffusers accelerate scipy torchvision 2>&1 | tail -10

# Configuration
IMAGENET_PATH="/n/netscratch/ydu_lab/Lab/jslee/imagenet"
USE_MINING="${USE_MINING:-false}"
GAMMA="${GAMMA:-0.1}"
RESULTS_DIR="projects/diff-EqM/results/imagenet_smoke_test"
REPO_ROOT="/n/home03/mkrasnow/research-repo"

# Verify ImageNet data exists (flat structure: class dirs directly in IMAGENET_PATH)
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet data not found at $IMAGENET_PATH" >&2
    exit 1
fi

mkdir -p "$WORK_DIR/$RESULTS_DIR"

# Build training command (single GPU, torchrun needed for DDP init)
TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path $IMAGENET_PATH \
    --global-batch-size 64 \
    --epochs 1 \
    --image-size 256 \
    --ckpt-every 999999 \
    --results-dir $RESULTS_DIR"

# Add mining arguments if enabled
if [ "$USE_MINING" = "true" ]; then
    echo "=== Mining ENABLED (gamma=$GAMMA) ===" >&2
    TRAIN_CMD="$TRAIN_CMD \
        --use-mining \
        --gamma $GAMMA \
        --mining-epsilon 0.3 \
        --mining-steps 3 \
        --neg-margin 5.0 \
        --mine-every 5"
else
    echo "=== Mining DISABLED (vanilla EqM) ===" >&2
fi

echo "=== Starting 1-epoch smoke test ===" >&2
echo "=== Command: $TRAIN_CMD ===" >&2
echo "=== Start time: $(date) ===" >&2
echo "=== Job ID: $SLURM_JOB_ID ===" >&2
echo "=== Node: $(hostname) ===" >&2

eval $TRAIN_CMD 2>&1 | tee train.log

TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit code: $TRAIN_EXIT ===" >&2
echo "=== Training end time: $(date) ===" >&2

# Verify checkpoint was saved
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

# Sync results back to persistent storage
echo "=== Syncing results ===" >&2
mkdir -p "$REPO_ROOT/projects/diff-EqM/results/"
rsync -av "$WORK_DIR/$RESULTS_DIR/" "$REPO_ROOT/projects/diff-EqM/results/imagenet_smoke_test/" 2>&1 | tail -10
cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet_smoke_${SLURM_JOB_ID}.log" 2>/dev/null || true

# Cleanup
echo "=== Cleaning up work directory ===" >&2
rm -rf "$WORK_DIR"

echo "=== Smoke test done (exit=$TRAIN_EXIT) ===" >&2
exit $TRAIN_EXIT
