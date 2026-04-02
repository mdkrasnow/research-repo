#!/bin/bash
#SBATCH --job-name=eqm-imagenet
#SBATCH --output=slurm/logs/eqm-imagenet_%A.out
#SBATCH --error=slurm/logs/eqm-imagenet_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# EqM-B/2 ImageNet 256x256 training (80 epochs, 4x A100 DDP)
# Expected: ~36-48h for 80 epochs with global batch size 256

set -euo pipefail

# Load required modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPUs
echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }
echo "=== Visible GPUs: ${CUDA_VISIBLE_DEVICES:-all} ===" >&2

# Set working directory
WORK_DIR="/tmp/eqm-imagenet-${SLURM_JOB_ID}"
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
IMAGENET_PATH="${IMAGENET_PATH:-/n/home03/mkrasnow/imagenet100/train}"
USE_MINING="${USE_MINING:-false}"
GAMMA="${GAMMA:-0.1}"
RESULTS_DIR="${RESULTS_DIR:-projects/diff-EqM/results/imagenet}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"

# Verify ImageNet data exists (flat structure: class dirs directly in IMAGENET_PATH)
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet data not found at $IMAGENET_PATH" >&2
    exit 1
fi
NUM_CLASSES=$(ls -d "$IMAGENET_PATH"/n*/ 2>/dev/null | wc -l)
echo "=== Found $NUM_CLASSES class directories in $IMAGENET_PATH ===" >&2

# Create results directory
mkdir -p "$WORK_DIR/$RESULTS_DIR"

# Random port to avoid collisions
MASTER_PORT=$((29500 + RANDOM % 1000))

# Build training command
TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path $IMAGENET_PATH \
    --num-classes ${NUM_CLASSES:-100} \
    --global-batch-size 256 \
    --epochs 80 \
    --image-size 256 \
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

echo "=== Starting ImageNet training ===" >&2
echo "=== Command: $TRAIN_CMD ===" >&2
echo "=== Start time: $(date) ===" >&2
echo "=== Job ID: $SLURM_JOB_ID ===" >&2
echo "=== Node: $(hostname) ===" >&2

eval $TRAIN_CMD 2>&1 | tee train.log

TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit code: $TRAIN_EXIT ===" >&2
echo "=== Training end time: $(date) ===" >&2

# Sync results back to persistent storage regardless of exit code
echo "=== Syncing results to persistent storage ===" >&2
mkdir -p "$REPO_ROOT/projects/diff-EqM/results/"
rsync -av "$WORK_DIR/$RESULTS_DIR/" "$REPO_ROOT/projects/diff-EqM/results/imagenet/" 2>&1 | tail -20
cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet_train_${SLURM_JOB_ID}.log" 2>/dev/null || true

# Cleanup
echo "=== Cleaning up work directory ===" >&2
rm -rf "$WORK_DIR"

echo "=== Done (exit=$TRAIN_EXIT) ===" >&2
exit $TRAIN_EXIT
