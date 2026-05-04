#!/bin/bash
#SBATCH --job-name=stageb-in1k-smoke
#SBATCH --output=slurm/logs/stageb-in1k-smoke_%j.out
#SBATCH --error=slurm/logs/stageb-in1k-smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=8:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Stage B IN-1K SMOKE: 3 epochs vanilla EqM-B/2 on full ImageNet-1K-256 to
# measure steps/sec + verify pipeline. NO FID at the end (just throughput).
# Use the steps/sec to extrapolate full 80ep walltime budget before any
# multi-day commit.

set -uo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/stageb-in1k-smoke-${SLURM_JOB_ID}"
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
    git fetch --unshallow origin 2>&1 | tail -3
    git checkout "${GIT_SHA}" 2>&1 | tail -3 || exit 1
}

pip install --quiet timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
IN1K_TRAIN="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"
RESULTS_SUBDIR="projects/diff-EqM/results/stage_b_in1k_smoke_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR/$RESULTS_SUBDIR"

if [ ! -d "$IN1K_TRAIN" ]; then
    echo "ERROR: IN-1K data not found at $IN1K_TRAIN" >&2
    exit 1
fi
FOUND=$(ls -d "$IN1K_TRAIN"/*/ 2>/dev/null | wc -l)
echo "=== Found $FOUND class dirs in $IN1K_TRAIN ===" >&2

MASTER_PORT=$((29500 + RANDOM % 1000))

# Vanilla EqM-B/2, 3 epochs, bs=256 (4 GPU DDP). Same hyperparameters
# as Stage B vanilla; only --epochs and --num-classes differ from
# the IN-100 baseline.
TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path $IN1K_TRAIN \
    --num-classes 1000 \
    --global-batch-size 256 \
    --epochs 3 \
    --image-size 256 \
    --global-seed 0 \
    --log-every 50 \
    --results-dir $RESULTS_SUBDIR"

echo "=== Smoke command: $TRAIN_CMD ===" >&2
echo "=== Start: $(date) ===" >&2
eval $TRAIN_CMD 2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit: $TRAIN_EXIT (end $(date)) ===" >&2

# Persist train.log for throughput analysis.
PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/stage_b_in1k_smoke_${SLURM_JOB_ID}"
mkdir -p "$PERSIST_DIR"
cp train.log "$PERSIST_DIR/" 2>/dev/null || true
rsync -a --include='*/' --include='*.log' --include='*.txt' --exclude='*' \
    "$WORK_DIR/$RESULTS_SUBDIR/" "$PERSIST_DIR/" 2>&1 | tail -5 || true

# Extract steps/sec for grep-friendly summary.
echo "=== Throughput summary ===" >&2
grep -E "Steps/Sec" train.log | tail -10 >&2 || echo "(no throughput lines found)"

echo "=== Done smoke ===" >&2
exit $TRAIN_EXIT
