#!/bin/bash
#SBATCH --job-name=stageb-in1k-v02-smoke-ext
#SBATCH --output=slurm/logs/stageb-in1k-v02-smoke-ext_%j.out
#SBATCH --error=slurm/logs/stageb-in1k-v02-smoke-ext_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=10:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Extended v02 IN-1K smoke: 5 epochs (~25K steps) to confirm cosine-neg
# loss drops below 1.0 (CIFAR drop point ~step 22K). If neg drops, full
# 80ep run is green-lit; if still saturated, mining is broken at IN-1K
# scale and full runs should be killed.

set -uo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/stageb-in1k-v02-smoke-ext-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -3 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -3 || {
    git fetch --unshallow origin 2>&1 | tail -3
    git checkout "${GIT_SHA}" 2>&1 | tail -3 || exit 1
}

pip install --quiet timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -3

REPO_ROOT="/n/home03/mkrasnow/research-repo"
IN1K_TRAIN="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"
RESULTS_SUBDIR="projects/diff-EqM/results/stage_b_in1k_v02_smoke_ext_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR/$RESULTS_SUBDIR"

MASTER_PORT=$((29500 + RANDOM % 1000))

TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path $IN1K_TRAIN \
    --num-classes 1000 \
    --global-batch-size 256 \
    --epochs 5 \
    --image-size 256 \
    --global-seed 0 \
    --log-every 100 \
    --use-mining \
    --mining-flavor v02 \
    --gamma 1.0 \
    --mining-epsilon 0.3 \
    --mining-steps 3 \
    --mining-lr 0.05 \
    --mine-every 5 \
    --lambda-pos 1.0 \
    --lambda-neg 1.0 \
    --pos-sigma 0.05 \
    --neg-cos-margin 0.0 \
    --results-dir $RESULTS_SUBDIR"

echo "=== Extended v02 smoke (5 epochs, ~25K steps) ===" >&2
echo "=== Start: $(date) ===" >&2
eval $TRAIN_CMD 2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit: $TRAIN_EXIT (end $(date)) ===" >&2

PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/stage_b_in1k_v02_smoke_ext_${SLURM_JOB_ID}"
mkdir -p "$PERSIST_DIR"
cp train.log "$PERSIST_DIR/" 2>/dev/null || true
rsync -a --include='*/' --include='*.log' --include='*.txt' --exclude='*' \
    "$WORK_DIR/$RESULTS_SUBDIR/" "$PERSIST_DIR/" 2>&1 | tail -5 || true

echo "=== neg trajectory (every 1000 steps) ===" >&2
grep -E "Steps/Sec" train.log | awk 'NR%10==1' | tail -30 >&2 || true

exit $TRAIN_EXIT
