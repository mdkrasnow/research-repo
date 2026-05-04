#!/bin/bash
#SBATCH --job-name=stageb-in1k-v02-smoke
#SBATCH --output=slurm/logs/stageb-in1k-v02-smoke_%j.out
#SBATCH --error=slurm/logs/stageb-in1k-v02-smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=8:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Stage B IN-1K v02 smoke: 2 epochs, EqM-B/2 + v02 cosine-contrastive,
# full ImageNet-1K-256, 4-GPU DDP. Verifies the v02 code path runs at
# scale, exposes any DDP / latent-space-epsilon issues, and produces
# pos/neg diagnostic values to guide hyperparameter calibration.
# Defaults: mining_epsilon=0.1 (latent), pos_sigma=0.05, mine_every=5.

set -uo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/stageb-in1k-v02-smoke-${SLURM_JOB_ID}"
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
RESULTS_SUBDIR="projects/diff-EqM/results/stage_b_in1k_v02_smoke_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR/$RESULTS_SUBDIR"

if [ ! -d "$IN1K_TRAIN" ]; then
    echo "ERROR: IN-1K data not found at $IN1K_TRAIN" >&2
    exit 1
fi

MASTER_PORT=$((29500 + RANDOM % 1000))

# v02 EqM-B/2: 2 epochs, bs=256, mining_flavor=v02 with CIFAR-tuned defaults.
# Latent-space hyperparameters (epsilon=0.1, pos_sigma=0.05) are educated
# guesses; smoke will print pos/neg diag values so we can calibrate.
TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path $IN1K_TRAIN \
    --num-classes 1000 \
    --global-batch-size 256 \
    --epochs 2 \
    --image-size 256 \
    --global-seed 0 \
    --log-every 50 \
    --use-mining \
    --mining-flavor v02 \
    --gamma 1.0 \
    --mining-epsilon 0.1 \
    --mining-steps 3 \
    --mining-lr 0.01 \
    --mine-every 5 \
    --lambda-pos 1.0 \
    --lambda-neg 1.0 \
    --pos-sigma 0.05 \
    --neg-cos-margin 0.0 \
    --results-dir $RESULTS_SUBDIR"

echo "=== v02 smoke command: $TRAIN_CMD ===" >&2
echo "=== Start: $(date) ===" >&2
eval $TRAIN_CMD 2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit: $TRAIN_EXIT (end $(date)) ===" >&2

PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/stage_b_in1k_v02_smoke_${SLURM_JOB_ID}"
mkdir -p "$PERSIST_DIR"
cp train.log "$PERSIST_DIR/" 2>/dev/null || true
rsync -a --include='*/' --include='*.log' --include='*.txt' --exclude='*' \
    "$WORK_DIR/$RESULTS_SUBDIR/" "$PERSIST_DIR/" 2>&1 | tail -5 || true

echo "=== v02 diag summary (last 10 logged steps) ===" >&2
grep -E "Steps/Sec|v02\(pos" train.log | tail -10 >&2 || echo "(no diag lines)"

echo "=== Done v02 smoke ===" >&2
exit $TRAIN_EXIT
