#!/bin/bash
#SBATCH --job-name=eqm-1k-v10-scale
#SBATCH --output=slurm/logs/eqm-1k-v10-scale_%A.out
#SBATCH --error=slurm/logs/eqm-1k-v10-scale_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --gres=gpu:4

# v10 mining EqM 80ep IN-1K scaling. Parameterized.
# Required: MODEL (EqM-S/2 | EqM-B/2 | EqM-L/2 | EqM-XL/2), SEED
# Optional: EPOCHS (default 80), GLOBAL_BATCH_SIZE (256), GAMMA (0.1),
#           V10_K (1), V10_EPS_RADIUS (0.3), MINING_LR (0.05),
#           RESULTS_TAG (auto from model), RESUME_CKPT
#
# Smoke usage: EPOCHS=1 GLOBAL_BATCH_SIZE=64 --time=01:00:00

set -uo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

MODEL="${MODEL:?ERROR: MODEL env var must be set}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-80}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
GAMMA="${GAMMA:-0.1}"
V10_K="${V10_K:-1}"
V10_EPS_RADIUS="${V10_EPS_RADIUS:-0.3}"
MINING_LR="${MINING_LR:-0.05}"
MODEL_TAG=$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
RESULTS_TAG="${RESULTS_TAG:-v10_${MODEL_TAG}_seed${SEED}}"

WORK_DIR="/tmp/eqm-1k-v10-${RESULTS_TAG}-${SLURM_JOB_ID}"
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
RESULTS_DIR="projects/diff-EqM/results/imagenet1k_${EPOCHS}ep_${RESULTS_TAG}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
PERSISTENT_RESULTS="$REPO_ROOT/$RESULTS_DIR"

if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet-1K data not found at $IMAGENET_PATH" >&2
    exit 1
fi

mkdir -p "$WORK_DIR/$RESULTS_DIR"
mkdir -p "$PERSISTENT_RESULTS"

sync_checkpoints() {
    if [ -d "$WORK_DIR/$RESULTS_DIR" ]; then
        rsync -a --include='*/' --include='*.pt' --exclude='*' \
            "$WORK_DIR/$RESULTS_DIR/" "$PERSISTENT_RESULTS/" 2>/dev/null || true
        echo "=== Synced ckpts at $(date) ===" >&2
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
    sync_checkpoints
    cp "$WORK_DIR/train.log" "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet1k_${EPOCHS}ep_${RESULTS_TAG}_${SLURM_JOB_ID}.log" 2>/dev/null || true
    kill $SYNC_PID 2>/dev/null || true
}
trap cleanup SIGTERM SIGINT EXIT

MASTER_PORT=$((29500 + RANDOM % 1000))

TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model $MODEL \
    --data-path $IMAGENET_PATH \
    --num-classes 1000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --epochs $EPOCHS \
    --image-size 256 \
    --results-dir $RESULTS_DIR \
    --ckpt-every 5000 \
    --global-seed $SEED \
    --use-mining \
    --mining-flavor v10 \
    --gamma $GAMMA \
    --v10-K $V10_K \
    --v10-eps-radius $V10_EPS_RADIUS \
    --mining-lr $MINING_LR \
    --mine-every 1"

if [ -n "${RESUME_CKPT:-}" ] && [ -f "$RESUME_CKPT" ]; then
    echo "=== Resuming from $RESUME_CKPT ===" >&2
    LOCAL_CKPT="$WORK_DIR/resume_ckpt.pt"
    cp "$RESUME_CKPT" "$LOCAL_CKPT"
    TRAIN_CMD="$TRAIN_CMD --ckpt $LOCAL_CKPT"
fi

echo "=== v10 scaling: MODEL=$MODEL SEED=$SEED EPOCHS=$EPOCHS GBS=$GLOBAL_BATCH_SIZE ===" >&2
echo "=== mining: gamma=$GAMMA K=$V10_K eps=$V10_EPS_RADIUS lr=$MINING_LR ===" >&2
echo "=== Job: $SLURM_JOB_ID Node: $(hostname) Start: $(date) ===" >&2

set +e
eval $TRAIN_CMD 2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
echo "=== Training exit: $TRAIN_EXIT end: $(date) ===" >&2

rsync -av "$WORK_DIR/$RESULTS_DIR/" "$PERSISTENT_RESULTS/" 2>&1 | tail -20
cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet1k_${EPOCHS}ep_${RESULTS_TAG}_${SLURM_JOB_ID}.log" 2>/dev/null || true

rm -rf "$WORK_DIR"
exit $TRAIN_EXIT
