#!/bin/bash
#SBATCH --job-name=stageb-v02-in1k-80ep
#SBATCH --output=slurm/logs/stageb-v02-in1k-80ep_%j.out
#SBATCH --error=slurm/logs/stageb-v02-in1k-80ep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Stage B v02 EqM-B/2 + cosine-contrastive, 80 epochs, full IN-1K-256.
# Throughput estimate: 3.23 sps (v02 smoke fast node) -> ~33h for 80ep.
# Slow-node ~1.64 sps -> ~67h (will need resume across two jobs).
# CIFAR-validated hyperparameters (12.96±0.70 vs vanilla 15.06±0.90).
# Pass SEED env to override default 0.

set -uo pipefail

SEED="${SEED:-0}"

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

nvidia-smi || exit 1

WORK_DIR="/tmp/stageb-v02-in1k-80ep-${SLURM_JOB_ID}-seed${SEED}"
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
RESULTS_SUBDIR="projects/diff-EqM/results/stage_b_v02_in1k_80ep_seed${SEED}"
PERSIST_DIR="$REPO_ROOT/$RESULTS_SUBDIR"
mkdir -p "$WORK_DIR/$RESULTS_SUBDIR" "$PERSIST_DIR"

MASTER_PORT=$((29500 + RANDOM % 1000))

rsync_loop() {
    while true; do
        sleep 900
        rsync -a --quiet --include='*/' --include='*.pt' --include='*.log' \
            --include='*.txt' --include='*.json' --exclude='*' \
            "$WORK_DIR/$RESULTS_SUBDIR/" "$PERSIST_DIR/" 2>/dev/null
        echo "[rsync] $(date +%H:%M:%S)" >&2
    done
}
rsync_loop &
RSYNC_PID=$!
trap "kill $RSYNC_PID 2>/dev/null; rsync -a --include='*/' --include='*.pt' --include='*.log' --include='*.txt' --include='*.json' --exclude='*' '$WORK_DIR/$RESULTS_SUBDIR/' '$PERSIST_DIR/' 2>&1 | tail -5" EXIT

TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path $IN1K_TRAIN \
    --num-classes 1000 \
    --global-batch-size 256 \
    --epochs 80 \
    --image-size 256 \
    --global-seed $SEED \
    --log-every 200 \
    --ckpt-every 25000 \
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

if [ -n "${RESUME_CKPT:-}" ] && [ -f "$RESUME_CKPT" ]; then
    cp "$RESUME_CKPT" "$WORK_DIR/resume_ckpt.pt"
    TRAIN_CMD="$TRAIN_CMD --ckpt $WORK_DIR/resume_ckpt.pt"
    echo "=== Resuming from $RESUME_CKPT ===" >&2
fi

echo "=== v02 80ep seed=$SEED start: $(date) ===" >&2
eval $TRAIN_CMD 2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit: $TRAIN_EXIT (end $(date)) ===" >&2

kill "$RSYNC_PID" 2>/dev/null
rsync -a --include='*/' --include='*.pt' --include='*.log' --include='*.txt' --include='*.json' --exclude='*' \
    "$WORK_DIR/$RESULTS_SUBDIR/" "$PERSIST_DIR/" 2>&1 | tail -5

exit $TRAIN_EXIT
