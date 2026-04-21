#!/bin/bash
#SBATCH --job-name=cifar-dganm-ext
#SBATCH --output=slurm/logs/cifar-dganm-ext_%A_%a.out
#SBATCH --error=slurm/logs/cifar-dganm-ext_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3

# DG-ANM CIFAR extension run — 3 seeds to 150 epochs + 10K final FID for
# apples-to-apples with the vanilla seed study.
#
# Previous run (6821254_3,4,5) hit walltime at e108 before final rsync, so
# checkpoints were LOST. This sbatch fixes:
#   - 8h walltime (was 6h — DG-ANM @ 188s/epoch needs ~7.8h for 150ep + eval)
#   - Background rsync every 10 min — checkpoints reach persistent storage
#     even if training times out
#   - From scratch (no resume; no e108 checkpoint to resume from)

set -u  # intentional: NO -e, so background rsync can run even if training fails

SEED="${SLURM_ARRAY_TASK_ID:-0}"
echo "=== DG-ANM extend seed=$SEED ===" >&2

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/cifar-dganm-ext-${SLURM_JOB_ID}-${SEED}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0

echo "=== Clone at SHA $GIT_SHA ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5
git checkout "$GIT_SHA" 2>&1 | tail -3 || {
    git fetch --unshallow origin 2>&1 | tail -3
    git checkout "$GIT_SHA" 2>&1 | tail -3
}

pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -3

REPO_ROOT="/n/home03/mkrasnow/research-repo"
REF_STATS="$REPO_ROOT/projects/diff-EqM/results/cifar10_inception_stats.npz"
OUTPUT_DIR="$WORK_DIR/projects/diff-EqM/results/cifar_dganm_ext_${SLURM_JOB_ID}_seed${SEED}"
PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/cifar_dganm_ext_${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "$OUTPUT_DIR" "$PERSIST_DIR"
if [ -f "$REF_STATS" ]; then
    mkdir -p "$WORK_DIR/projects/diff-EqM/results"
    cp "$REF_STATS" "$WORK_DIR/projects/diff-EqM/results/cifar10_inception_stats.npz"
fi

# Launch background rsync loop: every 10 min, sync OUTPUT_DIR -> PERSIST_DIR.
# This ensures checkpoints land even if training times out.
rsync_loop() {
    while true; do
        sleep 600
        rsync -a --quiet "$OUTPUT_DIR/" "$PERSIST_DIR/" 2>/dev/null
        echo "[rsync-loop] sync $(date +%H:%M:%S)" >&2
    done
}
rsync_loop &
RSYNC_PID=$!
echo "=== Background rsync loop PID $RSYNC_PID ===" >&2

trap "echo 'Trap: final rsync'; kill $RSYNC_PID 2>/dev/null; rsync -a \"$OUTPUT_DIR/\" \"$PERSIST_DIR/\" 2>&1 | tail -5" EXIT

echo "=== Training DG-ANM EqM CIFAR (seed=$SEED, margin=50) ===" >&2
python projects/diff-EqM/experiments/train_cifar_dganm_unet.py \
    --output-dir "$OUTPUT_DIR" \
    --data-dir ./data \
    --epochs 150 \
    --batch-size 128 \
    --lr 2e-4 \
    --seed "$SEED" \
    --eval-every-epochs 25 \
    --eval-fid-samples 2000 \
    --final-fid-samples 10000 \
    --euler-num-steps 100 \
    --gamma 6.0 \
    --mining-epsilon 0.8 \
    --mining-steps 3 \
    --mine-every 5 \
    --neg-margin 50.0 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit: $TRAIN_EXIT ===" >&2

# Final explicit rsync (trap will also fire on EXIT)
kill "$RSYNC_PID" 2>/dev/null
rsync -a "$OUTPUT_DIR/" "$PERSIST_DIR/" 2>&1 | tail -5

echo "=== Done seed=$SEED exit=$TRAIN_EXIT ===" >&2
