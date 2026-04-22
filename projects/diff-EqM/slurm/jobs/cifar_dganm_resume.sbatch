#!/bin/bash
#SBATCH --job-name=cifar-dganm-res
#SBATCH --output=slurm/logs/cifar-dganm-res_%A_%a.out
#SBATCH --error=slurm/logs/cifar-dganm-res_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3

# Resume DG-ANM CIFAR extension from e125 checkpoints persisted by 7024903 run.
# Trains e125 -> e150 (25 epochs, ~1.3h at 186s/epoch) and runs 10K final FID.

set -u

SEED="${SLURM_ARRAY_TASK_ID:-0}"

# Map seed -> existing persisted result dir (has e125 checkpoint.pt)
case "$SEED" in
    0) PRIOR_DIR="cifar_dganm_ext_7266197_seed0" ;;
    1) PRIOR_DIR="cifar_dganm_ext_7284533_seed1" ;;
    2) PRIOR_DIR="cifar_dganm_ext_7024903_seed2" ;;
    *) echo "Bad seed $SEED"; exit 1 ;;
esac
echo "=== DG-ANM resume seed=$SEED from $PRIOR_DIR ===" >&2

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/cifar-dganm-res-${SLURM_JOB_ID}-${SEED}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0

git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -3
git checkout "$GIT_SHA" 2>&1 | tail -3 || {
    git fetch --unshallow origin 2>&1 | tail -3
    git checkout "$GIT_SHA" 2>&1 | tail -3
}

pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -3

REPO_ROOT="/n/home03/mkrasnow/research-repo"
REF_STATS="$REPO_ROOT/projects/diff-EqM/results/cifar10_inception_stats.npz"
PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/$PRIOR_DIR"
OUTPUT_DIR="$WORK_DIR/projects/diff-EqM/results/$PRIOR_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy the e125 checkpoint into work dir so training writes to the same place
rsync -a "$PERSIST_DIR/" "$OUTPUT_DIR/" 2>&1 | tail -3
echo "=== Resume checkpoint: $(ls -lh $OUTPUT_DIR/checkpoint.pt) ===" >&2

if [ -f "$REF_STATS" ]; then
    mkdir -p "$WORK_DIR/projects/diff-EqM/results"
    cp "$REF_STATS" "$WORK_DIR/projects/diff-EqM/results/cifar10_inception_stats.npz"
fi

# Background rsync loop (same pattern as extension run)
rsync_loop() {
    while true; do
        sleep 300
        rsync -a --quiet "$OUTPUT_DIR/" "$PERSIST_DIR/" 2>/dev/null
        echo "[rsync-loop] sync $(date +%H:%M:%S)" >&2
    done
}
rsync_loop &
RSYNC_PID=$!
trap "echo 'Trap: final rsync'; kill $RSYNC_PID 2>/dev/null; rsync -a \"$OUTPUT_DIR/\" \"$PERSIST_DIR/\" 2>&1 | tail -3" EXIT

echo "=== Training DG-ANM resume seed=$SEED e125 -> e150 ===" >&2
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
    --resume "$OUTPUT_DIR/checkpoint.pt" \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit: $TRAIN_EXIT ===" >&2

kill "$RSYNC_PID" 2>/dev/null
rsync -a "$OUTPUT_DIR/" "$PERSIST_DIR/" 2>&1 | tail -3
echo "=== Done seed=$SEED exit=$TRAIN_EXIT ===" >&2
