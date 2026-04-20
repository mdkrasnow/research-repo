#!/bin/bash
#SBATCH --job-name=cifar-seed-study
#SBATCH --output=slurm/logs/cifar-seed-study_%A_%a.out
#SBATCH --error=slurm/logs/cifar-seed-study_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%6

# Fast 3+3 CIFAR seed study to discriminate method signal from seed noise.
#
# Array index -> variant/seed:
#   0,1,2 : vanilla EqM (FM UNet), seed={0,1,2}
#   3,4,5 : DG-ANM EqM (FM UNet), same seeds, with neg_margin=50 (our diag
#           showed ||field|| ~47 on this UNet so margin=5 is saturated; 50
#           produces real gradient signal).
#
# Both groups: 150 epochs, bs=128, lr=2e-4 (matches Step B smoke test that
# hit FID 9.72). Emits:
#   vanilla: "cifar10_eqm_fid: <value>"
#   dganm:   "cifar10_dganm_fid: <value>"

set -euo pipefail

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -lt 3 ]; then
    VARIANT="vanilla"
    SEED="$IDX"
else
    VARIANT="dganm"
    SEED=$((IDX - 3))
fi
echo "=== variant=$VARIANT seed=$SEED (array_idx=$IDX) ===" >&2

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/cifar-seed-study-${SLURM_JOB_ID}-${IDX}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0

echo "=== Clone at SHA $GIT_SHA ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5 || exit 1
git checkout "$GIT_SHA" 2>&1 | tail -3 || {
    git fetch --unshallow origin 2>&1 | tail -3
    git checkout "$GIT_SHA" 2>&1 | tail -3 || exit 1
}

pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -3

REPO_ROOT="/n/home03/mkrasnow/research-repo"
REF_STATS="$REPO_ROOT/projects/diff-EqM/results/cifar10_inception_stats.npz"
OUTPUT_DIR="$WORK_DIR/projects/diff-EqM/results/cifar_seed_study_${SLURM_JOB_ID}_${VARIANT}_seed${SEED}"
mkdir -p "$OUTPUT_DIR"
if [ -f "$REF_STATS" ]; then
    mkdir -p "$WORK_DIR/projects/diff-EqM/results"
    cp "$REF_STATS" "$WORK_DIR/projects/diff-EqM/results/cifar10_inception_stats.npz"
fi

# Early-persist directory (per our Stage B lesson: never lose trained models)
PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/cifar_seed_study_${SLURM_JOB_ID}_${VARIANT}_seed${SEED}"
mkdir -p "$PERSIST_DIR"

COMMON_ARGS="--output-dir $OUTPUT_DIR \
    --data-dir ./data \
    --epochs 150 \
    --batch-size 128 \
    --lr 2e-4 \
    --seed $SEED \
    --eval-every-epochs 50 \
    --eval-fid-samples 2000 \
    --final-fid-samples 10000 \
    --euler-num-steps 100"

if [ "$VARIANT" = "vanilla" ]; then
    echo "=== Training VANILLA EqM CIFAR (seed=$SEED) ===" >&2
    python projects/diff-EqM/experiments/train_cifar_eqm_unet.py \
        $COMMON_ARGS --sampler euler \
        2>&1 | tee "$OUTPUT_DIR/train.log"
else
    echo "=== Training DG-ANM EqM CIFAR (seed=$SEED, margin=50) ===" >&2
    # margin=50 based on diagnostic: ||field|| on this UNet is ~47-49, so
    # margin=50 ensures ReLU bites on ~100% of samples -> real gradient.
    python projects/diff-EqM/experiments/train_cifar_dganm_unet.py \
        $COMMON_ARGS \
        --gamma 6.0 \
        --mining-epsilon 0.8 \
        --mining-steps 3 \
        --mine-every 5 \
        --neg-margin 50.0 \
        2>&1 | tee "$OUTPUT_DIR/train.log"
fi

# Persist (early, before anything can fail)
rsync -av "$OUTPUT_DIR/" "$PERSIST_DIR/" 2>&1 | tail -5 || echo "WARN rsync failed"

echo "=== Done variant=$VARIANT seed=$SEED ===" >&2
