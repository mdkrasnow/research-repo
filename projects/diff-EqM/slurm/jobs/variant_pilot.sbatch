#!/bin/bash
#SBATCH --job-name=variant-pilot
#SBATCH --output=slurm/logs/variant-pilot_%A_%a.out
#SBATCH --error=slurm/logs/variant-pilot_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Generic pilot runner for DG-ANM variants. One config per array task.
#
# IMPORTANT: mining variants (v01-v04, v06) require >=40G GPU. The
# `gpu_test` partition has 20G cards and will OOM at bs=128 — use
# --partition=gpu for those. Non-mining variants (v00, v05) fit in 20G.
# History: jobs 8898137 (v03) and 8898147 (v06) OOM'd on gpu_test.
# The caller is responsible for:
#   - passing CONFIG_PATH (path under repo, relative) or CONFIGS_DIR + VARIANT_LIST
#   - setting GIT_SHA + GIT_URL for the clone
#   - ensuring the config's variant/output_dir/seed are correct
#
# Expects one of:
#   CONFIG_PATH=projects/diff-EqM/configs/variants/v03_noised_negatives.json
# or:
#   CONFIGS_DIR=projects/diff-EqM/configs/variants VARIANT=v03_noised_negatives
#
# Output: checkpoints + train_log.tsv in $OUTPUT_DIR, plus a grep-friendly
# "cifar10_variant_fid[<variant>]: <fid>" final line.

set -u

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
WORK_DIR="/tmp/variant-pilot-${SLURM_JOB_ID}-${TASK_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0

echo "=== Clone at SHA $GIT_SHA (task=$TASK_ID) ===" >&2
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

# Resolve config path.
if [ -n "${CONFIG_PATH:-}" ]; then
    CFG="$CONFIG_PATH"
elif [ -n "${VARIANT:-}" ]; then
    CFG="${CONFIGS_DIR:-projects/diff-EqM/configs/variants}/${VARIANT}.json"
else
    echo "ERROR: must pass CONFIG_PATH or VARIANT env var"; exit 2
fi
echo "=== Config: $CFG ===" >&2
cat "$CFG" >&2

# Extract variant + seed offset so we can build an output_dir. We allow
# callers to pass SEED env var to override the seed in the config.
VARIANT_NAME=$(python -c "import json,sys; print(json.load(open('$CFG'))['variant'])")
SEED="${SEED:-$TASK_ID}"

OUTPUT_DIR="$WORK_DIR/projects/diff-EqM/results/variant_${VARIANT_NAME}_${SLURM_JOB_ID}_seed${SEED}"
PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/variant_${VARIANT_NAME}_${SLURM_JOB_ID}_seed${SEED}"
mkdir -p "$OUTPUT_DIR" "$PERSIST_DIR"
if [ -f "$REF_STATS" ]; then
    mkdir -p "$WORK_DIR/projects/diff-EqM/results"
    cp "$REF_STATS" "$WORK_DIR/projects/diff-EqM/results/cifar10_inception_stats.npz"
fi

# Background rsync loop — preserve checkpoints if training times out.
rsync_loop() {
    while true; do
        sleep 600
        rsync -a --quiet "$OUTPUT_DIR/" "$PERSIST_DIR/" 2>/dev/null
        echo "[rsync-loop] sync $(date +%H:%M:%S)" >&2
    done
}
rsync_loop &
RSYNC_PID=$!
trap "echo 'Trap: final rsync'; kill $RSYNC_PID 2>/dev/null; rsync -a \"$OUTPUT_DIR/\" \"$PERSIST_DIR/\" 2>&1 | tail -5" EXIT

echo "=== Training variant=$VARIANT_NAME seed=$SEED ===" >&2
python projects/diff-EqM/experiments/run_variant.py \
    --config "$CFG" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}

kill "$RSYNC_PID" 2>/dev/null
rsync -a "$OUTPUT_DIR/" "$PERSIST_DIR/" 2>&1 | tail -5

echo "=== Done variant=$VARIANT_NAME seed=$SEED exit=$TRAIN_EXIT ===" >&2
exit $TRAIN_EXIT
