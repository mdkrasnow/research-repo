#!/bin/bash
#SBATCH --job-name=stageb-van-in100
#SBATCH --output=slurm/logs/stageb-van-in100_%A_%a.out
#SBATCH --error=slurm/logs/stageb-van-in100_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --array=0-2%2

# Stage B — vanilla EqM-B/2 80ep on the same IN-100 subset as our prior
# 80ep run (vanilla FID 121.24 at 1 seed, job 3300851). This is the
# baseline for the 3-seed DG-ANM vs vanilla comparison.
#
# Array index = seed. Throttled to 2 concurrent (%2) to respect per-user
# GPU limits; will step through 0, 1, 2.
#
# Emits per-epoch FID during training (from train_imagenet.py) and a
# final FID line for grep.

set -euo pipefail

SEED="${SLURM_ARRAY_TASK_ID:-0}"

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/stageb-van-in100-${SLURM_JOB_ID}-seed${SEED}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} (seed=${SEED}) ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || {
    echo "SHA not in last 50 commits, fetching full..." >&2
    git fetch --unshallow origin 2>&1 | tail -5
    git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1
}

echo "=== Installing dependencies ===" >&2
pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -5

# Same IN-100 subset used by prior 80ep runs + autoresearch pilots.
REPO_ROOT="/n/home03/mkrasnow/research-repo"
IN100_DATA="/n/home03/mkrasnow/in100_subset"
REF_STATS="$REPO_ROOT/projects/diff-EqM/results/in100_reference_stats.npz"
RESULTS_SUBDIR="projects/diff-EqM/results/stage_b_vanilla_in100_80ep_seed${SEED}"
mkdir -p "$WORK_DIR/$RESULTS_SUBDIR"

if [ ! -d "$IN100_DATA" ]; then
    echo "ERROR: IN-100 data not found at $IN100_DATA" >&2
    exit 1
fi
FOUND_CLASSES=$(ls -d "$IN100_DATA"/*/ 2>/dev/null | wc -l)
echo "=== Found $FOUND_CLASSES class directories in $IN100_DATA ===" >&2

MASTER_PORT=$((29500 + RANDOM % 1000))

# Vanilla EqM-B/2, 80 epochs, bs=256 (global, 4 GPU DDP).
# Hyperparams match our prior vanilla IN-100 80ep run (FID 121.24).
TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 \
    --data-path $IN100_DATA \
    --num-classes 100 \
    --global-batch-size 256 \
    --epochs 80 \
    --image-size 256 \
    --global-seed $SEED \
    --results-dir $RESULTS_SUBDIR"

# Resume from checkpoint if provided (long-running runs may need it).
if [ -n "${RESUME_CKPT:-}" ] && [ -f "$RESUME_CKPT" ]; then
    echo "=== Resuming from checkpoint: $RESUME_CKPT ===" >&2
    LOCAL_CKPT="$WORK_DIR/resume_ckpt.pt"
    cp "$RESUME_CKPT" "$LOCAL_CKPT"
    TRAIN_CMD="$TRAIN_CMD --ckpt $LOCAL_CKPT"
fi

echo "=== Mining DISABLED (vanilla EqM baseline) ===" >&2
echo "=== Seed: $SEED ===" >&2
echo "=== Command: $TRAIN_CMD ===" >&2
echo "=== Start: $(date), Job: $SLURM_JOB_ID, Array: ${SLURM_ARRAY_TASK_ID:-na} ===" >&2

eval $TRAIN_CMD 2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit: $TRAIN_EXIT (end $(date)) ===" >&2

# ---------------- Persist checkpoints FIRST (before sampling) ----------------
# Lesson from the 6540081 failure: sampler crashes (e.g. --num-classes mismatch)
# were killing checkpoint persistence via set -euo pipefail. Persist first so
# training is never wasted if sampling has a bug.
PERSIST_DIR="$REPO_ROOT/projects/diff-EqM/results/stage_b_vanilla_in100_80ep_seed${SEED}"
mkdir -p "$PERSIST_DIR"
echo "=== Early persist: syncing checkpoints to $PERSIST_DIR ===" >&2
rsync -av --include='*/' --include='*.pt' --include='*.json' --include='*.log' --exclude='*' \
    "$WORK_DIR/$RESULTS_SUBDIR/" "$PERSIST_DIR/" 2>&1 | tail -10 || echo "WARN: early rsync failed" >&2
cp train.log "$PERSIST_DIR/" 2>/dev/null || true

# ---------------- FID Evaluation ----------------
# Sample 50K images, compute FID against IN-100 reference stats.
# Wrap in a subshell so sampler failures don't abort the final rsync.
echo "=== Sampling + FID eval (50K samples) ===" >&2
SAMPLES_DIR="$WORK_DIR/$RESULTS_SUBDIR/samples"
mkdir -p "$SAMPLES_DIR"

CKPT_DIR=$(find "$WORK_DIR/$RESULTS_SUBDIR" -name "*.pt" -path "*/checkpoints/*" | sort | tail -1)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: no final checkpoint found for FID eval" >&2
else
    echo "Using checkpoint: $CKPT_DIR" >&2
    # BUGFIX: must pass --num-classes 100 to match the IN-100 training setup;
    # sample_gd.py defaults to 1000 (IN-1K) which causes a y_embedder size
    # mismatch on state_dict load.
    set +e
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$((MASTER_PORT+1)) \
        projects/diff-EqM/eqm-upstream/sample_gd.py \
        --model EqM-B/2 \
        --num-classes 100 \
        --image-size 256 \
        --ckpt "$CKPT_DIR" \
        --num-sampling-steps 250 \
        --cfg-scale 1.0 \
        --global-batch-size 256 \
        --num-fid-samples 50000 \
        --folder "$SAMPLES_DIR" \
        2>&1 | tee sample.log
    SAMPLE_EXIT=${PIPESTATUS[0]}
    echo "=== Sample exit: $SAMPLE_EXIT ===" >&2

    # FID compute against precomputed IN-100 stats.
    python -c "
import numpy as np, sys
from pytorch_fid.fid_score import calculate_fid_given_paths
ref_stats = '$REF_STATS'
samples_npz = '${SAMPLES_DIR}.npz'
fid = calculate_fid_given_paths([ref_stats, samples_npz], batch_size=64, device='cuda', dims=2048)
print(f'imagenet100_fid: {fid:.4f}')
" 2>&1 | tee fid.log
    FID_EXIT=${PIPESTATUS[0]}
    echo "=== FID exit: $FID_EXIT ===" >&2
    set -e
fi

# ---------------- Final persist (samples + logs) ----------------
echo "=== Final persist ===" >&2
rsync -av "$WORK_DIR/$RESULTS_SUBDIR/" "$PERSIST_DIR/" 2>&1 | tail -10 || echo "WARN: final rsync failed" >&2
cp train.log sample.log fid.log "$PERSIST_DIR/" 2>/dev/null || true

echo "=== Done seed=$SEED ===" >&2
