#!/bin/bash
#SBATCH --job-name=eqm-imagenet-fid
#SBATCH --output=slurm/logs/eqm-imagenet-fid_%A.out
#SBATCH --error=slurm/logs/eqm-imagenet-fid_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# FID evaluation for EqM-B/2 on ImageNet 256x256
# Generates 50K samples with DDP (4x A100), then computes FID with ADM evaluator
# Expected: ~2-3h sampling + ~30min FID computation

set -euo pipefail

# Load required modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPUs
echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

# Set working directory
WORK_DIR="/tmp/eqm-imagenet-fid-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository at specific git SHA
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || {
    echo "SHA not in last 50 commits, fetching full..." >&2
    git fetch --unshallow origin 2>&1 | tail -5
    git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1
}

# Install dependencies
echo "=== Installing dependencies ===" >&2
pip install timm diffusers accelerate scipy torchvision wandb 2>&1 | tail -10

# Configuration
CHECKPOINT_PATH="${CHECKPOINT_PATH:?ERROR: CHECKPOINT_PATH env var must be set}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
PERSISTENT_CACHE="$REPO_ROOT/projects/diff-EqM/results/.cache"
SAMPLES_DIR="$WORK_DIR/samples"

mkdir -p "$SAMPLES_DIR"
mkdir -p "$PERSISTENT_CACHE"

# === STEP 1: Sync checkpoint from persistent storage ===
echo "=== Syncing checkpoint: $CHECKPOINT_PATH ===" >&2
if [ ! -f "$CHECKPOINT_PATH" ]; then
    # Try relative to persistent storage
    FULL_CKPT="$REPO_ROOT/$CHECKPOINT_PATH"
    if [ -f "$FULL_CKPT" ]; then
        CHECKPOINT_PATH="$FULL_CKPT"
    else
        echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH or $FULL_CKPT" >&2
        exit 1
    fi
fi

# Copy checkpoint to work directory for faster access
LOCAL_CKPT="$WORK_DIR/checkpoint.pt"
cp "$CHECKPOINT_PATH" "$LOCAL_CKPT"
echo "=== Checkpoint copied to $LOCAL_CKPT ($(du -h "$LOCAL_CKPT" | cut -f1)) ===" >&2

# === STEP 2: Download/cache ImageNet reference stats ===
REF_STATS="$PERSISTENT_CACHE/VIRTUAL_imagenet256_labeled.npz"
if [ ! -f "$REF_STATS" ]; then
    echo "=== Downloading ImageNet 256x256 reference stats (one-time) ===" >&2
    wget -q -O "$REF_STATS" \
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz" \
        || { echo "ERROR: Failed to download reference stats"; exit 1; }
    echo "=== Reference stats downloaded ($(du -h "$REF_STATS" | cut -f1)) ===" >&2
else
    echo "=== Using cached reference stats ===" >&2
fi

# Copy reference stats to work dir
cp "$REF_STATS" "$WORK_DIR/VIRTUAL_imagenet256_labeled.npz"

# === STEP 3: Download/cache ADM evaluator ===
EVALUATOR_PY="$PERSISTENT_CACHE/evaluator.py"
if [ ! -f "$EVALUATOR_PY" ]; then
    echo "=== Downloading ADM evaluator.py ===" >&2
    wget -q -O "$EVALUATOR_PY" \
        "https://raw.githubusercontent.com/openai/guided-diffusion/main/evaluations/evaluator.py" \
        || { echo "ERROR: Failed to download evaluator.py"; exit 1; }
    echo "=== ADM evaluator downloaded ===" >&2
else
    echo "=== Using cached ADM evaluator ===" >&2
fi

cp "$EVALUATOR_PY" "$WORK_DIR/evaluator.py"

# === STEP 4: Generate 50K samples with DDP ===
echo "=== Generating 50K samples (4x A100 DDP) ===" >&2
echo "=== Sampling start time: $(date) ===" >&2

MASTER_PORT=$((29500 + RANDOM % 1000))
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
echo "=== Using $NUM_GPUS GPU(s) for sampling ===" >&2
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    projects/diff-EqM/eqm-upstream/sample_gd.py \
    --model EqM-B/2 \
    --ckpt "$LOCAL_CKPT" \
    --sampler gd \
    --stepsize 0.003 \
    --num-sampling-steps 250 \
    --cfg-scale 1.0 \
    --num-fid-samples 50000 \
    --folder "$SAMPLES_DIR" \
    2>&1 | tee sample.log

SAMPLE_EXIT=${PIPESTATUS[0]}
echo "=== Sampling exit code: $SAMPLE_EXIT ===" >&2
echo "=== Sampling end time: $(date) ===" >&2

if [ $SAMPLE_EXIT -ne 0 ]; then
    echo "ERROR: Sampling failed" >&2
    cp sample.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet_fid_${SLURM_JOB_ID}_sample.log" 2>/dev/null || true
    exit $SAMPLE_EXIT
fi

# Check that samples.npz was generated
SAMPLES_NPZ="$SAMPLES_DIR/samples.npz"
if [ ! -f "$SAMPLES_NPZ" ]; then
    # Some versions output to the folder directly; look for .npz files
    SAMPLES_NPZ=$(find "$SAMPLES_DIR" -name "*.npz" -type f | head -1)
    if [ -z "$SAMPLES_NPZ" ]; then
        echo "ERROR: No .npz file found in $SAMPLES_DIR after sampling" >&2
        ls -la "$SAMPLES_DIR/" >&2
        exit 1
    fi
fi
echo "=== Samples file: $SAMPLES_NPZ ($(du -h "$SAMPLES_NPZ" | cut -f1)) ===" >&2

# === STEP 5: Compute FID with ADM evaluator ===
echo "=== Computing FID ===" >&2
echo "=== FID start time: $(date) ===" >&2

python evaluator.py \
    "$WORK_DIR/VIRTUAL_imagenet256_labeled.npz" \
    "$SAMPLES_NPZ" \
    2>&1 | tee fid.log

FID_EXIT=${PIPESTATUS[0]}
echo "=== FID exit code: $FID_EXIT ===" >&2
echo "=== FID end time: $(date) ===" >&2

# Parse and print FID result
echo "=== FID RESULTS ===" >&2
grep -i "fid" fid.log || echo "(no FID line found in output)" >&2
echo "==================" >&2

# === STEP 6: Sync results back ===
echo "=== Syncing results ===" >&2
RESULTS_OUT="$REPO_ROOT/projects/diff-EqM/results/imagenet_fid_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_OUT"
cp fid.log "$RESULTS_OUT/" 2>/dev/null || true
cp sample.log "$RESULTS_OUT/" 2>/dev/null || true
cp "$SAMPLES_NPZ" "$RESULTS_OUT/" 2>/dev/null || true
cp sample.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet_fid_${SLURM_JOB_ID}_sample.log" 2>/dev/null || true
cp fid.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/imagenet_fid_${SLURM_JOB_ID}_fid.log" 2>/dev/null || true

# Cleanup
echo "=== Cleaning up work directory ===" >&2
rm -rf "$WORK_DIR"

echo "=== Done (exit=$FID_EXIT) ===" >&2
exit $FID_EXIT
