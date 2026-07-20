#!/bin/bash
#SBATCH --job-name=eqm-stage1
#SBATCH --output=projects/masked-EqM/slurm/logs/stage1-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/stage1-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=18:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1

# Stage 1 matched longer-training curve (2026-07-20): does the cutoff-0.10
# Fourier delta_G effect grow with more training? Retrains gaussian/mask/
# gaussian+mask-1:1 matched seeds from scratch through 5 epochs (identical
# config to the existing 1-epoch corruption_sanity.sbatch runs except
# --epochs), saving an epoch-boundary checkpoint every epoch (train.py
# epoch{N}.pt, added 2026-07-20) so one run yields epochs 1/2/3/4/5 directly.
# Retrain-from-scratch, not resume: train.py's --ckpt path does not restore
# train_steps/epoch/RNG state, so resuming would silently replay the
# original epoch-0 shuffle order instead of continuing -- see pipeline.json
# decision note 2026-07-20.
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/masked-eqm-stage1-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?GIT_SHA must be set}"

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

cd projects/masked-EqM

echo "=== Installing dependencies ===" >&2
pip install --quiet timm diffusers accelerate torchdiffeq wandb torchvision 2>&1 | tail -10

IMAGENET_PATH="${IMAGENET_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
CORRUPTION_MODE="${CORRUPTION_MODE:-gaussian}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
RESULTS_ROOT="${RESULTS_ROOT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be set (e.g. stage1_gaussian_seed0)}"
RESULTS_DIR="$RESULTS_ROOT/${RUN_TAG}_job${SLURM_JOB_ID}"

if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet data not found at $IMAGENET_PATH" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

export ENTITY="${ENTITY:-none}"
export PROJECT="${PROJECT:-none}"

MASTER_PORT=$((29500 + RANDOM % 1000))

TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=$MASTER_PORT \
    train.py \
    --model EqM-B/2 \
    --data-path $IMAGENET_PATH \
    --num-classes 1000 \
    --global-batch-size 32 \
    --epochs ${EPOCHS:-5} \
    --image-size 256 \
    --log-every 50 \
    --ckpt-every 5000 \
    --results-dir $RESULTS_DIR \
    --corruption-mode $CORRUPTION_MODE \
    --mask-prob ${MASK_PROB:-0.5} \
    --fourier-cutoff ${FOURIER_CUTOFF:-0.25} \
    --blur-sigma ${BLUR_SIGMA:-1.0} \
    --downsample-factor ${DOWNSAMPLE_FACTOR:-4.0} \
    --gaussian-weight ${GAUSSIAN_WEIGHT:-1.0} \
    --mask-weight ${MASK_WEIGHT:-0.0} \
    --fourier-weight ${FOURIER_WEIGHT:-0.0} \
    --blur-weight ${BLUR_WEIGHT:-0.0} \
    --downsample-weight ${DOWNSAMPLE_WEIGHT:-0.0} \
    --global-seed ${GLOBAL_SEED:-0}"

echo "=== run_tag=$RUN_TAG corruption_mode=$CORRUPTION_MODE ===" >&2
echo "=== Command: $TRAIN_CMD ===" >&2
echo "=== Start time: $(date) ===" >&2
echo "=== Job ID: $SLURM_JOB_ID ===" >&2
echo "=== Node: $(hostname) ===" >&2

eval $TRAIN_CMD 2>&1 | tee train.log

TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit code: $TRAIN_EXIT ===" >&2
echo "=== Training end time: $(date) ===" >&2

cp train.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo "=== Cleaning up work directory ===" >&2
rm -rf "$WORK_DIR"

echo "=== Stage1 run done (tag=$RUN_TAG, exit=$TRAIN_EXIT) ===" >&2
exit $TRAIN_EXIT
