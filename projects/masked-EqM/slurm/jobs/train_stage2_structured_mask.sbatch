#!/bin/bash
#SBATCH --job-name=eqm-stage2
#SBATCH --output=projects/masked-EqM/slurm/logs/stage2-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/stage2-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=11:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Stage 2 structured-mask corruption (2026-07-22): 1-epoch controlled
# comparison of gaussian-only / structured-mask-only / gaussian+structured-
# mask-1:1 EqM-B/2, screening seeds 0-2. Mirrors corruption_sanity.sbatch's
# config exactly (same batch size, optimizer, LR schedule, image size,
# single epoch) so this is a clean matched comparison, not a re-baseline.
# See documentation/stage2_structured_mask_proposal_2026-07-22.md for the
# full variant proposal + mechanism check (mandatory per AGENTS.md, written
# before this file). Default RESULTS_ROOT points at home03 (not holylabs)
# from the start this round -- Stage 1 hit 3 separate holylabs quota
# incidents overnight (see pipeline.json debugging_notes 2026-07-21), the
# third of which was the shared ydu_lab quota pinned at 100% by other lab
# members' usage and not fixable unilaterally; routing around holylabs
# entirely avoids repeating that failure mode.
#
# FIX 2026-07-22 (4th home03 quota incident): originally shipped with
# --ckpt-every 5000 (copied from corruption_sanity.sbatch), which saved a
# ~2GB step-checkpoint every 5000 steps in addition to the final epoch01.pt
# unconditional save. Across up to 8 concurrent/sequential 1-epoch arms this
# accumulated ~42GB of entirely redundant intermediate checkpoints (Stage 2
# only ever evaluates the single end-of-epoch checkpoint, never an
# intermediate one) and filled home03's 95GB quota to 100%, which in turn
# caused a long-running job to stall ~46min then die and caused new
# submissions to fail within seconds of starting. Disabled periodic
# checkpointing entirely (matches train_stage1_longer.sbatch's own fix from
# the same root cause) -- CKPT_EVERY is now overridable but defaults off.
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

echo "=== GPU verification ===" >&2
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/masked-eqm-stage2-${SLURM_JOB_ID}"
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
CORRUPTION_MODE="${CORRUPTION_MODE:?CORRUPTION_MODE must be set (gaussian|structured_mask|mixture)}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
RESULTS_ROOT="${RESULTS_ROOT:-/n/home03/mkrasnow/masked-EqM-stage2-results}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be set (e.g. stage2_gaussian_seed0)}"
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
    --epochs 1 \
    --image-size 256 \
    --log-every 50 \
    --ckpt-every ${CKPT_EVERY:-999999999} \
    --results-dir $RESULTS_DIR \
    --corruption-mode $CORRUPTION_MODE \
    --mask-prob ${MASK_PROB:-0.5} \
    --gaussian-weight ${GAUSSIAN_WEIGHT:-1.0} \
    --structured-mask-weight ${STRUCTURED_MASK_WEIGHT:-0.0} \
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

echo "=== Stage2 run done (tag=$RUN_TAG, exit=$TRAIN_EXIT) ===" >&2
exit $TRAIN_EXIT
