#!/bin/bash
#SBATCH --job-name=exp3-gen
#SBATCH --output=slurm/logs/exp3-gen_%A.out
#SBATCH --error=slurm/logs/exp3-gen_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=8:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Exp3 generation for ONE arm, with the FIXED shared schedule (identical labels
# + per-sample seeds across vanilla/anm). Mirrors imagenet1k_fid_eval.sbatch.
#
# Required env: CKPT_PATH, TAG (vanilla|anm), SCHEDULE_REL, GEN_DIR_REL
# Optional: MODEL (EqM-B/2), STEPSIZE, NUM_SAMPLING_STEPS, SAMPLE_BATCH_SIZE, OVERWRITE

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/exp3-gen-${TAG}-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git init . && git remote add origin "$GIT_URL"
git fetch --depth=50 origin main 2>&1 | tail -3
git checkout "$GIT_SHA" 2>&1 | tail -3 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "$GIT_SHA"; }

pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
CKPT_PATH="${CKPT_PATH:?CKPT_PATH required}"
TAG="${TAG:?TAG required}"
SCHEDULE_REL="${SCHEDULE_REL:?SCHEDULE_REL required}"
GEN_DIR_REL="${GEN_DIR_REL:?GEN_DIR_REL required}"     # e.g. projects/.../gen/anm
MODEL="${MODEL:-EqM-B/2}"
STEPSIZE="${STEPSIZE:-0.003}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"
OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then OVERWRITE_FLAG="--overwrite"; fi

# resolve ckpt: accept a file, or a dir holding .pt files anywhere beneath it
# (the *.pt often live nested at <dir>/000-.../checkpoints/), pick highest-sorted.
echo "=== resolving ckpt: $CKPT_PATH ==="
if [ -f "$REPO_ROOT/$CKPT_PATH" ]; then
    CKPT_PATH="$REPO_ROOT/$CKPT_PATH"
elif [ -f "$CKPT_PATH" ]; then
    :
elif [ -d "$REPO_ROOT/$CKPT_PATH" ]; then
    CKPT_PATH=$(find "$REPO_ROOT/$CKPT_PATH" -name '*.pt' 2>/dev/null | sort -V | tail -1)
elif [ -d "$CKPT_PATH" ]; then
    CKPT_PATH=$(find "$CKPT_PATH" -name '*.pt' 2>/dev/null | sort -V | tail -1)
fi
[ -n "$CKPT_PATH" ] && [ -f "$CKPT_PATH" ] || { echo "ERROR: ckpt not found for $TAG: '$CKPT_PATH'"; exit 1; }
CKPT_ID="$CKPT_PATH"   # real identity for provenance (before /tmp copy)
echo "=== arm=$TAG resolved ckpt=$CKPT_PATH ; /tmp free: $(df -h "$WORK_DIR" | tail -1 | awk '{print $4}') ==="
LOCAL_CKPT="$WORK_DIR/ckpt.pt"
cp "$CKPT_PATH" "$LOCAL_CKPT" || { echo "ERROR: ckpt copy failed (disk?)"; exit 1; }
echo "=== arm=$TAG ckpt copied ($(du -h "$LOCAL_CKPT"|cut -f1)) ==="

PERSIST_GEN="$REPO_ROOT/$GEN_DIR_REL"; mkdir -p "$PERSIST_GEN"
SCHEDULE_ABS="$REPO_ROOT/$SCHEDULE_REL"
[ -f "$SCHEDULE_ABS" ] || { echo "ERROR: schedule missing: $SCHEDULE_ABS"; exit 1; }

MASTER_PORT=$((29500 + RANDOM % 1000))
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "=== generating arm=$TAG on $NUM_GPUS GPU(s), start $(date) ==="

# Unbuffered so a worker's traceback flushes to the log BEFORE torchrun kills the
# job (prior failures lost the real error to block-buffering). HF cache shared so
# the VAE isn't re-fetched per rank; disable telemetry noise.
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1

python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/exp3_fidelity_diversity/sample_scheduled.py \
    --model "$MODEL" --ckpt "$LOCAL_CKPT" --ckpt-id "$CKPT_ID" --tag "$TAG" \
    --schedule "$SCHEDULE_ABS" --folder "$PERSIST_GEN" \
    --num-classes 1000 --sampler gd --stepsize "$STEPSIZE" \
    --num-sampling-steps "$NUM_SAMPLING_STEPS" --cfg-scale 1.0 \
    --batch-size "$SAMPLE_BATCH_SIZE" $OVERWRITE_FLAG 2>&1 | tee gen.log

EXIT=${PIPESTATUS[0]}
N_PNG=$(find "$PERSIST_GEN" -name '*.png' | wc -l)
echo "=== arm=$TAG exit=$EXIT pngs=$N_PNG end $(date) ==="
cp gen.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/exp3-gen-${TAG}_${SLURM_JOB_ID}.log" 2>/dev/null || true
rm -rf "$WORK_DIR"
exit $EXIT
