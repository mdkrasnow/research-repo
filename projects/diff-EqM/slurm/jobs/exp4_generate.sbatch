#!/bin/bash
#SBATCH --job-name=exp4-gen
#SBATCH --output=slurm/logs/exp4-gen_%A.out
#SBATCH --error=slurm/logs/exp4-gen_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=6:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# Experiment 4: generate one checkpoint's sample set with the FIXED eval protocol
# (gd / stepsize 0.003 / 250 steps / cfg 1.0 / EMA) and a FIXED label schedule
# (--label-seed). Identical (global_seed, world_size=4, label_seed) across ALL
# checkpoints => identical noise + labels => fair comparison.
#
# Required env: CHECKPOINT_PATH, OUT_NPZ (destination .npz path under results)
# Optional: MODEL (EqM-B/2), NUM_FID_SAMPLES (50000), SAMPLE_BATCH_SIZE (128),
#           GLOBAL_SEED (0), LABEL_SEED (12345)

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/exp4-gen-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"
git init . 2>&1; git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "${GIT_SHA}" || exit 1; }
pip install timm diffusers accelerate scipy torchvision pillow 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?set CHECKPOINT_PATH}"
OUT_NPZ="${OUT_NPZ:?set OUT_NPZ (e.g. projects/diff-EqM/results/experiment4/samples/anm_step_seed0.npz)}"
[ -f "$CHECKPOINT_PATH" ] || CHECKPOINT_PATH="$REPO_ROOT/$CHECKPOINT_PATH"
[ -f "$CHECKPOINT_PATH" ] || { echo "ERROR: ckpt not found"; exit 1; }
LOCAL_CKPT="$WORK_DIR/checkpoint.pt"; cp "$CHECKPOINT_PATH" "$LOCAL_CKPT"

SAMPLES_DIR="$WORK_DIR/samples"; mkdir -p "$SAMPLES_DIR"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-128}"
MASTER_PORT=$((29500 + RANDOM % 1000))
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')

python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    projects/diff-EqM/eqm-upstream/sample_gd.py \
    --model "${MODEL:-EqM-B/2}" --ckpt "$LOCAL_CKPT" --num-classes 1000 \
    --sampler gd --stepsize 0.003 --num-sampling-steps 250 --cfg-scale 1.0 \
    --global-seed "${GLOBAL_SEED:-0}" --label-seed "${LABEL_SEED:-12345}" \
    --global-batch-size $SAMPLE_BATCH_SIZE --num-fid-samples $NUM_FID_SAMPLES \
    --folder "$SAMPLES_DIR" 2>&1 | tee sample.log
[ ${PIPESTATUS[0]} -eq 0 ] || { cp sample.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/exp4-gen_${SLURM_JOB_ID}.log"; exit 1; }

# pack pngs -> single .npz (key arr_0), the format the audit consumes
DEST="$REPO_ROOT/$OUT_NPZ"; mkdir -p "$(dirname "$DEST")"
python -c "
import sys; sys.path.insert(0,'projects/diff-EqM/eqm-upstream')
from sample_gd import create_npz_from_sample_folder
import shutil
p = create_npz_from_sample_folder('$SAMPLES_DIR', num=$NUM_FID_SAMPLES)
shutil.move(p, '$DEST')
print('wrote', '$DEST')
"
echo "=== samples -> $OUT_NPZ ==="
# NOTE: class-conditional NN needs per-sample labels aligned to this npz. Saving
# the label schedule from DDP is a follow-up; until then the audit uses global
# (class-agnostic) NN and logs that it did so.
