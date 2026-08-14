#!/bin/bash
#SBATCH --job-name=eqm-eval-fid
#SBATCH --output=projects/masked-EqM/slurm/logs/eval-fid-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/eval-fid-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# Sanity-scale FID (see eval_fid.py). Single GPU, no DDP. Not the paper-scale
# 50k-sample protocol -- a few thousand samples to check masking doesn't wreck
# unconditional generation. CKPT, OUT_NAME set via --export.
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/masked-eqm-fid-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_SHA="${GIT_SHA:?GIT_SHA must be set}"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:?SOURCE_ARCHIVE must point to the submitted source archive}"
test -f "$SOURCE_ARCHIVE"
tar -xzf "$SOURCE_ARCHIVE" -C "$WORK_DIR"
cd "$WORK_DIR/projects/masked-EqM"
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision pytorch-fid 2>&1 | tail -10

CKPT="${CKPT:?CKPT must be set}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
# No default. A shared default (`fid`) meant two arms invoked without OUT_NAME
# both wrote $OUT_DIR/fid.json and $OUT_DIR/fid_samples/, so whichever finished
# second silently destroyed the other's result. Correctness used to live in the
# caller (the queue script exports OUT_NAME); requiring it here makes the
# collision unrepresentable instead of merely unlikely.
OUT_NAME="${OUT_NAME:?OUT_NAME must be set -- a shared default silently overwrites the other arm FID result}"
EBM="${EBM:-none}"
MODEL="${MODEL:-EqM-B/2}"

OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
mkdir -p "$OUT_DIR"

python eval_fid.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    --model "$MODEL" \
    --num-samples "${NUM_SAMPLES:-2000}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --ebm "$EBM" \
    --num-sampling-steps "${NUM_SAMPLING_STEPS:-250}" \
    --stepsize "${STEPSIZE:-0.003}" \
    --work-dir "$WORK_DIR/fid_work" \
    --out "$OUT_DIR/${OUT_NAME}.json"

echo "=== FID eval done, output: $OUT_DIR/${OUT_NAME}.json ==="

# smoke-time sample probe (AGENTS.md): persist a few generated images for
# eyeball verification before trusting the FID number alone.
SAMPLES_DIR="$OUT_DIR/${OUT_NAME}_samples"
mkdir -p "$SAMPLES_DIR"
cp "$WORK_DIR"/fid_work/generated/00000{0,1,2,3,4,5,6,7}.png "$SAMPLES_DIR/" 2>/dev/null || true

rm -rf "$WORK_DIR"
