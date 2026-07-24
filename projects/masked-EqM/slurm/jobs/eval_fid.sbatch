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

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?GIT_SHA must be set}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || {
    git fetch --unshallow origin 2>&1 | tail -5
    git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1
}

cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision pytorch-fid 2>&1 | tail -10

CKPT="${CKPT:?CKPT must be set}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
OUT_NAME="${OUT_NAME:-fid}"
EBM="${EBM:-none}"

OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
mkdir -p "$OUT_DIR"

python eval_fid.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    --num-samples "${NUM_SAMPLES:-2000}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --ebm "$EBM" \
    --num-sampling-steps "${NUM_SAMPLING_STEPS:-250}" \
    --work-dir "$WORK_DIR/fid_work" \
    --out "$OUT_DIR/${OUT_NAME}.json"

echo "=== FID eval done, output: $OUT_DIR/${OUT_NAME}.json ==="

# smoke-time sample probe (AGENTS.md): persist a few generated images for
# eyeball verification before trusting the FID number alone.
SAMPLES_DIR="$OUT_DIR/${OUT_NAME}_samples"
mkdir -p "$SAMPLES_DIR"
cp "$WORK_DIR"/fid_work/generated/00000{0,1,2,3,4,5,6,7}.png "$SAMPLES_DIR/" 2>/dev/null || true

rm -rf "$WORK_DIR"
