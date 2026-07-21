#!/bin/bash
#SBATCH --job-name=eqm-eval-fourier
#SBATCH --output=projects/masked-EqM/slurm/logs/eval-fourier-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/eval-fourier-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Fourier-recovery eval (see eval_fourier_recovery.py). Single GPU, no DDP.
# CKPT, CUTOFF/CUTOFF_GRID, OUT_NAME set via --export.
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/masked-eqm-fourier-${SLURM_JOB_ID}"
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
pip install --quiet timm diffusers accelerate torchvision lpips 2>&1 | tail -10

CKPT="${CKPT:?CKPT must be set}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
OUT_NAME="${OUT_NAME:-fourier_recovery}"
CUTOFF_ARGS="--cutoff ${CUTOFF:-0.25}"
if [ -n "${CUTOFF_GRID:-}" ]; then
    # CUTOFF_GRID is semicolon-separated (not comma) because sbatch --export
    # itself uses commas to split NAME=VALUE pairs -- a comma-separated
    # value here silently corrupts the export list and can hang submission.
    CUTOFF_ARGS="--cutoff-grid $(echo "$CUTOFF_GRID" | tr ';' ',')"
fi
VAE_ORACLE_FLAG=""
if [ "${VAE_ROUNDTRIP_ORACLE:-0}" = "1" ]; then
    VAE_ORACLE_FLAG="--vae-roundtrip-oracle"
fi
SAVE_IMAGES_FLAG=""
if [ "${SAVE_IMAGES:-0}" = "1" ]; then
    SAVE_IMAGES_FLAG="--save-images"
fi
MANIFEST_ARGS=""
if [ -n "${MANIFEST:-}" ]; then
    MANIFEST_ARGS="--manifest ${MANIFEST}"
fi

OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
mkdir -p "$OUT_DIR"

python eval_fourier_recovery.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    $CUTOFF_ARGS \
    --num-images "${NUM_IMAGES:-1024}" \
    --batch-size "${BATCH_SIZE:-16}" \
    $VAE_ORACLE_FLAG \
    $SAVE_IMAGES_FLAG \
    $MANIFEST_ARGS \
    --out "$OUT_DIR/${OUT_NAME}.json"

echo "=== Fourier-recovery eval done, output: $OUT_DIR/${OUT_NAME}.json ==="
rm -rf "$WORK_DIR"
