#!/bin/bash
#SBATCH --job-name=eqm-eval-recovery
#SBATCH --output=projects/masked-EqM/slurm/logs/eval-recovery-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/eval-recovery-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Masked-recovery eval (see eval_masked_recovery.py). Single GPU, no DDP.
# CKPT, HARD_CONSTRAIN, OUT_NAME set via --export.
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/masked-eqm-eval-${SLURM_JOB_ID}"
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
pip install --quiet timm diffusers accelerate torchvision 2>&1 | tail -10

CKPT="${CKPT:?CKPT must be set}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
OUT_NAME="${OUT_NAME:-masked_recovery}"
HARD_CONSTRAIN_FLAG=""
if [ "${HARD_CONSTRAIN:-0}" = "1" ]; then
    HARD_CONSTRAIN_FLAG="--hard-constrain"
fi
VAE_ORACLE_FLAG=""
if [ "${VAE_ROUNDTRIP_ORACLE:-0}" = "1" ]; then
    VAE_ORACLE_FLAG="--vae-roundtrip-oracle"
fi

OUT_DIR="/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results"
mkdir -p "$OUT_DIR"

python eval_masked_recovery.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    --mask-prob "${MASK_PROB:-0.5}" \
    --num-images "${NUM_IMAGES:-256}" \
    --batch-size "${BATCH_SIZE:-16}" \
    $HARD_CONSTRAIN_FLAG \
    $VAE_ORACLE_FLAG \
    --out "$OUT_DIR/${OUT_NAME}.json"

echo "=== Eval done, output: $OUT_DIR/${OUT_NAME}.json ==="
rm -rf "$WORK_DIR"
