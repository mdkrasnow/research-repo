#!/bin/bash
#SBATCH --job-name=eqm-eval-fourier-conv
#SBATCH --output=projects/masked-EqM/slurm/logs/eval-fourier-conv-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/eval-fourier-conv-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Recovery convergence-curve eval (single trajectory, records LPIPS/MSE at
# multiple step counts) -- see eval_fourier_recovery.py --record-steps.
# CKPT, CUTOFF, OUT_NAME, RECORD_STEPS set via --export.
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/masked-eqm-fourier-conv-${SLURM_JOB_ID}"
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
OUT_NAME="${OUT_NAME:?OUT_NAME must be set}"
CUTOFF="${CUTOFF:-0.10}"
# RECORD_STEPS/SAVE_STEPS are semicolon-separated (not comma) because sbatch
# --export itself uses commas to split NAME=VALUE pairs -- see CUTOFF_GRID
# precedent in eval_fourier_recovery.sbatch.
RECORD_STEPS_ARG="$(echo "${RECORD_STEPS:-0;25;50;100;250;500;1000}" | tr ';' ',')"

SAVE_IMAGES_FLAG=""
SAVE_STEPS_ARG=""
if [ "${SAVE_IMAGES:-0}" = "1" ]; then
    SAVE_IMAGES_FLAG="--save-images"
    if [ -n "${SAVE_STEPS:-}" ]; then
        SAVE_STEPS_ARG="--save-steps $(echo "$SAVE_STEPS" | tr ';' ',')"
    fi
fi

MANIFEST_ARGS=""
if [ -n "${MANIFEST:-}" ]; then
    MANIFEST_ARGS="--manifest ${MANIFEST}"
fi

SUBSET_ARGS=""
if [ -n "${SUBSET_INDICES:-}" ]; then
    SUBSET_ARGS="--subset-indices ${SUBSET_INDICES}"
fi

OUT_DIR="/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results"
mkdir -p "$OUT_DIR"

python eval_fourier_recovery.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    --cutoff "$CUTOFF" \
    --record-steps "$RECORD_STEPS_ARG" \
    --num-images "${NUM_IMAGES:-1024}" \
    --batch-size "${BATCH_SIZE:-16}" \
    $SAVE_IMAGES_FLAG \
    $SAVE_STEPS_ARG \
    $MANIFEST_ARGS \
    $SUBSET_ARGS \
    --out "$OUT_DIR/${OUT_NAME}.json"

echo "=== Fourier-convergence eval done, output: $OUT_DIR/${OUT_NAME}.json ==="
rm -rf "$WORK_DIR"
