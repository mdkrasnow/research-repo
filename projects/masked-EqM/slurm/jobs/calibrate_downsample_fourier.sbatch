#!/bin/bash
#SBATCH --job-name=eqm-calib-ds-fc
#SBATCH --output=projects/masked-EqM/slurm/logs/calib-dsfc-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/calib-dsfc-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# One-shot calibration: finds downsample factor and fourier cutoff whose
# pixel-space LPIPS vs clean matches the existing p=0.5 mask task's LPIPS
# (see calibrate_downsample_fourier.py).
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/masked-eqm-calib-dsfc-${SLURM_JOB_ID}"
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

VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
OUT_DIR="/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results"
mkdir -p "$OUT_DIR"

python calibrate_downsample_fourier.py \
    --data-path "$VAL_PATH" \
    --mask-prob "${MASK_PROB:-0.5}" \
    --num-images "${NUM_IMAGES:-128}" \
    --batch-size "${BATCH_SIZE:-16}" \
    | tee "$OUT_DIR/downsample_fourier_calibration.log"

echo "=== Calibration done, log: $OUT_DIR/downsample_fourier_calibration.log ==="
rm -rf "$WORK_DIR"
