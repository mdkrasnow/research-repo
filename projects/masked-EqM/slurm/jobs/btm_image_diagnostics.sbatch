#!/bin/bash
#SBATCH --job-name=btm-imgdiag
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-imgdiag-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-imgdiag-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail

# FD calibration / gradient-noise / weak-conservation probes on a FROZEN
# ImageNet B/2 scalar-energy checkpoint.  No training, no optimizer, no weight
# writes.  Single GPU, so gpu_test MIG slices are fine.
#
# Env: GIT_SHA (required), CKPT (optional -- omit for an init-scale run),
#      RUN_TAG, DO (calibrate,gradnoise,weak), BATCH, FD_K, N_GRAD_BATCHES.

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
RUN_TAG="${RUN_TAG:-btm_imgdiag}"
DO="${DO:-calibrate,gradnoise,weak}"
BATCH="${BATCH:-8}"
FD_K="${FD_K:-4}"
N_GRAD_BATCHES="${N_GRAD_BATCHES:-64}"
N_WEAK_BATCHES="${N_WEAK_BATCHES:-32}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
WORK_DIR="/tmp/btm-imgdiag-${SLURM_JOB_ID}"
OUT_REL="results/btm/imgdiag/${RUN_TAG}_job${SLURM_JOB_ID}.json"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin '+refs/heads/*:refs/remotes/origin/*'
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision pytest

python -m pytest tests/test_btm_math.py tests/test_btm_image_conventions.py -q

CKPT_ARG=""
if [ -n "${CKPT:-}" ]; then CKPT_ARG="--ckpt ${CKPT}"; fi
mkdir -p "$(dirname "$OUT_REL")"
python -m experiments.btm.image_diagnostics \
  --data-path "$IMAGENET_PATH" --out "$OUT_REL" --do "$DO" \
  --batch "$BATCH" --fd-k "$FD_K" --n-grad-batches "$N_GRAD_BATCHES" \
  --n-weak-batches "$N_WEAK_BATCHES" $CKPT_ARG 2>&1 | tee diag.log

mkdir -p "$REPO_ROOT/projects/masked-EqM/results/btm/imgdiag"
cp "$OUT_REL" "$REPO_ROOT/projects/masked-EqM/$OUT_REL"
cp diag.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
