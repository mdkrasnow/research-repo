#!/bin/bash
#SBATCH --job-name=exp3-ref
#SBATCH --output=slurm/logs/exp3-ref_%A.out
#SBATCH --error=slurm/logs/exp3-ref_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=6:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1

# Exp3 one-time fixed reference cache (inception feats + per-class stats +
# real classifier histogram). Controls the "re-shuffled reference" confound.
#
# Env: OUT_DIR (rel, default projects/diff-EqM/results/exp3/reference)
# Opt: NUM_CLASSES, AGGREGATE_N, REF_PER_CLASS, REF_SEED, IMAGENET_TRAIN

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/exp3-ref-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git init . && git remote add origin "$GIT_URL"
git fetch --depth=50 origin main 2>&1 | tail -3
git checkout "$GIT_SHA" 2>&1 | tail -3 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "$GIT_SHA"; }

pip install timm diffusers accelerate scipy torchvision pytorch-fid 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
IMAGENET_TRAIN="${IMAGENET_TRAIN:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
OUT_DIR="${OUT_DIR:-projects/diff-EqM/results/exp3/reference}"
NUM_CLASSES="${NUM_CLASSES:-1000}"
AGGREGATE_N="${AGGREGATE_N:-50000}"
REF_PER_CLASS="${REF_PER_CLASS:-100}"
REF_SEED="${REF_SEED:-0}"

cd "$WORK_DIR"
python projects/diff-EqM/experiments/exp3_fidelity_diversity/build_reference.py \
    --imagenet-train "$IMAGENET_TRAIN" \
    --out-dir "$WORK_DIR/$OUT_DIR" \
    --num-classes "$NUM_CLASSES" \
    --aggregate-n "$AGGREGATE_N" \
    --ref-per-class "$REF_PER_CLASS" \
    --seed "$REF_SEED" 2>&1 | tee ref.log

mkdir -p "$REPO_ROOT/$OUT_DIR"
rsync -a "$WORK_DIR/$OUT_DIR/" "$REPO_ROOT/$OUT_DIR/"
cp ref.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/exp3-ref_${SLURM_JOB_ID}.log" 2>/dev/null || true
rm -rf "$WORK_DIR"
echo "=== exp3-ref done -> $OUT_DIR ==="
