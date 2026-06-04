#!/bin/bash
#SBATCH --job-name=exp3-metrics
#SBATCH --output=slurm/logs/exp3-metrics_%A.out
#SBATCH --error=slurm/logs/exp3-metrics_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=6:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1

# Exp3 metrics phase: extract Inception features + resnet50 preds for both arms,
# compute FID/KID/PRDC/per-class/classifier-histogram, deltas, plots, verdict.
#
# Required env: RUN_DIR_REL (contains gen/{vanilla,anm} + schedule.json),
#               REF_DIR_REL, VANILLA_CKPT, ANM_CKPT

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/exp3-metrics-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git init . && git remote add origin "$GIT_URL"
git fetch --depth=50 origin main 2>&1 | tail -3
git checkout "$GIT_SHA" 2>&1 | tail -3 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "$GIT_SHA"; }

pip install timm diffusers accelerate scipy torchvision pytorch-fid matplotlib 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
RUN_DIR_REL="${RUN_DIR_REL:?RUN_DIR_REL required}"
REF_DIR_REL="${REF_DIR_REL:-projects/diff-EqM/results/exp3/reference}"
VANILLA_CKPT="${VANILLA_CKPT:-}"
ANM_CKPT="${ANM_CKPT:-}"
PRDC_K="${PRDC_K:-5}"
# OUT_DIR_REL lets metrics WRITE outputs to a different filesystem than it READS
# gen/reference from. Needed when the gen PNGs live on a quota-capped fileset
# (holylabs group quota): point OUT_DIR_REL at home so the small CSV/npy/plots
# outputs always have room even when the read-side fs is over quota. Reads
# (gen PNGs, reference) never consume quota. Defaults to RUN_DIR_REL (in-place).
OUT_DIR_REL="${OUT_DIR_REL:-$RUN_DIR_REL}"

RUN_DIR="$REPO_ROOT/$RUN_DIR_REL"
OUT_DIR="$REPO_ROOT/$OUT_DIR_REL"; mkdir -p "$OUT_DIR"
cd "$WORK_DIR"
python projects/diff-EqM/experiments/exp3_fidelity_diversity/eval_fidelity_diversity.py \
    --gen-root "$RUN_DIR/gen" \
    --schedule "$RUN_DIR/schedule.json" \
    --reference-dir "$REPO_ROOT/$REF_DIR_REL" \
    --out "$OUT_DIR" \
    --vanilla-ckpt "$VANILLA_CKPT" --anm-ckpt "$ANM_CKPT" \
    --sampler gd --nfe 250 --step-size 0.003 --cfg-scale 1.0 \
    --prdc-k "$PRDC_K" 2>&1 | tee metrics.log

cp metrics.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/exp3-metrics_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo "=== exp3-metrics done; read $RUN_DIR_REL -> wrote $OUT_DIR_REL ==="
rm -rf "$WORK_DIR"
