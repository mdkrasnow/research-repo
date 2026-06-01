#!/bin/bash
#SBATCH --job-name=exp4-refs
#SBATCH --output=slurm/logs/exp4-refs_%A.out
#SBATCH --error=slurm/logs/exp4-refs_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=8:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1

# Experiment 4: build val + train reference banks (FIXES B1 -- project had no
# val-reference FID). Produces unified {mu,sigma} inception stats, raw inception
# feats (KID), and DINOv2 NN banks for BOTH splits with equal class-balanced
# counts. Run ONCE before the audit.
#
# Optional env: COUNT (default 50000), PER_CLASS (50), NO_THUMBS (set to 1 to skip)

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/exp4-refs-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"
git init . 2>&1; git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "${GIT_SHA}" || exit 1; }

pip install timm diffusers scipy torchvision pillow open_clip_torch 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
OUT="$REPO_ROOT/projects/diff-EqM/results"
TRAIN_ROOT="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"
VAL_ROOT="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val"
COUNT="${COUNT:-50000}"; PER_CLASS="${PER_CLASS:-50}"
THUMB_FLAG=""; [ "${NO_THUMBS:-0}" = "1" ] && THUMB_FLAG="--no-thumbs"

[ -d "$VAL_ROOT" ] || { echo "ERROR: val split missing at $VAL_ROOT (B-blocker)"; exit 1; }

python projects/diff-EqM/experiments/experiment4/build_references.py \
    --train-root "$TRAIN_ROOT" --val-root "$VAL_ROOT" --out "$OUT" \
    --count "$COUNT" --per-class "$PER_CLASS" --image-size 256 $THUMB_FLAG \
    2>&1 | tee build_refs.log
cp build_refs.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/exp4-refs_${SLURM_JOB_ID}.log" || true
echo "=== references written to $OUT ==="
