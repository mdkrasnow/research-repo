#!/bin/bash
#SBATCH --job-name=exp4-smoke
#SBATCH --output=slurm/logs/exp4-smoke_%A.out
#SBATCH --error=slurm/logs/exp4-smoke_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Experiment 4 cluster smoke: (1) plumbing via local_smoke.py (stub backbone),
# then (2) a REAL-backbone sanity -- extract inception + dinov2 features from 16
# fake images and assert finite/correct shapes. Catches dep/weights/download
# issues before the full run. Smoke numbers are meaningless.

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/exp4-smoke-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"
git init . 2>&1; git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "${GIT_SHA}" || exit 1; }
pip install timm scipy torchvision pillow matplotlib open_clip_torch 2>&1 | tail -5

echo "=== (1) plumbing smoke (stub backbone) ==="
python projects/diff-EqM/experiments/experiment4/local_smoke.py

echo "=== (2) real-backbone feature sanity ==="
python -c "
import sys, numpy as np
sys.path.insert(0,'projects/diff-EqM/experiments/experiment4')
import features as feat
imgs = (np.random.rand(16,256,256,3)*255).astype('uint8')
inc = feat.build_extractor('inception_pool3', device='cuda')
f1 = inc.extract(imgs, batch_size=8); assert f1.shape==(16,2048) and np.isfinite(f1).all(), f1.shape
dino = feat.build_extractor('dinov2_vitl14', device='cuda')
f2 = dino.extract(imgs, batch_size=8); assert f2.shape[0]==16 and np.isfinite(f2).all(), f2.shape
print('REAL-BACKBONE SANITY PASS', f1.shape, f2.shape)
"
echo "=== exp4 smoke PASS ==="
