#!/bin/bash
#SBATCH --job-name=eqm-forensics
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/forensics-%A_%a.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/forensics-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=03:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
ROOT="${CHECKPOINT_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer40_direct_seed0_ckpt50k_job36359213/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints}"
OUT_ROOT="${OUT_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_checkpoint_forensics}"
DATA="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
STEPS=(1450000 1500000 1550000 1600000)
STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
WORK="/tmp/eqm-forensics-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK" "$OUT_ROOT"
cd "$WORK"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision
python experiments/direct_energy/checkpoint_forensics.py \
  --data-path "$DATA" --ckpt "$ROOT/$(printf '%07d' "$STEP").pt" \
  --label "direct_${STEP}" --output "$OUT_ROOT/direct_${STEP}.json"
rm -rf "$WORK"
