#!/bin/bash
#SBATCH --job-name=eqm-forensic-fid
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/forensic-fid-%A_%a.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/forensic-fid-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
ROOT="${CHECKPOINT_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer40_direct_seed0_ckpt50k_job36359213/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints}"
OUT_ROOT="${OUT_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_checkpoint_forensics/fid500}"
VAL="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
STEPS=(1450000 1500000 1550000 1600000)
WORK="/tmp/eqm-forensic-fid-${SLURM_JOB_ID}"
mkdir -p "$WORK" "$OUT_ROOT"
cd "$WORK"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision pytorch-fid
for STEP in "${STEPS[@]}"; do
  python eval_fid.py --ckpt "$ROOT/$(printf '%07d' "$STEP").pt" \
    --data-path "$VAL" --num-samples 500 --batch-size 16 --ebm direct \
    --num-sampling-steps 250 --stepsize "${STEPSIZE:-0.003}" --seed 20260803 \
    --work-dir "$WORK/fid_work_${STEP}" --out "$OUT_ROOT/direct_${STEP}.json"
  mkdir -p "$OUT_ROOT/direct_${STEP}_samples"
  cp "$WORK/fid_work_${STEP}"/generated/00000{0,1,2,3,4,5,6,7}.png \
    "$OUT_ROOT/direct_${STEP}_samples/" 2>/dev/null || true
done
rm -rf "$WORK"
