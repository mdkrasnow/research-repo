#!/bin/bash
#SBATCH --job-name=eqm-state-delta
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/state-delta-%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/state-delta-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --time=01:00:00
#SBATCH --partition=shared
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
GIT_SHA="${GIT_SHA:?set GIT_SHA}"
ROOT="${CHECKPOINT_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer40_direct_seed0_ckpt50k_job36359213/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints}"
OUT_ROOT="${OUT_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_checkpoint_forensics}"
WORK="/tmp/eqm-state-delta-${SLURM_JOB_ID}"
mkdir -p "$WORK" "$OUT_ROOT"
cd "$WORK"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm torchvision
python experiments/direct_energy/checkpoint_state_deltas.py \
  --ckpt "$ROOT/1450000.pt" --ckpt "$ROOT/1500000.pt" \
  --ckpt "$ROOT/1550000.pt" --ckpt "$ROOT/1600000.pt" \
  --output "$OUT_ROOT/state_deltas.json"
rm -rf "$WORK"
