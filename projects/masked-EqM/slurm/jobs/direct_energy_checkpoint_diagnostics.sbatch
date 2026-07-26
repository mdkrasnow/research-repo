#!/bin/bash
#SBATCH --job-name=eqm-direct-diag
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/direct-diag-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/direct-diag-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi
WORK_DIR="/tmp/masked-eqm-diag-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
GIT_SHA="${GIT_SHA:?}"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision
spec_args=()
IFS=';' read -ra specs <<< "${CKPT_SPECS:?}"
for spec in "${specs[@]}"; do spec_args+=(--ckpt "$spec"); done
python experiments/direct_energy/checkpoint_diagnostics.py \
  --data-path /n/holylabs/ydu_lab/Lab/raywang4/imagenet/train \
  --output "${OUT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results/direct_energy_campaign/training_diagnostics/checkpoint_metrics.jsonl}" \
  "${spec_args[@]}"
rm -rf "$WORK_DIR"
