#!/bin/bash
#SBATCH --job-name=eqm-armijo-audit
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/armijo-audit-%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/armijo-audit-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
WORK_DIR="/tmp/masked-eqm-armijo-audit-${SLURM_JOB_ID}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
RESULT_ROOT="/n/home03/mkrasnow/energy_aware_sampling/epoch15_armijo"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?GIT_SHA is required}"
DIRECT_CKPT="/n/home03/mkrasnow/direct_energy_longer_retry/longer15_direct_seed0_job35524667/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/epoch15.pt"
DOT_CKPT="/n/home03/mkrasnow/direct_energy_longer_retry/longer15_dot_seed0_job35524601/000-EqM-B-2-Linear-velocity-None-ebm-dot/checkpoints/epoch15.pt"

mkdir -p "$WORK_DIR" "$RESULT_ROOT"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers torchvision
python tests/test_energy_only.py
python energy_aware_sampling/evaluate.py \
  --mode audit --samples 32 --batch-size 4 --steps 4 --initial-step 0.003 \
  --direct "$DIRECT_CKPT" --dot "$DOT_CKPT" \
  --output "$RESULT_ROOT/sign_audit_${SLURM_JOB_ID}.json"
cp "$RESULT_ROOT/sign_audit_${SLURM_JOB_ID}.json" "$REPO_ROOT/projects/masked-EqM/results/energy_aware_sampling_sign_audit_${SLURM_JOB_ID}.json" || true
rm -rf "$WORK_DIR"
