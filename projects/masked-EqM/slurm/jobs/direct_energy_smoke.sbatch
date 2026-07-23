#!/bin/bash
#SBATCH --job-name=eqm-direct-energy-smoke
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-energy-smoke-%j.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-energy-smoke-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# GPU implementation smoke only: no dataset, checkpoints, or training metric.
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

WORK_DIR="/tmp/masked-eqm-direct-smoke-${SLURM_JOB_ID}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?GIT_SHA must be set}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }

cd projects/masked-EqM
pip install --quiet timm diffusers torchvision
python tests/test_scalar_ebm.py 2>&1 | tee direct_energy_smoke.log
python tests/test_direct_eval_samplers.py 2>&1 | tee -a direct_energy_smoke.log

mkdir -p "$REPO_ROOT/projects/masked-EqM/slurm/logs"
cp direct_energy_smoke.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/direct_energy_smoke_${SLURM_JOB_ID}.log"
rm -rf "$WORK_DIR"
