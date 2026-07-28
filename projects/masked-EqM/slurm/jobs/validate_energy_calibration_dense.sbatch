#!/bin/bash
#SBATCH --job-name=eqm-energy-dense-validation
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-dense-%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/energy-dense-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
GIT_SHA="${GIT_SHA:?set GIT_SHA}"
WORK_DIR="$(mktemp -d /tmp/masked-eqm-energy-dense-${SLURM_JOB_ID}-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT
git -C "$WORK_DIR" init; git -C "$WORK_DIR" remote add origin "${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
git -C "$WORK_DIR" fetch --depth=50 origin main; git -C "$WORK_DIR" checkout "$GIT_SHA"
cd "$WORK_DIR/projects/masked-EqM"
pip install --quiet timm diffusers accelerate torchdiffeq torchvision pandas pyarrow matplotlib seaborn
python -m energy_monotonicity.validate_energy_calibration_dense \
  --monotonicity-output-dir /n/home03/mkrasnow/energy_monotonicity_full \
  --output-dir /n/home03/mkrasnow/energy_calibration_dense_validation --subset 256 --batch-size 8 --device cuda
