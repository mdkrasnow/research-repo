#!/bin/bash
#SBATCH --job-name=eqm-armijo-cal
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/armijo-cal-%A_%a.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/armijo-cal-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
MULTIPLIERS=(0.25 0.5 1 2 4 8)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
MULTIPLIER="${MULTIPLIERS[$TASK_ID]}"
WORK_DIR="/tmp/masked-eqm-armijo-cal-${ARRAY_JOB_ID}_${TASK_ID}"
RESULT_ROOT="/n/home03/mkrasnow/energy_aware_sampling/epoch15_armijo/calibration"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?GIT_SHA is required}"
CHECKPOINT="/n/home03/mkrasnow/direct_energy_longer_retry/longer15_direct_seed0_job35524667/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/epoch15.pt"
REAL="/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/val"
STEP="$(python -c "print(0.0017 * ${MULTIPLIER})")"
TAG="m${MULTIPLIER//./p}"
mkdir -p "$WORK_DIR" "$RESULT_ROOT"
cd "$WORK_DIR"; git init .; git remote add origin "$GIT_URL"; git fetch --depth=50 origin main; git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers torchvision pytorch-fid
python energy_aware_sampling/calibrate_fid.py --checkpoint "$CHECKPOINT" --real "$REAL" --output "$RESULT_ROOT/$TAG" --multiplier "$MULTIPLIER" --step "$STEP" --samples 1024 --steps 100 --batch-size 4 --fid-batch-size 16
cp "$RESULT_ROOT/$TAG/metrics.json" "/n/home03/mkrasnow/research-repo/projects/masked-EqM/results/energy_aware_calibration_${ARRAY_JOB_ID}_${TASK_ID}.json" || true
rm -rf "$WORK_DIR"
