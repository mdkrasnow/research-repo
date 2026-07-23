#!/bin/bash
#SBATCH --job-name=eqm-direct-sweep
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-sweep-%A_%a.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-sweep-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

ARMS=(none dot direct)
EBM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
RUN_TAG="${RUN_TAG:?set RUN_TAG, e.g. stepsize_0p0015}"
STEP_SIZE="${STEP_SIZE:-0.003}"
STEPS="${STEPS:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
WORK_DIR="/tmp/masked-eqm-sweep-${SLURM_JOB_ID}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?}"
CKPT="$REPO_ROOT/projects/masked-EqM/results/direct_energy_campaign/stage3_pilot_v2/${EBM}_seed0/step1000.pt"
RESULTS="results/direct_energy_campaign/sampler_sweeps/${RUN_TAG}/${EBM}_seed0"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers torchvision
python experiments/direct_energy/sample_probe.py \
  --ckpt "$CKPT" --output "$RESULTS" --ebm "$EBM" \
  --stepsize "$STEP_SIZE" --steps "$STEPS" --num-samples "$NUM_SAMPLES" \
  2>&1 | tee sweep.log
mkdir -p "$REPO_ROOT/projects/masked-EqM/results/direct_energy_campaign/sampler_sweeps/${RUN_TAG}"
rsync -a "$RESULTS/" "$REPO_ROOT/projects/masked-EqM/$RESULTS/"
cp sweep.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/direct_sweep_${RUN_TAG}_${EBM}_${SLURM_JOB_ID}.log"
rm -rf "$WORK_DIR"
