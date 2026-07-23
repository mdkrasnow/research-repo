#!/bin/bash
#SBATCH --job-name=eqm-direct-offtraj
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-offtraj-%A_%a.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-offtraj-%A_%a.err
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
WORK_DIR="/tmp/masked-eqm-offtraj-${SLURM_JOB_ID}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?}"
CKPT="$REPO_ROOT/projects/masked-EqM/results/direct_energy_campaign/stage3_pilot_v2/${EBM}_seed0/step1000.pt"
RESULTS="results/direct_energy_campaign/stage8_off_trajectory/${EBM}_seed0.json"
mkdir -p "$WORK_DIR"; cd "$WORK_DIR"; git init .; git remote add origin "$GIT_URL"; git fetch --depth=50 origin main; git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }; cd projects/masked-EqM
pip install --quiet timm diffusers torchvision
python experiments/direct_energy/off_trajectory.py --data-path "${IMAGENET_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}" --ckpt "$CKPT" --output "$RESULTS" --ebm "$EBM" 2>&1 | tee offtraj.log
mkdir -p "$REPO_ROOT/projects/masked-EqM/results/direct_energy_campaign/stage8_off_trajectory"; cp "$RESULTS" "$REPO_ROOT/projects/masked-EqM/$RESULTS"; cp offtraj.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/direct_offtraj_${EBM}_${SLURM_JOB_ID}.log"; rm -rf "$WORK_DIR"
