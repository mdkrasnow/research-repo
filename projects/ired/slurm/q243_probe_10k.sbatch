#!/bin/bash
#SBATCH --job-name=ired_q243_10k
#SBATCH --output=projects/ired/slurm/logs/%x_%j.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

WORK_DIR="${TMPDIR:-/tmp}/ired-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

nvidia-smi  # verify GPU

git clone "$GIT_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout "$GIT_SHA"

python projects/ired/experiments/matrix_inversion_mining.py \
    --config projects/ired/configs/q243_hard_state_traj_div_diag_10k.json \
    --seed 0

# Rsync results back to persistent storage
REPO_ROOT="/n/home03/mkrasnow/research-repo"
rsync -av "$WORK_DIR/projects/ired/results/" "$REPO_ROOT/projects/ired/results/" \
    --include='*/' --include='*.pt' --include='*.json' --include='*.md' --exclude='*'

cd /
rm -rf "$WORK_DIR"
