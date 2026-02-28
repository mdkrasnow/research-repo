#!/bin/bash
#SBATCH --job-name=ired_q239_100k
#SBATCH --output=projects/ired/slurm/logs/%x_%j_%a.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-7%2

set -euo pipefail

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

WORK_DIR="${TMPDIR:-/tmp}/ired-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

nvidia-smi

git clone "$GIT_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout "$GIT_SHA"

python projects/ired/experiments/matrix_inversion_mining.py \
    --config projects/ired/configs/q239_trajectory_cvar_100k.json \
    --seed $SLURM_ARRAY_TASK_ID

REPO_ROOT="/n/home03/mkrasnow/research-repo"
rsync -av "$WORK_DIR/projects/ired/results/" "$REPO_ROOT/projects/ired/results/" \
    --include='*/' --include='*.pt' --include='*.json' --include='*.md' --exclude='*'

cd /
rm -rf "$WORK_DIR"
