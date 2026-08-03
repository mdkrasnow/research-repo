#!/bin/bash
#SBATCH --job-name=eqm-mask-smoke
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/masked-field-smoke-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/masked-field-smoke-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CONFIG_REL="${CONFIG_REL:?set CONFIG_REL}"
RESUME_CKPT="${RESUME_CKPT:-}"
WORK_DIR="/tmp/masked-eqm-field-smoke-${SLURM_JOB_ID}"
MASTER_PORT=$((29500 + RANDOM % 1000))
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=80 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchdiffeq wandb torchvision

COMMAND=(python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port="$MASTER_PORT" -m masked_field_shaping.train_continuation --config "$WORK_DIR/$CONFIG_REL")
if [[ -n "$RESUME_CKPT" ]]; then
  COMMAND+=(--resume "$RESUME_CKPT")
fi
"${COMMAND[@]}"
