#!/bin/bash
#SBATCH --job-name=eqm-mask-eval
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/masked-field-eval-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/masked-field-eval-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CONFIG_REL="${CONFIG_REL:?set CONFIG_REL}"
TASK="${TASK:?set TASK to bank, recovery, generation, shared_d0, apply_shared_d0, or aggregate}"
WORK_DIR="/tmp/masked-eqm-field-eval-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=80 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
python -m pip install --quiet \
  timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' \
  accelerate torchdiffeq wandb torchvision lpips pytorch-fid scipy pandas matplotlib tabulate

case "$TASK" in
  bank) python -m masked_field_shaping.build_evaluation_bank --config "$WORK_DIR/$CONFIG_REL" ;;
  recovery) python -m masked_field_shaping.evaluate_recovery --config "$WORK_DIR/$CONFIG_REL" ;;
  generation) python -m masked_field_shaping.evaluate_generation --config "$WORK_DIR/$CONFIG_REL" ;;
  shared_d0) python -m masked_field_shaping.shared_d0 --config "$WORK_DIR/$CONFIG_REL" ;;
  apply_shared_d0) python -m masked_field_shaping.apply_shared_d0 --config "$WORK_DIR/$CONFIG_REL" ;;
  aggregate) python -m masked_field_shaping.aggregate_results --manifest "$WORK_DIR/$CONFIG_REL" ;;
  *) echo "unknown TASK=$TASK" >&2; exit 2 ;;
esac
