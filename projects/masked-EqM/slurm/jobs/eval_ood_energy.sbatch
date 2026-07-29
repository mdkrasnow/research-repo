#!/bin/bash
#SBATCH --job-name=eqm-ood
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/ood-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/ood-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

WORK_DIR="/tmp/masked-eqm-ood-${SLURM_JOB_ID}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?}"
DOT_CKPT="${DOT_CKPT:-/n/home03/mkrasnow/direct_energy_longer_retry/longer15_dot_seed0_job35524601/000-EqM-B-2-Linear-velocity-None-ebm-dot/checkpoints/epoch15.pt}"
DIRECT_CKPT="${DIRECT_CKPT:-/n/home03/mkrasnow/direct_energy_longer_retry/longer15_direct_seed0_job35524667/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/epoch15.pt}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
DATA_ROOT="${DATA_ROOT:-/n/home03/mkrasnow/research-repo/data}"
OUT_DIR="${OUT_DIR:-/n/home03/mkrasnow/masked-EqM-ood}"
NUM_IMAGES="${NUM_IMAGES:-10000}"
mkdir -p "$WORK_DIR" "$DATA_ROOT" "$OUT_DIR"
cd "$WORK_DIR"
git init -q . && git remote add origin "$GIT_URL"
git fetch --depth=50 origin main && git checkout "$GIT_SHA"
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision scikit-learn

for arm in dot direct; do
  ckpt="${DOT_CKPT}"; [ "$arm" = direct ] && ckpt="${DIRECT_CKPT}"
  python eval_ood_energy.py --ckpt "$ckpt" --ebm "$arm" --model EqM-B/2 \
    --id-root "$VAL_PATH" --data-root "$DATA_ROOT" --ood cifar10 constant --num-images "$NUM_IMAGES" \
    --batch-size 8 --workers 4 --out "$OUT_DIR/epoch15_${arm}.json"
done
rm -rf "$WORK_DIR"
