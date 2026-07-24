#!/bin/bash
#SBATCH --job-name=eqm-frozen-prior-smoke
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=0-8%2
#SBATCH --output=projects/masked-EqM/slurm/logs/frozen-prior-smoke-%A_%a.out
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true; module load cuda/11.8.0-fasrc01 2>/dev/null || true
WORK=/tmp/frozen-prior-smoke-${SLURM_JOB_ID}; mkdir -p "$WORK"; cd "$WORK"
git clone --depth=1 "${GIT_URL:?}" repo; cd repo; git fetch --depth=1 origin "${GIT_SHA:?}"; git checkout "$GIT_SHA"
cd projects/masked-EqM
ID=$(python -c "import json;print(json.load(open('configs/frozen_prior_constraint_checkpoints.json'))['arms'][$SLURM_ARRAY_TASK_ID]['id'])")
CKPT=$(python -c "import json;print(json.load(open('configs/frozen_prior_constraint_checkpoints.json'))['arms'][$SLURM_ARRAY_TASK_ID]['path'])")
python -m pip install --quiet timm diffusers accelerate torchvision lpips
python frozen_prior_constraint.py --checkpoint "$CKPT" --checkpoint-id "$ID" --data-path /n/holylabs/ydu_lab/Lab/raywang4/imagenet/val --sample-manifest configs/frozen_prior_constraint_pilot_locked_v4.json --manifest-split pilot --mask-family combined --projection-mode hard --steps 5 --num-shards 1 --output-dir /n/home03/mkrasnow/frozen-prior-smoke --strict --resume
