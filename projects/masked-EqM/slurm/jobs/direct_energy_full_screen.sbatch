#!/bin/bash
#SBATCH --job-name=eqm-direct-full
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-full-%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-full-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi || exit 1

WORK_DIR="/tmp/masked-eqm-direct-full-${SLURM_JOB_ID}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?}"
EBM="${EBM:?set EBM to none, dot, or direct}"
GLOBAL_SEED="${GLOBAL_SEED:?set GLOBAL_SEED}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results/direct_energy_full}"
RESULTS_DIR="$RESULTS_ROOT/${RUN_TAG}_job${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$RESULTS_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchdiffeq wandb torchvision

export ENTITY="${ENTITY:-none}"
export PROJECT="${PROJECT:-none}"
MASTER_PORT=$((29500 + RANDOM % 1000))
TRAIN_CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=$MASTER_PORT train.py --model EqM-B/2 --data-path $IMAGENET_PATH --num-classes 1000 --global-batch-size 32 --epochs 1 --image-size 256 --log-every 50 --ckpt-every 999999999 --results-dir $RESULTS_DIR --ebm $EBM --global-seed $GLOBAL_SEED"
echo "$TRAIN_CMD"
eval "$TRAIN_CMD" 2>&1 | tee train.log
EXIT_CODE=${PIPESTATUS[0]}
cp train.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
