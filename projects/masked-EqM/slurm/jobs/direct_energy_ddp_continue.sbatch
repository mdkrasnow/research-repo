#!/bin/bash
#SBATCH --job-name=eqm-ddp-cont
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/ddp-cont-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/ddp-cont-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320GB
#SBATCH --time=3-00:00:00
#SBATCH --partition=seas_gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:4
#SBATCH --exclusive
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
EBM="${EBM:?set EBM}"
GLOBAL_SEED="${GLOBAL_SEED:?set GLOBAL_SEED}"
CKPT="${CKPT:?set CKPT}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
EPOCHS="${1:-30}"
SAVE_EPOCHS="${2:-30}"
CKPT_EVERY="${CKPT_EVERY:-50000}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-}"
GRAD_LOG_EVERY="${GRAD_LOG_EVERY:-0}"
export ENTITY="${ENTITY:-none}"
export PROJECT="${PROJECT:-none}"
WORK_DIR="/tmp/masked-eqm-ddp-cont-${SLURM_JOB_ID}"
RESULTS_DIR="$RESULTS_ROOT/${RUN_TAG}_job${SLURM_JOB_ID}"
MASTER_PORT=$((29500 + RANDOM % 1000))

mkdir -p "$WORK_DIR" "$RESULTS_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchdiffeq wandb torchvision

CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT train.py --model EqM-B/2 --data-path $IMAGENET_PATH --num-classes 1000 --global-batch-size 32 --epochs $EPOCHS --save-epochs $SAVE_EPOCHS --image-size 256 --log-every 50 --ckpt-every $CKPT_EVERY --results-dir $RESULTS_DIR --ebm $EBM --global-seed $GLOBAL_SEED --ckpt $CKPT --num-workers 4 --grad-log-every $GRAD_LOG_EVERY"
if [[ -n "$MAX_GRAD_NORM" ]]; then
  CMD+=" --max-grad-norm $MAX_GRAD_NORM"
fi
echo "Continuation configuration: EPOCHS=$EPOCHS SAVE_EPOCHS=$SAVE_EPOCHS CKPT_EVERY=$CKPT_EVERY CKPT=$CKPT RESULTS_DIR=$RESULTS_DIR MAX_GRAD_NORM=${MAX_GRAD_NORM:-disabled} GRAD_LOG_EVERY=$GRAD_LOG_EVERY"
echo "$CMD"
eval "$CMD" 2>&1 | tee train.log
EXIT_CODE=${PIPESTATUS[0]}
cp train.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
