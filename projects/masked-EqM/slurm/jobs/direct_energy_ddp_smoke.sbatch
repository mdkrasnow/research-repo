#!/bin/bash
#SBATCH --job-name=eqm-ddp-smoke
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/ddp-smoke-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/ddp-smoke-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320GB
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_requeue
#SBATCH --constraint=h200
#SBATCH --gres=gpu:4
#SBATCH --exclusive
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
EBM="${EBM:?set EBM}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
CKPT="${CKPT:?set CKPT}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/home03/mkrasnow/direct_energy_ddp_smoke}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
WORK_DIR="/tmp/masked-eqm-ddp-${SLURM_JOB_ID}"
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

CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT train.py --model EqM-B/2 --data-path $IMAGENET_PATH --num-classes 1000 --global-batch-size 32 --epochs 1 --save-epochs 999 --image-size 256 --log-every 50 --ckpt-every 999999999 --max-steps 200 --results-dir $RESULTS_DIR --ebm $EBM --global-seed $GLOBAL_SEED --ckpt $CKPT --num-workers 4"
echo "$CMD"
eval "$CMD" 2>&1 | tee ddp_smoke.log
cp ddp_smoke.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
