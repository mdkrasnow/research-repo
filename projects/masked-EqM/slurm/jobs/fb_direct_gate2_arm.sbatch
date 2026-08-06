#!/bin/bash
#SBATCH --job-name=fb-direct-gate2
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb-direct-gate2-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb-direct-gate2-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=02:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
EBM="${EBM:?set EBM}"
GLOBAL_SEED="${GLOBAL_SEED:?set GLOBAL_SEED}"
CKPT="${CKPT:?set CKPT}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
MAX_STEPS="${MAX_STEPS:-2000}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-6.87141}"
GRAD_LOG_EVERY="${GRAD_LOG_EVERY:-200}"
FB_EXACT_AUDIT_EVERY="${FB_EXACT_AUDIT_EVERY:-200}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/fb_direct_gate2}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
WORK_DIR="/tmp/masked-eqm-fbgate2-${SLURM_JOB_ID}"
RESULTS_DIR="$RESULTS_ROOT/${RUN_TAG}_job${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$RESULTS_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchdiffeq wandb torchvision

CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=$((29500 + RANDOM % 1000)) train.py --model EqM-B/2 --data-path $IMAGENET_PATH --num-classes 1000 --global-batch-size 32 --epochs 999 --max-steps $MAX_STEPS --image-size 256 --log-every 50 --ckpt-every 100000 --results-dir $RESULTS_DIR --ebm $EBM --global-seed $GLOBAL_SEED --ckpt $CKPT --num-workers 4 --grad-log-every $GRAD_LOG_EVERY --max-grad-norm $MAX_GRAD_NORM"
if [[ "$EBM" == "forward-backwards-direct" ]]; then
  CMD+=" --fb-exact-audit-every $FB_EXACT_AUDIT_EVERY"
fi
echo "Gate 2 arm configuration: EBM=$EBM MAX_STEPS=$MAX_STEPS MAX_GRAD_NORM=$MAX_GRAD_NORM GLOBAL_SEED=$GLOBAL_SEED CKPT=$CKPT RESULTS_DIR=$RESULTS_DIR"
echo "$CMD"
eval "$CMD" 2>&1 | tee train.log
EXIT_CODE=${PIPESTATUS[0]}
cp train.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
