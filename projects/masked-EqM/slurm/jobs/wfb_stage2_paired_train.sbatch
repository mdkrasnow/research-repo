#!/bin/bash
#SBATCH --job-name=eqm-wfb-stage2
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage2-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage2-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320GB
#SBATCH --time=04:00:00
#SBATCH --partition=seas_gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:4
#SBATCH --exclusive
set -euo pipefail

# WFB-EqM Stage 2: very short paired training test (spec Section 9). Copied from the
# production direct_energy_ddp_continue.sbatch template so ARM A (exact-fwrev) and ARM B
# (wfb-backward) share IDENTICAL infra (nproc_per_node, global-batch-size, data pipeline,
# checkpoint-loading path) -- only the backward-mode flags + RUN_TAG differ between the two
# submissions. GLOBAL_SEED must be passed IDENTICAL across both arms for RNG/data-ordering
# parity (spec: "Initialize BOTH arms from the exact same pre-instability checkpoint...
# RNG seeds, data ordering... identical").

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
GLOBAL_SEED="${GLOBAL_SEED:?set GLOBAL_SEED}"
CKPT="${CKPT:?set CKPT}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
ARM="${ARM:?set ARM to 'exact' or 'wfb'}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-6.87141}"
GRAD_LOG_EVERY="${GRAD_LOG_EVERY:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_STEPS="${MAX_STEPS:-500}"
WFB_RHO="${WFB_RHO:-1e-4}"
WFB_K="${WFB_K:-12}"
export ENTITY="${ENTITY:-none}"
export PROJECT="${PROJECT:-none}"
WORK_DIR="/tmp/masked-eqm-wfbstage2-${SLURM_JOB_ID}"
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

echo "=== FP64 CPU unit tests (must pass before any GPU work) ==="
python -u tests/test_fb_direct_exact_hvp.py

CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT train.py --model EqM-B/2 --data-path $IMAGENET_PATH --num-classes 1000 --global-batch-size 32 --epochs 999 --image-size 256 --log-every 10 --ckpt-every 100000 --results-dir $RESULTS_DIR --ebm direct --global-seed $GLOBAL_SEED --ckpt $CKPT --num-workers 4 --grad-log-every $GRAD_LOG_EVERY --weight-decay $WEIGHT_DECAY --max-grad-norm $MAX_GRAD_NORM --max-steps $MAX_STEPS"
if [[ "$ARM" == "exact" ]]; then
  CMD+=" --exact-fwrev"
elif [[ "$ARM" == "wfb" ]]; then
  CMD+=" --wfb-backward --wfb-rho $WFB_RHO --wfb-k $WFB_K"
else
  echo "ARM must be 'exact' or 'wfb', got: $ARM" >&2
  exit 2
fi
echo "Stage 2 configuration: ARM=$ARM GLOBAL_SEED=$GLOBAL_SEED CKPT=$CKPT MAX_STEPS=$MAX_STEPS MAX_GRAD_NORM=$MAX_GRAD_NORM WEIGHT_DECAY=$WEIGHT_DECAY GRAD_LOG_EVERY=$GRAD_LOG_EVERY WFB_RHO=$WFB_RHO WFB_K=$WFB_K RESULTS_DIR=$RESULTS_DIR"
echo "$CMD"
eval "$CMD" 2>&1 | tee train.log
EXIT_CODE=${PIPESTATUS[0]}
cp train.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
