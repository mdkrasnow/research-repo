#!/bin/bash
#SBATCH --job-name=btm-image
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-image-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-image-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4
set -euo pipefail

# Experiment II: corrected-BTM arms on the real ImageNet B/2 EqM recipe.
#
# One arm per job.  Required env:
#   GIT_SHA     commit to run
#   BTM_MODE    btm_vector | btm_scalar_exact | btm_scalar_action_exact
#               | btm_scalar_fd_directional | btm_scalar_fd_action
#   EBM         none (vector arm) | direct (every scalar arm)
#   GLOBAL_SEED seed
#   RUN_TAG     human tag for the results dir
# Optional: EPOCHS SAVE_EPOCHS BTM_TC FD_EPS FD_K MAX_STEPS CKPT NPROC
#
# multi-GPU DDP => seas_gpu (full A100/H200).  gpu_test MIG slices cannot do
# multi-rank NCCL ("Duplicate GPU detected"); use NPROC=1 there.

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
REPO_ROOT="/n/home03/mkrasnow/research-repo"
BTM_MODE="${BTM_MODE:?set BTM_MODE (use 'none' for the vanilla-EqM control path)}"
EBM="${EBM:?set EBM}"
GLOBAL_SEED="${GLOBAL_SEED:?set GLOBAL_SEED}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
EPOCHS="${EPOCHS:-80}"
SAVE_EPOCHS="${SAVE_EPOCHS:-1,2,5,10,20,40,80}"
BTM_TC="${BTM_TC:-0.8}"
BTM_INTERPOLANT="${BTM_INTERPOLANT:-self_stopping}"
FD_EPS="${FD_EPS:-1e-3}"
FD_K="${FD_K:-1}"
NPROC="${NPROC:-4}"
LOG_EVERY="${LOG_EVERY:-50}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MAX_STEPS_ARG=""
if [ -n "${MAX_STEPS:-}" ]; then MAX_STEPS_ARG="--max-steps ${MAX_STEPS}"; fi
CKPT_ARG=""
if [ -n "${CKPT:-}" ]; then CKPT_ARG="--ckpt ${CKPT}"; fi

IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/btm}"
RESULTS_DIR="$RESULTS_ROOT/${RUN_TAG}_job${SLURM_JOB_ID}"
WORK_DIR="/tmp/btm-image-${SLURM_JOB_ID}"
export ENTITY="${ENTITY:-none}" PROJECT="${PROJECT:-none}"

mkdir -p "$WORK_DIR" "$RESULTS_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin '+refs/heads/*:refs/remotes/origin/*'
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchdiffeq wandb torchvision pytest

# Hard gate: the BTM mathematics must be validated on THIS commit before any
# GPU-hours are spent on it.
python -m pytest tests/test_btm_math.py tests/test_btm_image_conventions.py -q

# BTM_MODE=none runs the repository's ORIGINAL training path untouched -- the
# control that separates "our BTM branch hangs" from "4-GPU DDP hangs here".
if [ "$BTM_MODE" = "none" ]; then
  BTM_ARGS=""
else
  BTM_ARGS="--btm-mode $BTM_MODE --btm-interpolant $BTM_INTERPOLANT --btm-tc $BTM_TC --fd-eps $FD_EPS --fd-k $FD_K --btm-eval-every ${BTM_EVAL_EVERY:-500}"
fi

MASTER_PORT=$((29500 + RANDOM % 1000))
CMD="python -m torch.distributed.run --nnodes=1 --nproc_per_node=$NPROC --master_port=$MASTER_PORT train.py \
  --model EqM-B/2 --data-path $IMAGENET_PATH --num-classes 1000 \
  --global-batch-size $GLOBAL_BATCH --epochs $EPOCHS --save-epochs $SAVE_EPOCHS \
  --image-size 256 --log-every $LOG_EVERY --ckpt-every 999999999 \
  --results-dir $RESULTS_DIR --ebm $EBM --global-seed $GLOBAL_SEED \
  --grad-log-every ${GRAD_LOG_EVERY:-50} $BTM_ARGS $MAX_STEPS_ARG $CKPT_ARG"
echo "$CMD"
eval "$CMD" 2>&1 | tee train.log
EXIT_CODE=${PIPESTATUS[0]}
cp train.log "$REPO_ROOT/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
