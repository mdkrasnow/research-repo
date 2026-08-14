#!/bin/bash
#SBATCH --job-name=fb-direct-wfb-stage3a
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage3a-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage3a-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

# WFB-EqM Stage 3A: frozen-checkpoint mechanism discriminator.
#
# Decides H1 (Gauss-Newton LOCAL MODEL is wrong -> repair = LM damping) vs
# H2 (local model is right for the WRONG MINIBATCH -> repair = larger model
# batch / independent acceptance; damping cannot help) vs H3 (both), for the
# certified CG-FBGN algorithm whose 300-step run reduced every one of its own
# minibatches while driving the fixed held-out probe 10.625 -> 14.881.
#
# NO training, NO optimizer, NO parameter persistence: every candidate step is
# apply -> evaluate -> revert with bitwise restoration asserted. Single GPU.
#
# Checkpoints are passed as repeatable NAME=PATH pairs via CKPT_SPECS (space
# separated), so the same sbatch covers the start checkpoint and any of the
# 300-step run's saved step100/step200/step300 checkpoints.

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT_SPECS="${CKPT_SPECS:?set CKPT_SPECS to space-separated NAME=PATH pairs}"
RUN_TAG="${RUN_TAG:-wfb_stage3a}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/wfb_stage3a}"
# TRAIN_STEPS is the probe-slice offset: it MUST match the 300-step runs'
# --max-steps so that pool[TRAIN_STEPS:TRAIN_STEPS+8] is byte-identical to the
# probe those runs reported. Do not change it to "save time" -- that silently
# swaps the probe and destroys comparability with the trajectory being explained.
TRAIN_STEPS="${TRAIN_STEPS:-300}"
N_MODEL_BATCHES="${N_MODEL_BATCHES:-8}"
N_TRUST_BATCHES="${N_TRUST_BATCHES:-8}"
N_PROBE_BATCHES="${N_PROBE_BATCHES:-8}"
WFB_RHO="${WFB_RHO:-1e-4}"
CG_TOL="${CG_TOL:-1e-2}"
CG_MAX_ITERS="${CG_MAX_ITERS:-500}"
ETAS="${ETAS:-1.0 0.5 0.25 0.125}"
LAMBDA_SWEEP="${LAMBDA_SWEEP:-10 100}"
LAMBDA_SWEEP_BATCHES="${LAMBDA_SWEEP_BATCHES:-4}"

WORK_DIR="/tmp/masked-eqm-wfbstage3a-${SLURM_JOB_ID}"
RESULTS_DIR="$RESULTS_ROOT/${RUN_TAG}_job${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$RESULTS_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision

echo "=== FP64 CPU unit tests (must pass before any GPU work) ==="
python -u tests/test_fb_direct_exact_hvp.py
echo "=== FP64 CPU Stage 3A metric tests (apply/revert exactness, D_B order, R_B->1, d_V vs finite difference) ==="
python -u tests/test_stage3a_discriminator_metrics.py

CKPT_ARGS=""
for spec in $CKPT_SPECS; do CKPT_ARGS="$CKPT_ARGS --ckpt $spec"; done

echo "Stage 3A configuration: RUN_TAG=$RUN_TAG SEED=$GLOBAL_SEED TRAIN_STEPS=$TRAIN_STEPS RHO=$WFB_RHO CG_TOL=$CG_TOL ETAS='$ETAS' LAMBDA_SWEEP='$LAMBDA_SWEEP' RESULTS_DIR=$RESULTS_DIR"
echo "Checkpoints: $CKPT_SPECS"

python -u experiments/direct_energy/wfb_stage3a_mechanism_discriminator.py \
  $CKPT_ARGS \
  --data-path "$IMAGENET_PATH" \
  --seed "$GLOBAL_SEED" \
  --train-steps "$TRAIN_STEPS" \
  --n-model-batches "$N_MODEL_BATCHES" \
  --n-trust-batches "$N_TRUST_BATCHES" \
  --n-probe-batches "$N_PROBE_BATCHES" \
  --rho "$WFB_RHO" \
  --cg-tol "$CG_TOL" \
  --cg-max-iters "$CG_MAX_ITERS" \
  --etas $ETAS \
  --lambda-sweep $LAMBDA_SWEEP \
  --lambda-sweep-batches "$LAMBDA_SWEEP_BATCHES" \
  --results-dir "$RESULTS_DIR" \
  --run-tag "$RUN_TAG"

echo "Stage 3A done -> $RESULTS_DIR"
cd /
rm -rf "$WORK_DIR"
