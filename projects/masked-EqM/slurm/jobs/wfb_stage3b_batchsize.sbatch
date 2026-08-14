#!/bin/bash
#SBATCH --job-name=fb-direct-wfb-stage3b
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage3b-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage3b-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

# WFB-EqM Stage 3B: the batch-size discriminator for H2.
#
# Stage 3A (job 38985448) classified the CG-FBGN failure at the healthy `start`
# checkpoint: the certified direction is essentially ORTHOGONAL to the
# independent-batch gradient (C_V = 0.00108) and is an ascent direction on 4/8
# batches (d_V is eta-INDEPENDENT, so no step size and no damping can fix it).
# The pre-registered H1 repair -- damping -- fixed the local model (R_B -10.7 ->
# +0.15, D_B 3.40 -> 0.91) while driving C_V NEGATIVE, i.e. it failed its own
# pre-registered falsifier.
#
# The only remaining pre-registered intervention is the H2 branch: enlarge the
# stochastic model batch. This job tests its PREREQUISITE first and cheaply --
# does the plain minibatch gradient's alignment with the population gradient
# actually rise with B, faster than the sqrt(B) that pure noise-averaging gives?
# -- and only then pays for the stacked GN solve at B_model = 32.
#
# NO training, NO optimizer, NO parameter persistence. The gradient scan never
# touches theta; the stacked-GN arm uses apply -> evaluate -> revert with bitwise
# restoration asserted. The deterministic global probe is evaluated for REPORTING
# ONLY -- never for acceptance, damping, step selection, or retries.

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT="${CKPT:?set CKPT to the start checkpoint}"
RUN_TAG="${RUN_TAG:-wfb_stage3b}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/wfb_stage3b}"
# TRAIN_STEPS is the bank offset. It MUST stay 300 so pool[300:308] is the
# byte-identical probe the 300-step runs reported and pool[308:324] are Stage
# 3A's untouched B/V banks. Changing it silently swaps every bank.
TRAIN_STEPS="${TRAIN_STEPS:-300}"
N_REF="${N_REF:-32}"
BATCH_SIZES="${BATCH_SIZES:-8,16,32,64,128}"
N_REP="${N_REP:-8}"
STACKED_GN="${STACKED_GN:-32}"
N_GN_BATCHES="${N_GN_BATCHES:-4}"
WFB_RHO="${WFB_RHO:-1e-4}"
# Stage 3A showed the 300-step run's 1e-2 tolerance bought ~200 CG iterations of
# fidelity to a system that is ~99.9% batch-idiosyncratic. Per the directive's
# "CG should be INEXACT" point, start at a true relative residual of ~0.05.
CG_TOL="${CG_TOL:-5e-2}"
CG_MAX_ITERS="${CG_MAX_ITERS:-250}"

WORK_DIR="/tmp/masked-eqm-wfbstage3b-${SLURM_JOB_ID}"
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
echo "=== FP64 CPU Stage 3A metric tests ==="
python -u tests/test_stage3a_discriminator_metrics.py
echo "=== FP64 CPU stacked-microbatch GN tests (stacked A == explicit M M^T; J=1 reduces"
echo "    to mixed_gram_mv; solve satisfies (A+lam I)u=r; and stacked != average-of-solves) ==="
python -u tests/test_stacked_microbatch_gn.py

echo "Stage 3B configuration: RUN_TAG=$RUN_TAG SEED=$GLOBAL_SEED TRAIN_STEPS=$TRAIN_STEPS N_REF=$N_REF BATCH_SIZES=$BATCH_SIZES N_REP=$N_REP STACKED_GN=$STACKED_GN CG_TOL=$CG_TOL RESULTS_DIR=$RESULTS_DIR"
echo "Checkpoint: $CKPT"

python -u experiments/direct_energy/wfb_stage3b_batchsize_alignment.py \
  --ckpt "$CKPT" \
  --data-path "$IMAGENET_PATH" \
  --seed "$GLOBAL_SEED" \
  --train-steps "$TRAIN_STEPS" \
  --n-ref "$N_REF" \
  --batch-sizes "$BATCH_SIZES" \
  --n-rep "$N_REP" \
  --stacked-gn "$STACKED_GN" \
  --n-gn-batches "$N_GN_BATCHES" \
  --rho "$WFB_RHO" \
  --cg-tol "$CG_TOL" \
  --cg-max-iters "$CG_MAX_ITERS" \
  --out-dir "$RESULTS_DIR"

echo "Stage 3B done -> $RESULTS_DIR"
cd /
rm -rf "$WORK_DIR"
