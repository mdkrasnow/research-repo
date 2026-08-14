#!/bin/bash
#SBATCH --job-name=fb-direct-wfb-stage2-6a
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage2-6a-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage2-6a-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

# WFB-EqM Stage 2.6a: FBGN Lanczos residual-certificate k-convergence check
# (reviewer derivation, 2026-08-12), reconciling Stage 2.5's >1 induced-field-
# gain measurements at alpha=1/k=12. No training/weight updates persisted --
# same zero-update causal-diagnostic design as Stage 1/2.5. Confirms
# rho_m = -ell_m and ||q_m||/||r|| <= 1 + ||rho_m||/||r|| hold on the real
# model at k in {12,24,48,96}, and reports the field-gain trend vs k.

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT_DIRECT_LATE="${CKPT_DIRECT_LATE:?set CKPT_DIRECT_LATE}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
OUT_NAME="${OUT_NAME:-wfb_stage2_6_residual_certificate}"
WORK_DIR="/tmp/masked-eqm-wfbstage26a-${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision

echo "=== FP64 CPU unit tests (must pass before any GPU work, incl. residual-certificate test) ==="
python -u tests/test_fb_direct_exact_hvp.py

python -u experiments/direct_energy/wfb_stage2_6_residual_certificate.py \
    --ckpt-direct "$CKPT_DIRECT_LATE" \
    --data-path "$IMAGENET_PATH" \
    --batch-size "${BATCH_SIZE:-8}" \
    --pool-size "${POOL_SIZE:-256}" \
    --spike-frac "${SPIKE_FRAC:-0.06}" \
    --num-control "${NUM_CONTROL:-8}" \
    --sweep-n "${SWEEP_N:-4}" \
    --rho "${RHO:-1e-4}" \
    --lambda-max-num-iters "${LAMBDA_MAX_ITERS:-20}" \
    --out "$OUT_DIR/${OUT_NAME}_${SLURM_JOB_ID}.json" 2>&1 | tee wfbstage26a.log
EXIT_CODE=${PIPESTATUS[0]}
cp wfbstage26a.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb_stage2_6_residual_certificate_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
