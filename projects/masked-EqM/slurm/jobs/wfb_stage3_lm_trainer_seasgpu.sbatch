#!/bin/bash
#SBATCH --job-name=fb-direct-wfb-stage3
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage3-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage3-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
set -euo pipefail

# WFB-EqM Stage 3, seas_gpu variant (single GPU, no --exclusive/--constraint --
# shares a partially-free 4-GPU node): identical to wfb_stage3_lm_trainer.sbatch,
# only the partition differs. Used to overflow past gpu_test's ~2-concurrent-job
# QOS cap (per AGENTS.md partition-diversification guidance) when gpu_test
# slots are occupied by longer-running jobs.

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT_DIRECT_LATE="${CKPT_DIRECT_LATE:?set CKPT_DIRECT_LATE}"
ALPHA="${ALPHA:?set ALPHA to 0.0, 0.5, or 1.0}"
RUN_TAG="${RUN_TAG:?set RUN_TAG}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/wfb_stage3}"
MAX_STEPS="${MAX_STEPS:-300}"
WFB_RHO="${WFB_RHO:-1e-4}"
WFB_K="${WFB_K:-96}"
ARMIJO_C1="${ARMIJO_C1:-1e-4}"
BACKTRACK_FACTOR="${BACKTRACK_FACTOR:-0.5}"
MAX_BACKTRACKS="${MAX_BACKTRACKS:-6}"
SOLVER="${SOLVER:-lanczos}"
CG_TOL="${CG_TOL:-1e-3}"
CG_MAX_ITERS="${CG_MAX_ITERS:-500}"
ADAPTIVE_DAMPING_FLAG=""
if [[ "${ADAPTIVE_DAMPING:-0}" == "1" ]]; then
  ADAPTIVE_DAMPING_FLAG="--adaptive-damping"
fi
WORK_DIR="/tmp/masked-eqm-wfbstage3-${SLURM_JOB_ID}"
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

echo "Stage 3 configuration: ALPHA=$ALPHA RUN_TAG=$RUN_TAG GLOBAL_SEED=$GLOBAL_SEED MAX_STEPS=$MAX_STEPS WFB_RHO=$WFB_RHO WFB_K=$WFB_K ARMIJO_C1=$ARMIJO_C1 BACKTRACK_FACTOR=$BACKTRACK_FACTOR MAX_BACKTRACKS=$MAX_BACKTRACKS SOLVER=$SOLVER CG_TOL=$CG_TOL CG_MAX_ITERS=$CG_MAX_ITERS RESULTS_DIR=$RESULTS_DIR"

python -u experiments/direct_energy/wfb_stage3_lm_trainer.py \
    --ckpt-direct "$CKPT_DIRECT_LATE" \
    --data-path "$IMAGENET_PATH" \
    --alpha "$ALPHA" \
    --max-steps "$MAX_STEPS" \
    --rho "$WFB_RHO" \
    --k "$WFB_K" \
    --armijo-c1 "$ARMIJO_C1" \
    --backtrack-factor "$BACKTRACK_FACTOR" \
    --max-backtracks "$MAX_BACKTRACKS" \
    --solver "$SOLVER" \
    --cg-tol "$CG_TOL" \
    --cg-max-iters "$CG_MAX_ITERS" \
    --seed "$GLOBAL_SEED" \
    --results-dir "$RESULTS_DIR" \
    --run-tag "$RUN_TAG" \
    $ADAPTIVE_DAMPING_FLAG 2>&1 | tee wfbstage3.log
EXIT_CODE=${PIPESTATUS[0]}
cp wfbstage3.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/${RUN_TAG}_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
