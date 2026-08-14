#!/bin/bash
#SBATCH --job-name=fb-direct-longvjp
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/longvjp-diag-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/longvjp-diag-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT_DIRECT_EARLY="${CKPT_DIRECT_EARLY:?set CKPT_DIRECT_EARLY}"
CKPT_DIRECT_MID="${CKPT_DIRECT_MID:?set CKPT_DIRECT_MID}"
CKPT_DIRECT_LATE="${CKPT_DIRECT_LATE:?set CKPT_DIRECT_LATE}"
CKPT_NONE_EARLY="${CKPT_NONE_EARLY:?set CKPT_NONE_EARLY}"
CKPT_NONE_MID="${CKPT_NONE_MID:?set CKPT_NONE_MID}"
CKPT_NONE_LATE="${CKPT_NONE_LATE:?set CKPT_NONE_LATE}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
OUT_NAME="${OUT_NAME:-longitudinal_jacobian_diagnostic}"
WORK_DIR="/tmp/masked-eqm-longvjpdiag-${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision

echo "=== FP64 CPU unit tests (must pass before any GPU work) ==="
python tests/test_fb_direct_exact_hvp.py

python experiments/direct_energy/longitudinal_jacobian_diagnostic.py \
    --ckpt-direct "$CKPT_DIRECT_EARLY" "$CKPT_DIRECT_MID" "$CKPT_DIRECT_LATE" \
    --ckpt-none "$CKPT_NONE_EARLY" "$CKPT_NONE_MID" "$CKPT_NONE_LATE" \
    --data-path "$IMAGENET_PATH" \
    --pool-size "${POOL_SIZE:-2560}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --spike-frac "${SPIKE_FRAC:-0.025}" \
    --num-control "${NUM_CONTROL:-64}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-2000}" \
    --sv-k "${SV_K:-8}" \
    --sv-iters "${SV_ITERS:-12}" \
    --out "$OUT_DIR/${OUT_NAME}_${SLURM_JOB_ID}.json" 2>&1 | tee longvjpdiag.log
EXIT_CODE=${PIPESTATUS[0]}
cp longvjpdiag.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/longitudinal_jacobian_diagnostic_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
