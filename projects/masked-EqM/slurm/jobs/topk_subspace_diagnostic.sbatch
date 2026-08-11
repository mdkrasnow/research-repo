#!/bin/bash
#SBATCH --job-name=fb-direct-topk-subspace
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/topk-subspace-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/topk-subspace-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=16:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT_DIRECT_LATE="${CKPT_DIRECT_LATE:?set CKPT_DIRECT_LATE}"
CKPT_NONE_LATE="${CKPT_NONE_LATE:?set CKPT_NONE_LATE}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
OUT_NAME="${OUT_NAME:-topk_subspace_diagnostic}"
WORK_DIR="/tmp/masked-eqm-topksubspace-${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision

echo "=== FP64 CPU unit tests (must pass before any GPU work, incl. block-subspace validation) ==="
python tests/test_fb_direct_exact_hvp.py

python experiments/direct_energy/topk_subspace_diagnostic.py \
    --ckpt-direct "$CKPT_DIRECT_LATE" \
    --ckpt-none "$CKPT_NONE_LATE" \
    --data-path "$IMAGENET_PATH" \
    --batch-size "${BATCH_SIZE:-8}" \
    --k "${SV_K:-8}" \
    --k2 "${SV_K2:-0}" \
    --subspace-iters "${SV_ITERS:-30}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES:-2000}" \
    --clip-threshold "${CLIP_THRESHOLD:-6.87141}" \
    --out "$OUT_DIR/${OUT_NAME}_${SLURM_JOB_ID}.json" 2>&1 | tee topksubspace.log
EXIT_CODE=${PIPESTATUS[0]}
cp topksubspace.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/topk_subspace_diagnostic_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
