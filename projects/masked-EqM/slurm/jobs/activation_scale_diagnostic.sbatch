#!/bin/bash
#SBATCH --job-name=fb-direct-actscale
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/actscale-diag-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/actscale-diag-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
# ZPATHZ token, not a shell metacharacter -- see curvature_clip_diagnostic.sbatch.
CKPTS_LAMBDA0="${CKPTS_LAMBDA0:?set CKPTS_LAMBDA0 (ZPATHZ-joined paths)}"
CKPTS_GP="${CKPTS_GP:?set CKPTS_GP (ZPATHZ-joined paths)}"
CKPTS_LAMBDA0="${CKPTS_LAMBDA0//ZPATHZ/ }"
CKPTS_GP="${CKPTS_GP//ZPATHZ/ }"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
OUT_NAME="${OUT_NAME:-activation_scale_diagnostic}"
WORK_DIR="/tmp/masked-eqm-actscalediag-${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision

python experiments/direct_energy/activation_scale_diagnostic.py \
    --ckpts-lambda0 $CKPTS_LAMBDA0 \
    --ckpts-gp $CKPTS_GP \
    --data-path "$IMAGENET_PATH" \
    --num-batches "${NUM_BATCHES:-5}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --out "$OUT_DIR/${OUT_NAME}_${SLURM_JOB_ID}.json" 2>&1 | tee actscalediag.log
EXIT_CODE=${PIPESTATUS[0]}
cp actscalediag.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/activation_scale_diagnostic_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
