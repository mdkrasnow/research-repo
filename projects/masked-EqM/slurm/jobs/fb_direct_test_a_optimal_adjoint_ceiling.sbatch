#!/bin/bash
#SBATCH --job-name=fb-direct-testA
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb-direct-testA-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb-direct-testA-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT="${CKPT:?set CKPT}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
OUT_NAME="${OUT_NAME:-fb_direct_testA}"
WORK_DIR="/tmp/masked-eqm-fbtestA-${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision

python experiments/direct_energy/fb_direct_test_a_optimal_adjoint_ceiling.py \
    --ckpt "$CKPT" \
    --data-path "$IMAGENET_PATH" \
    --num-batches "${NUM_BATCHES:-20}" \
    --cg-iters "${CG_ITERS:-20}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --out "$OUT_DIR/${OUT_NAME}_${SLURM_JOB_ID}.json" 2>&1 | tee testA.log
EXIT_CODE=${PIPESTATUS[0]}
cp testA.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb_direct_testA_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
