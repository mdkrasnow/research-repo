#!/bin/bash
#SBATCH --job-name=eqm-stage2-eval
#SBATCH --output=projects/masked-EqM/slurm/logs/stage2-full-eval-%x_%A_%a.out
#SBATCH --error=projects/masked-EqM/slurm/logs/stage2-full-eval-%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=05:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%6

# Full preregistered Stage 2 evaluation, one array task per matched
# checkpoint. Each task runs the frozen cutoff-0.10 Fourier recovery
# evaluation, structured-mask recovery, and sanity-scale 2K FID. The
# array covers 3 training seeds x {gaussian, structured-mask,
# gaussian+structured-mask 1:1}; the three arms are the negative control,
# positive/specialist control, and treatment respectively.
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true

nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

TASK_ID="${SLURM_ARRAY_TASK_ID}"
RUN_TAGS=(
    "gaussian_seed0"
    "gaussian_seed1"
    "gaussian_seed2"
    "structmask_seed0"
    "structmask_seed1"
    "structmask_seed2"
    "gm_seed0"
    "gm_seed1"
    "gm_seed2"
)
CKPTS=(
    "/n/home03/mkrasnow/masked-EqM-stage2-results/stage2_gaussian_seed0_v2_job34382647/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch01.pt"
    "/n/home03/mkrasnow/masked-EqM-stage2-results/stage2_gaussian_seed1_v2_job34382651/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch01.pt"
    "/n/home03/mkrasnow/masked-EqM-stage2-results/stage2_gaussian_seed2_job34315134/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch01.pt"
    "/n/home03/mkrasnow/masked-EqM-stage2-results/stage2_structmask_seed0_v2_job34382652/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch01.pt"
    "/n/home03/mkrasnow/masked-EqM-stage2-results/stage2_structmask_seed1_v2_job34382653/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch01.pt"
    "/n/home03/mkrasnow/masked-EqM-stage2-results/stage2_structmask_seed2_v2_job34382654/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch01.pt"
    "/n/home03/mkrasnow/masked-EqM-stage2-results/stage2_gm_seed0_v2_job34382655/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch01.pt"
    "/n/home03/mkrasnow/masked-EqM-stage2-results/stage2_gm_seed1_v2_job34382922/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch01.pt"
    "/n/home03/mkrasnow/masked-EqM-stage2-results/stage2_gm_seed2_v2_job34382924/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch01.pt"
)

RUN_TAG="${RUN_TAGS[$TASK_ID]}"
CKPT="${CKPTS[$TASK_ID]}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
MANIFEST="/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results/manifest_fourier_repl_1024.json"
OUT_DIR="${OUT_DIR:-/n/home03/mkrasnow/masked-EqM-stage2-eval}"
WORK_DIR="/tmp/masked-eqm-stage2-eval-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"

test -f "$CKPT"
test -f "$MANIFEST"
mkdir -p "$OUT_DIR" "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?GIT_SHA must be set}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "$GIT_SHA" 2>&1 | tail -5 || {
    git fetch --unshallow origin 2>&1 | tail -5
    git checkout "$GIT_SHA" 2>&1 | tail -5 || exit 1
}

cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision lpips pytorch-fid 2>&1 | tail -10

echo "=== Stage 2 full eval: task=$TASK_ID tag=$RUN_TAG ckpt=$CKPT ==="
echo "=== Frozen Fourier recovery: cutoff=0.10, n=1024, steps=250 ==="
python eval_fourier_recovery.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    --cutoff 0.10 \
    --num-images 1024 \
    --batch-size 16 \
    --manifest "$MANIFEST" \
    --out "$OUT_DIR/stage2_${RUN_TAG}_fourier.json"

echo "=== Structured-mask recovery: mask_prob=0.5, n=256, steps=250 ==="
python eval_masked_recovery.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    --mask-prob 0.5 \
    --mask-mode structured \
    --num-images 256 \
    --batch-size 16 \
    --out "$OUT_DIR/stage2_${RUN_TAG}_maskrecovery.json"

echo "=== Sanity FID: n=2000, steps=250 ==="
python eval_fid.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    --num-samples 2000 \
    --batch-size 32 \
    --num-sampling-steps 250 \
    --work-dir "$WORK_DIR/fid_work" \
    --out "$OUT_DIR/stage2_${RUN_TAG}_fid.json"

# Persist 16 generated examples for the mandatory smoke-time sample probe.
SAMPLES_DIR="$OUT_DIR/stage2_${RUN_TAG}_fid_samples"
mkdir -p "$SAMPLES_DIR"
cp "$WORK_DIR"/fid_work/generated/00000{0,1,2,3,4,5,6,7,8,9}.png "$SAMPLES_DIR/"
cp "$WORK_DIR"/fid_work/generated/00001{0,1,2,3,4,5}.png "$SAMPLES_DIR/"

echo "=== Stage 2 full eval complete: $RUN_TAG ==="
rm -rf "$WORK_DIR"
