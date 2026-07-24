#!/bin/bash
#SBATCH --job-name=eqm-direct-full-eval
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-full-eval-%A_%a.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-full-eval-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%2
set -euo pipefail

# Matched post-training FID evaluation for the user-directed real-scale
# screen.  It is submitted with an afterok dependency on all nine training
# jobs, so a failed training run cannot be silently evaluated as a success.
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi || exit 1

TASK_ID="${SLURM_ARRAY_TASK_ID}"
RUN_TAGS=(
    direct_full_none_seed0 direct_full_dot_seed0 direct_full_direct_seed0
    direct_full_none_seed1 direct_full_dot_seed1 direct_full_direct_seed1
    direct_full_none_seed2 direct_full_dot_seed2 direct_full_direct_seed2
)
JOB_IDS=(
    34721182 34721183 34796838
    34796863 34796866 34796868
    34796869 34796870 34796872
)
EBMS=(none dot direct none dot direct none dot direct)

RUN_TAG="${RUN_TAGS[$TASK_ID]}"
TRAIN_JOB_ID="${JOB_IDS[$TASK_ID]}"
EBM="${EBMS[$TASK_ID]}"
RESULTS_ROOT="${RESULTS_ROOT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results/direct_energy_full}"
CKPT="$RESULTS_ROOT/${RUN_TAG}_job${TRAIN_JOB_ID}/000-EqM-B-2-Linear-velocity-None-ebm-${EBM}/checkpoints/epoch01.pt"
OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results/direct_energy_campaign/full_eval}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
WORK_DIR="/tmp/masked-eqm-direct-full-eval-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?set GIT_SHA to the direct-aware evaluation commit}"

test -f "$CKPT"
mkdir -p "$OUT_DIR" "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision pytorch-fid

python eval_fid.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    --num-samples "${NUM_SAMPLES:-2000}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --num-sampling-steps "${NUM_SAMPLING_STEPS:-250}" \
    --ebm "$EBM" \
    --work-dir "$WORK_DIR/fid_work" \
    --out "$OUT_DIR/${RUN_TAG}_fid.json"

SAMPLES_DIR="$OUT_DIR/${RUN_TAG}_samples"
mkdir -p "$SAMPLES_DIR"
cp "$WORK_DIR"/fid_work/generated/00000{0,1,2,3,4,5,6,7}.png "$SAMPLES_DIR/" 2>/dev/null || true
cp "$WORK_DIR"/fid_work/generated/00001{0,1,2,3,4,5,6,7}.png "$SAMPLES_DIR/" 2>/dev/null || true

rm -rf "$WORK_DIR"
