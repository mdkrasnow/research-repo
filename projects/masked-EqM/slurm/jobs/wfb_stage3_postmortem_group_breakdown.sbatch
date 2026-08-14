#!/bin/bash
#SBATCH --job-name=fb-direct-wfb-stage3-postmortem
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage3-postmortem-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb-stage3-postmortem-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

# WFB-EqM Stage 3 postmortem: backbone vs head breakdown of the lambda_max
# growth observed in the CG-FBGN full run (job 38909657) -- top-k singular
# subspace of the field-parameter Jacobian, computed separately for backbone
# and energy-head parameter groups, on the step100 (before most of the
# growth) and step300 (after) checkpoints from that run. No training.

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT_BEFORE="${CKPT_BEFORE:?set CKPT_BEFORE}"
CKPT_AFTER="${CKPT_AFTER:?set CKPT_AFTER}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
WORK_DIR="/tmp/masked-eqm-wfbstage3postmortem-${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision

python -u experiments/direct_energy/wfb_stage3_postmortem_group_breakdown.py \
    --ckpt-before "$CKPT_BEFORE" \
    --ckpt-after "$CKPT_AFTER" \
    --data-path "$IMAGENET_PATH" \
    --batch-size "${BATCH_SIZE:-8}" \
    --n-probe-batches "${N_PROBE_BATCHES:-4}" \
    --k "${K:-4}" \
    --num-iters "${NUM_ITERS:-40}" 2>&1 | tee wfbstage3postmortem.log
EXIT_CODE=${PIPESTATUS[0]}
cp wfbstage3postmortem.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/wfb_stage3_postmortem_group_breakdown_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
