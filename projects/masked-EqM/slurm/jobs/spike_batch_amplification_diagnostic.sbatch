#!/bin/bash
#SBATCH --job-name=fb-direct-spikeamp
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/spikeamp-diag-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/spikeamp-diag-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT_DIRECT="${CKPT_DIRECT:?set CKPT_DIRECT}"
CKPT_NONE="${CKPT_NONE:?set CKPT_NONE}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
OUT_DIR="${OUT_DIR:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/eval_results}"
OUT_NAME="${OUT_NAME:-spike_batch_amplification_diagnostic}"
WORK_DIR="/tmp/masked-eqm-spikeampdiag-${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR" "$OUT_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision

python experiments/direct_energy/spike_batch_amplification_diagnostic.py \
    --ckpt-direct "$CKPT_DIRECT" \
    --ckpt-none "$CKPT_NONE" \
    --data-path "$IMAGENET_PATH" \
    --pool-size "${POOL_SIZE:-500}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --spike-frac "${SPIKE_FRAC:-0.02}" \
    --num-ordinary "${NUM_ORDINARY:-10}" \
    --num-directional-batches "${NUM_DIRECTIONAL_BATCHES:-3}" \
    --eps "${EPS:-0.001}" \
    --out "$OUT_DIR/${OUT_NAME}_${SLURM_JOB_ID}.json" 2>&1 | tee spikeampdiag.log
EXIT_CODE=${PIPESTATUS[0]}
cp spikeampdiag.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/spike_batch_amplification_diagnostic_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
