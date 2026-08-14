#!/bin/bash
#SBATCH --job-name=fb-direct-memprobe
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb-direct-memprobe-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb-direct-memprobe-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
CKPT="${CKPT:?set CKPT}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/ILSVRC2012_img_train}"
WORK_DIR="/tmp/masked-eqm-fbmemprobe-${SLURM_JOB_ID}"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin https://github.com/mdkrasnow/research-repo.git
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' 'transformers==4.40.2' accelerate torchvision

python experiments/direct_energy/fb_direct_testA_memory_probe.py \
    --ckpt "$CKPT" \
    --data-path "$IMAGENET_PATH" \
    --batch-size "${BATCH_SIZE:-8}" 2>&1 | tee memprobe.log
EXIT_CODE=${PIPESTATUS[0]}
cp memprobe.log "/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/fb_direct_memprobe_${SLURM_JOB_ID}.log" || true
rm -rf "$WORK_DIR"
exit "$EXIT_CODE"
