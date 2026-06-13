#!/bin/bash
#SBATCH --job-name=rcanm-step1
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/rc-hpm/slurm/logs/rcanm-step1_%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/rc-hpm/slurm/logs/rcanm-step1_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# RC-ANM Step 1 premise probe — NO-CLONE variant: runs directly from the
# rsynced cluster repo (GitHub push of rewritten history was too large/flaky).
set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi >&2 || { echo "no GPU"; exit 1; }
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/imagenet1k_80ep_v10_b2_lambda10_seed0/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt}"
IMAGENET_PATH="${IMAGENET_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train}"
EPS_GRID="${EPS_GRID:-0.25,0.5,1.0,1.5}"
BATCHES="${BATCHES:-40}"; BATCH_SIZE="${BATCH_SIZE:-32}"

pip install -q timm diffusers accelerate scipy torchvision 2>&1 | tail -2

[ -f "$CKPT" ] || { echo "ERROR ckpt $CKPT" >&2; exit 1; }
[ -d "$IMAGENET_PATH" ] || { echo "ERROR imagenet $IMAGENET_PATH" >&2; exit 1; }

cd /n/home03/mkrasnow/research-repo/projects/rc-hpm
mkdir -p results slurm/logs
python3 experiments/rcanm_step1_probe.py \
    --ckpt "$CKPT" --data-path "$IMAGENET_PATH" \
    --out "results/rcanm_step1_verdict.json" \
    --batches "$BATCHES" --batch-size "$BATCH_SIZE" --eps-grid "$EPS_GRID" \
    2>&1 | tee results/rcanm_step1.log
echo "=== verdict ===" >&2; cat results/rcanm_step1_verdict.json >&2
