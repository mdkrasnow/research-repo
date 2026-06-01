#!/bin/bash
#SBATCH --job-name=exp4-van-resume
#SBATCH --output=slurm/logs/exp4-van-resume_%A.out
#SBATCH --error=slurm/logs/exp4-van-resume_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# Experiment 4 fairness fix: resume vanilla EqM-B/2 seed0 from 0380000.pt (76ep)
# to 80 epochs (405000 steps) so the step-matched arm is 80ep-vs-80ep against
# v10 final.pt (405000), removing the 4-epoch confound. ~25K steps, ~3h.
# Output: results/imagenet1k_80ep_vanilla_resume405k/000-.../checkpoints/{0405000.pt,final.pt}

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "ERROR: No GPU"; exit 1; }

WORK_DIR="/tmp/exp4-van-resume-${SLURM_JOB_ID}"; mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes"
git init . 2>&1; git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || { git fetch --unshallow origin 2>&1 | tail -3; git checkout "${GIT_SHA}" || exit 1; }
pip install timm diffusers accelerate scipy torchvision 2>&1 | tail -5

REPO_ROOT="/n/home03/mkrasnow/research-repo"
IMAGENET_PATH="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train"
RESULTS_DIR="projects/diff-EqM/results/imagenet1k_80ep_vanilla_resume405k"
# trusted vanilla seed0 380K ckpt (FID 31.41 baseline)
RESUME_CKPT="${RESUME_CKPT:-$REPO_ROOT/projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt}"
[ -f "$RESUME_CKPT" ] || { echo "ERROR: resume ckpt not found: $RESUME_CKPT"; exit 1; }
[ -d "$IMAGENET_PATH" ] || { echo "ERROR: ImageNet train missing"; exit 1; }

mkdir -p "$WORK_DIR/$RESULTS_DIR"
LOCAL_CKPT="$WORK_DIR/resume_ckpt.pt"; cp "$RESUME_CKPT" "$LOCAL_CKPT"
MASTER_PORT=$((29500 + RANDOM % 1000))

python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/train_imagenet.py \
    --model EqM-B/2 --data-path "$IMAGENET_PATH" --num-classes 1000 \
    --global-batch-size 256 --epochs 80 --image-size 256 \
    --global-seed 0 --results-dir "$RESULTS_DIR" --ckpt "$LOCAL_CKPT" \
    2>&1 | tee train.log
EXIT=${PIPESTATUS[0]}

mkdir -p "$REPO_ROOT/$RESULTS_DIR"
rsync -av "$WORK_DIR/$RESULTS_DIR/" "$REPO_ROOT/$RESULTS_DIR/" 2>&1 | tail -10
cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/exp4-van-resume_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo "=== resume exit $EXIT; expect 0405000.pt + final.pt in $RESULTS_DIR ==="
exit $EXIT
