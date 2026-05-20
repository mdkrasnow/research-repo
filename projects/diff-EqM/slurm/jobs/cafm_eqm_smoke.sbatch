#!/bin/bash
#SBATCH --job-name=cafm-eqm-smoke
#SBATCH --output=slurm/logs/cafm-eqm-smoke_%j.out
#SBATCH --error=slurm/logs/cafm-eqm-smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:4

# CAFM-EqM smoke: 500-step end-to-end validation of the port. ~6-15 min on
# 4xA100. Goal: verify losses finite, discriminator integrates, no NaN.
# Followed by full 10-epoch run (cafm_eqm_b2_in256.sbatch) if smoke passes.

set -uo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || exit 1

WORK_DIR="/tmp/cafm-eqm-smoke-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -3 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -3 || {
    git fetch --unshallow origin 2>&1 | tail -3
    git checkout "${GIT_SHA}" 2>&1 | tail -3 || exit 1
}

# Clone Lin AFM/CAFM repo + download sit.py.
git clone --depth=1 https://github.com/ByteDance-Seed/Adversarial-Flow-Models.git \
    projects/diff-EqM/external/Adversarial-Flow-Models 2>&1 | tail -3
curl -fL -o projects/diff-EqM/external/Adversarial-Flow-Models/models/cafm/sit/sit.py \
    https://raw.githubusercontent.com/willisma/SiT/main/models.py 2>&1 | tail -3 || exit 1

# Pre-stage Lin's models/cafm/ to a uniquely-named local package dir to dodge
# the EqM `models.py` namespace collision. Done once here (before DDP fork) so
# the 4 DDP ranks don't race on shutil.copytree.
STAGING="projects/diff-EqM/external/cafm_pkg"
rm -rf "$STAGING"
mkdir -p "$STAGING"
touch "$STAGING/__init__.py"
cp -r projects/diff-EqM/external/Adversarial-Flow-Models/models/cafm "$STAGING/cafm"
# Add __init__.py to every subdirectory so the copied tree is a real package.
find "$STAGING/cafm" -type d -exec touch {}/__init__.py \;
echo "=== Pre-staged cafm_pkg at $STAGING ===" >&2
ls -la "$STAGING/cafm/sit" >&2 || true
ls -la "$STAGING/cafm/jvp" >&2 || true

pip install --quiet timm diffusers accelerate scipy torchvision pytorch-fid omegaconf 2>&1 | tail -3
pip install --quiet -r projects/diff-EqM/external/Adversarial-Flow-Models/requirements_cafm.txt 2>&1 | tail -3 || true

REPO_ROOT="/n/home03/mkrasnow/research-repo"
RESULTS_SUBDIR="projects/diff-EqM/results/cafm_eqm_smoke"
PERSIST_DIR="$REPO_ROOT/$RESULTS_SUBDIR"
mkdir -p "$WORK_DIR/$RESULTS_SUBDIR" "$PERSIST_DIR"

# Stage vanilla EqM-B/2 80ep ckpt for CAFM post-training. Config references it
# at projects/diff-EqM/results/... relative to CWD; CWD is $WORK_DIR, so we
# create the same relative path inside WORK_DIR pointing at the persistent ckpt.
CKPT_REL="projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt"
CKPT_SRC="$REPO_ROOT/$CKPT_REL"
CKPT_DST="$WORK_DIR/$CKPT_REL"
mkdir -p "$(dirname "$CKPT_DST")"
if [ -f "$CKPT_SRC" ]; then
    ln -sf "$CKPT_SRC" "$CKPT_DST"
    echo "=== Linked vanilla EqM ckpt: $(du -h "$CKPT_SRC" | cut -f1) ===" >&2
else
    echo "ERROR: vanilla EqM ckpt not found at $CKPT_SRC" >&2
    exit 1
fi

MASTER_PORT=$((29500 + RANDOM % 1000))

ENABLE_V10="${ENABLE_V10:-0}"
V10_FLAG=""
if [ "$ENABLE_V10" = "1" ]; then
    V10_FLAG="--enable-v10"
fi

echo "=== CAFM-EqM smoke start: $(date)  ENABLE_V10=$ENABLE_V10 ===" >&2

python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/cafm_eqm/train_cafm_eqm.py \
    --config projects/diff-EqM/configs/cafm/smoke_cluster.yaml \
    $V10_FLAG \
    2>&1 | tee train.log

EXIT=$?

# Persist
rsync -av --include='*/' --include='*.pt' --include='*.log' --include='*.txt' \
    --include='*.json' --exclude='*' \
    "$WORK_DIR/$RESULTS_SUBDIR/" "$PERSIST_DIR/" 2>&1 | tail -5
cp train.log "$PERSIST_DIR/smoke_train_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo "=== CAFM-EqM smoke done: exit=$EXIT $(date) ===" >&2
exit $EXIT
