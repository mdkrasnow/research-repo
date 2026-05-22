#!/bin/bash
#SBATCH --job-name=cafm-eqm-b2-in256-10ep
#SBATCH --output=slurm/logs/cafm-eqm-b2-in256-10ep_%j.out
#SBATCH --error=slurm/logs/cafm-eqm-b2-in256-10ep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=72:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:4

# CAFM post-training of vanilla EqM-B/2 80ep (FID 31.41) for 10 epochs.
# Estimated 10 ep on IN-1K with batch 256 + N=16 disc updates per gen ≈ 24-36h.
# Pass SEED env to override default 0.
# Pass ENABLE_V10=1 to add v10 PGD mining in gen step (Phase 2; default OFF for Phase 1b).

set -uo pipefail

SEED="${SEED:-0}"
ENABLE_V10="${ENABLE_V10:-0}"
CONFIG="${CONFIG:-projects/diff-EqM/configs/cafm/eqm_b2_in256_cafm.yaml}"

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || exit 1

WORK_DIR="/tmp/cafm-eqm-${SLURM_JOB_ID}-seed${SEED}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repo at submitted SHA (standard pattern per CLAUDE.md SLURM rules).
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

# Clone Lin AFM/CAFM repo for discriminator class import.
git clone --depth=1 https://github.com/ByteDance-Seed/Adversarial-Flow-Models.git \
    projects/diff-EqM/external/Adversarial-Flow-Models 2>&1 | tail -3
# Per Lin's CAFM README: download sit.py from SiT repo.
curl -fL -o projects/diff-EqM/external/Adversarial-Flow-Models/models/cafm/sit/sit.py \
    https://raw.githubusercontent.com/willisma/SiT/main/models.py \
    2>&1 | tail -3 || exit 1

# Pre-stage Lin's models/cafm/ to a uniquely-named local package dir to dodge
# the EqM `models.py` namespace collision. Single-threaded here so 4 DDP ranks
# don't race on shutil.copytree.
STAGING="projects/diff-EqM/external/cafm_pkg"
rm -rf "$STAGING"
mkdir -p "$STAGING"
touch "$STAGING/__init__.py"
cp -r projects/diff-EqM/external/Adversarial-Flow-Models/models/cafm "$STAGING/cafm"
find "$STAGING/cafm" -type d -exec touch {}/__init__.py \;
echo "=== Pre-staged cafm_pkg at $STAGING ===" >&2

# Install deps.
pip install --quiet timm diffusers accelerate scipy torchvision pytorch-fid omegaconf 2>&1 | tail -3
pip install --quiet -r projects/diff-EqM/external/Adversarial-Flow-Models/requirements_cafm.txt 2>&1 | tail -3 || true

# Persist setup (background rsync per CLAUDE.md pattern).
REPO_ROOT="/n/home03/mkrasnow/research-repo"
RESULTS_SUBDIR="projects/diff-EqM/results/cafm_eqm_b2_in256_seed${SEED}"
PERSIST_DIR="$REPO_ROOT/$RESULTS_SUBDIR"
mkdir -p "$WORK_DIR/$RESULTS_SUBDIR" "$PERSIST_DIR"

# Stage vanilla EqM-B/2 80ep ckpt. Config references it at projects/diff-EqM/results/...
# relative to CWD ($WORK_DIR); symlink into WORK_DIR pointing at persistent ckpt.
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

rsync_loop() {
    while true; do
        sleep 300
        rsync -a --quiet --include='*/' --include='*.pt' --include='*.log' \
            --include='*.txt' --include='*.json' --exclude='*' \
            "$WORK_DIR/$RESULTS_SUBDIR/" "$PERSIST_DIR/" 2>/dev/null
    done
}
rsync_loop &
RSYNC_PID=$!
trap "kill $RSYNC_PID 2>/dev/null; sleep 2; rsync -av --include='*/' --include='*.pt' --include='*.log' --include='*.txt' --include='*.json' --exclude='*' '$WORK_DIR/$RESULTS_SUBDIR/' '$PERSIST_DIR/'" EXIT

MASTER_PORT=$((29500 + RANDOM % 1000))

V10_FLAG=""
if [ "$ENABLE_V10" = "1" ]; then
    V10_FLAG="--enable-v10"
fi

echo "=== CAFM-EqM training start: $(date) ===" >&2
echo "  SEED=$SEED  ENABLE_V10=$ENABLE_V10  CONFIG=$CONFIG" >&2

python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
    projects/diff-EqM/experiments/cafm_eqm/train_cafm_eqm.py \
    --config "$CONFIG" \
    $V10_FLAG \
    2>&1 | tee train.log
