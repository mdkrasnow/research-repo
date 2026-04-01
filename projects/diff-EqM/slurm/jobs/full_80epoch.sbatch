#!/bin/bash
#SBATCH --job-name=dganm-80ep
#SBATCH --output=slurm/logs/dganm-80ep_%A.out
#SBATCH --error=slurm/logs/dganm-80ep_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Full 80-epoch DG-ANM training + FID evaluation
# Expected: ~8-10h training + ~1h FID eval (50K samples)

# Load required modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU
nvidia-smi || { echo "No GPU available"; exit 1; }

# Set working directory
WORK_DIR="/tmp/dganm-80ep-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository at specific git SHA
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} ===" >&2
git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || { echo "SHA not in last 50 commits, fetching full..."; git fetch --unshallow origin 2>&1 | tail -5; git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1; }

# Install dependencies
pip install timm diffusers accelerate scipy 2>&1 | tail -5

# === PHASE 1: Full 80-epoch training ===
CONFIG_PATH="${CONFIG_PATH:-projects/diff-EqM/configs/full_80epoch.json}"

echo "=== Starting 80-epoch training with config: ${CONFIG_PATH} ===" >&2
echo "=== Start time: $(date) ===" >&2
python -u projects/diff-EqM/experiments/train_dganm.py \
    --config "${CONFIG_PATH}" \
    --seed "${SEED:-0}" \
    2>&1 | tee train.log

TRAIN_EXIT=$?
echo "=== Training exit code: $TRAIN_EXIT ===" >&2
echo "=== Training end time: $(date) ===" >&2

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed" >&2
    # Sync whatever we have
    REPO_ROOT="/n/home03/mkrasnow/research-repo"
    rsync -av "$WORK_DIR/projects/diff-EqM/results/" "$REPO_ROOT/projects/diff-EqM/results/" 2>&1 | tail -10
    cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/train_80ep_${SLURM_JOB_ID}.log" 2>/dev/null || true
    exit $TRAIN_EXIT
fi

# === PHASE 2: Proxy metric evaluation ===
EVAL_CONFIG="${EVAL_CONFIG:-projects/diff-EqM/configs/eval.json}"

# Dynamically read output_dir from training config to find the checkpoint
python3 -c "
import json
with open('${CONFIG_PATH}') as f:
    train_cfg = json.load(f)
output_dir = train_cfg.get('output_dir', 'projects/diff-EqM/results/baseline')
with open('$EVAL_CONFIG') as f:
    cfg = json.load(f)
cfg['checkpoint_path'] = output_dir + '/final_model.pt'
with open('/tmp/eval_80ep.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'Proxy eval checkpoint: {cfg[\"checkpoint_path\"]}')
"

echo "=== Running proxy metric evaluation ===" >&2
python -u projects/diff-EqM/experiments/evaluate.py \
    --config /tmp/eval_80ep.json \
    2>&1 | tee eval.log

# === PHASE 3: FID evaluation (50K samples) ===
FID_CONFIG="${FID_CONFIG:-projects/diff-EqM/configs/eval_fid_full.json}"

# Sync reference stats if available
REPO_ROOT="/n/home03/mkrasnow/research-repo"
if [ -f "$REPO_ROOT/projects/diff-EqM/results/cifar10_inception_stats.npz" ]; then
    mkdir -p "$WORK_DIR/projects/diff-EqM/results/"
    cp "$REPO_ROOT/projects/diff-EqM/results/cifar10_inception_stats.npz" \
       "$WORK_DIR/projects/diff-EqM/results/" 2>/dev/null || true
fi

echo "=== Starting FID evaluation (50K samples) ===" >&2
echo "=== FID start time: $(date) ===" >&2
python -u projects/diff-EqM/experiments/evaluate_fid.py \
    --config "${FID_CONFIG}" \
    2>&1 | tee fid_eval.log

FID_EXIT=$?
echo "=== FID evaluation exit code: $FID_EXIT ===" >&2
echo "=== FID end time: $(date) ===" >&2

# === Sync everything back ===
echo "=== Syncing results ===" >&2
rsync -av "$WORK_DIR/projects/diff-EqM/results/" "$REPO_ROOT/projects/diff-EqM/results/" 2>&1 | tail -10
cp train.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/train_80ep_${SLURM_JOB_ID}.log" 2>/dev/null || true
cp eval.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/eval_80ep_${SLURM_JOB_ID}.log" 2>/dev/null || true
cp fid_eval.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/fid_80ep_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo "=== All done ===" >&2
echo "=== Total end time: $(date) ===" >&2
exit $FID_EXIT
