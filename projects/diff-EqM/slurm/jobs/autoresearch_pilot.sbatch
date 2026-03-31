#!/bin/bash
#SBATCH --job-name=diff-eqm-pilot
#SBATCH --output=slurm/logs/diff-eqm-pilot_%A.out
#SBATCH --error=slurm/logs/diff-eqm-pilot_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# Load required modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU
nvidia-smi || { echo "No GPU available"; exit 1; }

# Set working directory
WORK_DIR="/tmp/diff-eqm-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository at specific git SHA
GIT_URL="${GIT_URL:-https://github.com/mkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

echo "=== Cloning repo at SHA ${GIT_SHA} ===" >&2
git clone --depth 1 "$GIT_URL" . 2>&1 | tail -20 || exit 1
git fetch origin "${GIT_SHA}" 2>&1 | tail -5
git checkout "${GIT_SHA}" 2>&1 | tail -5

# Install dependencies if needed
pip install timm diffusers accelerate torchdiffeq 2>&1 | tail -5

# Config path is passed as environment variable
CONFIG_PATH="${CONFIG_PATH:-projects/diff-EqM/configs/baseline.json}"

echo "=== Starting training with config: ${CONFIG_PATH} ===" >&2
python -u projects/diff-EqM/experiments/train_dganm.py \
    --config "${CONFIG_PATH}" \
    --seed "${SEED:-0}" \
    2>&1 | tee train.log

TRAIN_EXIT=$?
echo "=== Training exit code: $TRAIN_EXIT ===" >&2

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed" >&2
    exit $TRAIN_EXIT
fi

# Run evaluation
EVAL_CONFIG="${EVAL_CONFIG:-projects/diff-EqM/configs/eval.json}"
echo "=== Running evaluation ===" >&2
python -u projects/diff-EqM/experiments/evaluate.py \
    --config "${EVAL_CONFIG}" \
    2>&1 | tee eval.log

EVAL_EXIT=$?
echo "=== Evaluation exit code: $EVAL_EXIT ===" >&2

# Sync results back to persistent storage
echo "=== Syncing results ===" >&2
REPO_ROOT="/n/home03/mkrasnow/research-repo"
rsync -av "$WORK_DIR/projects/diff-EqM/results/" "$REPO_ROOT/projects/diff-EqM/results/" 2>&1 | tail -10

# Also sync the eval log so we can parse the metric
cp eval.log "$REPO_ROOT/projects/diff-EqM/slurm/logs/eval_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo "=== Done ===" >&2
exit $EVAL_EXIT
