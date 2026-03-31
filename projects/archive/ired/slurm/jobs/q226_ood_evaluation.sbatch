#!/bin/bash
#SBATCH --job-name=ired-q226
#SBATCH --output=slurm/logs/q226_ood_evaluation_%A_%a.out
#SBATCH --error=slurm/logs/q226_ood_evaluation_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%2

# Load required modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Set working directory
WORK_DIR="/tmp/ired-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository at specific git SHA
# GIT_URL and GIT_SHA are passed via SLURM --export
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git clone --depth 1 "$GIT_URL" . 2>&1 | tail -20 || exit 1
git fetch origin "${GIT_SHA}" 2>&1 | tail -5
git checkout "${GIT_SHA}" 2>&1 | tail -5

# Run the OOD evaluation
echo "=== Starting q226 OOD evaluation (seed ${SLURM_ARRAY_TASK_ID}) ===" >&2
python -u projects/ired/experiments/matrix_inversion_mining.py \
    --config projects/ired/configs/q226_ood_evaluation.json \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    2>&1 | tee q226_eval_seed${SLURM_ARRAY_TASK_ID}.log

EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ===" >&2

# Sync results back to persistent storage before exiting
echo "=== Syncing results back to persistent storage ===" >&2
REPO_ROOT="/n/home03/mkrasnow/research-repo"
rsync -av "$WORK_DIR/projects/ired/results/" "$REPO_ROOT/projects/ired/results/" 2>&1 | tail -10

exit $EXIT_CODE
