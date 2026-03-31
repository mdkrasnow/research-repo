#!/bin/bash
#SBATCH --job-name=ired-q225
#SBATCH --output=slurm/logs/q225_tam_ctl_full_100k_%A_%a.out
#SBATCH --error=slurm/logs/q225_tam_ctl_full_100k_%A_%a.err
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
GIT_URL="${GIT_URL:-https://github.com/mkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"

export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git clone --depth 1 "$GIT_URL" . 2>&1 | tail -20 || exit 1
git fetch origin "${GIT_SHA}" 2>&1 | tail -5
git checkout "${GIT_SHA}" 2>&1 | tail -5

# Verify checkpoint structure
echo "=== Repository structure ===" >&2
ls -la projects/ired/results/ 2>&1 | head -20 || echo "No results dir yet"

# Run the experiment
echo "=== Starting q225 training (seed ${SLURM_ARRAY_TASK_ID}) ===" >&2
python -u projects/ired/experiments/matrix_inversion_mining.py \
    --config projects/ired/configs/q225_tam_ctl_full_100k.json \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    2>&1 | tee q225_train_seed${SLURM_ARRAY_TASK_ID}.log

EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ===" >&2

# Verify checkpoint was created
echo "=== Final checkpoint verification ===" >&2
if [ -f "projects/ired/results/ds_inverse/q225_tam_ctl_full_100k/seed_${SLURM_ARRAY_TASK_ID}/final_model.pt" ]; then
    ls -lh "projects/ired/results/ds_inverse/q225_tam_ctl_full_100k/seed_${SLURM_ARRAY_TASK_ID}/final_model.pt"
    echo "✓✓✓ SUCCESS: Checkpoint saved"
else
    echo "✗✗✗ ERROR: Checkpoint not found"
    ls -la "projects/ired/results/ds_inverse/q225_tam_ctl_full_100k/" 2>&1 || echo "No output dir"
fi

# Sync results back to persistent storage before exiting
echo "=== Syncing results back to persistent storage ===" >&2
REPO_ROOT="/n/home03/mkrasnow/research-repo"
rsync -av "$WORK_DIR/projects/ired/results/" "$REPO_ROOT/projects/ired/results/" 2>&1 | tail -10

exit $EXIT_CODE
