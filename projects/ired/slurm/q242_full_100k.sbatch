#!/bin/bash
#SBATCH --job-name=ired_q242_100k
#SBATCH --output=projects/ired/slurm/logs/%x_%j.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
set -euo pipefail

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"
SEED=0
REPO_ROOT="/n/home03/mkrasnow/research-repo"

WORK_DIR="${TMPDIR:-/tmp}/ired-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

nvidia-smi

# Clone with retry, fall back to local copy
for attempt in 1 2 3; do
    if timeout 120 git clone "$GIT_URL" "$WORK_DIR" 2>/dev/null; then
        break
    fi
    echo "git clone attempt $attempt failed, retrying..."
    rm -rf "$WORK_DIR" && mkdir -p "$WORK_DIR"
    if [ "$attempt" -eq 3 ]; then
        echo "git clone failed, copying from local repo"
        cp -r "$REPO_ROOT/." "$WORK_DIR/"
    fi
done

cd "$WORK_DIR"
git checkout "$GIT_SHA" 2>/dev/null || true

python projects/ired/experiments/matrix_inversion_mining.py \
    --config projects/ired/configs/q242_hard_state_replay_unc_100k.json \
    --seed "$SEED"

rsync -av "$WORK_DIR/projects/ired/results/" "$REPO_ROOT/projects/ired/results/" \
    --include='*/' --include='*.pt' --include='*.json' --include='*.md' --exclude='*'

cd /
rm -rf "$WORK_DIR"
