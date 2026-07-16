#!/bin/bash
#SBATCH --job-name=eqm-grid-assets
#SBATCH --output=projects/masked-EqM/slurm/logs/grid-assets-%x_%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/grid-assets-%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# One-off: save clean + Fourier-corrupted PNGs (cutoffs 0.05/0.10/0.15) for
# the fixed index set used in the severity-sweep qualitative grids. Reuses
# eval_fourier_recovery.py's manifest/encode/corrupt determinism verbatim.
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

WORK_DIR="/tmp/masked-eqm-gridassets-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?GIT_SHA must be set}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -10 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || {
    git fetch --unshallow origin 2>&1 | tail -5
    git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1
}
cd projects/masked-EqM
pip install --quiet timm diffusers accelerate torchvision 2>&1 | tail -10

cp "$REPO_ROOT_INDICES" ./indices_input.json 2>/dev/null || true

VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
MANIFEST="${MANIFEST:?MANIFEST must be set}"
INDICES_FILE="${INDICES_FILE:?INDICES_FILE must be set}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set}"

INDICES_JSON=$(cat "$INDICES_FILE")

python save_clean_corrupted.py \
    --data-path "$VAL_PATH" \
    --manifest "$MANIFEST" \
    --indices "$INDICES_JSON" \
    --cutoffs "0.05,0.1,0.15" \
    --out-dir "$OUT_DIR"

echo "=== done ==="
rm -rf "$WORK_DIR"
