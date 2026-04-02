#!/bin/bash
#SBATCH --job-name=dl-imagenet100
#SBATCH --output=slurm/logs/dl-imagenet100_%A.out
#SBATCH --error=slurm/logs/dl-imagenet100_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:0

# Download ImageNet-100 to home directory
# Expected: ~1-2 hours for 17 GB download + conversion

set -euo pipefail

module load python/3.10.13-fasrc01

# Install datasets if needed
pip install datasets pillow 2>&1 | tail -5

OUTPUT_DIR="/n/home03/mkrasnow/imagenet100"

echo "=== Starting ImageNet-100 download ===" >&2
echo "=== Output: $OUTPUT_DIR ===" >&2
echo "=== Start time: $(date) ===" >&2

# Clone repo for the download script
WORK_DIR="/tmp/dl-imagenet100-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-HEAD}"
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

git init . 2>&1
git remote add origin "$GIT_URL" 2>&1
git fetch --depth=50 origin main 2>&1 | tail -5 || exit 1
git checkout "${GIT_SHA}" 2>&1 | tail -5 || exit 1

python -u projects/diff-EqM/scripts/download_imagenet100.py \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee download.log

DL_EXIT=$?
echo "=== Download exit code: $DL_EXIT ===" >&2
echo "=== End time: $(date) ===" >&2

# Verify
if [ $DL_EXIT -eq 0 ]; then
    echo "=== Verifying dataset ===" >&2
    ls -la "$OUTPUT_DIR/train/" | head -5
    echo "Classes: $(ls -d $OUTPUT_DIR/train/class_*/ | wc -l)"
    echo "Train images: $(find $OUTPUT_DIR/train/ -name '*.jpg' | wc -l)"
fi

cp download.log /n/home03/mkrasnow/research-repo/projects/diff-EqM/slurm/logs/dl_imagenet100_${SLURM_JOB_ID}.log 2>/dev/null || true
rm -rf "$WORK_DIR"

echo "=== Done ===" >&2
exit $DL_EXIT
