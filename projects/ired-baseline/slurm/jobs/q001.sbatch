#!/bin/bash
#SBATCH --job-name=ired-baseline_q001
#SBATCH --output=projects/ired-baseline/slurm/logs/%x_%j.out
#SBATCH --error=projects/ired-baseline/slurm/logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Configuration (passed via environment variables)
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

# Create work directory for this job
WORK_DIR="${TMPDIR:-/tmp}/ired-baseline-job-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

echo "Git URL: $GIT_URL"
echo "Git SHA: $GIT_SHA"
echo "Work directory: $WORK_DIR"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Clone repository
echo "Cloning repository..."
git clone "$GIT_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout "$GIT_SHA"

echo "Repository cloned and checked out to: $(git rev-parse --short HEAD)"
echo "Repository info:"
git log -1 --oneline

# Run pilot experiment
echo "Starting IRED baseline pilot test..."
python projects/ired-baseline/experiments/run_baseline.py \
    --config projects/ired-baseline/configs/q001_pilot.json \
    --output-dir projects/ired-baseline/runs/q001_pilot

echo "Pilot test completed at $(date)"

# Copy results to persistent storage before cleanup
REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying results to persistent storage..."
if [ -d "$WORK_DIR/projects/ired-baseline/runs" ]; then
    mkdir -p "$REPO_ROOT/projects/ired-baseline/runs"
    rsync -av "$WORK_DIR/projects/ired-baseline/runs/" "$REPO_ROOT/projects/ired-baseline/runs/"
    echo "Results copied to $REPO_ROOT/projects/ired-baseline/runs/"
else
    echo "Warning: No runs directory found at $WORK_DIR/projects/ired-baseline/runs"
fi

# Cleanup
echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"
