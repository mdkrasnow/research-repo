#!/bin/bash
#SBATCH --job-name=ired_q224_eval_robustness
#SBATCH --output=projects/ired/slurm/logs/%x_%j.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Configuration (passed via environment variables)
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:-main}"

# Create work directory for this job
WORK_DIR="${TMPDIR:-/tmp}/ired-eval-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"

echo "Git URL: $GIT_URL"
echo "Git SHA: $GIT_SHA"
echo "Work directory: $WORK_DIR"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Verify GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU info:"
    nvidia-smi
fi

# Clone repository
echo "Cloning repository..."
git clone "$GIT_URL" "$WORK_DIR"
cd "$WORK_DIR"
git checkout "$GIT_SHA"

echo "Repository cloned and checked out to: $(git rev-parse --short HEAD)"
echo "Repository info:"
git log -1 --oneline

# Create output directory
mkdir -p projects/ired/results/evaluation

echo ""
echo "=========================================="
echo "Running TAM-CTL Robustness Evaluation"
echo "=========================================="
echo ""

# Level 1: Baseline difficulty (ood=false)
echo "Running Level 1 evaluation: Baseline difficulty (ood=false)..."
python projects/ired/experiments/evaluate_robustness.py \
    --config projects/ired/configs/q224_eval_difficulty_L1_baseline.json

echo ""
echo "Level 1 evaluation completed at $(date)"
echo ""

# Level 2: Hard difficulty (ood=true)
echo "Running Level 2 evaluation: Hard difficulty (ood=true)..."
python projects/ired/experiments/evaluate_robustness.py \
    --config projects/ired/configs/q224_eval_difficulty_L2_hard.json

echo ""
echo "Level 2 evaluation completed at $(date)"
echo ""

echo "=========================================="
echo "All evaluations completed"
echo "=========================================="

# Copy results to persistent storage before cleanup
REPO_ROOT="/n/home03/mkrasnow/research-repo"
echo "Copying results to persistent storage..."
if [ -d "$WORK_DIR/projects/ired/results" ]; then
    mkdir -p "$REPO_ROOT/projects/ired/results"
    # Explicitly include .pt files and all results
    rsync -av --include='*.pt' --include='*.json' --include='*.md' --include='*/' --exclude='*' "$WORK_DIR/projects/ired/results/" "$REPO_ROOT/projects/ired/results/"
    echo "Results copied to $REPO_ROOT/projects/ired/results/"
    # Also list what was copied for verification
    echo "Contents of copied results:"
    find "$REPO_ROOT/projects/ired/results" -type f -newer "$WORK_DIR/projects/ired/results" 2>/dev/null | head -20
else
    echo "Warning: No results directory found at $WORK_DIR/projects/ired/results"
fi

# Cleanup
echo "Cleaning up work directory..."
cd /
rm -rf "$WORK_DIR"

echo "Job completed at $(date)"
