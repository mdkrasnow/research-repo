#!/bin/bash
#SBATCH --job-name=ired_q001_baseline
#SBATCH --output=projects/ired/slurm/logs/%x_%j.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules (adjust based on cluster environment)
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

# Navigate to repo root
cd "${SLURM_SUBMIT_DIR}"

# Verify GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU info:"
    nvidia-smi
fi

# Run experiment
echo "Starting matrix inversion baseline experiment..."
python experiments/matrix_inversion_mining.py \
    --config configs/q001_baseline.json

echo "Experiment completed at $(date)"
