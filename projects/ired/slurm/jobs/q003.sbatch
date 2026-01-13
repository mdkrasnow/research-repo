#!/bin/bash
#SBATCH --job-name=ired_q003_adversarial
#SBATCH --output=projects/ired/slurm/logs/%x_%j.out
#SBATCH --error=projects/ired/slurm/logs/%x_%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

cd "${SLURM_SUBMIT_DIR}"

# Run experiment
echo "Starting matrix inversion with adversarial negative mining..."
python experiments/matrix_inversion_mining.py \
    --config configs/q003_adversarial.json

echo "Experiment completed at $(date)"
