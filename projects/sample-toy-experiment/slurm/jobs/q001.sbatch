#!/bin/bash
#SBATCH --job-name=sample-toy-experiment_q001
#SBATCH --output=projects/sample-toy-experiment/slurm/logs/%x_%j.out
#SBATCH --error=projects/sample-toy-experiment/slurm/logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 projects/sample-toy-experiment/experiments/coinflip.py --config projects/sample-toy-experiment/configs/q001.json
