#!/bin/bash
#SBATCH --job-name=none-fid2k
#SBATCH --output=projects/masked-EqM/slurm/logs/none-fid2k-%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/none-fid2k-%A.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

exec "$(dirname "$0")/eval_fid.sbatch"
