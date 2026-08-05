#!/bin/bash
#SBATCH --job-name=eqm-mask-shared-d0
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/masked-field-shared-d0-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/masked-field-shared-d0-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterok:37389012

exec bash /n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/jobs/masked_field_shaping_eval.sbatch
