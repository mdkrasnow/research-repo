#!/bin/bash
#SBATCH --job-name=none-fid2k
#SBATCH --output=projects/masked-EqM/slurm/logs/none-fid2k-%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/none-fid2k-%A.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# Arm-specific default so a direct submission of this wrapper cannot collide
# with the `direct` arm's output. eval_fid.sbatch now requires OUT_NAME.
export OUT_NAME="${OUT_NAME:-none_energy_fid2k}"
exec bash /n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/jobs/eval_fid.sbatch
