#!/bin/bash
#SBATCH --job-name=direct-fid2k
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-fid2k-%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-fid2k-%A.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
# Final direct epoch-80 sanity FID. The continuation completed before this
# submission; the shared evaluator is single-GPU compatible with gpu_test.
# Arm-specific default so a direct submission of this wrapper cannot collide
# with the `none` arm's output. eval_fid.sbatch now requires OUT_NAME.
export OUT_NAME="${OUT_NAME:-direct_energy_fid2k}"
exec bash /n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/jobs/eval_fid.sbatch
