#!/bin/bash
#SBATCH --job-name=direct-fid2k
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-fid2k-%A.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-fid2k-%A.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --dependency=afterok:36847271

# Final direct epoch-80 sanity FID, released only after the continuation exits
# successfully.  The shared evaluator is single-GPU compatible with gpu_test.
exec bash /n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/jobs/eval_fid.sbatch
