#!/bin/bash
#SBATCH --job-name=eqm-epoch02-launch
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/epoch02-launch-%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/epoch02-launch-%j.err
#SBATCH --partition=shared
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=04:00:00
set -euo pipefail

while squeue -h -j 35377629,35377631 | grep -q .; do
  sleep 120
done

diag=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/jobs/direct_energy_checkpoint_diagnostics.sbatch
base=/n/home03/mkrasnow/direct_energy_longer_retry
out=/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/masked-EqM/results/direct_energy_campaign/training_diagnostics/checkpoint_metrics_epoch02.jsonl
nck="$base/longer_none_seed0_interleaved2_job35337285/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch02.pt"
dck="$base/longer_dot_seed0_interleaved2_job35337286/000-EqM-B-2-Linear-velocity-None-ebm-dot/checkpoints/epoch02.pt"
eck="$base/longer_direct_seed0_interleaved2_job35337288/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/epoch02.pt"
specs="none_seed0_epoch02=$nck;dot_seed0_epoch02=$dck;direct_seed0_epoch02=$eck"
sbatch --parsable --export=ALL,GIT_SHA=58b7229,CKPT_SPECS="$specs",OUT="$out" "$diag"
