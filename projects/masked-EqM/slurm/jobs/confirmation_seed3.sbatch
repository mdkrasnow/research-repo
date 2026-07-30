#!/bin/bash
#SBATCH --job-name=eqm-energy-confirm-3
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=12:00:00
OUT=/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/energy_candidate_ranking/confirmation_20260803_actual1
EVAL_CONFIG=configs/energy_candidate_ranking_confirmation_seed3.json
source /n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/jobs/evaluate_energy_candidate_ranking_pilot.sbatch
