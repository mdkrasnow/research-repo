#!/bin/bash
#SBATCH --job-name=eqm-energy-confirm-2
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=12:00:00
OUT=/n/holylabs/ydu_lab/Lab/mkrasnow_eqm/energy_candidate_ranking/confirmation_20260802_actual3
EVAL_CONFIG=configs/energy_candidate_ranking_confirmation_seed2.json
source "$(dirname "$0")/evaluate_energy_candidate_ranking_pilot.sbatch"
