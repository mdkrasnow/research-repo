#!/bin/bash
#SBATCH --job-name=maze-gpu
#SBATCH --output=slurm/logs/maze-gpu_%A.out
#SBATCH --error=slurm/logs/maze-gpu_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Maze-EqM GPU scale-up: regen data, train a WIDER EqM on harder (c7 15x15) mazes
# natively, multi-seed (START_SEED), then run the metacognition sweep on it.
# Single-GPU, in-place (data regenerated on node — deterministic, gitignored locally).
#   START_SEED=0 sbatch projects/diff-EqM/slurm/jobs/maze_gpu.sbatch

set -euo pipefail
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
nvidia-smi || { echo "no gpu"; exit 1; }
REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT/projects/diff-EqM/experiments/maze_eqm"
pip install torch numpy 2>&1 | tail -2
START_SEED="${START_SEED:-0}"
WIDTH="${WIDTH:-128}"
EPOCHS="${EPOCHS:-80}"
export PYTHONUNBUFFERED=1

# regen data (deterministic) if absent
[ -f data/maze_c7_train.npz ]  || python gen_maze_data.py --cells 7  --n 5000 --out data/maze_c7_train.npz --seed 71
[ -f data/maze_c7_test.npz ]   || python gen_maze_data.py --cells 7  --n 1000 --out data/maze_c7_test.npz  --seed 72
[ -f data/maze_c10_ood.npz ]   || python gen_maze_data.py --cells 10 --n 1000 --out data/maze_c10_ood.npz  --seed 10

echo "=== TRAIN width=$WIDTH epochs=$EPOCHS seed=$START_SEED on c7 (15x15) $(date) ==="
python eqm_maze.py --train data/maze_c7_train.npz --test data/maze_c7_test.npz \
    --epochs "$EPOCHS" --width "$WIDTH" --eta 0.02 --steps 200 --eval-every 10 \
    --seed "$START_SEED" --out "runs/maze_gpu_s${START_SEED}/model.pt"

echo "=== METACOG (c10 OOD) seed=$START_SEED $(date) ==="
python maze_metacog.py --ckpt "runs/maze_gpu_s${START_SEED}/model.pt" \
    --data data/maze_c10_ood.npz --steps 25 --eta 0.01 --R 4 --n 800 --seed "$START_SEED" \
    --out "runs/maze_gpu_metacog_s${START_SEED}"
echo "=== DONE seed=$START_SEED $(date) ==="
