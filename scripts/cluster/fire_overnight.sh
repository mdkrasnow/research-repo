#!/usr/bin/env bash
# Fire the prepared overnight cluster jobs the moment SSH is reachable.
# Idempotent: writes a marker after a successful submit; re-runs are no-ops.
# Submits: maze GPU scale-up (3 seeds) + online 50k adaptive (3 seeds).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MARKER="$ROOT/.overnight_fired"
CK="projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt"
RR="/n/home03/mkrasnow/research-repo"

if [ -f "$MARKER" ]; then echo "ALREADY_FIRED $(cat "$MARKER")"; exit 0; fi

# 1) liveness
if ! timeout 30 bash scripts/cluster/ssh.sh "echo OK" 2>/dev/null | grep -q OK; then
  echo "SSH_DOWN"; exit 7
fi
echo "SSH up — firing overnight jobs"

# 2) sync code + slurm jobs (no push needed; in-place)
CFG=.claude/cluster.local.json
USER=$(python3 -c "import json;print(json.load(open('$CFG'))['ssh']['user'])")
HOST=$(python3 -c "import json;print(json.load(open('$CFG'))['ssh']['host'])")
PORT=$(python3 -c "import json;print(json.load(open('$CFG'))['ssh'].get('port',22))")
CP=$(python3 -c "import json,os;print(os.path.expanduser(json.load(open('$CFG'))['ssh']['control_path']))")
SSH="ssh -p $PORT -o BatchMode=yes -o ControlPath=$CP"
rsync -az --exclude 'data/' --exclude 'runs/' --exclude '__pycache__/' -e "$SSH" \
  projects/diff-EqM/experiments/maze_eqm/ "$USER@$HOST:$RR/projects/diff-EqM/experiments/maze_eqm/" 2>&1 | tail -1
rsync -az -e "$SSH" projects/diff-EqM/slurm/jobs/ "$USER@$HOST:$RR/projects/diff-EqM/slurm/jobs/" 2>&1 | tail -1

IDS=""
# 3) maze GPU scale-up, 3 seeds (single GPU each, gpu partition)
for S in 0 1 2; do
  O=$(bash scripts/cluster/ssh.sh "cd $RR && sbatch -p gpu --gres=gpu:1 -c 7 --mem=32G -t 06:00:00 -J maze-gpu-s$S --export=ALL,START_SEED=$S,WIDTH=128,EPOCHS=80 projects/diff-EqM/slurm/jobs/maze_gpu.sbatch" 2>&1 | tail -1)
  echo "maze s$S: $O"; IDS="$IDS maze-s$S=$(echo "$O"|awk '{print $4}')"
done
# 4) online 50k adaptive, 3 seeds (4 GPU each)
for S in 0 1 2; do
  O=$(bash scripts/cluster/ssh.sh "cd $RR && sbatch -p gpu --gres=gpu:4 -t 12:00:00 -J online50k-s$S --export=ALL,CKPT_PATH=$CK,START_SEED=$S,NUM_SLOTS=50000,FLAG_FRAC=0.3 projects/diff-EqM/slurm/jobs/online_seed.sbatch" 2>&1 | tail -1)
  echo "online s$S: $O"; IDS="$IDS online-s$S=$(echo "$O"|awk '{print $4}')"
done
echo "$IDS" > "$MARKER"
echo "FIRED:$IDS"
