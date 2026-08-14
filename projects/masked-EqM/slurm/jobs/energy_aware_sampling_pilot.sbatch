#!/bin/bash
#SBATCH --job-name=eqm-armijo-pilot
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/armijo-pilot-%j.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/armijo-pilot-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true; module load cuda/11.8.0-fasrc01 2>/dev/null || true
W=/tmp/eqm-armijo-pilot-$SLURM_JOB_ID; R=/n/home03/mkrasnow/energy_aware_sampling/epoch15_armijo
mkdir -p "$W" "$R"; cd "$W"; git init .; git remote add origin "${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"; git fetch --depth=50 origin main; git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }; cd projects/masked-EqM
pip install --quiet timm diffusers torchvision pytorch-fid
read BEST_MULTIPLIER BEST_STEP <<EOF
$(python - <<'PY'
import json
from pathlib import Path
root=Path('/n/home03/mkrasnow/energy_aware_sampling/epoch15_armijo/calibration')
rows=[]
for path in root.glob('m*/metrics.json'):
 d=json.loads(path.read_text())
 if d['finite_samples']==d['samples']:
  rows.append((d['fid'],d['multiplier'],d['step']))
if not rows: raise SystemExit('no finite calibration metrics')
_,m,step=min(rows)
print(m,step)
PY
)
EOF
echo "Selected fixed reference multiplier=$BEST_MULTIPLIER step=$BEST_STEP"
python energy_aware_sampling/smoke_pipeline.py --direct /n/home03/mkrasnow/direct_energy_longer_retry/longer15_direct_seed0_job35524667/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/epoch15.pt --dot /n/home03/mkrasnow/direct_energy_longer_retry/longer15_dot_seed0_job35524601/000-EqM-B-2-Linear-velocity-None-ebm-dot/checkpoints/epoch15.pt --real /n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k/val --output "$R/pilot_${SLURM_JOB_ID}_m${BEST_MULTIPLIER}" --samples 1024 --steps 100 --step "$BEST_STEP" --seed 20260730
