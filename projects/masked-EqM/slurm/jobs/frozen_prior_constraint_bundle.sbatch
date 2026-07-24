#!/bin/bash
# Bundled, resumable frozen-prior evaluation.  Intentionally has no --array:
# FASRC counts every array index toward the submit-job cap.
# Submit with STAGE=smoke|pilot|final and BUNDLE_ID.  Final additionally needs
# SOFT_RHO (locked after pilot) and may set SHARD_START/SHARD_END.
#SBATCH --job-name=eqm-frozen-prior
#SBATCH --output=projects/masked-EqM/slurm/logs/frozen-prior-%x_%j.out
#SBATCH --error=projects/masked-EqM/slurm/logs/frozen-prior-%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-23:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
set -euo pipefail

STAGE="${STAGE:?set STAGE=smoke|pilot|final}"; BUNDLE_ID="${BUNDLE_ID:-0}"
case "$STAGE" in smoke|pilot|final) ;; *) echo "bad STAGE=$STAGE"; exit 2;; esac
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi
WORK="/tmp/frozen-prior-${SLURM_JOB_ID}"; mkdir -p "$WORK"; cd "$WORK"
git init repo >/dev/null; cd repo
git remote add origin "${GIT_URL:?}"; git fetch --depth=1 origin "${GIT_SHA:?}"; git checkout --detach "$GIT_SHA"
cd projects/masked-EqM
python -m pip install --quiet timm diffusers accelerate torchvision lpips

VAL=/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val
OUT_ROOT="${OUT_ROOT:-/n/home03/mkrasnow/frozen-prior-constraint}"
mkdir -p "$OUT_ROOT/$STAGE"
mapfile -t ARMS < <(python -c 'import json; [print(x["id"]+"|"+x["path"]) for x in json.load(open("configs/frozen_prior_constraint_checkpoints.json"))["arms"]]')
run_one() {
  local line="$1" mode="$2" rho="$3" shard="$4" n_shards="$5" manifest="$6" steps="$7"
  local id="${line%%|*}" ckpt="${line#*|}"
  echo "stage=$STAGE bundle=$BUNDLE_ID id=$id mode=$mode rho=$rho shard=$shard/$n_shards"
  python frozen_prior_constraint.py --checkpoint "$ckpt" --checkpoint-id "$id" --data-path "$VAL" \
    --sample-manifest "$manifest" --manifest-split "${8}" --projection-mode "$mode" \
    --mask-family "${9:-all}" \
    --projection-strength "$rho" --steps "$steps" --shard-id "$shard" --num-shards "$n_shards" \
    --output-dir "$OUT_ROOT/$STAGE" --git-commit "$GIT_SHA" --strict --resume
}

if [ "$STAGE" = smoke ]; then
  # First representative checkpoint of each primary arm; 16 pilot examples,
  # combined mask, none and hard projection, 5 steps.
  for i in 0 3 6; do
    for mode in none hard; do run_one "${ARMS[$i]}" "$mode" 1 0 16 configs/frozen_prior_constraint_pilot_locked_v4.json 5 pilot combined; done
  done
elif [ "$STAGE" = pilot ]; then
  # BUNDLE_ID is matched training seed 0, 1, or 2.  One job contains all
  # three arms and the entire preregistered soft-rho grid.
  for i in "$BUNDLE_ID" "$((BUNDLE_ID+3))" "$((BUNDLE_ID+6))"; do
    run_one "${ARMS[$i]}" none 0 0 1 configs/frozen_prior_constraint_pilot_locked_v4.json 250 pilot
    run_one "${ARMS[$i]}" hard 1 0 1 configs/frozen_prior_constraint_pilot_locked_v4.json 250 pilot
    for rho in .25 .5 .75; do run_one "${ARMS[$i]}" soft "$rho" 0 1 configs/frozen_prior_constraint_pilot_locked_v4.json 250 pilot; done
  done
else
  # BUNDLE_ID maps directly to one of the nine locked checkpoints.  Bundle a
  # contiguous set of shards sequentially so a submission consumes one slot.
  : "${SOFT_RHO:?lock pilot rho before final}"
  START="${SHARD_START:-0}"; END="${SHARD_END:-0}"; N="${NUM_SHARDS:-8}"
  for shard in $(seq "$START" "$END"); do
    for mode in none hard soft; do
      rho=0; [ "$mode" = hard ] && rho=1; [ "$mode" = soft ] && rho="$SOFT_RHO"
      run_one "${ARMS[$BUNDLE_ID]}" "$mode" "$rho" "$shard" "$N" configs/frozen_prior_constraint_final_locked_v4.json 250 final
    done
  done
fi
echo "completed stage=$STAGE bundle=$BUNDLE_ID host=$(hostname) end=$(date -Is) out=$OUT_ROOT/$STAGE"
