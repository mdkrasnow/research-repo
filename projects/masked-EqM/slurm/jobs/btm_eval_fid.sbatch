#!/bin/bash
#SBATCH --job-name=btm-fid
#SBATCH --output=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-fid-%A.out
#SBATCH --error=/n/home03/mkrasnow/research-repo/projects/masked-EqM/slurm/logs/btm-fid-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
set -euo pipefail

# Sanity-scale FID for BTM arms.  Single GPU, so gpu_test MIG slices are fine.
#
# Two deliberate departures from slurm/jobs/eval_fid.sbatch:
#
# 1. It clones at GIT_SHA instead of unpacking a SOURCE_ARCHIVE.  The archive
#    directory only holds a stale sha (3ffaeb5, Aug 5); the BTM commits have no
#    archive, and the clone prelude is what every other BTM job already uses.
#
# 2. TOTAL_TIME is the knob, not NUM_SAMPLING_STEPS.  The legacy default is
#    250 x 0.003 = 0.75, tuned for legacy EqM.  BTM's self-stopping interpolant
#    has Idot_1 = 0, so the drift VANISHES at the data end: under-integrating
#    truncates transport, over-integrating is harmless.  Inheriting T = 0.75
#    unexamined would handicap every BTM arm at once and look like a method
#    failure.  Step size is held FIXED so that varying T changes the integration
#    horizon and not the discretization error.
#
# Env: GIT_SHA CKPT OUT_NAME (all required); EBM TOTAL_TIME STEPSIZE
#      NUM_SAMPLES MODEL VAL_PATH.

module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi || { echo "ERROR: no GPU"; exit 1; }

GIT_SHA="${GIT_SHA:?set GIT_SHA}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
CKPT="${CKPT:?set CKPT}"
# No default: a shared default name means two arms silently overwrite each
# other's result file.  Same reasoning as eval_fid.sbatch.
# NOTE: no apostrophe in this message.  Bash quote-processes the word in
# ${VAR:?word} even inside double quotes, so a lone ' opens a quote and the
# whole script dies with "unexpected EOF" at parse time -- before the first
# command runs, which is why it fails in 2 seconds with no useful output.
OUT_NAME="${OUT_NAME:?set OUT_NAME -- a shared default silently overwrites the other arm FID result}"
EBM="${EBM:-none}"
MODEL="${MODEL:-EqM-B/2}"
STEPSIZE="${STEPSIZE:-0.003}"
TOTAL_TIME="${TOTAL_TIME:-0.75}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VAL_PATH="${VAL_PATH:-/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val}"
# netscratch, NOT holylabs: the holylabs GROUP quota is exhausted (EDQUOT), and
# it killed two completed FID computations at the final write.  netscratch has
# 579T free and is where every BTM training run already writes.
OUT_DIR="${OUT_DIR:-/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/btm/eval_results}"

# Integer step count from the requested horizon, then report the horizon that
# was ACTUALLY integrated -- rounding must never be silent, because the whole
# point of the sweep is to compare horizons.
NUM_STEPS=$(python3 -c "print(int(round($TOTAL_TIME / $STEPSIZE)))")
EFF_T=$(python3 -c "print($NUM_STEPS * $STEPSIZE)")
echo "FID: ckpt=$CKPT ebm=$EBM requested_T=$TOTAL_TIME effective_T=$EFF_T steps=$NUM_STEPS stepsize=$STEPSIZE"

WORK_DIR="/tmp/btm-fid-${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin '+refs/heads/*:refs/remotes/origin/*'
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM
pip install --quiet timm 'diffusers==0.27.2' 'huggingface_hub==0.23.5' \
    'transformers==4.40.2' accelerate torchvision pytorch-fid 2>&1 | tail -5

mkdir -p "$OUT_DIR"
python eval_fid.py \
    --ckpt "$CKPT" \
    --data-path "$VAL_PATH" \
    --model "$MODEL" \
    --ebm "$EBM" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --num-sampling-steps "$NUM_STEPS" \
    --stepsize "$STEPSIZE" \
    --work-dir "$WORK_DIR/fid_work" \
    --out "$OUT_DIR/${OUT_NAME}.json"

echo "wrote $OUT_DIR/${OUT_NAME}.json"
rm -rf "$WORK_DIR"
