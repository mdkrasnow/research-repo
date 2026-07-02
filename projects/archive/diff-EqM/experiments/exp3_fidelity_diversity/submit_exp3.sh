#!/usr/bin/env bash
# Experiment 3 launcher: build the shared schedule locally, rsync to cluster,
# then submit  reference -> generate(vanilla) + generate(anm) -> metrics  with
# SLURM dependencies. Vanilla & ANM share the identical schedule.json.
#
# Usage (full run, lambda=0.3 = best ANM arm):
#   bash projects/diff-EqM/experiments/exp3_fidelity_diversity/submit_exp3.sh \
#     --vanilla-ckpt projects/diff-EqM/results/stage_b_vanilla_in1k_80ep_seed0/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/0380000.pt \
#     --anm-ckpt    projects/diff-EqM/results/imagenet1k_80ep_v10_b2_lambda03_seed0 \
#     --num-classes 1000 --samples-per-class 50 --out exp3/full_lambda03_vs_vanilla
#
# Smoke (plumbing only, ~1000 samples):
#   bash .../submit_exp3.sh --smoke --vanilla-ckpt <p> --anm-ckpt <p> --out exp3/smoke
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults ----
NUM_CLASSES=1000; SAMPLES_PER_CLASS=50; BASE_SEED=0; SHUFFLE_SEED=0
MODEL="EqM-B/2"; STEPSIZE=0.003; NUM_SAMPLING_STEPS=250; SAMPLE_BATCH_SIZE=64
VANILLA_CKPT=""; ANM_CKPT=""; OUT_REL=""; SMOKE=0; BUILD_REFERENCE=1
REF_DIR_REL="projects/diff-EqM/results/exp3/reference"; OVERWRITE=0

while [ $# -gt 0 ]; do case "$1" in
    --vanilla-ckpt) VANILLA_CKPT="$2"; shift 2;;
    --anm-ckpt) ANM_CKPT="$2"; shift 2;;
    --num-classes) NUM_CLASSES="$2"; shift 2;;
    --samples-per-class) SAMPLES_PER_CLASS="$2"; shift 2;;
    --base-seed) BASE_SEED="$2"; shift 2;;
    --label-schedule-seed) SHUFFLE_SEED="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --stepsize) STEPSIZE="$2"; shift 2;;
    --num-sampling-steps) NUM_SAMPLING_STEPS="$2"; shift 2;;
    --sample-batch-size) SAMPLE_BATCH_SIZE="$2"; shift 2;;
    --out) OUT_REL="$2"; shift 2;;
    --ref-dir) REF_DIR_REL="$2"; shift 2;;
    --no-build-reference) BUILD_REFERENCE=0; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --smoke) SMOKE=1; shift;;
    *) echo "unknown arg: $1"; exit 1;;
esac; done

[ -n "$VANILLA_CKPT" ] || { echo "ERROR: --vanilla-ckpt required"; exit 1; }
[ -n "$ANM_CKPT" ]     || { echo "ERROR: --anm-ckpt required"; exit 1; }
[ -n "$OUT_REL" ]      || { echo "ERROR: --out required (rel to results/)"; exit 1; }

if [ "$SMOKE" = "1" ]; then
    NUM_CLASSES=100; SAMPLES_PER_CLASS=10
    echo "### SMOKE MODE: ${NUM_CLASSES}x${SAMPLES_PER_CLASS} samples. FID/KID/PRDC are PLUMBING-ONLY. ###"
fi

RUN_DIR_REL="projects/diff-EqM/results/$OUT_REL"
LOCAL_RUN_DIR="$REPO_ROOT/$RUN_DIR_REL"
mkdir -p "$LOCAL_RUN_DIR"

# ---- 1. build shared schedule locally (numpy only) ----
python projects/diff-EqM/experiments/exp3_fidelity_diversity/schedule.py \
    --num-classes "$NUM_CLASSES" --samples-per-class "$SAMPLES_PER_CLASS" \
    --base-seed "$BASE_SEED" --shuffle-seed "$SHUFFLE_SEED" \
    --out "$LOCAL_RUN_DIR/schedule.json"

GIT_SHA="$(git rev-parse HEAD)"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
ahead="$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)"
if [ "$ahead" -gt 0 ]; then echo "ERROR: $ahead commits ahead of origin/main. Push first."; exit 1; fi

CONTROL_PATH="$HOME/.ssh/cc-research-repo-%r@%h:%p"
SSH="ssh -o ControlPath=$CONTROL_PATH"
CLUSTER="mkrasnow@login.rc.fas.harvard.edu"
REMOTE_ROOT="/n/home03/mkrasnow/research-repo"

# rsync slurm + schedule.json to cluster
rsync -az -e "$SSH" projects/diff-EqM/slurm/ "$CLUSTER:$REMOTE_ROOT/projects/diff-EqM/slurm/"
$SSH "$CLUSTER" "mkdir -p $REMOTE_ROOT/$RUN_DIR_REL"
rsync -az -e "$SSH" "$LOCAL_RUN_DIR/schedule.json" "$CLUSTER:$REMOTE_ROOT/$RUN_DIR_REL/schedule.json"

SUB="cd $REMOTE_ROOT &&"
COMMON_EXPORT="GIT_URL=$GIT_URL,GIT_SHA=$GIT_SHA"

echo "=== submitting Exp3 chain (SHA $GIT_SHA) ==="

# ---- 2. reference (optional) ----
DEP_REF=""
if [ "$BUILD_REFERENCE" = "1" ]; then
    REF_AGG=$((NUM_CLASSES * SAMPLES_PER_CLASS)); [ "$SMOKE" = "1" ] || REF_AGG=50000
    REF_PER_CLASS=100; [ "$SMOKE" = "1" ] && REF_PER_CLASS=20
    REF_JOB=$(bash scripts/cluster/ssh.sh "$SUB sbatch --parsable \
        --export=$COMMON_EXPORT,OUT_DIR=$REF_DIR_REL,NUM_CLASSES=$NUM_CLASSES,AGGREGATE_N=$REF_AGG,REF_PER_CLASS=$REF_PER_CLASS \
        projects/diff-EqM/slurm/jobs/exp3_reference_features.sbatch")
    echo "reference job: $REF_JOB"
    DEP_REF="--dependency=afterok:$REF_JOB"
fi

# ---- 3. generate both arms (depend on reference) ----
GEN_V=$(bash scripts/cluster/ssh.sh "$SUB sbatch --parsable $DEP_REF \
    --export=$COMMON_EXPORT,CKPT_PATH=$VANILLA_CKPT,TAG=vanilla,SCHEDULE_REL=$RUN_DIR_REL/schedule.json,GEN_DIR_REL=$RUN_DIR_REL/gen/vanilla,MODEL=$MODEL,STEPSIZE=$STEPSIZE,NUM_SAMPLING_STEPS=$NUM_SAMPLING_STEPS,SAMPLE_BATCH_SIZE=$SAMPLE_BATCH_SIZE,OVERWRITE=$OVERWRITE \
    projects/diff-EqM/slurm/jobs/exp3_generate.sbatch")
GEN_A=$(bash scripts/cluster/ssh.sh "$SUB sbatch --parsable $DEP_REF \
    --export=$COMMON_EXPORT,CKPT_PATH=$ANM_CKPT,TAG=anm,SCHEDULE_REL=$RUN_DIR_REL/schedule.json,GEN_DIR_REL=$RUN_DIR_REL/gen/anm,MODEL=$MODEL,STEPSIZE=$STEPSIZE,NUM_SAMPLING_STEPS=$NUM_SAMPLING_STEPS,SAMPLE_BATCH_SIZE=$SAMPLE_BATCH_SIZE,OVERWRITE=$OVERWRITE \
    projects/diff-EqM/slurm/jobs/exp3_generate.sbatch")
echo "generate vanilla: $GEN_V   anm: $GEN_A"

# ---- 4. metrics (depend on both generates) ----
MET=$(bash scripts/cluster/ssh.sh "$SUB sbatch --parsable --dependency=afterok:$GEN_V:$GEN_A \
    --export=$COMMON_EXPORT,RUN_DIR_REL=$RUN_DIR_REL,REF_DIR_REL=$REF_DIR_REL,VANILLA_CKPT=$VANILLA_CKPT,ANM_CKPT=$ANM_CKPT \
    projects/diff-EqM/slurm/jobs/exp3_metrics.sbatch")
echo "metrics: $MET"

echo ""
echo "=== Exp3 submitted. Add active_runs entries to pipeline.json for: ${REF_JOB:-none} $GEN_V $GEN_A $MET ==="
echo "=== Results will land in $RUN_DIR_REL (aggregate_metrics.csv/json, class_metrics.csv, plots/, README.md) ==="
