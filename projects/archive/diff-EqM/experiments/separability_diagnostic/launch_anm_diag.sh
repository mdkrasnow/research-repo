#!/bin/bash
# Task 4 — ANM/v10 small diagnostic launcher (PREP ONLY; does NOT auto-submit).
#
# Purpose: run the SAME separability diagnostic on an ANM/v10 checkpoint at small
# scale (512-1000 samples) to test whether ANM training changes the gradient-
# metacognition signal vs vanilla B/2. Reuses sep_diag_local.sbatch unchanged.
#
# This script ECHOES the exact sbatch command and exits. To actually submit, run
# it with SUBMIT=1 (explicit opt-in), or copy/paste the printed command.
#
# Env (all optional): V10_CKPT, NUM_SAMPLES (512), NUM_REAL (5000), BATCH_SIZE (32),
#   PARTITION (gpu_test), RUN_TAG (b2_v10_anm).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
V10_CKPT="${V10_CKPT:-projects/diff-EqM/results/imagenet1k_80ep_v10_b2_seed1/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
NUM_REAL="${NUM_REAL:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RUN_TAG="${RUN_TAG:-b2_v10_anm}"
SBATCH="projects/diff-EqM/slurm/jobs/sep_diag_local.sbatch"

CMD="cd $REPO_ROOT && CKPT_PATH=$V10_CKPT NUM_SAMPLES=$NUM_SAMPLES NUM_REAL=$NUM_REAL \\
    BATCH_SIZE=$BATCH_SIZE RUN_TAG=$RUN_TAG MODEL=EqM-B/2 \\
    sbatch -p ${PARTITION:-gpu_test} -c 7 --mem=60G $SBATCH"

echo "=== ANM/v10 separability diagnostic (PREP) ==="
echo "ckpt:        $V10_CKPT"
echo "samples:     $NUM_SAMPLES  real_ref: $NUM_REAL  batch: $BATCH_SIZE"
echo "run_tag:     $RUN_TAG  -> runs/$RUN_TAG/"
echo "compare vs:  runs/b2_vanilla/ (vanilla baseline)"
echo ""
echo "Command:"
echo "  $CMD"
echo ""
if [ "${SUBMIT:-0}" = "1" ]; then
    echo "SUBMIT=1 set -> submitting now."
    eval "$CMD"
else
    echo "PREP ONLY. To submit: re-run with SUBMIT=1, or paste the command above."
    echo "After it finishes, compare signal vs vanilla:"
    echo "  python projects/diff-EqM/experiments/separability_diagnostic/baseline_table.py \\"
    echo "      --folder projects/diff-EqM/experiments/separability_diagnostic/runs/$RUN_TAG"
fi
