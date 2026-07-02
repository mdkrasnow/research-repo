#!/bin/bash
#SBATCH --job-name=sep-diag-posthoc
#SBATCH --output=slurm/logs/sep-diag-posthoc_%A.out
#SBATCH --error=slurm/logs/sep-diag-posthoc_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --partition=seas_compute

# Post-hoc analysis ladder. CPU-ONLY (no GPU): waits for the main diagnostic's
# VERDICT.txt to appear in RUN_FOLDER/results/, then runs robustness_analysis.py
# + dynamics_probe.py from cached logs/labels/scores and writes a combined
# POSTHOC_SUMMARY.md. Does NOT regenerate samples/features/labels.
#
#   cd <REPO_ROOT> && RUN_FOLDER=projects/diff-EqM/experiments/separability_diagnostic/runs/b2_vanilla \
#       sbatch projects/diff-EqM/slurm/jobs/sep_diag_posthoc.sbatch
#
# Optional env: WAIT_SECS (default 21600 = 6h), LABEL_QUANTILES.

set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true

REPO_ROOT="${REPO_ROOT:-/n/home03/mkrasnow/research-repo}"
cd "$REPO_ROOT"
RUN_FOLDER="${RUN_FOLDER:?RUN_FOLDER required (repo-relative or absolute)}"
case "$RUN_FOLDER" in /*) : ;; *) RUN_FOLDER="$REPO_ROOT/$RUN_FOLDER" ;; esac
WAIT_SECS="${WAIT_SECS:-21600}"
LABEL_QUANTILES="${LABEL_QUANTILES:-0.2,0.25,0.3,0.35}"
SD="$REPO_ROOT/projects/diff-EqM/experiments/separability_diagnostic"
VERDICT="$RUN_FOLDER/results/VERDICT.txt"
export PYTHONUNBUFFERED=1

pip install numpy 2>&1 | tail -1 || true

echo "=== waiting for $VERDICT (timeout ${WAIT_SECS}s) $(date) ==="
WAITED=0
while [ ! -f "$VERDICT" ]; do
    if [ "$WAITED" -ge "$WAIT_SECS" ]; then
        echo "TIMEOUT after ${WAITED}s: main diagnostic VERDICT.txt never appeared at $VERDICT."
        echo "Exiting cleanly without running posthoc (no cached outputs to analyze)."
        exit 0
    fi
    sleep 60; WAITED=$((WAITED + 60))
done
echo "=== VERDICT present; running posthoc ladder $(date) ==="

POST="$RUN_FOLDER/results/POSTHOC_SUMMARY.md"
{
    echo "# Post-hoc analysis ladder — combined summary"
    echo ""
    echo "Run folder: $RUN_FOLDER"
    echo "Generated: $(date)"
    echo ""
} > "$POST"

RB_OK=1; DP_OK=1
echo "=== robustness_analysis.py $(date) ==="
python "$SD/robustness_analysis.py" --folder "$RUN_FOLDER" \
    --label-quantiles "$LABEL_QUANTILES" 2>&1 | tee /tmp/rb_${SLURM_JOB_ID}.log || RB_OK=0

echo "=== dynamics_probe.py $(date) ==="
python "$SD/dynamics_probe.py" --folder "$RUN_FOLDER" 2>&1 | tee /tmp/dp_${SLURM_JOB_ID}.log || DP_OK=0

{
    echo "## 1. Endpoint separability robustness (label-threshold sweep)"
    echo ""
    if [ "$RB_OK" = 1 ] && [ -f "$RUN_FOLDER/results/robustness/ROBUSTNESS_SUMMARY.md" ]; then
        cat "$RUN_FOLDER/results/robustness/ROBUSTNESS_SUMMARY.md"
    else
        echo "(robustness_analysis.py failed — see slurm log)"
    fi
    echo ""
    echo "## 2. Learned dynamics probe (trajectory shape)"
    echo ""
    if [ "$DP_OK" = 1 ] && [ -f "$RUN_FOLDER/results/dynamics_probe/DYNAMICS_PROBE_SUMMARY.md" ]; then
        cat "$RUN_FOLDER/results/dynamics_probe/DYNAMICS_PROBE_SUMMARY.md"
    else
        echo "(dynamics_probe.py failed — see slurm log)"
    fi
    echo ""
    echo "## Interpretation"
    echo "- Endpoint KILL + dynamics GREEN  => failure is invisible at the stop but"
    echo "  visible in the relaxation dynamics; pursue METACOGNITIVE_RESCUE_SPEC.md."
    echo "- Endpoint + dynamics both KILL    => no usable inference-time signal; stop."
    echo "- Endpoint GREEN                    => simplest readout already works."
} >> "$POST"

echo "=== DONE $(date); wrote $POST ==="
cat "$POST" || true
cp /tmp/rb_${SLURM_JOB_ID}.log /tmp/dp_${SLURM_JOB_ID}.log \
   "$REPO_ROOT/projects/diff-EqM/slurm/logs/" 2>/dev/null || true
