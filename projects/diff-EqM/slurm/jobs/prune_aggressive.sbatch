#!/bin/bash
#SBATCH --job-name=prune-aggr
#SBATCH --output=slurm/logs/prune-aggr_%A.out
#SBATCH --error=slurm/logs/prune-aggr_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=48:00:00
#SBATCH --partition=shared

# Aggressive standing ckpt pruner (deployed 2026-06-04 after home hit 100% with the
# lenient prune_all_active still running). Lower steady-state floor on the 95G home:
#   - Smoke dirs (*smoke*): delete ALL ckpts.
#   - Full run dirs (*80ep*): keep ONLY {0065000.pt} anchor (early-FID multi-fidelity
#     pair) + latest 2 (latest-2 not latest-1: guards against deleting a mid-write ckpt).
#     Drops the 0005000.pt anchor the old pruner kept (~2GB/dir reclaimed).
#   - Bridge/variant dirs (variants/bridge*): keep latest 1 checkpoint.pt + final.pt only.
#   - Rsync temp files (.*.pt.*) deleted everywhere.
# Runs every 180s. Exits when user has no train jobs in queue.
# NOTE: prunes ALL run dirs regardless of which experiment owns them (authorized).

set -uo pipefail

RESULTS_ROOT="/n/home03/mkrasnow/research-repo/projects/diff-EqM/results"
ANCHORS="0065000.pt"
USER_TAG="mkrasnow"

echo "=== prune-aggr started $(date) ===" >&2

while true; do
    ACTIVE=$(squeue -u "$USER_TAG" -h -o '%j' 2>/dev/null | grep -cE 'eqm-1k|eqm-imagenet|prune-v10|eqm-v10|eqm-van|eqm-ab|variant-pilot' || true)
    if [ "$ACTIVE" -eq 0 ]; then
        echo "=== No active train jobs, exiting $(date) ===" >&2
        exit 0
    fi

    # Smoke dirs: delete all ckpts.
    for d in $RESULTS_ROOT/*smoke*/000-*/checkpoints; do
        [ -d "$d" ] || continue
        N=$(ls "$d"/*.pt 2>/dev/null | wc -l)
        [ "$N" -gt 0 ] && { rm -f "$d"/*.pt; echo "$(date) smoke pruned $N from $d" >&2; }
    done

    # Rsync temp cleanup everywhere.
    find "$RESULTS_ROOT" -name '.*.pt.*' -type f -delete 2>/dev/null || true

    # Full 80ep dirs: keep {65000 anchor} + latest 2.
    for d in $RESULTS_ROOT/*80ep*/000-*/checkpoints; do
        [ -d "$d" ] || continue
        cd "$d"
        ls *.pt 2>/dev/null | sort > /tmp/pa_all_$$.txt
        KEEP_LAST=$(tail -2 /tmp/pa_all_$$.txt)
        for f in $ANCHORS $KEEP_LAST; do echo "$f"; done | sort -u > /tmp/pa_keep_$$.txt
        DEL=$(comm -23 /tmp/pa_all_$$.txt /tmp/pa_keep_$$.txt)
        if [ -n "$DEL" ]; then
            N=$(echo "$DEL" | wc -l)
            echo "$(date) full pruned $N from $d" >&2
            echo "$DEL" | xargs -r rm -f
        fi
        rm -f /tmp/pa_all_$$.txt /tmp/pa_keep_$$.txt
        cd /
    done

    # Bridge/variant dirs: keep latest checkpoint.pt + final.pt; drop older numbered ckpts.
    for d in $RESULTS_ROOT/variants/bridge*; do
        [ -d "$d" ] || continue
        find "$d" -name '0*.pt' -type f -delete 2>/dev/null || true
    done

    sleep 180
done
