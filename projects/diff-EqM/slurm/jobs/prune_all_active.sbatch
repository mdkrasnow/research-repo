#!/bin/bash
#SBATCH --job-name=prune-all
#SBATCH --output=slurm/logs/prune-all_%A.out
#SBATCH --error=slurm/logs/prune-all_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=48:00:00
#SBATCH --partition=shared

# Generic multi-dir ckpt pruner.
# - Smoke dirs (imagenet1k_*ep_*smoke*): delete ALL ckpts (smokes never need them).
# - Full run dirs (imagenet1k_*ep_*): keep anchors {5000, 65000} + latest 2.
# Runs every 5min. Exits when user has no v10 / vanilla / smoke jobs in queue.

set -uo pipefail

RESULTS_ROOT="/n/home03/mkrasnow/research-repo/projects/diff-EqM/results"
ANCHORS="0005000.pt 0065000.pt"
USER_TAG="mkrasnow"

echo "=== prune-all started $(date) ===" >&2

while true; do
    # Check if any relevant jobs still running/pending
    ACTIVE=$(squeue -u "$USER_TAG" -h -o '%j' 2>/dev/null | grep -cE 'eqm-1k|eqm-imagenet|prune-v10' || true)
    if [ "$ACTIVE" -eq 0 ]; then
        echo "=== No active jobs, exiting $(date) ===" >&2
        exit 0
    fi

    # Smoke dirs: delete all ckpts
    for d in $RESULTS_ROOT/imagenet1k_*smoke*/000-*/checkpoints; do
        [ -d "$d" ] || continue
        N=$(ls "$d"/*.pt 2>/dev/null | wc -l)
        if [ "$N" -gt 0 ]; then
            rm -f "$d"/*.pt
            echo "$(date) smoke pruned $N from $d" >&2
        fi
    done

    # Full run dirs: keep anchors + latest 2
    for d in $RESULTS_ROOT/imagenet1k_80ep_*/000-*/checkpoints; do
        [ -d "$d" ] || continue
        cd "$d"
        ls *.pt 2>/dev/null | sort > /tmp/prune_all_$$.txt
        KEEP_LAST=$(tail -2 /tmp/prune_all_$$.txt)
        for f in $ANCHORS $KEEP_LAST; do echo "$f"; done | sort -u > /tmp/prune_keep_$$.txt
        DEL=$(comm -23 /tmp/prune_all_$$.txt /tmp/prune_keep_$$.txt)
        if [ -n "$DEL" ]; then
            N=$(echo "$DEL" | wc -l)
            echo "$(date) full pruned $N from $d" >&2
            echo "$DEL" | xargs -r rm -f
        fi
        rm -f /tmp/prune_all_$$.txt /tmp/prune_keep_$$.txt
        cd /
    done

    sleep 300
done
