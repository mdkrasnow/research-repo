#!/usr/bin/env bash
# Autonomous metacog watcher — runs as a CPU SLURM job so it survives the laptop
# session. Polls squeue; when the segmented smokes finish it dumps their meta;
# when the selection screen finishes it runs the aggregator (CPU scipy) and
# writes POLICY_SWEEP_RESULTS.txt. Does NOT launch any new GPU compute (promotion
# stays human-gated). Exits when no mc_ jobs remain. Marker files signal milestones.
set -uo pipefail
RR=/n/home03/mkrasnow/research-repo
D=$RR/projects/diff-EqM/experiments/separability_diagnostic
META=$D/runs/b2_vanilla/metacog
REF=$RR/projects/diff-EqM/results/in1k_reference_stats.npz
LOG=$META/WATCHER.log
mkdir -p "$META"
module load python/3.10.13-fasrc01 cuda/11.8.0-fasrc01 2>/dev/null
ts() { date '+%F %T'; }
echo "$(ts) watcher start" >> "$LOG"
for i in $(seq 1 200); do          # 200 * 240s ~ 13h
  q=$(squeue -u mkrasnow -h -o '%j' 2>/dev/null)
  pend=$(echo "$q" | grep -c '^mc_')
  smoke=$(echo "$q" | grep -c 'mc_smoke')
  screen=$(echo "$q" | grep -Ec 'mc_(random|energy_path|probe_k50|stacked|smc_|multiread)')
  echo "$(ts) poll$i mc_=$pend smoke=$smoke screen=$screen" >> "$LOG"

  if [ "$smoke" -eq 0 ] && [ ! -f "$META/SMOKE_DONE" ]; then
    : > "$META/SMOKE_META.txt"
    for a in smoke_churn smoke_heun smoke_optsw smoke_alloc; do
      echo -n "$a: " >> "$META/SMOKE_META.txt"
      cat "$META/$a/meta_rank0.json" 2>/dev/null >> "$META/SMOKE_META.txt" || echo MISSING >> "$META/SMOKE_META.txt"
      echo >> "$META/SMOKE_META.txt"
    done
    touch "$META/SMOKE_DONE"; echo "$(ts) SMOKE_DONE" >> "$LOG"
  fi

  if [ "$screen" -eq 0 ] && [ ! -f "$META/SCREEN_DONE" ]; then
    python "$D/aggregate_policy_sweep.py" --root "$META" --baseline probe_k50 \
      --ref-stats "$REF" --pareto-energy "$D/runs/b2_vanilla/pareto_r3energy" \
      > "$META/POLICY_SWEEP_RESULTS.txt" 2>&1
    touch "$META/SCREEN_DONE"; echo "$(ts) SCREEN_DONE aggregated" >> "$LOG"
  fi

  if [ "$pend" -eq 0 ]; then echo "$(ts) ALL_DONE" >> "$LOG"; break; fi
  sleep 240
done
echo "$(ts) watcher exit" >> "$LOG"
