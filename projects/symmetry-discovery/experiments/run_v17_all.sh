#!/bin/bash
# v17 MorphismGym ladder driver. Bounded parallelism (avoid CPU thrash on 4 cores).
# Usage: bash run_v17_all.sh phase0|phase1|phase2 [PAR]
set -u
cd "$(dirname "$0")"
PHASE="${1:-phase0}"
PAR="${2:-2}"
export OMP_NUM_THREADS=2

run_pool() {  # read commands from stdin, run PAR at a time
  local par="$1"; shift
  cat | xargs -P "$par" -I{} bash -c '{}'
}

if [ "$PHASE" = "phase0" ]; then
  {
    echo "python v17_run_calibration.py --part 0A --seed 0"
    echo "python v17_run_calibration.py --part 0C --seed 0"
    echo "python v17_run_calibration.py --part 0D --seed 0"
    echo "python v17_run_calibration.py --part 0B --seed 0"
    echo "python v17_run_calibration.py --part 0B --seed 1"
  } | run_pool "$PAR"
  python v17_collect_results.py --phase 0

elif [ "$PHASE" = "phase1" ]; then
  TASKS="single_rotation single_hue single_scale multi_independent multi_composed decoy_pressure impossible_control"
  {
    for t in $TASKS; do for s in 0 1 2; do
      echo "python v17_run_discovery.py --task $t --seed $s"
    done; done
  } | run_pool "$PAR"
  python v17_collect_results.py --phase 1

elif [ "$PHASE" = "phase2" ]; then
  TASKS="${V17_PAYOFF_TASKS:-multi_independent}"
  {
    for t in $TASKS; do for s in 0 1 2; do for p in classifier eqm_lite; do
      echo "python v17_run_payoff.py --task $t --seed $s --proxy $p"
    done; done; done
  } | run_pool "$PAR"
  python v17_collect_results.py --phase 2
fi
echo "DONE $PHASE"
