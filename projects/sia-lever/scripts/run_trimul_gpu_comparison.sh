#!/usr/bin/env bash
# One-command SIA-W+H (paper-style) vs SIA-Lever comparison on the TriMul GPU-kernel task.
#
# Runs everywhere (CPU fallback) and uses real GPU kernels + CUDA-event timing automatically when a
# CUDA device + Triton are present (--real-latency). On a GPU box this is the closest faithful run of
# the SIA paper's TriMul kernel-optimization task that the public artifacts allow (the paper's exact
# W+H code is not public — see documentation/reproduction_limits.md).
#
# Steps: env -> kernel bench -> build measured cache -> build SIA custom-task data -> headroom probe
#        -> [optional] base/LoRA gpt-oss selector rollouts -> compare -> table/plot.
#
# Flags:
#   --real-latency        measure kernel latencies on the device (CUDA events on GPU)
#   --reps <n>            cache reps per config (default 10)
#   --lever-rollouts <g>  glob of gpt-oss rollout jsonl -> adds the learned SIA-Lever column
#   --dry-run             print steps only
set -euo pipefail
cd "$(dirname "$0")/.."

REPS=10; REAL=""; ROLL=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --real-latency) REAL="--real-latency";;
    --reps) REPS="$2"; shift;;
    --lever-rollouts) ROLL="$2"; shift;;
    --dry-run) DRY=1;;
    *) echo "unknown flag $1"; exit 1;;
  esac; shift
done
run(){ echo "+ $*"; [ "$DRY" = 1 ] || eval "$*"; }

CACHE="gpt_oss/data/out/kernel_cache.jsonl"

echo "=== [1/6] env + kernel benchmark ==="
run "python3 gpt_oss/check_env.py | sed -n '1,12p'"
run "python3 experiments/trimul_gpu.py --N 32 --K 32 --C 8 --reps 50"

echo "=== [2/6] build MEASURED kernel cache (lever outcomes from real reruns) ==="
run "python3 experiments/trimul_task.py --reps $REPS $REAL --out $CACHE"

echo "=== [3/6] build SIA custom-task data (public traces + private outcomes) ==="
run "python3 sia_task_trimul/build_task_data.py --cache $CACHE"

echo "=== [4/6] headroom probe (is the task trivially threshold-solvable?) ==="
run "python3 experiments/trace_difficulty_probe.py --cache $CACHE --eval-seeds 3 | sed -n '/HEADROOM/,/====/p' || true"

echo "=== [5/6] (optional) learned SIA-Lever rollouts ==="
if [ -n "$ROLL" ]; then
  echo "using provided rollouts glob: $ROLL"
else
  echo "no --lever-rollouts given: learned column omitted (needs GPU/endpoint gpt-oss rollouts)."
fi

echo "=== [6/6] SIA-W+H vs SIA-Lever comparison ==="
run "python3 gpt_oss/eval/compare_sia_wh_vs_lever.py --cache $CACHE ${ROLL:+--lever-rollouts $ROLL}"
echo "Done. See results/trimul_sia_wh_vs_lever.md + plots/trimul_sia_wh_vs_lever.png"
