#!/usr/bin/env bash
# End-to-end SIA-Lever-120B comparison.
# Steps: env check -> CPU regression -> build datasets -> base gpt-oss eval -> (SFT LoRA) ->
#        adapter eval -> policy comparison -> final table/plot.
#
# Flags:
#   --dry-run         plan only; no API/training calls
#   --skip-train      do not train; expect --adapter to be provided (or skip adapter column)
#   --adapter <path>  use an existing adapter dir (serve/load) instead of training
#   --limit <n>       cap episodes for quick runs
#   --model <id>      served model id (endpoint mode)
#   --base-url <url>  OpenAI-compatible endpoint
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0; SKIP_TRAIN=0; ADAPTER=""; LIMIT=""; MODEL="${GPT_OSS_MODEL:-gpt-oss-120b}"; BASEURL="${GPT_OSS_BASE_URL:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1;;
    --skip-train) SKIP_TRAIN=1;;
    --adapter) ADAPTER="$2"; shift;;
    --limit) LIMIT="--limit $2"; shift;;
    --model) MODEL="$2"; shift;;
    --base-url) BASEURL="$2"; shift;;
    *) echo "unknown flag $1"; exit 1;;
  esac; shift
done
run(){ echo "+ $*"; [ "$DRY" = 1 ] || eval "$*"; }

echo "=== [1/7] env check ==="
run "python3 gpt_oss/check_env.py"

echo "=== [2/7] CPU regression (phenomenon intact) ==="
run "bash scripts/run_cpu_regression.sh >/tmp/sialever_cpu.log 2>&1 && tail -1 /tmp/sialever_cpu.log"

echo "=== [3/7] build datasets from measured cache ==="
run "python3 gpt_oss/data/build_sft_dataset.py"
run "python3 gpt_oss/data/build_dpo_dataset.py"
run "python3 sia_task/build_task_data.py"

echo "=== [4/7] base gpt-oss-120b selector eval ==="
BASE_ROLL="results/gpt_oss/base_rollouts_latest.jsonl"
run "python3 gpt_oss/rollout/rollout_base.py --model '$MODEL' ${BASEURL:+--base-url $BASEURL} $LIMIT --tag base"
run "python3 gpt_oss/eval/eval_selector.py --rollouts 'results/gpt_oss/base_rollouts_*.jsonl' --tag base"

echo "=== [5/7] adapter (train or reuse) ==="
if [ -n "$ADAPTER" ]; then
  echo "using provided adapter: $ADAPTER"
elif [ "$SKIP_TRAIN" = 1 ]; then
  echo "skip-train set and no --adapter: adapter column will be omitted"
else
  run "bash scripts/run_gpt_oss_lora_sft.sh"
  ADAPTER="$(ls -dt adapters/gpt_oss_120b/lever_sft_* 2>/dev/null | head -1 || true)"
fi

echo "=== [6/7] adapter eval ==="
if [ -n "$ADAPTER" ]; then
  run "python3 gpt_oss/rollout/rollout_adapter.py --adapter '$ADAPTER' --local --base-model '${GPT_OSS_MODEL_PATH:-openai/gpt-oss-120b}' $LIMIT --tag sft"
  run "python3 gpt_oss/eval/eval_adapter.py --adapter-rollouts 'results/gpt_oss/sft_rollouts_*.jsonl' --base-rollouts 'results/gpt_oss/base_rollouts_*.jsonl' --tag sft"
fi

echo "=== [7/7] policy comparison + demo report ==="
run "python3 gpt_oss/eval/compare_policies.py --base-rollouts 'results/gpt_oss/base_rollouts_*.jsonl' --adapter-rollouts 'results/gpt_oss/sft_rollouts_*.jsonl'"
run "python3 scripts/make_demo_report.py"
echo "Done. See results/DEMO_REPORT.md (all figures), results/final_comparison.md, plots/final_comparison.png"
