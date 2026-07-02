#!/usr/bin/env bash
# Kernel-task lever comparison (SIA paper's GPU-kernel task: TriMul Triton).
# CPU stub by default; real gpt-oss + GPU when --endpoint + device cuda.
#
#   bash scripts/run_kernel_comparison.sh                       # CPU stub (no model/GPU)
#   bash scripts/run_kernel_comparison.sh --endpoint            # gpt-oss writes kernels (GPU)
#
# Env (GPU mode): GPT_OSS_BASE_URL, GPT_OSS_MODEL, GPT_OSS_API_KEY|NEBIUS_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."

ENDPOINT=0; DEVICE="cpu"; SHAPE="32 32 8"; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --endpoint) ENDPOINT=1; DEVICE="cuda"; SHAPE="128 128 32";;
    --device) DEVICE="$2"; shift;;
    --shape) SHAPE="$2 $3 $4"; shift 3;;
    *) EXTRA="$EXTRA $1";;
  esac; shift
done

echo "=== [1/3] env check ==="
python3 kernel_task/check_env.py | tail -8

echo "=== [2/3] CPU stub regression (phenomenon wired) ==="
python3 kernel_task/tests/test_cpu_stub.py | tail -2

echo "=== [3/3] comparison ==="
if [ "$ENDPOINT" = 1 ]; then
  python3 kernel_task/run.py --endpoint --device "$DEVICE" --shape $SHAPE \
    --model "${GPT_OSS_MODEL:-gpt-oss-120b}" ${GPT_OSS_BASE_URL:+--base-url "$GPT_OSS_BASE_URL"} $EXTRA
else
  python3 kernel_task/run.py --device "$DEVICE" --shape $SHAPE $EXTRA
fi
echo "Done. See results/kernel/comparison.md"
