#!/usr/bin/env bash
# Serve base gpt-oss-120b + a LoRA adapter with vLLM (OpenAI-compatible, with LoRA enabled).
# The adapter is requested by name as the "model" field in chat completions.
set -euo pipefail

BASE="${GPT_OSS_MODEL_PATH:-openai/gpt-oss-120b}"
ADAPTER_PATH="${1:?usage: serve_vllm_adapter.sh <adapter_dir> [adapter_name]}"
ADAPTER_NAME="${2:-lever_lora}"
TP="${TP_SIZE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
PORT="${PORT:-8001}"

echo "Serving $BASE + LoRA '$ADAPTER_NAME' from $ADAPTER_PATH (tp=$TP, port=$PORT)"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$BASE" \
  --enable-lora \
  --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
  --tensor-parallel-size "${TP:-1}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --port "$PORT"

# Query the adapter:
#   export GPT_OSS_BASE_URL=http://localhost:8001/v1  GPT_OSS_API_KEY=dummy
#   python gpt_oss/rollout/rollout_adapter.py --endpoint --model lever_lora --tag sft
