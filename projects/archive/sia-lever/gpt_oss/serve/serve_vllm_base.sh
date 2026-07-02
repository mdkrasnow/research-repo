#!/usr/bin/env bash
# Serve base gpt-oss-120b with vLLM on an OpenAI-compatible endpoint (for fast rollouts).
# Adjust --tensor-parallel-size to your H200 count. Model id can be a local path or HF id.
set -euo pipefail

MODEL="${GPT_OSS_MODEL_PATH:-${1:-openai/gpt-oss-120b}}"
TP="${TP_SIZE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
PORT="${PORT:-8000}"

echo "Serving base model: $MODEL  (tensor-parallel=$TP, port=$PORT)"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "${TP:-1}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --port "$PORT"

# Then point the client at it:
#   export GPT_OSS_BASE_URL=http://localhost:8000/v1
#   export GPT_OSS_API_KEY=dummy   GPT_OSS_MODEL=$MODEL
#   python gpt_oss/smoke_infer.py
