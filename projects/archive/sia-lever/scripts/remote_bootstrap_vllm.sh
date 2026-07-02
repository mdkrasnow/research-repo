#!/usr/bin/env bash
# Runs ON the Nebius H200 VM. Installs vLLM, then serves the LoRA adapter's base (bf16 gpt-oss-120b)
# + the adapter, OpenAI-compatible on :8001. Adapter dir is rsync'd to ~/adapter by the orchestrator.
#
# gpt-oss-120b is MoE (~117B total / ~5B active); bf16 ~235 GB -> needs >=4x H200 (TP). The base for
# the TF-trained adapter is unsloth/gpt-oss-120b-BF16. Requires HF_TOKEN in env for the weights.
set -euo pipefail

ADAPTER_DIR="${ADAPTER_DIR:-$HOME/adapter}"          # rsync'd checkpoint dir (has adapter_model.safetensors)
# Default to native MXFP4 gpt-oss-120b (~63GB, fits 1x H200). Adapter is ATTENTION-ONLY (q/k/v/o),
# so it applies cleanly on the quantized base — vLLM computes LoRA in bf16 regardless of base quant.
# For an exact bf16-base match instead, set BASE_MODEL=unsloth/gpt-oss-120b-BF16 (needs 8x H200).
BASE="${BASE_MODEL:-openai/gpt-oss-120b}"
NAME="${ADAPTER_NAME:-lever_lora}"
PORT="${PORT:-8001}"
TP="${TP_SIZE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

echo "[vm] GPUs: $(nvidia-smi -L 2>/dev/null | wc -l) ; TP=$TP ; base=$BASE"

# gpt-oss MoE LoRA serving needs vLLM >= 0.15.0 (gpt-oss LoRA support landed there).
if ! python3 -c "import vllm,sys;v=tuple(int(x) for x in vllm.__version__.split('.')[:2]);sys.exit(0 if v>=(0,15) else 1)" 2>/dev/null; then
  echo "[vm] installing vLLM >= 0.15.0 ..."
  python3 -m pip install -q --upgrade pip
  python3 -m pip install -q "vllm>=0.15.0" "huggingface_hub[cli]"
fi

# the checkpoint subdir vLLM wants (the ftckpt_* folder with adapter_model.safetensors)
APATH="$(dirname "$(find "$ADAPTER_DIR" -name adapter_model.safetensors | head -1)")"
echo "[vm] adapter path = $APATH"

echo "[vm] launching vLLM (base + LoRA '$NAME') on :$PORT ..."
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$BASE" \
  --enable-lora \
  --lora-modules "${NAME}=${APATH}" \
  --tensor-parallel-size "${TP:-4}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --port "$PORT"
