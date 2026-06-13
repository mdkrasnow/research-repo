#!/usr/bin/env bash
#SBATCH --job-name=gemma-serve
#SBATCH --partition=gpu_requeue        # gpu_requeue (full card, bf16) | gpu_test (20GB MIG -> set QUANT=4bit)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=serve_gemma_%j.out
#
# Serve Gemma 4 12B (OpenAI-compatible) for the kernel-task agent.
# Gemma 4 12B: dense 12B, 256K ctx, Apache-2.0 (NOT gated), bf16 26.7GB / 8bit 13.4GB / Q4 6.7GB.
#   MODEL  : default google/gemma-4-12b-it
#   QUANT  : ""(bf16, full card -> gpu_requeue) | "4bit"(gpu_test 20GB MIG) | "8bit"
#   PORT   : 8000
# After it's up, on the same node (or via tunnel) point the kernel run at it:
#   GPT_OSS_BASE_URL=http://$(hostname):8000/v1  GPT_OSS_MODEL=google/gemma-4-12b-it \
#     bash scripts/run_kernel_comparison.sh --endpoint
set -euo pipefail
MODEL="${MODEL:-google/gemma-4-12b-it}"
QUANT="${QUANT:-}"
PORT="${PORT:-8000}"

module load python/3.10 cuda 2>/dev/null || true
# expects: pip install vllm  (and HF_TOKEN for gated Gemma)
ARGS=(--port "$PORT" --max-model-len 4096 --gpu-memory-utilization 0.90)
case "$QUANT" in
  4bit) ARGS+=(--quantization bitsandbytes --load-format bitsandbytes) ;;
  8bit) ARGS+=(--quantization bitsandbytes --load-format bitsandbytes) ;;  # 8-bit via bnb config
  *)    : ;;                                                                # bf16 (full card)
esac

echo "serving $MODEL  quant='${QUANT:-bf16}'  on $(hostname):$PORT"
echo "set GPT_OSS_BASE_URL=http://$(hostname):$PORT/v1  GPT_OSS_MODEL=$MODEL"
exec vllm serve "$MODEL" "${ARGS[@]}"
