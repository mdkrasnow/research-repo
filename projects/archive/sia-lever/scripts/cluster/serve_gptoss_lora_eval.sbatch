#!/usr/bin/env bash
#SBATCH --job-name=sia-gptoss-lora
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=sia_gptoss_lora_%j.out
#
# Self-contained: serve gpt-oss-120b (native MXFP4, ~63GB, fits 1x H200) + our LoRA adapter via
# vLLM OpenAI-compatible on localhost, then run base-vs-LoRA selector eval against a rung cache.
# Harvard-cluster alternative to the Nebius Token-Factory serving blocker. First-to-serve wins.
#
# Env knobs (export before sbatch, or edit):
#   WORK     : project dir on netscratch (default below)
#   ADAPTER  : LoRA checkpoint dir (has adapter_model.safetensors)
#   CACHE    : measured cache to score against
#   TAG      : rung tag (names rollouts/markers)
#   BASE     : base model id (default native MXFP4 gpt-oss-120b — Hopper only)
#   HF_TOKEN : optional; only needed if HF gates the weights (try ungated first)
set -uo pipefail

WORK="${WORK:-/n/netscratch/ydu_lab/Lab/mkrasnow/sia-lever}"
ADAPTER="${ADAPTER:-$WORK/adapters/gpt_oss_120b/hard_ftjob-d12e84aa9d5349ccb08404f7b17e3003}"
CACHE="${CACHE:-$WORK/gpt_oss/data/out/hard_cache.jsonl}"
TAG="${TAG:-hard}"
BASE="${BASE:-openai/gpt-oss-120b}"
PORT="${PORT:-8001}"
NAME="lever_${TAG}"

export HF_HOME="${HF_HOME:-/n/netscratch/ydu_lab/Lab/mkrasnow/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_CACHE_ROOT="$HF_HOME/vllm"
mkdir -p "$HF_HOME"
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

cd "$WORK"
module load python/3.10 cuda 2>/dev/null || true

# --- venv on netscratch (home quota too small) ---
VENV="/n/netscratch/ydu_lab/Lab/mkrasnow/venvs/vllm015"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -c "import vllm,sys;v=tuple(int(x) for x in vllm.__version__.split('.')[:2]);sys.exit(0 if v>=(0,15) else 1)" 2>/dev/null \
  || pip install -q --upgrade pip "vllm>=0.15.0" "huggingface_hub" "openai>=1.40"
# vLLM's default torch wheel targets cu130 (needs driver >=13); seas_gpu H200 nodes ship driver 12.9.
# Force torch built for CUDA 12.9 so it matches the node driver. (pip ==2.11.0 ignores +cuXXX tag, so
# the +cu129 local tag + --force-reinstall is required to overwrite the cu130 wheel.)
python -c "import torch,sys;sys.exit(0 if 'cu129' in torch.__version__ or 'cu128' in torch.__version__ else 1)" 2>/dev/null \
  || pip install -q --force-reinstall --index-url https://download.pytorch.org/whl/cu129 "torch==2.11.0+cu129"

echo "[job] node=$(hostname) gpus=$(nvidia-smi -L | wc -l) base=$BASE adapter=$ADAPTER"
APATH="$(dirname "$(find "$ADAPTER" -name adapter_model.safetensors | head -1)")"
echo "[job] adapter path=$APATH cache=$CACHE"

# --- launch vLLM server in background ---
python -m vllm.entrypoints.openai.api_server \
  --model "$BASE" \
  --enable-lora \
  --lora-modules "${NAME}=${APATH}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --port "$PORT" > "vllm_${TAG}_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

# --- wait for /v1/models (model load can take several min) ---
export GPT_OSS_BASE_URL="http://localhost:${PORT}/v1"
export GPT_OSS_API_KEY="dummy"
echo "[job] waiting for vLLM /v1/models ..."
UP=0
for i in $(seq 1 120); do            # 120 * 15s = 30 min cap
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then UP=1; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[job] vLLM died — see vllm log"; tail -40 "vllm_${TAG}_${SLURM_JOB_ID}.log"; exit 1; fi
  sleep 15
done
[ "$UP" = 1 ] || { echo "[job] vLLM never came up"; tail -40 "vllm_${TAG}_${SLURM_JOB_ID}.log"; exit 1; }
echo "[job] vLLM up. models:"; curl -s "http://localhost:${PORT}/v1/models"

# --- smoke: one base chat call before trusting the adapter path ---
python gpt_oss/rollout/rollout_base.py --model "$BASE" --base-url "$GPT_OSS_BASE_URL" --limit 1 --tag smoke_${TAG} --cache "$CACHE" || { echo "[job] base smoke failed"; exit 1; }

# --- full rollouts: base + LoRA ---
echo "[job] rollout BASE ..."
python gpt_oss/rollout/rollout_base.py --model "$BASE"  --base-url "$GPT_OSS_BASE_URL" --tag "base_${TAG}" --eval-seeds 3 --cache "$CACHE"
echo "[job] rollout LoRA ($NAME) ..."
python gpt_oss/rollout/rollout_base.py --model "$NAME"  --base-url "$GPT_OSS_BASE_URL" --tag "lora_${TAG}" --eval-seeds 3 --cache "$CACHE"

# --- score ---
python gpt_oss/eval/eval_selector.py --rollouts "results/gpt_oss/base_${TAG}_rollouts_*.jsonl" --tag "base_${TAG}" --cache "$CACHE"
python gpt_oss/eval/eval_selector.py --rollouts "results/gpt_oss/lora_${TAG}_rollouts_*.jsonl" --tag "lora_${TAG}" --cache "$CACHE"
python gpt_oss/eval/eval_adapter.py \
  --base-rollouts "results/gpt_oss/base_${TAG}_rollouts_*.jsonl" \
  --adapter-rollouts "results/gpt_oss/lora_${TAG}_rollouts_*.jsonl" --tag "$TAG" --cache "$CACHE"

echo "DONE tag=$TAG job=$SLURM_JOB_ID" > "harvard_lora_eval_done_${TAG}.txt"
echo "[job] complete."
