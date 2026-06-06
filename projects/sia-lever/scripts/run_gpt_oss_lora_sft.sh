#!/usr/bin/env bash
# Train SFT LoRA on gpt-oss-120b. Auto-picks single vs multi-GPU launch.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-adapters/gpt_oss_120b/lever_sft_$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${GPT_OSS_MODEL_PATH:-openai/gpt-oss-120b}"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
EXTRA="${EXTRA:-}"   # e.g. EXTRA="--smoke"

echo "SFT LoRA: base=$BASE out=$OUT gpus=$NGPU"
python3 gpt_oss/data/build_sft_dataset.py

if [ "${NGPU:-0}" -ge 2 ]; then
  accelerate launch --config_file gpt_oss/train/accelerate_config_multih200.yaml \
    gpt_oss/train/train_lora_sft.py --base-model "$BASE" --out "$OUT" $EXTRA
else
  # single GPU: try bf16 LoRA, fall back to QLoRA on OOM
  python3 gpt_oss/train/train_lora_sft.py --base-model "$BASE" --out "$OUT" $EXTRA \
    || python3 gpt_oss/train/train_lora_sft.py --base-model "$BASE" --out "$OUT" --qlora $EXTRA
fi
echo "adapter -> $OUT"
