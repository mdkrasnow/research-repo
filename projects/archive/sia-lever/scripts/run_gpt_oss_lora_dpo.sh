#!/usr/bin/env bash
# Train DPO LoRA on gpt-oss-120b (chosen=correct lever, rejected=measured-worse action).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-adapters/gpt_oss_120b/lever_dpo_$(date -u +%Y%m%dT%H%M%SZ)}"
BASE="${GPT_OSS_MODEL_PATH:-openai/gpt-oss-120b}"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
EXTRA="${EXTRA:-}"

python3 gpt_oss/data/build_dpo_dataset.py
if [ "${NGPU:-0}" -ge 2 ]; then
  accelerate launch --config_file gpt_oss/train/accelerate_config_multih200.yaml \
    gpt_oss/train/train_lora_dpo.py --base-model "$BASE" --out "$OUT" $EXTRA
else
  python3 gpt_oss/train/train_lora_dpo.py --base-model "$BASE" --out "$OUT" $EXTRA \
    || python3 gpt_oss/train/train_lora_dpo.py --base-model "$BASE" --out "$OUT" --qlora $EXTRA
fi
echo "adapter -> $OUT"
