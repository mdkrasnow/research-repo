#!/usr/bin/env bash
# Autonomous per-rung driver: poll a Token Factory LoRA job to completion, then run base-vs-LoRA
# selector eval against the rung's measured cache. Needs NEBIUS_API_KEY + GPT_OSS_BASE_URL in env.
#
# Usage: driver_lora_eval.sh <job_json> <cache_jsonl> <tag>
#   job_json    : gpt_oss/data/out/tf_finetune_job_<tag>.json   (has job_id)
#   cache_jsonl : measured cache to score against (e.g. gpt_oss/data/out/hard_cache.jsonl)
#   tag         : rung tag (names rollouts/markers), e.g. hard
set -uo pipefail
cd "$(dirname "$0")/.."

JOB_FILE="${1:?job_json required}"
CACHE="${2:?cache_jsonl required}"
TAG="${3:?tag required}"

export GPT_OSS_BASE_URL="${GPT_OSS_BASE_URL:-https://api.tokenfactory.nebius.com/v1/}"
export GPT_OSS_API_KEY="${GPT_OSS_API_KEY:-$NEBIUS_API_KEY}"
BASE_MODEL="openai/gpt-oss-120b"
MARKER="gpt_oss/data/out/lora_eval_done_${TAG}.txt"
JOB_ID="$(python3 -c "import json;print(json.load(open('$JOB_FILE'))['job_id'])")"

_status(){ python3 - "$JOB_ID" "$1" <<'PY'
import json,os,sys,requests
jid,field=sys.argv[1],sys.argv[2]
key=os.getenv("GPT_OSS_API_KEY"); base=os.getenv("GPT_OSS_BASE_URL").rstrip("/")
j=requests.get(f"{base}/fine_tuning/jobs/{jid}",headers={"Authorization":f"Bearer {key}"},timeout=60).json()
print(j.get(field,"") or "")
PY
}

echo "[$TAG] polling $JOB_ID (cache=$CACHE)"
for i in $(seq 1 120); do          # 120 * 30s = 60 min cap
  STATUS="$(_status status)"
  echo "[$TAG] poll $i: status=$STATUS"
  case "$STATUS" in
    succeeded) break ;;
    failed|cancelled) echo "$TAG job $STATUS — abort" > "$MARKER"; exit 1 ;;
  esac
  sleep 30
done

FT_MODEL="$(_status fine_tuned_model)"
if [ -z "$FT_MODEL" ]; then echo "$TAG no fine_tuned model (timeout)" > "$MARKER"; exit 1; fi
echo "[$TAG] fine-tuned model = $FT_MODEL"

# base rollout may already exist from the parallel pre-launch; (re)run to be safe, tag-scoped
echo "[$TAG] rollout BASE ..."
python3 gpt_oss/rollout/rollout_base.py --model "$BASE_MODEL" --tag "base_${TAG}" --eval-seeds 3 --cache "$CACHE"
echo "[$TAG] rollout LoRA ..."
python3 gpt_oss/rollout/rollout_base.py --model "$FT_MODEL" --tag "lora_${TAG}" --eval-seeds 3 --cache "$CACHE"

echo "[$TAG] selector eval base + lora ..."
python3 gpt_oss/eval/eval_selector.py --rollouts "results/gpt_oss/base_${TAG}_rollouts_*.jsonl" --tag "base_${TAG}" --cache "$CACHE"
python3 gpt_oss/eval/eval_selector.py --rollouts "results/gpt_oss/lora_${TAG}_rollouts_*.jsonl" --tag "lora_${TAG}" --cache "$CACHE"
echo "[$TAG] base-vs-LoRA headline ..."
python3 gpt_oss/eval/eval_adapter.py \
  --base-rollouts "results/gpt_oss/base_${TAG}_rollouts_*.jsonl" \
  --adapter-rollouts "results/gpt_oss/lora_${TAG}_rollouts_*.jsonl" --tag "$TAG" --cache "$CACHE"

echo "DONE tag=$TAG model=$FT_MODEL" > "$MARKER"
echo "[$TAG] complete -> $MARKER"
