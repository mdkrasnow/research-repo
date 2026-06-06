#!/usr/bin/env bash
# Env check + one-trace smoke against gpt-oss-120b. Needs GPT_OSS_BASE_URL + key + GPT_OSS_MODEL.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 gpt_oss/check_env.py
echo
python3 gpt_oss/smoke_infer.py --model "${GPT_OSS_MODEL:-gpt-oss-120b}"
