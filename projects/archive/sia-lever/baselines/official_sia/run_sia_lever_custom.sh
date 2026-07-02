#!/usr/bin/env bash
# Run the official public SIA harness loop on our SIA-Lever custom task with a gpt-oss-120b target.
#
# IMPORTANT (honesty): the public SIA repo exposes only the HARNESS (H) self-improvement loop —
# it has NO weight-update (LoRA/RL) code (verified: 0 files match lora/peft/grpo/trl). So this run
# is an "official public SIA-H / harness-loop" baseline on our task, NOT a W or W+H run.
#
# Prereqs:
#   - baselines/vendor/sia installed (see EVALUATION_GUIDE): pip install -e '.[claude]' (or openhands)
#   - provider + profile registered for gpt-oss (copy gpt_oss/providers/*.example and
#     gpt_oss/profiles/*.example into the SIA config dir, drop the .example suffix)
#   - NEBIUS_API_KEY (or your endpoint key) exported
#   - python sia_task/build_task_data.py  (creates the task data)
set -euo pipefail
cd "$(dirname "$0")/../.."

TASK_DIR="$(pwd)/sia_task"
RUN_ID="${RUN_ID:-sia_lever_custom_gptoss_$(date -u +%Y%m%dT%H%M%SZ)}"
TARGET_PROFILE="${TARGET_PROFILE:-gpt-oss-target}"
MAX_GEN="${MAX_GEN:-3}"

python3 sia_task/build_task_data.py

echo "Running official SIA-H loop on SIA-Lever:"
echo "  task=$TASK_DIR run_id=$RUN_ID target=$TARGET_PROFILE max_gen=$MAX_GEN"

# Invoke the official CLI from the vendored repo. Flag names follow the public CLI; if your installed
# version differs, run `sia --help` and adjust. --no-web disables the dashboard for headless runs.
sia run \
  --task_dir "$TASK_DIR" \
  --run_id "$RUN_ID" \
  --max_gen "$MAX_GEN" \
  --target-profile "$TARGET_PROFILE" \
  --no-web \
  || { echo "[note] 'sia run' failed — check sia --help for the exact subcommand/flags in commit $(cat baselines/vendor/sia_commit.txt)"; exit 1; }

echo "Collect with: python3 baselines/official_sia/collect_runs.py --run-id $RUN_ID"
