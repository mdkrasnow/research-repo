#!/usr/bin/env bash
# Run the official public SIA-H loop on the BUNDLED LawBench task with a gpt-oss-120b target.
# This is the paper-territory stretch lane. Public SIA = harness loop only (no W updates).
#
# The LawBench task ships in baselines/vendor/sia/sia/tasks/lawbench. We point SIA at it and use a
# gpt-oss target profile. Paper reference numbers (same-split caveat applies!):
#   initial 13.5%   SIA-H 50.0%   SIA-W+H 70.1%
set -euo pipefail
cd "$(dirname "$0")/../.."

LAW_TASK="baselines/vendor/sia/sia/tasks/lawbench"
RUN_ID="${RUN_ID:-lawbench_sia_h_gptoss_$(date -u +%Y%m%dT%H%M%SZ)}"
TARGET_PROFILE="${TARGET_PROFILE:-gpt-oss-target}"
MAX_GEN="${MAX_GEN:-5}"

[ -d "$LAW_TASK" ] || { echo "LawBench task not found at $LAW_TASK (is the SIA repo vendored?)"; exit 1; }

echo "Official SIA-H on LawBench: run_id=$RUN_ID target=$TARGET_PROFILE max_gen=$MAX_GEN"
sia run \
  --task_dir "$(pwd)/$LAW_TASK" \
  --run_id "$RUN_ID" \
  --max_gen "$MAX_GEN" \
  --target-profile "$TARGET_PROFILE" \
  --no-web \
  || { echo "[note] adjust flags via 'sia --help' (commit $(cat baselines/vendor/sia_commit.txt))"; exit 1; }

echo "Then: python3 paper_benchmarks/lawbench/export_rollouts.py --run-id $RUN_ID"
