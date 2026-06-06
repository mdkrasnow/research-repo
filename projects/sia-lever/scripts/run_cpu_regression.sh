#!/usr/bin/env bash
# CPU regression: prove Phases 0-3 still pass. Run before every commit. No GPU, no network.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Phase 0/1: 4-stage episode (3 seeds) ==="
python3 experiments/run_seeds.py --seeds 3 --steps 800 | tail -6
echo "=== Phase 2: agentic-H structural verifier ==="
python3 harness/verifier.py | tail -3
echo "=== Phase 3: lever selector vs policies (3 seeds) ==="
python3 experiments/phase3.py --seeds 3 --steps 800 | tail -4
echo "=== SIA-Lever-120B data pipeline (offline) ==="
python3 gpt_oss/data/build_sft_dataset.py | tail -1
python3 gpt_oss/data/build_dpo_dataset.py | tail -1
python3 gpt_oss/eval/compare_policies.py | sed -n '/| Policy/,/oracle_best/p'
echo "=== CPU REGRESSION PASS ==="
