#!/usr/bin/env python3
"""Phase 2 Welch t-test on v10 vs vanilla 3-seed FIDs.

Auto-discovers results from completed_runs in pipeline.json.
Usage: python3 projects/diff-EqM/scripts/welch_phase2.py
"""
import json
import sys
from pathlib import Path

try:
    from scipy import stats
except ImportError:
    print("ERROR: scipy not installed. pip install scipy", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE = REPO_ROOT / "projects/diff-EqM/.state/pipeline.json"
GATE_MEAN_GAIN = 1.0
GATE_PVAL = 0.05


def collect_fids(state, run_id_prefix):
    """Find completed_runs entries whose run_id starts with prefix and has 'fid'."""
    results = []
    for r in state.get("completed_runs", []):
        rid = r.get("run_id", "")
        if rid.startswith(run_id_prefix) and "fid" in r and isinstance(r["fid"], (int, float)):
            results.append((rid, float(r["fid"])))
    return results


def main():
    state = json.loads(PIPELINE.read_text())

    v10 = collect_fids(state, "v10_in1k_seed")  # seed0 from Phase 1
    v10 += collect_fids(state, "v10_b2_seed")   # seeds 1+2 if logged that way
    vanilla = collect_fids(state, "stage_b_vanilla_in1k_80ep")
    vanilla += collect_fids(state, "vanilla_b2_in1k_80ep_seed")

    print(f"=== Phase 2 Welch t-test ===")
    print(f"v10 results: {len(v10)}")
    for rid, fid in v10:
        print(f"  {rid}: {fid:.4f}")
    print(f"vanilla results: {len(vanilla)}")
    for rid, fid in vanilla:
        print(f"  {rid}: {fid:.4f}")

    if len(v10) < 3 or len(vanilla) < 3:
        print(f"\nINCOMPLETE: need 3 seeds each. Have v10={len(v10)}, vanilla={len(vanilla)}")
        sys.exit(0)

    v10_fids = [f for _, f in v10[:3]]
    van_fids = [f for _, f in vanilla[:3]]

    v10_mean, v10_std = sum(v10_fids) / 3, (sum((x - sum(v10_fids)/3)**2 for x in v10_fids) / 2) ** 0.5
    van_mean, van_std = sum(van_fids) / 3, (sum((x - sum(van_fids)/3)**2 for x in van_fids) / 2) ** 0.5
    mean_gain = van_mean - v10_mean

    result = stats.ttest_ind(van_fids, v10_fids, equal_var=False, alternative='greater')
    t_stat = float(result[0])  # type: ignore[index]
    p_val = float(result[1])  # type: ignore[index]

    print(f"\nv10 FID:     mean={v10_mean:.4f}  std={v10_std:.4f}")
    print(f"vanilla FID: mean={van_mean:.4f}  std={van_std:.4f}")
    print(f"mean gain (vanilla - v10): {mean_gain:.4f} FID")
    print(f"Welch t-stat: {t_stat:.4f}")
    print(f"p-value (one-sided, vanilla > v10): {p_val:.4f}")

    pass_mean = mean_gain >= GATE_MEAN_GAIN
    pass_p = p_val < GATE_PVAL
    verdict = "PASS" if (pass_mean and pass_p) else "FAIL"
    print(f"\nGate (mean_gain >= {GATE_MEAN_GAIN}): {'PASS' if pass_mean else 'FAIL'}")
    print(f"Gate (p < {GATE_PVAL}):           {'PASS' if pass_p else 'FAIL'}")
    print(f"\n=== PHASE 2 VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
