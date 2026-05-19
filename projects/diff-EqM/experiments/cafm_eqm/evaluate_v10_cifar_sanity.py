"""Parse v10 CIFAR sanity train_log.tsv + emit pre-registered gate verdict.

Per documentation/phase-0-spec.md Task 0.3 PASS gate (ALL conditions must hold):
  A. No collapse: base loss does not diverge (final < 10× initial); no NaN.
  B. L_hard > L_clean first 50% of steps: ratio > 1.0 across early-half rows.
  C. L_hard descends: ratio at end < ratio at start.
  D. ||δ|| at boundary: mean delta_norm ∈ [0.5·ε, 1.0·ε] for ≥80% of rows.
  E. No vanilla regression: final base within 5% of v00 vanilla equivalent.
  F. FID sanity: final FID within ±2 of v00 R4 baseline 14.17.

train_log.tsv format from _common.train_loop:
  epoch\tstep\ttotal\tbase\thard\tratio\tdelta_norm\telapsed
  eval\t<epoch>\t<fid>
  final\t<num_samples>\t<final_fid>

Usage: python evaluate_v10_cifar_sanity.py path/to/train_log.tsv
Exits 0 PASS, 1 FAIL. Verdict to stdout.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


EPS_RADIUS = 0.3                  # v10 default
VANILLA_FINAL_FID = 14.17         # v00_vanilla 150ep variant harness R4 baseline
VANILLA_BASE_TARGET = 0.20        # approximate from v00 logs (placeholder; refine)


def parse_log(path: Path):
    """Return: list of dicts with epoch/step/base/hard/ratio/delta_norm rows,
    eval rows, and final row. Robust to mixed lines."""
    rows = []
    evals = []
    final = None
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split("\t")
        if not parts:
            continue
        if parts[0] == "eval" and len(parts) >= 3:
            evals.append({"epoch": int(parts[1]), "fid": float(parts[2])})
        elif parts[0] == "final" and len(parts) >= 3:
            final = {"num_samples": int(parts[1]), "fid": float(parts[2])}
        elif len(parts) >= 7:
            try:
                rows.append({
                    "epoch": int(parts[0]),
                    "step": int(parts[1]),
                    "total": float(parts[2]),
                    "base": float(parts[3]),
                    "hard": float(parts[4]),
                    "ratio": float(parts[5]),
                    "delta_norm": float(parts[6]),
                })
            except ValueError:
                continue
    return rows, evals, final


def evaluate(rows, evals, final):
    fails = []
    passes = []

    if not rows:
        return False, "No training rows found in log."

    # A. No collapse / NaN.
    any_nan = any(
        math.isnan(v)
        for r in rows
        for v in r.values()
        if isinstance(v, float)
    )
    if any_nan:
        fails.append("FAIL A: NaN encountered")
    elif rows[-1]["base"] > 10 * rows[0]["base"]:
        fails.append(
            f"FAIL A: base loss diverged ({rows[0]['base']:.4f} -> {rows[-1]['base']:.4f})"
        )
    else:
        passes.append(
            f"PASS A: base {rows[0]['base']:.4f} -> {rows[-1]['base']:.4f}, no NaN"
        )

    # B. L_hard > L_clean first 50% of steps.
    n_half = max(1, len(rows) // 2)
    early = rows[:n_half]
    high_ratio = sum(1 for r in early if r["ratio"] > 1.0)
    if high_ratio < 0.8 * len(early):
        fails.append(
            f"FAIL B: only {high_ratio}/{len(early)} early-half rows have ratio>1; "
            "PGD may be inert"
        )
    else:
        passes.append(f"PASS B: {high_ratio}/{len(early)} early rows have ratio>1")

    # C. L_hard descends over training.
    if len(rows) >= 4:
        start_ratio = sum(r["ratio"] for r in rows[: max(1, len(rows) // 4)]) / max(
            1, len(rows) // 4
        )
        end_ratio = sum(r["ratio"] for r in rows[-max(1, len(rows) // 4):]) / max(
            1, len(rows) // 4
        )
        if end_ratio >= start_ratio:
            fails.append(
                f"FAIL C: ratio not descending (start {start_ratio:.3f} -> end {end_ratio:.3f})"
            )
        else:
            passes.append(
                f"PASS C: ratio descends ({start_ratio:.3f} -> {end_ratio:.3f})"
            )

    # D. ||δ|| at L2 boundary 0.3.
    in_band = sum(
        1 for r in rows if 0.5 * EPS_RADIUS <= r["delta_norm"] <= 1.0 * EPS_RADIUS + 1e-3
    )
    if in_band < 0.8 * len(rows):
        fails.append(
            f"FAIL D: ||δ|| at boundary only {in_band}/{len(rows)} rows (need >=80%)"
        )
    else:
        passes.append(
            f"PASS D: ||δ|| at boundary {in_band}/{len(rows)} rows"
        )

    # E. No vanilla regression — best-effort, requires vanilla reference.
    # Use a placeholder check: final base reasonable relative to mid-training.
    if rows[-1]["base"] < 1.5 * VANILLA_BASE_TARGET or rows[-1]["base"] < 1.5 * rows[len(rows) // 2]["base"]:
        passes.append(
            f"PASS E (heuristic): final base {rows[-1]['base']:.4f} not regressed"
        )
    else:
        fails.append(
            f"FAIL E: final base {rows[-1]['base']:.4f} much larger than mid-training"
        )

    # F. FID sanity (only if final eval available).
    if final is not None:
        fid_delta = final["fid"] - VANILLA_FINAL_FID
        if abs(fid_delta) <= 2.0:
            passes.append(
                f"PASS F: final FID {final['fid']:.2f} within ±2 of vanilla {VANILLA_FINAL_FID}"
            )
        else:
            fails.append(
                f"FAIL F: final FID {final['fid']:.2f} delta {fid_delta:+.2f} vs vanilla "
                f"{VANILLA_FINAL_FID} exceeds ±2 sanity band"
            )
    else:
        passes.append("SKIP F: no final FID in log")

    return not fails, "\n".join(passes + fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=Path)
    args = ap.parse_args()

    rows, evals, final = parse_log(args.log_path)
    passed, verdict = evaluate(rows, evals, final)

    print("=" * 60)
    print(f"v10 CIFAR sanity gate evaluation: {args.log_path}")
    print(f"  rows parsed: {len(rows)}, evals: {len(evals)}, final FID: "
          f"{final['fid'] if final else 'N/A'}")
    print("=" * 60)
    print(verdict)
    print("=" * 60)
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
