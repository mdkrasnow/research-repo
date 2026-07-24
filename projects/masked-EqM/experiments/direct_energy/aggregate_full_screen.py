"""Aggregate the matched nine-run direct-energy full-screen FID results.

The script fails closed: a report is only written when all three matched seeds
are present for each of the none, dot, and direct arms.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


ARMS = ("none", "dot", "direct")
SEEDS = (0, 1, 2)


def confidence_interval(values: list[float]) -> tuple[float, float]:
    """Two-sided normal 95% CI; n=3 is reported as a descriptive interval."""
    average = mean(values)
    if len(values) < 2:
        return average, average
    half_width = 1.96 * stdev(values) / math.sqrt(len(values))
    return average - half_width, average + half_width


def load(input_dir: Path) -> dict[str, dict[int, dict]]:
    results: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            path = input_dir / f"direct_full_{arm}_seed{seed}_fid.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing required result: {path}")
            payload = json.loads(path.read_text())
            if not isinstance(payload.get("fid"), (int, float)):
                raise ValueError(f"missing numeric FID in {path}")
            results[arm][seed] = payload
    return results


def summarize(results: dict[str, dict[int, dict]]) -> dict:
    table = {}
    baseline = [float(results["none"][seed]["fid"]) for seed in SEEDS]
    for arm in ARMS:
        values = [float(results[arm][seed]["fid"]) for seed in SEEDS]
        paired_delta = [value - base for value, base in zip(values, baseline)]
        low, high = confidence_interval(values)
        table[arm] = {
            "per_seed_fid": {str(seed): value for seed, value in zip(SEEDS, values)},
            "mean_fid": mean(values),
            "std_fid": stdev(values) if len(values) > 1 else 0.0,
            "normal_95ci": [low, high],
            "paired_delta_vs_none": {
                str(seed): delta for seed, delta in zip(SEEDS, paired_delta)
            },
            "mean_paired_delta_vs_none": mean(paired_delta),
        }
    return table


def markdown(table: dict) -> str:
    rows = [
        "# Full direct-energy screen — FID aggregate",
        "",
        "All values are matched seed-0/1/2 2K-sample FIDs. Confidence intervals are descriptive normal 95% intervals (n=3).",
        "",
        "| Arm | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | 95% CI | Mean paired Δ vs none |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = table[arm]
        seed = item["per_seed_fid"]
        ci = item["normal_95ci"]
        rows.append(
            f"| {arm} | {seed['0']:.4f} | {seed['1']:.4f} | {seed['2']:.4f} | "
            f"{item['mean_fid']:.4f} ± {item['std_fid']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {item['mean_paired_delta_vs_none']:.4f} |"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    table = summarize(load(args.input_dir))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown(table))


if __name__ == "__main__":
    main()
