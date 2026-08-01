#!/usr/bin/env python3
"""Derive a preregistered clipping threshold from an unclipped JSONL run."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def calibrate(records, quantile=0.99, multiplier=2.0):
    norms = np.asarray([record["grad_norm"] for record in records], dtype=np.float64)
    if norms.size == 0 or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("gradient norms must be non-empty, finite, and positive")
    percentile = float(np.quantile(norms, quantile))
    return {
        "num_records": int(norms.size),
        "quantile": quantile,
        "multiplier": multiplier,
        "median_grad_norm": float(np.median(norms)),
        "p99_grad_norm": percentile,
        "max_observed_grad_norm": float(norms.max()),
        "max_grad_norm": multiplier * percentile,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--min-records", type=int, default=1000)
    args = parser.parse_args()

    if not 0 < args.quantile < 1 or not math.isfinite(args.multiplier) or args.multiplier <= 0:
        parser.error("quantile must be in (0, 1) and multiplier must be positive")
    records = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if len(records) < args.min_records:
        raise RuntimeError(f"expected at least {args.min_records} records, found {len(records)}")
    result = calibrate(records, args.quantile, args.multiplier)
    result["source_metrics"] = str(args.metrics.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
