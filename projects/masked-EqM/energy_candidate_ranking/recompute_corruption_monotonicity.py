"""Correct the corruption endpoint from saved fixed-candidate CSVs only.

The initial confirmation evaluator accidentally omitted the clean-to-first-
corruption transition.  This script makes no model calls and never changes a
candidate: it recomputes that endpoint over clean->0.25 and 0.25->0.75 for
real and all three generated families, with a source-cluster bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCORES = ("direct_energy", "direct_energy_zero_anchored", "dot_energy",
          "dot_energy_zero_anchored", "base_field_norm")


def cluster_boot(values: np.ndarray, sources: np.ndarray, repetitions: int, seed: int) -> tuple[float, list[float]]:
    units = np.unique(sources)
    by = [np.flatnonzero(sources == unit) for unit in units]
    rng = np.random.default_rng(seed)
    sampled = np.array([values[np.concatenate([by[j] for j in rng.integers(0, len(by), len(by))])].mean()
                        for _ in range(repetitions)])
    return float(values.mean()), [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]


def ordered_levels(config: dict) -> tuple[float, ...]:
    """Return the corruption ladder as a strictly increasing tuple starting at clean.

    The monotonicity endpoint is computed by zipping consecutive levels, so the
    ladder's ORDER is load-bearing: an unsorted config silently turns a
    "does energy increase with corruption" comparison into its converse.  The
    ordering is therefore a validated requirement of the config, not an
    incidental property of how it happened to be written.
    """
    severities = [float(s) for s in config["corruption_severities"]]
    if len(set(severities)) != len(severities):
        raise ValueError(f"corruption_severities must be unique, got {severities}")
    if any(s <= 0. for s in severities):
        raise ValueError(f"corruption_severities must be positive (0 is the clean anchor), got {severities}")
    if severities != sorted(severities):
        raise ValueError(
            "corruption_severities must be listed in strictly increasing order; "
            f"got {severities}. Ordering is required because the monotonicity "
            "endpoint compares consecutive levels pairwise."
        )
    return (0., *severities)


def main(args: argparse.Namespace) -> None:
    original = json.loads(args.metrics.read_text())
    rows = list(csv.DictReader(args.candidates.open()))
    by = {(row["group"], int(row["source_id"])): row for row in rows}
    sources = sorted({int(row["source_id"]) for row in rows})
    families = ("real", "generated_none", "generated_dot", "generated_direct")
    levels = ordered_levels(original["config"])
    result_rows = []
    for si, score in enumerate(SCORES):
        comparisons, clusters = [], []
        for family in families:
            for lo, hi in zip(levels[:-1], levels[1:]):
                lo_group = family if lo == 0 else f"{family}_corrupt_{lo:g}"
                hi_group = f"{family}_corrupt_{hi:g}"
                for source in sources:
                    comparisons.append(float(by[(hi_group, source)][score]) > float(by[(lo_group, source)][score]))
                    clusters.append(source)
        estimate, ci = cluster_boot(np.asarray(comparisons), np.asarray(clusters), original["config"]["bootstrap_replicates"], original["config"]["seed"] + 30 + si)
        result_rows.append({"score": score, "metric": "corruption_increases_all_families_clean_included",
                            # Explicit join key into the bank being corrected: consumers must not
                            # re-derive it by string surgery on `metric`.
                            "corrects_metric": "corruption_increases_all_families",
                            "estimate": estimate, "ci95": ci,
                            "pairs_per_source": len(comparisons) // len(sources), "families": list(families), "levels": list(levels)})
    args.output.write_text(json.dumps({"source_metrics": str(args.metrics),
                                       "source_metrics_resolved": str(args.metrics.resolve()),
                                       "source_config_seed": original["config"]["seed"],
                                       "correction": "included clean-to-first-corruption transition",
                                       "metrics": result_rows}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
