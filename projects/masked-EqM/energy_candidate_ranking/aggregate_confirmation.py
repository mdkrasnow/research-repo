"""Aggregate the three immutable direct-vs-dot confirmation banks.

This script intentionally consumes only completed ``metrics.json`` files.  It
does not select a seed, tune a score transformation, or rerun candidates.  The
primary comparison is raw direct energy minus raw dot energy (positive means
direct did better for the stated lower-is-better convention).  Uncertainty is
the nonparametric bootstrap over independently sampled candidate banks; with
three banks that uncertainty is necessarily coarse and is reported as such.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PRIMARY_SCORES = ("direct_energy", "dot_energy", "base_field_norm")
METRICS = (
    "spearman_quality_clustered",
    "pair_accuracy_clustered",
    "conditional_correct_lower",
    "corruption_increases_all_families",
)
REQUIRED_CONFIG = {
    "study": "three_bank_direct_minus_dot_confirmation",
    "num_per_group": 64,
    "reference_images": 256,
    "wrong_labels_per_real": 4,
    "t_eval": 1.0,
}


CLEAN_SUFFIX = "_clean_included"


def corrected_metric_name(row: dict) -> str:
    """Metric name in the bank that this correction row replaces.

    Newer sidecars carry the join key explicitly (``corrects_metric``).  Older
    ones only encode it in the metric name, in which case the suffix is stripped
    as a SUFFIX -- ``str.replace`` would delete any interior occurrence too.
    """
    if row.get("corrects_metric"):
        return row["corrects_metric"]
    metric = row["metric"]
    if not metric.endswith(CLEAN_SUFFIX):
        raise ValueError(f"correction row metric {metric!r} does not end with {CLEAN_SUFFIX!r}; "
                         "cannot determine which bank metric it corrects")
    return metric[: -len(CLEAN_SUFFIX)]


def same_metrics_file(recorded: str, resolved_input: Path) -> bool:
    """Does a recorded provenance path denote this input bank?

    Sidecars written before this join existed record a REPO-RELATIVE path
    (``projects/masked-EqM/results/.../metrics.json``), which cannot be resolved
    without knowing the cwd of the run that produced it.  Identity is therefore
    tested as: identical after resolution, or one path's components are a suffix
    of the other's.  Suffix agreement is still an identity test on the recorded
    provenance -- uniqueness of the match is enforced by the caller -- unlike a
    positional zip, which tests nothing at all.
    """
    recorded_path = Path(recorded)
    if recorded_path.is_absolute() and recorded_path.resolve() == resolved_input:
        return True
    a, b = recorded_path.parts, resolved_input.parts
    if not a or not b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return long[-len(short):] == short


def match_corrections(inputs: list[Path], banks: list[dict],
                      correction_paths: list[Path]) -> list[tuple[dict, Path]]:
    """Join corrections to banks on RECORDED PROVENANCE, never on argument position.

    The two file families glob in different orders, so a positional zip silently
    attaches each correction to the wrong bank and every downstream number is
    wrong while the run reports success.  Each sidecar records the metrics file
    it was computed from; that recorded path (and, when present, the source
    config seed) is the join key, and the join must be a bijection.
    """
    if len(correction_paths) != len(banks):
        raise ValueError("provide exactly one corruption correction per input bank")
    resolved_inputs = [Path(path).resolve() for path in inputs]
    if len(set(map(str, resolved_inputs))) != len(resolved_inputs):
        raise ValueError(f"an input bank was supplied more than once: {[str(p) for p in inputs]}")
    matched: list[tuple[dict, Path] | None] = [None] * len(banks)
    for correction_path in correction_paths:
        correction = json.loads(Path(correction_path).read_text())
        candidates = [correction.get("source_metrics_resolved"), correction.get("source_metrics")]
        source = next((c for c in candidates if c), None)
        if not source:
            raise ValueError(f"{correction_path} records no source_metrics provenance; cannot join")
        hits = {i for c in candidates if c for i, p in enumerate(resolved_inputs)
                if same_metrics_file(c, p)}
        if len(hits) != 1:
            raise ValueError(f"{correction_path} was computed from {source}, which does not resolve to "
                             f"exactly one of the input banks {[str(p) for p in inputs]} (matched {len(hits)})")
        index = hits.pop()
        if matched[index] is not None:
            raise ValueError(f"two corrections claim the same source bank {inputs[index]}")
        seed = correction.get("source_config_seed")
        if seed is not None and seed != banks[index]["config"].get("seed"):
            raise ValueError(f"{correction_path} records source seed {seed} but {inputs[index]} "
                             f"has seed {banks[index]['config'].get('seed')}")
        matched[index] = (correction, Path(correction_path))
    unmatched = [str(inputs[i]) for i, c in enumerate(matched) if c is None]
    if unmatched:
        raise ValueError(f"no corruption correction resolved to input banks: {unmatched}")
    return matched


def apply_correction(bank: dict, correction: dict, correction_path: Path) -> None:
    """Replace corrected rows in-place; a correction that matches nothing is an error."""
    replacements = {(row["score"], corrected_metric_name(row)):
                    {**row, "metric": corrected_metric_name(row)}
                    for row in correction["metrics"]}
    present = {(row["score"], row["metric"]) for row in bank["metrics"]}
    missing = sorted(set(replacements) - present)
    if missing:
        raise ValueError(f"{correction_path} contains correction rows with no matching bank row: {missing}")
    bank["metrics"] = [replacements.get((row["score"], row["metric"]), row) for row in bank["metrics"]]


def ci_over_banks(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    """Bootstrap the mean over immutable independent banks."""
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main(args: argparse.Namespace) -> None:
    banks = [json.loads(path.read_text()) for path in args.inputs]
    if args.corruption_corrections:
        corrections = match_corrections(args.inputs, banks, args.corruption_corrections)
        for bank, (correction, correction_path) in zip(banks, corrections):
            apply_correction(bank, correction, correction_path)
    for path, bank in zip(args.inputs, banks):
        bad = {key: (bank["config"].get(key), value) for key, value in REQUIRED_CONFIG.items()
               if bank["config"].get(key) != value}
        if bad:
            raise ValueError(f"{path} is not an immutable confirmation bank: {bad}")
        severities = bank["config"].get("corruption_severities")
        if severities is not None and list(severities) != sorted(severities):
            raise ValueError(f"{path} lists corruption_severities out of increasing order: {severities}; "
                             "the corruption monotonicity endpoint compares consecutive levels pairwise")
    seeds = [bank["config"]["seed"] for bank in banks]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"confirmation seeds must be unique, got {seeds}")

    point = {}
    for bank in banks:
        for row in bank["metrics"]:
            if row["score"] in PRIMARY_SCORES and row["metric"] in METRICS:
                point[(row["score"], row["metric"])] = point.get((row["score"], row["metric"]), []) + [row["estimate"]]
    missing = [(score, metric) for score in PRIMARY_SCORES for metric in METRICS
               if (score, metric) not in point or len(point[(score, metric)]) != len(banks)]
    if missing:
        raise ValueError(f"missing expected primary metric rows: {missing}")

    rows = []
    comparisons = []
    for mi, metric in enumerate(METRICS):
        direct = np.asarray(point[("direct_energy", metric)], dtype=float)
        dot = np.asarray(point[("dot_energy", metric)], dtype=float)
        base = np.asarray(point[("base_field_norm", metric)], dtype=float)
        for si, (name, values) in enumerate((("direct_energy", direct), ("dot_energy", dot), ("base_field_norm", base))):
            rows.append({"score": name, "metric": metric, "bank_estimates": values.tolist(),
                         "mean": float(values.mean()), "bank_bootstrap_ci95": ci_over_banks(values, args.bootstrap_replicates, args.seed + mi * 10 + si)})
        for name, delta in (("direct_minus_dot", direct-dot), ("direct_minus_base", direct-base)):
            comparisons.append({"comparison": name, "metric": metric, "bank_differences": delta.tolist(),
                                "mean_difference": float(delta.mean()),
                                "bank_bootstrap_ci95": ci_over_banks(delta, args.bootstrap_replicates, args.seed + mi * 10 + (1 if name.endswith("dot") else 2)),
                                "direct_wins_all_banks": bool(np.all(delta > 0)),
                                "direct_wins_bank_count": int(np.sum(delta > 0))})
    result = {"design": {"bank_seeds": seeds, "primary_scores": list(PRIMARY_SCORES), "metrics": list(METRICS),
                         "comparison_direction": "positive direct-minus-baseline means direct is better", "bootstrap": "resample complete banks", "bootstrap_replicates": args.bootstrap_replicates},
              "score_summary": rows, "direct_comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Three-bank direct-energy confirmation", "",
             "All rows use raw scalar energy. Positive comparison values mean direct is better under the lower-is-better convention. CIs resample the three complete, fixed candidate banks and are correspondingly coarse.", "",
             "| metric | direct mean [bank CI] | dot mean [bank CI] | base mean [bank CI] | direct-dot [bank CI] | direct wins |", "|---|---:|---:|---:|---:|---:|"]
    by_key = {(r["score"], r["metric"]): r for r in rows}
    by_comparison = {(r["comparison"], r["metric"]): r for r in comparisons}
    for metric in METRICS:
        def fmt(row: dict, key: str = "mean") -> str:
            low, high = row["bank_bootstrap_ci95"]
            return f"{row[key]:.3f} [{low:.3f}, {high:.3f}]"
        comparison = by_comparison[("direct_minus_dot", metric)]
        lines.append(f"| {metric} | {fmt(by_key[('direct_energy', metric)])} | {fmt(by_key[('dot_energy', metric)])} | {fmt(by_key[('base_field_norm', metric)])} | {fmt(comparison, 'mean_difference')} | {comparison['direct_wins_bank_count']}/{len(banks)} |")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--corruption-corrections", nargs="+", type=Path,
                        help="sidecars from recompute_corruption_monotonicity.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    main(parser.parse_args())
