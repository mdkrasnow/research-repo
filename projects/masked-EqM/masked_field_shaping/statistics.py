"""Predeclared paired image-cluster bootstrap and decision rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    paired_delta: float
    bootstrap_standard_error: float
    ci_lower: float
    ci_upper: float
    fraction_improved: float
    replicates: int
    seed: int


def paired_image_cluster_bootstrap(
    image_ids,
    control_recovery,
    treatment_recovery,
    *,
    replicates: int = 10_000,
    seed: int = 20260803,
) -> BootstrapResult:
    """Bootstrap image IDs; all corruption draws for an ID remain together."""
    image_ids = np.asarray(image_ids)
    control = np.asarray(control_recovery, dtype=np.float64)
    treatment = np.asarray(treatment_recovery, dtype=np.float64)
    if not (len(image_ids) == len(control) == len(treatment)):
        raise ValueError("image IDs and paired recovery arrays must have equal length")
    if replicates < 5_000:
        raise ValueError("the preregistered analysis requires at least 5,000 replicates")
    unique_ids, inverse = np.unique(image_ids, return_inverse=True)
    if unique_ids.size == 0:
        raise ValueError("no image clusters supplied")
    delta = treatment - control
    cluster_means = np.array([delta[inverse == i].mean() for i in range(unique_ids.size)])
    if np.any(np.bincount(inverse) != 2):
        raise ValueError("each image cluster must contain exactly two corruption draws")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 512):
        count = min(512, replicates - start)
        indices = rng.integers(0, unique_ids.size, size=(count, unique_ids.size))
        samples[start : start + count] = cluster_means[indices].mean(axis=1)
    return BootstrapResult(
        paired_delta=float(delta.mean()),
        bootstrap_standard_error=float(samples.std(ddof=1)),
        ci_lower=float(np.percentile(samples, 2.5)),
        ci_upper=float(np.percentile(samples, 97.5)),
        fraction_improved=float((delta > 0).mean()),
        replicates=replicates,
        seed=seed,
    )


def classify_result(ci_lower: float, fid_control: float, fid_treatment: float) -> dict:
    recovery_pass = ci_lower > 0.0
    fid_delta = fid_treatment - fid_control
    fid_pass = fid_delta <= 1.0
    if recovery_pass and fid_pass:
        decision = "PASS"
    elif recovery_pass:
        decision = "FIELD SHAPED BUT NOT USEFULLY"
    elif fid_pass:
        decision = "NO ROBUSTNESS EVIDENCE"
    else:
        decision = "FAIL"
    return {
        "decision": decision,
        "recovery_pass": recovery_pass,
        "fid_noninferior": fid_pass,
        "fid_delta": fid_delta,
        "thresholds": {"recovery_ci_lower_gt": 0.0, "fid_delta_lte": 1.0},
    }
