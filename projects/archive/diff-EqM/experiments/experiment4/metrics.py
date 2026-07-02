"""FID + KID over Inception features.

FID reuses experiments/evaluate_fid.py (compute_statistics, compute_fid) so the
Frechet distance is identical to the rest of the project. KID is implemented
here (the repo had none) as the standard unbiased polynomial-kernel MMD^2 used
by Binkowski et al. 2018, averaged over subsets for a stable estimate.

Reference stats use ONE unified .npz schema: keys 'mu' (D,) and 'sigma' (D,D),
optionally 'num_images'. This is the schema that fixes the historical
KeyError: mu mismatch between the two diverged FID code paths (see B1 in the
agent prompt).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate_fid import compute_fid, compute_statistics  # reuse exact FID math


def load_reference_stats(npz_path: str):
    data = np.load(npz_path)
    if "mu" not in data or "sigma" not in data:
        raise KeyError(
            f"reference stats {npz_path} missing 'mu'/'sigma' (got keys {list(data.keys())}). "
            "Rebuild with build_references.py to the unified schema."
        )
    return np.asarray(data["mu"]), np.asarray(data["sigma"])


def fid_against_reference(gen_features: np.ndarray, ref_mu, ref_sigma) -> float:
    mu, sigma = compute_statistics(gen_features)
    return compute_fid(mu, sigma, ref_mu, ref_sigma)


def _poly_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return (x @ y.T / d + 1.0) ** 3


def kid(
    x: np.ndarray,
    y: np.ndarray,
    n_subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Unbiased polynomial-kernel MMD^2 (KID). Returns (mean, std) over subsets.

    x = generated features, y = reference features. subset_size is clipped to
    the smaller bank so smoke (tiny banks) still runs.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = int(min(subset_size, x.shape[0], y.shape[0]))
    if m < 2:
        raise ValueError(f"KID needs >=2 samples per subset, got m={m}")
    ests = np.empty(n_subsets, dtype=np.float64)
    for i in range(n_subsets):
        xi = x[rng.choice(x.shape[0], m, replace=False)]
        yi = y[rng.choice(y.shape[0], m, replace=False)]
        kxx = _poly_kernel(xi, xi)
        kyy = _poly_kernel(yi, yi)
        kxy = _poly_kernel(xi, yi)
        # unbiased: drop diagonal of within-set kernels
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)
        term_xx = kxx.sum() / (m * (m - 1))
        term_yy = kyy.sum() / (m * (m - 1))
        term_xy = kxy.sum() * 2.0 / (m * m)
        ests[i] = term_xx + term_yy - term_xy
    return float(ests.mean()), float(ests.std())


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """95% bootstrap CI of the mean. Returns (lo, hi). NaN-safe for tiny n."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
