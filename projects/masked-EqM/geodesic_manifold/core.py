from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Calibration:
    mean_on: float
    mean_off: float
    alpha: float
    beta: float
    metric: str


def calibrate_linear(energy_on: torch.Tensor, energy_off: torch.Tensor) -> Calibration:
    """Calibrate lambda(E_on)=1 and lambda(E_off)=1000; never choose signs here."""
    on, off = float(energy_on.mean()), float(energy_off.mean())
    if not math.isfinite(on) or not math.isfinite(off) or off <= on:
        raise ValueError(f"fixed-sign energy ordering failed: on={on}, off={off}")
    alpha = 999.0 / (off - on)
    return Calibration(on, off, alpha, 1.0 - alpha * on, "linear")


def lambda_from_energy(energy: torch.Tensor, calibration: Calibration) -> torch.Tensor:
    if calibration.metric == "linear":
        return calibration.alpha * energy + calibration.beta
    if calibration.metric == "exp":
        return torch.exp(math.log(1000.0) * (energy - calibration.mean_on) /
                         (calibration.mean_off - calibration.mean_on))
    raise ValueError(calibration.metric)


def open_uniform_cubic_basis(points: int = 33, controls: int = 10) -> torch.Tensor:
    """Open-uniform cubic B-spline basis. Ten controls means eight free interiors."""
    if controls != 10:
        raise ValueError("this benchmark fixes 8 interior controls (10 including endpoints)")
    degree = 3
    knots = np.r_[np.zeros(degree + 1), np.linspace(0, 1, controls - degree + 1)[1:-1],
                  np.ones(degree + 1)]
    x = np.linspace(0, 1, points)
    basis = np.zeros((controls, len(x)), dtype=np.float64)
    for i in range(controls):
        basis[i] = ((knots[i] <= x) & (x < knots[i + 1])).astype(float)
    basis[-1, -1] = 1.0
    for p in range(1, degree + 1):
        next_basis = np.zeros_like(basis)
        for i in range(controls):
            if knots[i + p] > knots[i]:
                next_basis[i] += (x - knots[i]) / (knots[i + p] - knots[i]) * basis[i]
            if i + 1 < controls and knots[i + p + 1] > knots[i + 1]:
                next_basis[i] += (knots[i + p + 1] - x) / (knots[i + p + 1] - knots[i + 1]) * basis[i + 1]
        basis = next_basis
    basis = basis.T
    basis[0] = 0; basis[0, 0] = 1
    basis[-1] = 0; basis[-1, -1] = 1
    return torch.from_numpy(basis.astype(np.float32))


def path_from_controls(controls: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """[B,10,...] -> [B,33,...]."""
    return torch.einsum("pc,bc...->bp...", basis.to(controls), controls)


def kinetic_objective(path: torch.Tensor, lambda_at: callable) -> tuple[torch.Tensor, torch.Tensor]:
    mid = (path[:, :-1] + path[:, 1:]) / 2
    lam = lambda_at(mid)
    if lam.ndim != 2 or lam.shape != mid.shape[:2]:
        raise ValueError("metric must return one positive scalar per path segment")
    if not torch.isfinite(lam).all() or (lam <= 0).any():
        raise ValueError("non-positive/non-finite Riemannian metric")
    delta2 = (path[:, 1:] - path[:, :-1]).flatten(2).square().sum(2)
    return .5 * (lam * delta2 * 32.0).sum(1), lam


def normalized_manifold_metrics(path_features: np.ndarray, reference_features: np.ndarray,
                                radii: np.ndarray) -> dict[str, np.ndarray]:
    """Features include endpoints; excludes only k=0 endpoint by specification's 31 interiors."""
    interior = path_features[:, 1:32]
    distance = np.linalg.norm(interior[:, :, None] - reference_features[None, None], axis=-1)
    rho = (distance / np.maximum(radii[None, None], 1e-12)).min(2)
    excess = np.maximum(0, rho - 1).mean(1)
    precision = (rho <= 1).mean(1)
    d_rmse = np.sqrt(np.square(distance.min(2)).mean(1))
    length = np.linalg.norm(np.diff(path_features, axis=1), axis=-1).sum(1)
    endpoint = np.linalg.norm(path_features[:, -1] - path_features[:, 0], axis=-1)
    return {"excess": excess, "precision": precision, "d_rmse": d_rmse,
            "detour": length / np.maximum(endpoint, 1e-12), "rho": rho}


def kth_neighbor_radii(features: np.ndarray, k: int = 5) -> np.ndarray:
    distances = np.linalg.norm(features[:, None] - features[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return np.partition(distances, k - 1, axis=1)[:, k - 1]


def paired_bootstrap(dot: np.ndarray, direct: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    """Bootstrap the pooled direct-over-dot reduction, preserving endpoint pairs.

    With one frozen checkpoint pair this is intentionally pair-level only;
    seed-level inference requires additional independently trained models.
    Taking the ratio after each resampled mean implements the reported pooled
    quantity and stays well-defined when individual paths have zero excess.
    """
    if dot.shape != direct.shape:
        raise ValueError("direct/dot paths must share the exact endpoint bank")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(dot), size=(replicates, len(dot)))
    return ((dot[draws].mean(1) - direct[draws].mean(1)) /
            np.maximum(dot[draws].mean(1), 1e-12))
