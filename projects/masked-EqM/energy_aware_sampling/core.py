"""Armijo sampling primitives.  Model weights are never mutated here."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch


@dataclass
class SamplingStats:
    gradient_evaluations: int = 0
    energy_forwards: int = 0
    max_backtrack_samples: int = 0


def _field_and_energy(model, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
    """One field/energy evaluation; EqM internally enables only input autograd."""
    with torch.enable_grad():
        field, energy = model(x.detach(), t, y, get_energy=True, train=False)
    return field.detach(), energy.detach()


def _energy_only(model, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Scalar-only forward: no input-gradient or higher-order graph."""
    with torch.no_grad():
        return model(x, t, y, energy_only=True).detach()


def sign_audit(model, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> dict:
    """Empirically check the sampler convention field == -grad(E)."""
    x_for_grad = x.detach().requires_grad_(True)
    with torch.enable_grad():
        energy = model(x_for_grad, t, y, energy_only=True)
        negative_gradient = -torch.autograd.grad(energy.sum(), x_for_grad)[0]
    field, reported_energy = _field_and_energy(model, x, t, y)
    flat_field = field.flatten(1)
    flat_gradient = negative_gradient.detach().flatten(1)
    cosine = torch.nn.functional.cosine_similarity(flat_field, flat_gradient, dim=1)
    relative_error = (flat_field - flat_gradient).norm(dim=1) / flat_gradient.norm(dim=1).clamp_min(1e-12)
    energy_error = (reported_energy - energy.detach()).abs()
    return {
        "cosine_min": float(cosine.min()),
        "cosine_mean": float(cosine.mean()),
        "relative_error_max": float(relative_error.max()),
        "relative_error_mean": float(relative_error.mean()),
        "energy_error_max": float(energy_error.max()),
        "pass": bool(cosine.min() > 0.999 and relative_error.max() < 1e-4 and energy_error.max() < 1e-4),
    }


def fixed_sample(model, x0, y, steps: int, step_size: float) -> tuple[torch.Tensor, dict]:
    x = x0.detach().clone()
    t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    stats = SamplingStats()
    for _ in range(steps):
        field, _ = _field_and_energy(model, x, t, y)
        stats.gradient_evaluations += x.shape[0]
        x = (x + step_size * field).detach()
        t = t + step_size
    return x, asdict(stats)


def armijo_sample(model, x0, y, steps: int, initial_step: float, c=1e-4,
                  beta=0.5, growth=1.25, max_backtracks=8) -> tuple[torch.Tensor, dict]:
    """Independent Armijo backtracking with frozen failures at the hard cap."""
    x = x0.detach().clone()
    t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    trial_step = torch.full((x.shape[0],), initial_step, device=x.device, dtype=x.dtype)
    stats = SamplingStats()
    accepted_history = []
    for _ in range(steps):
        field, energy = _field_and_energy(model, x, t, y)
        stats.gradient_evaluations += x.shape[0]
        squared_norm = field.flatten(1).square().sum(dim=1)
        accepted = torch.zeros_like(trial_step)
        unresolved = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        for backtrack in range(max_backtracks + 1):
            indices = unresolved.nonzero(as_tuple=False).squeeze(1)
            if not len(indices):
                break
            alpha = trial_step[indices] * (beta ** backtrack)
            candidate = x[indices] + alpha[:, None, None, None] * field[indices]
            candidate_energy = _energy_only(model, candidate, t[indices], y[indices])
            stats.energy_forwards += len(indices)
            sufficient = candidate_energy <= energy[indices] - c * alpha * squared_norm[indices]
            passed = indices[sufficient]
            if len(passed):
                accepted[passed] = alpha[sufficient]
                unresolved[passed] = False
        stats.max_backtrack_samples += int(unresolved.sum())
        x = (x + accepted[:, None, None, None] * field).detach()
        # A cap failure freezes only that sample; it can recover on a later field.
        trial_step = torch.where(
            accepted > 0,
            torch.minimum(accepted * growth, torch.full_like(accepted, initial_step)),
            trial_step * beta,
        )
        accepted_history.append(accepted.detach().cpu())
        t = t + initial_step
    accepted_steps = torch.stack(accepted_history, dim=1)
    result = asdict(stats)
    result["accepted_steps"] = accepted_steps
    result["accepted_step_mean"] = float(accepted_steps.mean())
    result["accepted_step_median_by_iteration"] = accepted_steps.median(dim=0).values
    return x, result


def replay_sample(model, x0, y, schedule: torch.Tensor) -> tuple[torch.Tensor, dict]:
    x = x0.detach().clone()
    t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    stats = SamplingStats()
    for alpha in schedule.to(device=x.device, dtype=x.dtype):
        field, _ = _field_and_energy(model, x, t, y)
        stats.gradient_evaluations += x.shape[0]
        x = (x + alpha * field).detach()
        t = t + alpha
    return x, asdict(stats)
