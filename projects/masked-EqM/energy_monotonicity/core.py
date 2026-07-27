from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

Variant = Literal["none", "dot", "direct"]


@dataclass
class FieldResult:
    effective_field: torch.Tensor
    scalar_energy: torch.Tensor | None
    raw_output_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def _one_scalar_per_sample(output: torch.Tensor, batch_size: int) -> torch.Tensor:
    if output.ndim == 0:
        if batch_size != 1:
            raise ValueError("scalar model output coupled the batch")
        return output.reshape(1)
    if output.shape[0] != batch_size:
        raise ValueError(f"scalar output batch {output.shape[0]} != input batch {batch_size}")
    flattened = output.reshape(batch_size, -1)
    if flattened.shape[1] != 1:
        raise ValueError(f"direct output must contain exactly one scalar/sample, got {tuple(output.shape)}")
    return flattened[:, 0]


def _raw_backbone_output(model: torch.nn.Module, z: torch.Tensor, t: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
    """Run the existing vector stack without its dot input-gradient rewrite."""
    base = _unwrap(model)
    if base.ebm == "direct":
        raise ValueError("direct model has no vector output head")
    x = base.x_embedder(z) + base.pos_embed
    te = base.t_embedder(torch.zeros_like(t) if base.uncond else t)
    ye = base.y_embedder(labels, False)
    conditioning = te + ye
    for block in base.blocks:
        x = block(x, conditioning)
    output = base.unpatchify(base.final_layer(x, conditioning))
    if base.learn_sigma:
        output, _ = output.chunk(2, dim=1)
    return output


def get_effective_field(model: torch.nn.Module, variant: Variant, z: torch.Tensor,
                        labels: torch.Tensor | None = None,
                        gamma: torch.Tensor | None = None) -> FieldResult:
    """Extract exactly the preregistered field without parameter gradients."""
    if labels is None:
        raise ValueError("ground-truth labels are required for this class-conditional model")
    if gamma is None:
        gamma = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    base = _unwrap(model)
    if getattr(base, "ebm", None) != variant:
        raise ValueError(f"model ebm={getattr(base, 'ebm', None)!r}, requested variant={variant!r}")
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise ValueError("model parameters must be frozen before field extraction")

    if variant == "none":
        with torch.no_grad():
            raw = base(z, gamma, labels)
        return FieldResult(raw, None, tuple(raw.shape),
                           {"expected_raw_direction": "increasing_toward_data",
                            "canonical_energy_sign": -1})

    z_leaf = z.detach().requires_grad_(True)
    with torch.enable_grad():
        if variant == "dot":
            raw = _raw_backbone_output(base, z_leaf, gamma, labels)
            scalar = (z_leaf * raw).flatten(1).sum(1)
        else:
            tokens, conditioning = _direct_tokens(base, z_leaf, gamma, labels)
            raw = base.energy_head(tokens, conditioning)
            scalar = _one_scalar_per_sample(raw, z_leaf.shape[0])
        effective = torch.autograd.grad(
            scalar.sum(), z_leaf, create_graph=False, retain_graph=False,
            only_inputs=True,
        )[0]
    sign = -1 if variant == "dot" else 1
    return FieldResult(
        effective.detach(), scalar.detach(), tuple(raw.shape),
        {"expected_raw_direction": "increasing_toward_data" if variant == "dot"
         else "decreasing_toward_data", "canonical_energy_sign": sign},
    )


def _direct_tokens(model: torch.nn.Module, z: torch.Tensor, t: torch.Tensor,
                   labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = model.x_embedder(z) + model.pos_embed
    te = model.t_embedder(torch.zeros_like(t) if model.uncond else t)
    ye = model.y_embedder(labels, False)
    conditioning = te + ye
    for block in model.blocks:
        x = block(x, conditioning)
    return x, conditioning


def trapezoid_line_integral(fields: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Return path integrals with zero at point 0. Shapes: [T,G,...]."""
    if fields.shape != points.shape or fields.ndim < 3:
        raise ValueError("fields and points must share [trajectory,gamma,...] shape")
    increments = (
        0.5 * (fields[:, :-1] + fields[:, 1:]) *
        (points[:, 1:] - points[:, :-1])
    ).flatten(2).sum(2)
    zero = torch.zeros((fields.shape[0], 1), dtype=increments.dtype, device=increments.device)
    return torch.cat([zero, increments.cumsum(1)], dim=1)


def trajectory_metrics(energy: np.ndarray, gamma: np.ndarray,
                       fields: np.ndarray | None = None,
                       epsilon_minus_x: np.ndarray | None = None,
                       target: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if energy.ndim != 2 or energy.shape[1] != len(gamma):
        raise ValueError("energy must have shape [trajectory,gamma]")
    left, right = np.triu_indices(len(gamma), 1)
    differences = energy[:, left] - energy[:, right]
    adjacent = energy[:, :-1] - energy[:, 1:]
    violations = np.minimum(adjacent, 0.0)
    ranks = np.argsort(np.argsort(energy, axis=1), axis=1).astype(np.float64)
    gamma_ranks = np.arange(len(gamma), dtype=np.float64)
    ranks -= ranks.mean(1, keepdims=True)
    centered_gamma = gamma_ranks - gamma_ranks.mean()
    denom = np.sqrt((ranks ** 2).sum(1) * (centered_gamma ** 2).sum())
    result = {
        "ordering_accuracy": (differences > 0).mean(1),
        "tie_rate": (differences == 0).mean(1),
        "adjacent_accuracy": (adjacent > 0).mean(1),
        "perfect_trajectory": (adjacent > 0).all(1).astype(np.float64),
        "spearman": (ranks * centered_gamma).sum(1) / np.maximum(denom, 1e-30),
        "violation_count": (adjacent <= 0).sum(1),
        "violation_magnitude": -violations.sum(1),
        "total_energy_drop": energy[:, 0] - energy[:, -1],
        "nan_rate": np.isnan(energy).mean(1),
        "inf_rate": np.isinf(energy).mean(1),
    }
    if fields is not None:
        flat = fields.reshape(fields.shape[0], fields.shape[1], -1)
        norms = np.linalg.norm(flat, axis=2)
        result["zero_field_rate"] = (norms == 0).mean(1)
        if epsilon_minus_x is not None:
            direction = epsilon_minus_x.reshape(epsilon_minus_x.shape[0], -1)
            numerator = (flat * direction[:, None]).sum(2)
            denominator = norms * np.linalg.norm(direction, axis=1)[:, None]
            result["directional_alignment"] = (
                numerator / np.maximum(denominator, 1e-30)
            ).mean(1)
        if target is not None:
            result["field_target_mse"] = ((fields - target) ** 2).reshape(
                fields.shape[0], fields.shape[1], -1
            ).mean((1, 2))
    return result


def cluster_bootstrap(values: dict[str, np.ndarray], image_ids: np.ndarray,
                      replicates: int, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    variants = tuple(values)
    unique = np.unique(image_ids)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("image cluster ids must be contiguous from zero")
    cluster_values = {}
    for variant, array in values.items():
        cluster_values[variant] = np.array([
            array[image_ids == image_id].mean() for image_id in unique
        ])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)), dtype=np.int32)
    boot = {
        variant: cluster_values[variant][draws].mean(1)
        for variant in variants
    }
    return boot, draws
