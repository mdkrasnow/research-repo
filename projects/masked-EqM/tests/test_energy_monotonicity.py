from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from energy_monotonicity.core import (
    _one_scalar_per_sample,
    cluster_bootstrap,
    get_effective_field,
    trajectory_metrics,
    trapezoid_line_integral,
)
from energy_monotonicity.evaluate_energy_monotonicity import discover_checkpoints


class ToyEqM(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.ebm = variant
        self.uncond = True
        self.learn_sigma = False
        self.weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, z, gamma, labels):
        assert self.ebm == "none"
        return self.weight * z


class ToyDot(ToyEqM):
    def __init__(self):
        super().__init__("dot")
        self.x_embedder = nn.Identity()
        self.pos_embed = 0
        self.t_embedder = lambda t: torch.zeros((len(t), 1), device=t.device)
        self.y_embedder = lambda y, train: torch.zeros((len(y), 1), device=y.device)
        self.blocks = []
        self.final_layer = lambda x, c: self.weight * x
        self.unpatchify = nn.Identity()


class ToyHead(nn.Module):
    def forward(self, tokens, conditioning):
        return (tokens.square()).flatten(1).sum(1) / 2


class ToyDirect(ToyEqM):
    def __init__(self):
        super().__init__("direct")
        self.x_embedder = nn.Identity()
        self.pos_embed = 0
        self.t_embedder = lambda t: torch.zeros((len(t), 1), device=t.device)
        self.y_embedder = lambda y, train: torch.zeros((len(y), 1), device=y.device)
        self.blocks = []
        self.energy_head = ToyHead()


def test_synthetic_conservative_field_line_integral_matches_scalar():
    gamma = torch.linspace(0, 1, 101, dtype=torch.float64)
    points = gamma[None, :, None] ** 2
    fields = points.clone()
    integral = trapezoid_line_integral(fields, points)
    scalar_difference = points[..., 0].square() / 2
    assert torch.allclose(integral, scalar_difference, atol=1e-5)


def test_correct_reversed_and_constant_ordering():
    gamma = np.linspace(0, 1, 21)
    correct = (1 - gamma)[None]
    reversed_energy = gamma[None]
    constant = np.zeros((1, 21))
    assert trajectory_metrics(correct, gamma)["ordering_accuracy"][0] == 1
    assert trajectory_metrics(reversed_energy, gamma)["ordering_accuracy"][0] == 0
    metrics = trajectory_metrics(constant, gamma)
    assert metrics["ordering_accuracy"][0] == 0
    assert metrics["tie_rate"][0] == 1


def test_dot_gradient_correctness():
    model = ToyDot()
    z = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    labels = torch.zeros(2, dtype=torch.long)
    result = get_effective_field(model, "dot", z, labels)
    # z dot (2z) = 2|z|^2 -> gradient 4z
    assert torch.allclose(result.effective_field, 4 * z)
    assert torch.allclose(result.scalar_energy, 2 * z.square().flatten(1).sum(1))


def test_direct_gradient_correctness():
    model = ToyDirect()
    z = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    labels = torch.zeros(2, dtype=torch.long)
    result = get_effective_field(model, "direct", z, labels)
    assert torch.allclose(result.effective_field, z)
    assert torch.allclose(result.scalar_energy, z.square().flatten(1).sum(1) / 2)


def test_batch_independence():
    model = ToyDirect()
    labels = torch.zeros(2, dtype=torch.long)
    first = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]])
    second = first.clone()
    second[1] = 900
    grad_first = get_effective_field(model, "direct", first, labels).effective_field[0]
    grad_second = get_effective_field(model, "direct", second, labels).effective_field[0]
    assert torch.equal(grad_first, grad_second)


def test_direct_shape_reduction_rejects_batch_coupling():
    with pytest.raises(ValueError, match="exactly one scalar"):
        _one_scalar_per_sample(torch.ones(2, 3), 2)
    with pytest.raises(ValueError, match="coupled"):
        _one_scalar_per_sample(torch.tensor(1.0), 2)


def test_trapezoid_convergence():
    errors = []
    for count in (5, 21, 101):
        gamma = torch.linspace(0, 1, count, dtype=torch.float64)
        points = gamma[None, :, None]
        fields = points.pow(3)
        estimate = trapezoid_line_integral(fields, points)[0, -1]
        errors.append(abs(float(estimate) - 0.25))
    assert errors[2] < errors[1] < errors[0]


def test_deterministic_bank_seed_and_metrics():
    def bank(seed):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(100, generator=generator)[:8]
        noise = torch.randn((8, 2, 4), generator=generator)
        return indices, noise
    first = bank(123)
    second = bank(123)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    energy = first[1].mean(-1).numpy()
    gamma = np.array([0.0, 1.0])
    assert np.array_equal(
        trajectory_metrics(energy, gamma)["ordering_accuracy"],
        trajectory_metrics(second[1].mean(-1).numpy(), gamma)["ordering_accuracy"],
    )


def test_paired_cluster_bootstrap_reuses_draws():
    image_ids = np.repeat(np.arange(4), 2)
    values = {"none": np.arange(8.0), "dot": np.arange(8.0) + 1}
    boot, draws = cluster_bootstrap(values, image_ids, 100, 7)
    assert draws.shape == (100, 4)
    assert np.allclose(boot["dot"] - boot["none"], 1)


def test_checkpoint_labeling_mismatch_fails(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = {
        "epoch": 1, "step": 10,
        "args": {"ebm": "dot", "model": "EqM-S/2", "image_size": 256,
                 "num_classes": 1000, "uncond": True, "corruption_mode": "gaussian"},
    }
    torch.save(checkpoint, run / "epoch01.pt")
    with pytest.raises(ValueError, match="labeling failure"):
        discover_checkpoints(run, "none", [1])
