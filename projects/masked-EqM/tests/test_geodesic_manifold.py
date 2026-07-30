from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geodesic_manifold.core import (calibrate_linear, lambda_from_energy, open_uniform_cubic_basis,
    path_from_controls, kinetic_objective, normalized_manifold_metrics, paired_bootstrap)

def test_calibration_has_fixed_targets_and_rejects_bad_ordering():
    c = calibrate_linear(torch.tensor([1., 3.]), torch.tensor([5., 7.]))
    assert torch.allclose(lambda_from_energy(torch.tensor([2., 6.]), c), torch.tensor([1., 1000.]))
    with pytest.raises(ValueError, match="ordering"):
        calibrate_linear(torch.tensor([3.]), torch.tensor([2.]))

def test_spline_preserves_endpoints_and_energy_is_differentiable():
    basis = open_uniform_cubic_basis()
    assert torch.allclose(basis.sum(1), torch.ones(33), atol=1e-5)
    control = torch.zeros(2, 10, 1); control[:, -1] = 1
    control.requires_grad_(); path = path_from_controls(control, basis)
    objective, metric = kinetic_objective(path, lambda x: x.flatten(2).square().mean(2) + 1)
    objective.sum().backward()
    assert torch.equal(path[:, 0], control[:, 0]) and torch.equal(path[:, -1], control[:, -1])
    assert torch.isfinite(metric).all() and control.grad is not None

def test_manifold_metric_and_paired_bootstrap_are_paired():
    refs = np.array([[0., 0.], [2., 0.], [0., 2.]])
    path = np.linspace([0., 0.], [2., 0.], 33)[None]
    m = normalized_manifold_metrics(path, refs, np.ones(3))
    assert m["excess"].shape == (1,) and m["precision"][0] > .1
    boot = paired_bootstrap(np.array([2., 4.]), np.array([1., 2.]), 20, 7)
    assert np.allclose(boot, .5)
    # Some paths can be inside the estimated manifold exactly (zero excess).
    # Pooled resampling must remain finite rather than divide each zero by eps.
    boot = paired_bootstrap(np.array([0., 2.]), np.array([0., 1.]), 20, 7)
    assert np.isfinite(boot).all()
