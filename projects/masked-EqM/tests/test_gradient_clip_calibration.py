import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "direct_energy"))
from calibrate_gradient_clip import calibrate


def test_calibration_uses_twice_p99():
    records = [{"grad_norm": value} for value in range(1, 101)]
    result = calibrate(records)
    assert result["p99_grad_norm"] == pytest.approx(99.01)
    assert result["max_grad_norm"] == pytest.approx(198.02)
    assert result["median_grad_norm"] == pytest.approx(50.5)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_calibration_rejects_invalid_norms(value):
    with pytest.raises(ValueError):
        calibrate([{"grad_norm": value}])


def test_global_gradient_clip_bounds_oversized_gradient():
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([30.0, 40.0])
    original_norm = torch.nn.utils.clip_grad_norm_([parameter], max_norm=2.0)
    assert original_norm == pytest.approx(50.0)
    assert parameter.grad.norm().item() == pytest.approx(2.0)
