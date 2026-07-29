"""Regression coverage for scalar-only EqM evaluations used by line search."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import EqM


def make_model(ebm):
    model = EqM(input_size=4, patch_size=2, in_channels=4, hidden_size=32,
                depth=2, num_heads=4, mlp_ratio=2.0, class_dropout_prob=0.0,
                num_classes=10, learn_sigma=False, uncond=True, ebm=ebm)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    model.eval()
    return model


def main():
    torch.manual_seed(1)
    x = torch.randn(3, 4, 4, 4)
    t = torch.zeros(3)
    y = torch.tensor([1, 2, 3])
    for ebm in ("direct", "dot"):
        model = make_model(ebm)
        with torch.no_grad():
            scalar = model(x.clone(), t, y, energy_only=True)
        with torch.enable_grad():
            _, energy = model(x.clone(), t, y, get_energy=True)
        assert scalar.shape == (3,)
        assert not scalar.requires_grad
        assert torch.allclose(scalar, energy.detach(), atol=1e-5, rtol=1e-5)
        print(f"PASS {ebm}: scalar-only energy matches field-evaluation energy")


if __name__ == "__main__":
    main()
