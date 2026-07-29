"""Unit tests for per-sample Armijo semantics, independent of the transformer."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from energy_aware_sampling.core import armijo_sample, sign_audit


class QuadraticEnergy(torch.nn.Module):
    def forward(self, x, t, y, get_energy=False, energy_only=False, train=False):
        energy = 0.5 * x.flatten(1).square().sum(1)
        if energy_only:
            return energy
        field = -x
        return (field, energy) if get_energy else field


def main():
    model = QuadraticEnergy()
    x = torch.tensor([[[[1.0]]], [[[3.0]]]])
    y = torch.zeros(2, dtype=torch.long)
    audit = sign_audit(model, x, torch.zeros(2), y)
    assert audit["pass"], audit
    _, stats = armijo_sample(model, x, y, steps=1, initial_step=3.0, max_backtracks=8)
    accepted = stats["accepted_steps"][:, 0]
    assert torch.allclose(accepted, torch.tensor([1.5, 1.5]))
    assert stats["energy_forwards"] == 4
    print("PASS Armijo uses independent sufficient-decrease backtracking")


if __name__ == "__main__":
    main()
