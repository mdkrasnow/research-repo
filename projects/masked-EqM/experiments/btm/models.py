"""Toy networks for the five-atom benchmark.

`ScalarMLP` maps [B, d] -> [B] (the potential phi; the generative energy is
E = -phi).  `VectorMLP` maps [B, d] -> [B, d].  Widths are chosen so the two
have closely matched parameter counts, so Arm V is not silently advantaged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScalarMLP(nn.Module):
    def __init__(self, d: int = 2, width: int = 256, depth: int = 3):
        super().__init__()
        layers, dim = [], d
        for _ in range(depth):
            layers += [nn.Linear(dim, width), nn.SiLU()]
            dim = width
        layers += [nn.Linear(dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VectorMLP(nn.Module):
    def __init__(self, d: int = 2, width: int = 256, depth: int = 3):
        super().__init__()
        layers, dim = [], d
        for _ in range(depth):
            layers += [nn.Linear(dim, width), nn.SiLU()]
            dim = width
        layers += [nn.Linear(dim, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(arm: str, d: int = 2, width: int = 256, depth: int = 3,
                seed: int | None = None):
    """Shared-initialization helper: identical seed -> identical scalar init.

    Every scalar arm at a given seed gets bitwise-identical initial weights,
    which is what makes the paired G-vs-D comparison statistically powerful.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if arm in ("btm_vector", "eqm_legacy_vector"):
        return VectorMLP(d, width, depth)
    return ScalarMLP(d, width, depth)


@torch.no_grad()
def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
