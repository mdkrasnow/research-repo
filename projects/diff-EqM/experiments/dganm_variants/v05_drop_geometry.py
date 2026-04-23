"""v05_drop_geometry — v01 minus the feature-normal/tangent projector.

Motivation (Pidstrigach, NeurIPS 2022, "Score-Based Generative Models
Detect Manifolds"): learned scores already decompose into tangential
and normal components implicitly. The explicit P_N / P_T SVD projector
in v01 may be injecting stale geometry that the network would otherwise
learn on its own.

Concretely: same PGA mining as v01 (pixel-space, epsilon-L2 ball, ascend
-||field||), same margin-hinge, but no feature projector. If v05 matches
or beats v01, the geometry module is net-negative and should be dropped
from every downstream variant.

This is a pure ablation — not expected to beat vanilla on its own.
"""

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_loss, train_loop


def _mine_pixel(model, x, epsilon, mining_steps, mining_lr, device):
    B = x.shape[0]
    delta = torch.randn_like(x) * 0.01
    delta.requires_grad_(True)
    t_ones = torch.ones(B, device=device) * 999.0
    for _ in range(mining_steps):
        x_neg = x.detach() + delta
        field = model(x_neg, t_ones)
        obj = -field.flatten(1).norm(dim=1).mean()
        grad = torch.autograd.grad(obj, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta = delta + mining_lr * grad.sign()
            flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            delta = delta * torch.clamp(epsilon / (flat + 1e-8), max=1.0)
            delta = delta.detach().requires_grad_(True)
    return (x + delta).detach()


def step_fn(model, x, step, device, args: TrainArgs):
    e = args.extras or {}
    base = eqm_loss(model, x, device, eps=args.train_eps, a=args.a, gain=args.gain)
    neg_val = 0.0
    total = base

    gamma = e.get("gamma", 6.0)
    mine_every = e.get("mine_every", 5)
    if gamma > 0 and mine_every > 0 and (step % mine_every == 0):
        x_neg = _mine_pixel(
            model, x,
            epsilon=e.get("mining_epsilon", 0.8),
            mining_steps=e.get("mining_steps", 3),
            mining_lr=e.get("mining_lr", 0.01),
            device=device,
        )
        B = x_neg.size(0)
        t_ones = torch.ones(B, device=device) * 999.0
        field = model(x_neg, t_ones)
        field_norm = field.flatten(1).norm(dim=1)
        neg = F.relu(e.get("neg_margin", 50.0) - field_norm).mean()
        total = base + gamma * neg
        neg_val = neg.item()

    return total, {"base": base.item(), "neg": neg_val}


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "neg"])
