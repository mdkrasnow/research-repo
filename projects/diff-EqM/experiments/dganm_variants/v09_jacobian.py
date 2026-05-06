"""v09_jacobian — random-noise Jacobian regularizer (architecture-agnostic).

Diagnosis: v02 cosine-contrastive saturates on EqM-B/2 transformer because
|v|>>|J·δ| → cos(v(x), v(x+δ)) ≈ 1 for any small δ → PGA gradient vanishes.

Fix: drop cosine + drop PGA. Use unnormalized squared difference of velocity
under random Gaussian perturbation. Magnitude-aware, never saturates,
architecture-agnostic.

L_jac = ||v(x_t + δ) - v(x_t)||² / σ²    (penalize)

This is finite-difference Jacobian regularization (Sokolic et al. 2017,
arXiv:1605.08254) / input-noise regularization (Bishop 1995). Encourages
local Lipschitz continuity of the velocity field, which stabilizes ODE flow.

No mining loop; one extra forward per `jac_every` step.
"""

import torch

from ._common import TrainArgs, eqm_loss, train_loop


def step_fn(model, x, step, device, args: TrainArgs):
    e = args.extras or {}
    base = eqm_loss(model, x, device, eps=args.train_eps, a=args.a, gain=args.gain)
    total = base
    jac_val = anchor_norm = 0.0

    lam = e.get("jac_lambda", 0.1)
    sigma = e.get("jac_sigma", 0.05)
    every = e.get("jac_every", 5)
    if lam > 0 and every > 0 and (step % every == 0):
        B = x.size(0)
        t = torch.rand(B, device=device) * (1.0 - 2.0 * args.train_eps) + args.train_eps
        t_ = t.view(B, 1, 1, 1)
        x0 = torch.randn_like(x)
        x_t = (1.0 - t_) * x0 + t_ * x
        t_model = (t * 999.0).clamp_min(0.0)

        delta = sigma * torch.randn_like(x_t)
        v_anchor = model(x_t, t_model)
        v_pert = model(x_t + delta, t_model)

        diff = (v_pert - v_anchor).flatten(1)
        l_jac = (diff.pow(2).sum(dim=1) / (sigma * sigma + 1e-8)).mean()
        total = base + lam * l_jac
        jac_val = l_jac.item()
        anchor_norm = v_anchor.flatten(1).norm(dim=1).mean().item()

    return total, {"base": base.item(), "jac": jac_val, "anchor_norm": anchor_norm}


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "jac", "anchor_norm"])
