"""v04_ebm_contrastive — InfoNCE-with-one-negative on a velocity energy.

Motivation (Du, Li, Tenenbaum, Mordatch, "Improved Contrastive Divergence
Training of EBMs," ICML 2021): CD with PGD negatives is the canonical
recipe for avoiding saturation. Key idea: replace absolute-margin hinge
with a *relative* (logit-style) objective so the negative/positive gap
is what's optimized — there's no floor to saturate at.

Also (Luo et al. 2023, DCD, arXiv 2307.01668): diffusion-aware CD with
noise-conditional negatives dominates pure Langevin CD.

Concretely: define a per-sample energy
    E(x, t) = ||v_theta(x, t)||^2 / 2
and train with a binary-contrastive / InfoNCE-with-one-neg objective:
    L_cd = softplus( E(x_clean, t) - E(x_neg, t) )
This wants E(x_clean) < E(x_neg), i.e. *lower* velocity norm on data
and *higher* on negatives — a relative version of v01's hinge, with
no saturation.

Caveat from Du 2021: PGD-based EBM CD can be unstable without KL
regularization on the sampler chain. We keep the chain short (3 steps)
and add an L2 regularizer on the energy magnitude to dampen divergence.
"""

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_loss, train_loop


def _energy(model, x, t_model):
    v = model(x, t_model)
    return 0.5 * v.flatten(1).pow(2).sum(dim=1)


def _mine_by_ascending_energy(model, x_t, t_model, epsilon, steps, lr):
    B = x_t.shape[0]
    delta = torch.randn_like(x_t) * 0.01
    delta.requires_grad_(True)
    for _ in range(steps):
        e = _energy(model, x_t.detach() + delta, t_model).mean()
        grad = torch.autograd.grad(-e, delta, retain_graph=False)[0]  # ascend
        with torch.no_grad():
            delta = delta - lr * grad.sign()
            flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            delta = delta * torch.clamp(epsilon / (flat + 1e-8), max=1.0)
            delta = delta.detach().requires_grad_(True)
    return (x_t + delta).detach()


def step_fn(model, x, step, device, args: TrainArgs):
    e = args.extras or {}
    base = eqm_loss(model, x, device, eps=args.train_eps, a=args.a, gain=args.gain)
    total = base
    cd_val = reg_val = 0.0

    gamma = e.get("gamma", 0.5)
    mine_every = e.get("mine_every", 5)
    if gamma > 0 and mine_every > 0 and (step % mine_every == 0):
        B = x.size(0)
        t = torch.rand(B, device=device) * (1.0 - 2.0 * args.train_eps) + args.train_eps
        t_ = t.view(B, 1, 1, 1)
        x0 = torch.randn_like(x)
        x_t = (1.0 - t_) * x0 + t_ * x
        t_model = (t * 999.0).clamp_min(0.0)

        x_neg = _mine_by_ascending_energy(
            model, x_t, t_model,
            epsilon=e.get("mining_epsilon", 0.3),
            steps=e.get("mining_steps", 3),
            lr=e.get("mining_lr", 0.01),
        )
        e_pos = _energy(model, x_t, t_model)
        e_neg = _energy(model, x_neg, t_model)
        # softplus(e_pos - e_neg): wants e_pos << e_neg. Never saturates.
        cd = F.softplus(e_pos - e_neg).mean()
        # Du-style energy regularizer.
        reg = e.get("energy_reg", 1e-4) * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
        total = base + gamma * cd + reg
        cd_val = cd.item()
        reg_val = reg.item()

    return total, {"base": base.item(), "cd": cd_val, "reg": reg_val}


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "cd", "reg"])
