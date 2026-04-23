"""v03_noised_negatives — mine at t ~ U(0,1) instead of hardcoded t=1.

Motivation (Luo et al. 2023, "Training EBMs with Diffusion Contrastive
Divergences", arXiv 2307.01668): fixed-noise negatives cause distributional
mismatch — at t=1 the marginal is nearly pure Gaussian, so PGD on the field
finds a region the model already handles trivially. Noise-conditional
negatives dominate fixed-noise variants.

Concretely: sample t ~ U(eps,1-eps), diffuse x_0 to x_t via the FM path,
PGA-perturb x_t, and penalize velocity at (x_t + delta, t). The penalty
is the same margin-hinge as v01 for a minimal-change ablation — isolates
"what if we just fix the t schedule?" If this beats v01, the mining is
the right idea but at the wrong noise level.

NOTE: this still uses a hinge, which the review (VeCoR 2025, Du 2021)
argues is structurally wrong because ||v|| is unbounded at small t. v02
fixes the hinge; this variant isolates the t-schedule fix alone.
"""

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_ct, eqm_loss, train_loop


def _mine_noised(model, x, t, epsilon, mining_steps, mining_lr):
    """PGA on x_t (noised x), ascending ||field(x_t+delta, t)||."""
    B = x.shape[0]
    delta = torch.randn_like(x) * 0.01
    delta.requires_grad_(True)
    t_model = (t * 999.0).clamp_min(0.0)
    for _ in range(mining_steps):
        x_neg = x.detach() + delta
        field = model(x_neg, t_model)
        obj = -field.flatten(1).norm(dim=1).mean()  # ascend -||field||
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

    gamma = e.get("gamma", 2.0)
    mine_every = e.get("mine_every", 5)
    if gamma > 0 and mine_every > 0 and (step % mine_every == 0):
        B = x.size(0)
        t = torch.rand(B, device=device) * (1.0 - 2.0 * args.train_eps) + args.train_eps
        t_ = t.view(B, 1, 1, 1)
        x0 = torch.randn_like(x)
        x_t = (1.0 - t_) * x0 + t_ * x
        x_neg = _mine_noised(
            model, x_t, t,
            epsilon=e.get("mining_epsilon", 0.5),
            mining_steps=e.get("mining_steps", 3),
            mining_lr=e.get("mining_lr", 0.01),
        )
        t_model = (t * 999.0).clamp_min(0.0)
        field = model(x_neg, t_model)
        field_norm = field.flatten(1).norm(dim=1)
        # Scale the margin by c(t) so it's comparable across the t schedule.
        ct = eqm_ct(t, a=args.a, gain=args.gain)
        margin = e.get("neg_margin", 5.0) * ct
        neg = F.relu(margin - field_norm).mean()
        total = base + gamma * neg
        neg_val = neg.item()

    return total, {"base": base.item(), "neg": neg_val}


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "neg"])
