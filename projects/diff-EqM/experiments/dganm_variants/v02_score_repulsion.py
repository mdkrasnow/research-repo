"""v02_score_repulsion — VeCoR-style velocity contrastive regularization.

Motivation (Jiang et al. 2025, "VeCoR: Velocity Contrastive Regularization
for Flow Matching," arXiv 2511.18942): the closest prior art to DG-ANM.
Reports 22-35% relative FID reductions on SiT-XL/2 by replacing hinge
losses with a two-sided contrastive objective on the velocity field.

Also: "What is Adversarial Training for Diffusion Models?" (arXiv
2505.21742) argues the correct AT primitive for DMs is *equivariance*
of the velocity under small perturbations — penalizing raw norm (v01)
enforces invariance, which is known to hurt. Contrastive cosine avoids
this because it is sign/direction sensitive, not magnitude-clamping.

Concretely: at (x_t, t), form a small semantic-preserving perturbation
x_t^+ (pos, on-manifold-ish: gaussian noise scaled to stay inside the
forward-diffusion ball) and x_t^- (neg, off-manifold: PGA-mined to
maximize velocity discrepancy). The velocity at x_t^+ should *agree*
with the velocity at x_t (cosine close to 1); the velocity at x_t^-
should *disagree* (cosine pushed toward 0 or below). No fixed margin —
the objective is relative and cannot saturate.

Replaces the ReLU(m - ||v||) hinge with:
    L_contrast = (1 - cos(v(x_t), v(x_t^+))) + cos(v(x_t), v(x_t^-))_+

where (·)_+ is clamped to non-negative to avoid pushing unbounded-large
anti-alignment.
"""

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_loss, train_loop


def _cos_sim(a, b):
    a = a.flatten(1)
    b = b.flatten(1)
    return F.cosine_similarity(a, b, dim=1)


def _mine_adversarial(model, x_t, t_model, epsilon, mining_steps, mining_lr):
    """PGA to maximize cosine-dissimilarity wrt current velocity at x_t."""
    with torch.no_grad():
        v_anchor = model(x_t, t_model)
    B = x_t.shape[0]
    delta = torch.randn_like(x_t) * 0.01
    delta.requires_grad_(True)
    for _ in range(mining_steps):
        v_neg = model(x_t.detach() + delta, t_model)
        # Maximize (1 - cos) => minimize cos.
        obj = _cos_sim(v_neg, v_anchor).mean()
        grad = torch.autograd.grad(obj, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta = delta - mining_lr * grad.sign()  # descend cos
            flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            delta = delta * torch.clamp(epsilon / (flat + 1e-8), max=1.0)
            delta = delta.detach().requires_grad_(True)
    return (x_t + delta).detach()


def step_fn(model, x, step, device, args: TrainArgs):
    e = args.extras or {}
    base = eqm_loss(model, x, device, eps=args.train_eps, a=args.a, gain=args.gain)
    total = base
    pos_val = neg_val = 0.0

    gamma = e.get("gamma", 1.0)
    mine_every = e.get("mine_every", 5)
    if gamma > 0 and mine_every > 0 and (step % mine_every == 0):
        B = x.size(0)
        t = torch.rand(B, device=device) * (1.0 - 2.0 * args.train_eps) + args.train_eps
        t_ = t.view(B, 1, 1, 1)
        x0 = torch.randn_like(x)
        x_t = (1.0 - t_) * x0 + t_ * x
        t_model = (t * 999.0).clamp_min(0.0)

        pos_sigma = e.get("pos_sigma", 0.05)  # small on-manifold jitter
        x_pos = x_t + pos_sigma * torch.randn_like(x_t)
        x_neg = _mine_adversarial(
            model, x_t, t_model,
            epsilon=e.get("mining_epsilon", 0.3),
            mining_steps=e.get("mining_steps", 3),
            mining_lr=e.get("mining_lr", 0.01),
        )

        v_anchor = model(x_t, t_model)
        v_pos = model(x_pos, t_model)
        v_neg = model(x_neg, t_model)

        sim_pos = _cos_sim(v_pos, v_anchor)
        sim_neg = _cos_sim(v_neg, v_anchor)
        l_pos = (1.0 - sim_pos).mean()
        l_neg = F.relu(sim_neg - e.get("neg_cos_margin", 0.0)).mean()
        lam_pos = e.get("lambda_pos", 1.0)
        lam_neg = e.get("lambda_neg", 1.0)
        total = base + gamma * (lam_pos * l_pos + lam_neg * l_neg)
        pos_val = l_pos.item()
        neg_val = l_neg.item()

    return total, {"base": base.item(), "pos": pos_val, "neg": neg_val}


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "pos", "neg"])
