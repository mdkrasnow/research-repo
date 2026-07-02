"""v01_current — frozen reference of the current (broken) DG-ANM.

PGA mining in pixel space + feature-normal / feature-tangent projector +
margin-hinge on ||field|| at fixed t=1. Known to lose to vanilla by ~2x
FID on CIFAR-10 150ep. Kept for regression reference — every new variant
must beat this.

Diagnostic `diag_dganm_signal.py` shows the hinge saturates to 0 within
a handful of epochs, so the mining term is effectively inert during most
of training; this variant exists to document that failure mode, not to
propose a fix.
"""

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_loss, train_loop


def _estimate_local_geometry(features: torch.Tensor, k: int = 10):
    B, D = features.shape
    device = features.device
    k = min(k, B - 1)
    dists = torch.cdist(features, features)
    nn_idx = dists.topk(k=k + 1, largest=False).indices[:, 1:]

    P_T = torch.zeros(B, D, D, device=device)
    for i in range(B):
        nbrs = features[nn_idx[i]]
        centered = nbrs - features[i:i + 1]
        U, S, _ = torch.linalg.svd(centered.T, full_matrices=False)
        rank = (S > 1e-6).sum().item()
        U_T = U[:, :rank]
        P_T[i] = U_T @ U_T.T
    P_N = torch.eye(D, device=device).unsqueeze(0) - P_T
    return P_T, P_N


def _mine_negatives(model, x, features, P_N, P_T,
                    epsilon, mining_steps, mining_lr, device):
    B = x.shape[0]
    delta = torch.randn_like(x) * 0.01
    delta.requires_grad_(True)
    t_ones = torch.ones(B, device=device) * 999.0

    for _ in range(mining_steps):
        x_neg = x.detach() + delta
        field, neg_feats = model(x_neg, t_ones, return_features=True)
        field_norm = field.flatten(1).norm(dim=1)
        neg_features = neg_feats.flatten(2).mean(dim=2)
        delta_phi = neg_features - features.detach()
        delta_N = torch.bmm(P_N.detach(), delta_phi.unsqueeze(-1)).squeeze(-1)
        delta_T = torch.bmm(P_T.detach(), delta_phi.unsqueeze(-1)).squeeze(-1)
        obj = (delta_N.norm(dim=1) ** 2 - delta_T.norm(dim=1) ** 2
               - 0.1 * field_norm).mean()
        grad_delta = torch.autograd.grad(obj, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta = delta + mining_lr * grad_delta.sign()
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
        with torch.no_grad():
            t_ones = torch.ones(x.size(0), device=device) * 999.0
            _, feats = model(x, t_ones, return_features=True)
            anchor = feats.flatten(2).mean(dim=2)
            P_T, P_N = _estimate_local_geometry(anchor, k=e.get("geometry_k", 10))
        x_neg = _mine_negatives(
            model, x, anchor, P_N, P_T,
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
