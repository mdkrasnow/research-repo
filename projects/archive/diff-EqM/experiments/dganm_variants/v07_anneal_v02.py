"""v07_anneal_v02 — v02 score-repulsion with linear lambda decay to zero.

Hypothesis: aux losses on top of EqM may help escape early local optima
but hurt final FM convergence (v02 lands at 12.26 vs vanilla ~9.5 at
150ep, 1 seed). Linearly anneal lambda_pos and lambda_neg from their
config values down to 0.0 over the full training run, so the loss
becomes pure EqM by the last epoch. If this beats vanilla, the right
framing is "DG-ANM as curriculum, not regularizer."

CIFAR-10 with batch_size=128 -> ~391 steps/epoch. Total steps = epochs*391.
Progress = step / total_steps, clamped to [0, 1]. Annealed lambda =
config_lambda * (1 - progress).
"""

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_loss, train_loop
from .v02_score_repulsion import _cos_sim, _mine_adversarial


_STEPS_PER_EPOCH_CIFAR_BS128 = 391


def step_fn(model, x, step, device, args: TrainArgs):
    e = args.extras or {}
    base = eqm_loss(model, x, device, eps=args.train_eps, a=args.a, gain=args.gain)
    total = base
    pos_val = neg_val = 0.0
    anneal = 0.0

    gamma = e.get("gamma", 1.0)
    mine_every = e.get("mine_every", 5)
    if gamma > 0 and mine_every > 0 and (step % mine_every == 0):
        total_steps = max(1, args.epochs * _STEPS_PER_EPOCH_CIFAR_BS128)
        progress = min(1.0, step / total_steps)
        anneal = max(0.0, 1.0 - progress)

        B = x.size(0)
        t = torch.rand(B, device=device) * (1.0 - 2.0 * args.train_eps) + args.train_eps
        t_ = t.view(B, 1, 1, 1)
        x0 = torch.randn_like(x)
        x_t = (1.0 - t_) * x0 + t_ * x
        t_model = (t * 999.0).clamp_min(0.0)

        pos_sigma = e.get("pos_sigma", 0.05)
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
        lam_pos = e.get("lambda_pos", 1.0) * anneal
        lam_neg = e.get("lambda_neg", 1.0) * anneal
        total = base + gamma * (lam_pos * l_pos + lam_neg * l_neg)
        pos_val = l_pos.item()
        neg_val = l_neg.item()

    return total, {"base": base.item(), "pos": pos_val, "neg": neg_val, "anneal": anneal}


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "pos", "neg", "anneal"])
