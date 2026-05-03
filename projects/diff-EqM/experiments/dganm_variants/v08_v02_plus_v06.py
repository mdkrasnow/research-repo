"""v08_v02_plus_v06 — combined cosine-contrastive (v02) + reverse-ODE recovery (v06).

Hypothesis: v02 (12.96 ± 0.70) and v06 (13.73, 1 seed) target different
failure modes. v02 shapes the velocity *direction* on adversarially-mined
points; v06 shapes the velocity *integrated trajectory* on randomly-
perturbed points. If complementary, the sum should beat v02 alone.

Loss = base_FM + gamma_contrast * (lam_pos*(1-cos+) + lam_neg*relu(cos-)) +
       gamma_rec * ||short_reverse_ODE(x_t + delta) - x_clean||^2
"""

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_loss, train_loop
from .v02_score_repulsion import _cos_sim, _mine_adversarial
from .v06_diffusion_recovery import _short_reverse_ode


def step_fn(model, x, step, device, args: TrainArgs):
    e = args.extras or {}
    base = eqm_loss(model, x, device, eps=args.train_eps, a=args.a, gain=args.gain)
    total = base
    pos_val = neg_val = rec_val = 0.0

    mine_every = e.get("mine_every", 5)
    if mine_every > 0 and (step % mine_every == 0):
        B = x.size(0)

        # --- v02 contrastive branch (full t range) ---
        gamma_c = e.get("gamma_contrast", 1.0)
        if gamma_c > 0:
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
            l_pos = (1.0 - _cos_sim(v_pos, v_anchor)).mean()
            l_neg = F.relu(_cos_sim(v_neg, v_anchor) - e.get("neg_cos_margin", 0.0)).mean()
            lam_pos = e.get("lambda_pos", 1.0)
            lam_neg = e.get("lambda_neg", 1.0)
            total = total + gamma_c * (lam_pos * l_pos + lam_neg * l_neg)
            pos_val = l_pos.item()
            neg_val = l_neg.item()

        # --- v06 reverse-ODE recovery branch (t in [0.5, 1)) ---
        gamma_r = e.get("gamma_recovery", 0.3)
        ode_steps = e.get("ode_steps", 2)
        if gamma_r > 0:
            t_r = torch.rand(B, device=device) * 0.5 + 0.5 - args.train_eps
            t_r_ = t_r.view(B, 1, 1, 1)
            x0_r = torch.randn_like(x)
            x_tr = (1.0 - t_r_) * x0_r + t_r_ * x
            eps_r = e.get("perturb_eps", 0.2)
            delta = torch.randn_like(x_tr)
            flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            delta = delta * (eps_r / (flat + 1e-8))
            x_hat = _short_reverse_ode(
                model, x_tr + delta, t_r, ode_steps,
                a=args.a, gain=args.gain,
            )
            rec = F.mse_loss(x_hat, x.detach())
            total = total + gamma_r * rec
            rec_val = rec.item()

    return total, {"base": base.item(), "pos": pos_val, "neg": neg_val, "rec": rec_val}


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "pos", "neg", "rec"])
