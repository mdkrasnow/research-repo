"""v06_diffusion_recovery — penalize short reverse-ODE failure to reach data.

Motivation (Kim et al., "Consistency Trajectory Models," ICLR 2024,
arXiv 2310.02279; Kong et al., "ACT-Diffusion," CVPR 2024): off-manifold-
ness can be measured directly by running a short reverse ODE from a
perturbed state and comparing to the clean sample — no hinge, no margin,
a continuous and self-calibrating signal.

Concretely, for each training batch:
  1. Draw t ~ U(eps, 1-eps), form x_t via the FM forward path with x_1 = clean.
  2. Perturb: x_t^- = x_t + delta, delta bounded in an epsilon L2 ball
     (mild random perturbation; no PGD).
  3. Roll the learned velocity forward from t to 1 for k Euler steps to
     obtain x_hat_from_neg.
  4. Loss = || x_hat_from_neg - x_clean ||^2.

This is the *ODE-reconstruction* loss: if the velocity field is wrong
off-manifold, the reconstructed endpoint won't match the data sample.
Gradient flows through all k ODE steps, so the model is trained to have
a well-behaved field *off* the training distribution — directly what
DG-ANM wants, without a margin or mined adversary.
"""

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_ct, eqm_loss, train_loop


def _short_reverse_ode(model, x_start, t_start, num_steps, a, gain):
    """Roll learned velocity from t_start to 1 in num_steps Euler steps."""
    B = x_start.shape[0]
    x = x_start
    # dt per sample (each has its own starting t).
    remaining = (1.0 - t_start).view(B, 1, 1, 1)
    dt = remaining / max(num_steps, 1)
    for i in range(num_steps):
        frac = (i + 0.5) / max(num_steps, 1)
        t_i = t_start + (1.0 - t_start) * frac
        t_model = (t_i * 999.0).clamp_min(0.0)
        pred = model(x, t_model)
        ct = eqm_ct(t_i, a=a, gain=gain).view(B, 1, 1, 1).clamp_min(1e-3)
        v = pred / ct
        x = x + v * dt
    return x


def step_fn(model, x, step, device, args: TrainArgs):
    e = args.extras or {}
    base = eqm_loss(model, x, device, eps=args.train_eps, a=args.a, gain=args.gain)
    total = base
    rec_val = 0.0

    gamma = e.get("gamma", 0.3)
    mine_every = e.get("mine_every", 5)
    num_ode_steps = e.get("ode_steps", 2)
    if gamma > 0 and mine_every > 0 and (step % mine_every == 0):
        B = x.size(0)
        # Sample modest t so the ODE rollout distance is short (<= 0.5).
        t = torch.rand(B, device=device) * 0.5 + 0.5 - args.train_eps
        t_ = t.view(B, 1, 1, 1)
        x0 = torch.randn_like(x)
        x_t = (1.0 - t_) * x0 + t_ * x
        epsilon = e.get("perturb_eps", 0.2)
        delta = torch.randn_like(x_t)
        flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
        delta = delta * (epsilon / (flat + 1e-8))
        x_neg = x_t + delta

        x_hat = _short_reverse_ode(
            model, x_neg, t, num_ode_steps,
            a=args.a, gain=args.gain,
        )
        rec = F.mse_loss(x_hat, x.detach())
        total = base + gamma * rec
        rec_val = rec.item()

    return total, {"base": base.item(), "rec": rec_val}


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "rec"])
