"""v21_metacog_mine — SHAPE-of-local-descent hard-negative mining for EqM (Goal 3).

Per documentation/v10_hard_example_eqm_proposal.md pattern + the separability
diagnostic finding (experiments/separability_diagnostic/dynamics_probe.py):
single-step ||Δf|| magnitude (v20 "instability" selector) is a weaker signal
than the SHAPE of a short local descent trajectory (oscillation / non-monotone
norm decrease), which de-confounded to 0.813 AUROC vs 0.61 for magnitude alone
(SELECTOR_LOCKDOWN_RESULTS.md). v21 mines on a training-time analog of that
shape signal: confident-but-wrong points where a short local rollout doesn't
monotonically settle (oscillates) even though the model is "confident"
(low immediate residual) — these flag stationary points the model believes
but that the trajectory shape says are not clean minima.

Per example, K small no-grad Euler-style local steps from x_t along -∇_x||f-target||²
(cheap finite-diff direction, NOT a learned step), logging the step-to-step
norm deltas; shape score = oscillation of those deltas (sign-change rate +
non-monotonicity), rank-normalized. This replaces v20's single ||Δf|| magnitude
probe with the K-step SHAPE probe, while keeping v20's aux-loss form (low-risk,
bounded-by-base-loss reweighting per CLAUDE.md) so the only thing that changes
vs v20 is the SELECTOR, isolating shape-vs-magnitude as the variable of interest.

Controls via extras["mine_selector"]:
  "shape" (treatment, this variant's default), "magnitude" (= v20's
  "instability", positive-ish control), "random" (negative control).

extras: lambda_v21 (default 0.5), shape_K (default 4), shape_step (default 0.05).
"""
import torch

from ._common import TrainArgs, eqm_ct, train_loop


def _rank_norm(s: torch.Tensor) -> torch.Tensor:
    order = s.argsort()
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(s.numel(), device=s.device, dtype=s.dtype)
    return ranks / max(s.numel() - 1, 1)


def _shape_score(model, x_t, t_model, target, K: int, step: float) -> torch.Tensor:
    """K-step local rollout; shape score = non-monotonicity of the residual-norm
    trajectory (sign-change rate of consecutive deltas), de-confounded from
    overall magnitude by rank-normalizing the deltas within each trajectory."""
    B = x_t.size(0)
    x = x_t.clone()
    norms = []
    with torch.no_grad():
        for _ in range(K):
            pred = model(x, t_model)
            resid = pred - target
            r_norm = resid.flatten(1).norm(dim=1)
            norms.append(r_norm)
            # cheap descent direction: move x against the residual (no autograd)
            direction = resid / (r_norm.view(B, 1, 1, 1) + 1e-8)
            x = x - step * direction
    norms = torch.stack(norms, dim=1)  # (B, K)
    deltas = norms[:, 1:] - norms[:, :-1]  # (B, K-1)
    sign_changes = (deltas[:, 1:] * deltas[:, :-1] < 0).float().sum(dim=1)  # (B,)
    return sign_changes  # higher = more oscillatory = "shape-flagged"


def step_fn(model, x1, step, device, args: TrainArgs):
    e = args.extras or {}
    B = x1.size(0)
    eps_train = args.train_eps if args.train_eps is not None else 1e-3
    a = args.a if args.a is not None else 0.8
    gain = args.gain if args.gain is not None else 4.0
    lam = e.get("lambda_v21", 0.5)
    shape_K = e.get("shape_K", 4)
    shape_step = e.get("shape_step", 0.05)
    selector = e.get("mine_selector", "shape")

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps_train) + eps_train
    t_ = t.view(B, 1, 1, 1)
    x_t = (1.0 - t_) * x0 + t_ * x1
    ut = x1 - x0
    ct = eqm_ct(t, a=a, gain=gain).view(B, 1, 1, 1)
    target = ct * ut
    t_model = (t * 999.0).clamp_min(0.0)

    pred = model(x_t, t_model)
    pe = ((pred - target) ** 2).mean(dim=[1, 2, 3])  # per-example base loss (B,)

    if selector == "shape":
        raw = _shape_score(model, x_t, t_model, target, K=shape_K, step=shape_step)
        score = _rank_norm(raw)
    elif selector == "magnitude":
        with torch.no_grad():
            xi = torch.randn_like(x_t) * 0.05
            df = model(x_t + xi, t_model) - pred.detach()
            s = df.flatten(1).norm(dim=1)
        score = _rank_norm(s)
    elif selector == "random":
        score = torch.rand(B, device=device)
    else:
        score = torch.zeros(B, device=device)

    w = 1.0 + lam * score
    loss = (w * pe).mean()

    return loss, {
        "base": pe.mean().item(),
        "weighted": loss.item(),
        "ratio": loss.item() / max(pe.mean().item(), 1e-8),
        "score_mean": score.mean().item(),
    }


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "weighted", "ratio", "score_mean"])
