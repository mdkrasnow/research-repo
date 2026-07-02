"""v20_metacog_mine — metacognition-guided hard-example mining for EqM image gen.

The image-gen form of the v11 "metacognitive curriculum" idea (metacog_curriculum_proposal.md).
Image gen is unconditional, so there is no per-example "solve to fail" like Sudoku; instead we
mine the examples where the FIELD IS LOCALLY UNSTABLE — the same descent-instability signal the
trajectory-metacognition probe reads at inference, applied here at training time on REAL data.

Per training step: build the standard EqM path (x_t, target), take the per-example base loss,
and UPWEIGHT examples with high field-instability:
    s_i = || f(x_t + ξ_i) - f(x_t) ||         (ξ small random; one extra no-grad forward)
    w_i = 1 + λ · ŝ_i                          (ŝ = rank-normalized s to [0,1])
    L   = mean_i ( w_i · ||f(x_t_i) - target_i||² )

EqM-compatible / LOW-RISK: it is the BASE EqM loss, only reweighted (CLAUDE.md preferred class) —
no new loss geometry, bounded by the base loss. Distinct from v10 (PGD synthetic perturbations):
v20 reweights REAL examples by instability.

Controls via extras["mine_selector"]:
  "instability" (treatment), "random" (negative control: random weights, same λ),
  "loss" (baseline: weight by per-example base loss).

extras: lambda_v20 (default 0.5), xi_eps (default 0.05), mine_selector (default "instability").
"""
import torch

from ._common import TrainArgs, eqm_ct, train_loop


def _rank_norm(s: torch.Tensor) -> torch.Tensor:
    """rank-normalize to [0,1] (robust to scale/outliers)."""
    order = s.argsort()
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(s.numel(), device=s.device, dtype=s.dtype)
    return ranks / max(s.numel() - 1, 1)


def step_fn(model, x1, step, device, args: TrainArgs):
    e = args.extras or {}
    B = x1.size(0)
    eps_train = args.train_eps if args.train_eps is not None else 1e-3
    a = args.a if args.a is not None else 0.8
    gain = args.gain if args.gain is not None else 4.0
    lam = e.get("lambda_v20", 0.5)
    xi_eps = e.get("xi_eps", 0.05)
    selector = e.get("mine_selector", "instability")

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps_train) + eps_train
    t_ = t.view(B, 1, 1, 1)
    x_t = (1.0 - t_) * x0 + t_ * x1
    ut = x1 - x0
    ct = eqm_ct(t, a=a, gain=gain).view(B, 1, 1, 1)
    target = ct * ut
    t_model = (t * 999.0).clamp_min(0.0)

    pred = model(x_t, t_model)
    pe = ((pred - target) ** 2).mean(dim=[1, 2, 3])           # per-example base loss (B,)

    # ---- per-example hardness score ----
    if selector == "instability":
        with torch.no_grad():
            xi = torch.randn_like(x_t) * xi_eps
            df = model(x_t + xi, t_model) - pred.detach()
            s = df.flatten(1).norm(dim=1)                     # field sensitivity (B,)
        score = _rank_norm(s)
    elif selector == "loss":
        score = _rank_norm(pe.detach())
    elif selector == "random":
        score = torch.rand(B, device=device)
    else:
        score = torch.zeros(B, device=device)

    w = 1.0 + lam * score                                     # (B,) in [1, 1+λ]
    loss = (w * pe).mean()

    return loss, {
        "base": pe.mean().item(),
        "weighted": loss.item(),
        "ratio": loss.item() / max(pe.mean().item(), 1e-8),
        "score_mean": score.mean().item(),
    }


def train(args: TrainArgs) -> float:
    return train_loop(args, step_fn, diag_keys=["base", "weighted", "ratio", "score_mean"])
