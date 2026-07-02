"""v16_hybrid_mine_morph — v10 ANM hard-example mining + v17 discovered morphism augmentation.

The complementarity arm of the full-CIFAR v10 comparison ladder (documentation/v16_v10_comparison_ladder.md).
v10 mines an adversarial hard example in the NOISED (x_t) space to harden off-trajectory field robustness;
v17 augments with a FROZEN discovered on-manifold morphism policy in the DATA (x1) space. They act on
different axes, so they should compose:

    L = L_base(x_t) + lam_v10 * L_base(x_t + delta*)            # v10 mining (Madry PGA on EqM target)
                    + lam_aug * L_base(forward(policy(x1)))      # v17 discovered morphism aug

operator_mode: "discovered" (frozen v14-style policy) | "random" (uniform) | "none" (pure v10).

The claim this tests: discovery need not DETHRONE the hand-designed lever (v10) on its home turf; it can ADD
lift the lever lacks. Pre-registered best-publishable success = v16 (v10+disc) FID < v10-alone FID, same
protocol/seeds.

extras: lambda_v10 (0.1), mining_K (1), eps_radius (0.3), mining_lr (0.05), mine_every (1),
        lam_aug (0.3), operator_mode, discover_steps (400), anchor_n (1536), visible_zoom (1.5),
        use_ae (true), ae_weight (50.0), ae_steps (1500), discover_restarts (3).
Discovery diagnostics -> <output_dir>/operator_diag.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from ._common import TrainArgs, eqm_ct, eqm_loss, train_loop, build_cifar_loader
    from . import _multi_morphism as MM
except ImportError:  # top-level import (CPU ladder)
    from _common import TrainArgs, eqm_ct, eqm_loss, train_loop, build_cifar_loader
    import _multi_morphism as MM

_H: dict = {}


def _project_l2(delta, eps_radius):
    b = delta.size(0)
    flat = delta.flatten(1).norm(dim=1, keepdim=True).view(b, 1, 1, 1)
    return delta * torch.clamp(eps_radius / (flat + 1e-8), max=1.0)


def _mine_hard(model, x_t, t_model, target, K, eps_radius, lr):
    delta = _project_l2(torch.zeros_like(x_t).normal_(0.0, eps_radius / 2.0), eps_radius)
    for _ in range(K):
        delta = delta.detach().requires_grad_(True)
        loss_adv = ((model(x_t + delta, t_model) - target) ** 2).mean()
        g = torch.autograd.grad(loss_adv, delta)[0]
        with torch.no_grad():
            delta = _project_l2(delta + lr * g.sign(), eps_radius)
    return delta.detach()


def step_fn(model, x1, step, device, args: TrainArgs):
    e = args.extras or {}
    B = x1.size(0)
    eps_train = args.train_eps if args.train_eps is not None else 1e-3
    a = args.a if args.a is not None else 0.8
    gain = args.gain if args.gain is not None else 4.0

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps_train) + eps_train
    t_ = t.view(B, 1, 1, 1)
    x_t = (1.0 - t_) * x0 + t_ * x1
    ct = eqm_ct(t, a=a, gain=gain).view(B, 1, 1, 1)
    target = ct * (x1 - x0)
    t_model = (t * 999.0).clamp_min(0.0)

    loss_base = F.mse_loss(model(x_t, t_model), target)
    total = loss_base
    diag = {"base": loss_base.item(), "hard": 0.0, "aug": 0.0, "ratio": 0.0, "delta_norm": 0.0}

    # v10 mining term
    lam_v10 = e.get("lambda_v10", 0.1)
    if lam_v10 > 0 and (step % e.get("mine_every", 1) == 0):
        delta = _mine_hard(model, x_t, t_model, target, K=e.get("mining_K", 1),
                           eps_radius=e.get("eps_radius", 0.3), lr=e.get("mining_lr", 0.05))
        loss_hard = F.mse_loss(model(x_t + delta, t_model), target)
        total = total + lam_v10 * loss_hard
        diag["hard"] = loss_hard.item()
        diag["delta_norm"] = delta.flatten(1).norm(dim=1).mean().item()

    # v17 discovered-morphism aug term (data-space; fresh EqM forward via _common.eqm_loss)
    pol = _H.get("policy"); lam_aug = _H.get("lam_aug", 0.3)
    if pol is not None and lam_aug > 0:
        with torch.no_grad():
            x_aug = pol.sample_transform(x1)
        loss_aug = eqm_loss(model, x_aug.detach(), device, eps=args.train_eps, a=args.a, gain=args.gain)
        total = total + lam_aug * loss_aug
        diag["aug"] = loss_aug.item()

    diag["ratio"] = (diag["hard"] + diag["aug"]) / max(loss_base.item(), 1e-8)
    return total, diag


def _grab_real(args, n, device):
    loader = build_cifar_loader(args.data_dir, args.batch_size, args.num_workers)
    xs = []
    for x, _ in loader:
        xs.append(x)
        if sum(t.size(0) for t in xs) >= n:
            break
    return torch.cat(xs, 0)[:n].to(device)


def train(args: TrainArgs) -> float:
    e = args.extras or {}
    mode = e.get("operator_mode", "discovered")
    _H["lam_aug"] = e.get("lam_aug", 0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diag = {"mode": mode, "lam_aug": _H["lam_aug"], "lambda_v10": e.get("lambda_v10", 0.1)}

    if mode == "none":
        _H["policy"] = None
        print("[v16] mode=none (pure v10 mining, no morphism aug)", flush=True)
    else:
        steps = int(e.get("discover_steps", 400))
        dseed = int(e.get("discover_seed", args.seed))
        anchor_n = int(e.get("anchor_n", 1536))
        real = _grab_real(args, anchor_n, device)
        scorer = MM.AnchorScorer(real, seed=777)
        zoom = float(e.get("visible_zoom", 1.5))
        cc = int(round(32 / zoom)); off = (32 - cc) // 2
        visible = F.interpolate(real[:, :, off:off + cc, off:off + cc], size=32,
                                mode="bilinear", align_corners=False)
        ae = MM.train_robust_ae(real, steps=int(e.get("ae_steps", 1500)), seed=dseed) \
            if bool(e.get("use_ae", True)) else None
        aw = float(e.get("ae_weight", 50.0))
        if mode == "discovered":
            restarts = int(e.get("discover_restarts", 3))
            best = None
            for r in range(restarts):
                cand = MM.MorphismPolicy(MM.ALL_FAMILIES, depth=1).to(device)
                cd = MM.discover(cand, visible, scorer, steps=steps, seed=dseed + 101 * r,
                                 ae=ae, ae_weight=aw)
                print(f"[v16] discover restart {r}: decoy_usage={cd['decoy_usage']:.3f}", flush=True)
                if best is None or cd["decoy_usage"] < best[1]["decoy_usage"]:
                    best = (cand, cd)
            pol, d = best
            diag.update(d); diag["discover_restarts"] = restarts
            print(f"[v16] SELECTED decoy_usage={d['decoy_usage']:.3f}", flush=True)
        else:
            pol = MM.MorphismPolicy(MM.ALL_FAMILIES, depth=1).to(device)
            diag["family_weights"] = {f: 1.0 / pol.K for f in pol.families}
            diag["decoy_usage"] = len(MM.DECOY_FAMILIES) / pol.K
            print("[v16] RANDOM control", flush=True)
        for p in pol.parameters():
            p.requires_grad_(False)
        _H["policy"] = pol
        del scorer, real, visible
        if ae is not None:
            del ae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "operator_diag.json").write_text(json.dumps(diag, indent=2))
    return train_loop(args, step_fn, diag_keys=["base", "hard", "aug", "ratio", "delta_norm"])
