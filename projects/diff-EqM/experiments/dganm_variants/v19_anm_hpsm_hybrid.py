"""v19_anm_hpsm_hybrid — compose v10 ANM (hard-negative mining) with HPSM (hard-positive symmetry mining).

    L = L_base(x)
      + lam_anm * L_eqm(x_t + delta*)            # v10: hard NEGATIVE (reject invalid noised perturbation)
      + lam_hpsm * L_eqm(T*(x))                   # HPSM: hard POSITIVE (accept valid symmetry)
      + lam_consistency * commutator(x, T*(x))    # equivariance to the mined valid symmetry

Two adversaries on orthogonal axes: ANM hardens off-trajectory field robustness in NOISE space; HPSM
hardens equivariance to valid data symmetries in IMAGE space. The complementarity test (best-paper arm):
v19 < v10-alone. Reuses v10 mining via v16's helpers (no rewrite). lam_anm=0 -> pure HPSM; lam_hpsm=0 ->
pure v10.

extras: lam_anm (0.1), lam_hpsm (0.3), lam_consistency (0.1), v10 knobs (eps_radius, mining_lr, mining_K),
        hpsm_k (8), hpsm_mode, warmup_epochs (5), anchor_n, ae_steps, gap_reward, decoy_weight.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from ._common import TrainArgs, eqm_ct, eqm_loss, train_loop, build_cifar_loader
    from . import _multi_morphism as MM
    from . import hpsm_miner as HP
except ImportError:  # top-level import (CPU ladder)
    from _common import TrainArgs, eqm_ct, eqm_loss, train_loop, build_cifar_loader
    import _multi_morphism as MM
    import hpsm_miner as HP

_H: dict = {}


def _mine_delta(model, x_t, t_model, target, eps_radius=0.3, lr=0.05, K=1):
    """v10 ANM hard-negative: PGA on the EqM target in noise space (reused, not rewritten)."""
    d = torch.zeros_like(x_t).normal_(0, eps_radius / 2)
    n = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    d = d * (eps_radius / (n + 1e-8)).clamp(max=1.0)
    for _ in range(K):
        d = d.detach().requires_grad_(True)
        loss = ((model(x_t + d, t_model) - target) ** 2).mean()
        g = torch.autograd.grad(loss, d)[0]
        with torch.no_grad():
            d = d + lr * g.sign()
            n = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            d = d * (eps_radius / (n + 1e-8)).clamp(max=1.0)
    return d.detach()


def _grab_real(args, n, device):
    loader = build_cifar_loader(args.data_dir, args.batch_size, args.num_workers)
    xs = []
    for x, _ in loader:
        xs.append(x)
        if sum(t.size(0) for t in xs) >= n:
            break
    return torch.cat(xs, 0)[:n].to(device)


def step_fn(model, x1, step, device, args: TrainArgs):
    model._t_scale_999 = True  # real EqM UNet takes t in [0,999]; ensures commutator scales t right
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
    diag = {"base": loss_base.item(), "anm": 0.0, "hpsm": 0.0, "cons": 0.0,
            "anm_delta_norm": 0.0, "decoy": 0.0}

    lam_anm = _H.get("lam_anm", 0.1)
    if lam_anm > 0:
        delta = _mine_delta(model, x_t, t_model, target, K=_H.get("mining_K", 1),
                            eps_radius=_H.get("eps_radius", 0.3), lr=_H.get("mining_lr", 0.05))
        l_hard = F.mse_loss(model(x_t + delta, t_model), target)
        total = total + lam_anm * l_hard
        diag["anm"] = l_hard.item(); diag["anm_delta_norm"] = delta.flatten(1).norm(dim=1).mean().item()

    lam_h = _H.get("lam_hpsm", 0.3); lam_c = _H.get("lam_consistency", 0.1)
    spe = _H.get("steps_per_epoch", 390)
    if lam_h > 0 and step >= _H.get("warmup_epochs", 5) * spe and _H.get("scorer") is not None:
        with torch.no_grad():
            T_star, mdiag = HP.mine(model, x1, _H["scorer"], _H["ae"], K=_H.get("hpsm_k", 8),
                                    mode=_H.get("hpsm_mode", "loss_plus_comm"),
                                    w_gap=_H.get("w_gap", 2.0), w_decoy=_H.get("decoy_weight", 10.0), seed=step)
        fam, magv = T_star
        mag = torch.full((B,), float(magv), device=device)
        Tx = MM.apply_family(fam, x1, mag)
        l_hpsm = eqm_loss(model, Tx.detach(), device, eps=args.train_eps, a=args.a, gain=args.gain)
        total = total + lam_h * l_hpsm
        diag["hpsm"] = l_hpsm.item(); diag["decoy"] = mdiag["decoy_usage"]; diag["hpsm_family"] = fam
        if lam_c > 0:
            cons = HP.commutator_consistency(model, x1, fam, mag)
            total = total + lam_c * cons
            diag["cons"] = float(cons)

    diag["total"] = float(total)
    return total, diag


def train(args: TrainArgs) -> float:
    e = args.extras or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _H.clear()
    _H.update({"lam_anm": e.get("lam_anm", 0.1), "lam_hpsm": e.get("lam_hpsm", 0.3),
               "lam_consistency": e.get("lam_consistency", 0.1),
               "eps_radius": e.get("eps_radius", 0.3), "mining_lr": e.get("mining_lr", 0.05),
               "mining_K": int(e.get("mining_K", 1)), "hpsm_k": int(e.get("hpsm_k", 8)),
               "hpsm_mode": e.get("hpsm_mode", "loss_plus_comm"),
               "warmup_epochs": int(e.get("warmup_epochs", 5)),
               "w_gap": float(e.get("gap_reward", 2.0)), "decoy_weight": float(e.get("decoy_weight", 10.0)),
               "steps_per_epoch": max(1, 50000 // args.batch_size)})
    diag = {"variant": "v19_anm_hpsm_hybrid", **{k: _H[k] for k in ("lam_anm", "lam_hpsm", "lam_consistency")}}
    if _H["lam_hpsm"] > 0:
        real = _grab_real(args, int(e.get("anchor_n", 1536)), device)
        _H["scorer"] = MM.AnchorScorer(real, seed=777)
        _H["ae"] = MM.train_robust_ae(real, steps=int(e.get("ae_steps", 1500)), seed=args.seed) \
            if bool(e.get("use_ae", True)) else None
        del real
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        _H["scorer"] = None; _H["ae"] = None
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "operator_diag.json").write_text(json.dumps(diag, indent=2))
    return train_loop(args, step_fn,
                      diag_keys=["base", "anm", "hpsm", "cons", "anm_delta_norm", "decoy"])
