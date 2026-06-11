"""v18_hpsm_morph — Hard-Positive Symmetry Mining augmentation for EqM.

    L = L_base(x) + lam_hpsm * L_eqm(T*(x)) + lam_consistency * commutator(x, T*(x))

T* = the hardest VALID named symmetry for the current field (hpsm_miner.mine), re-mined online (cheap, K
candidates) until `freeze_policy_after` epochs, after which the dominant family is frozen. The consistency
term teaches the field to ACCEPT the valid symmetry (equivariance). lam_hpsm=0 -> exactly base.

extras: lam_hpsm (0.3), lam_consistency (0.1), hpsm_k (8), hpsm_mode (loss_plus_comm),
        warmup_epochs (5), freeze_policy_after (0=never), anchor_n (1536), ae_steps (1500),
        validity_weight, decoy_weight, gap_reward (w_gap, default 2.0), mine_every (1).
Diagnostics -> <output_dir>/operator_diag.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch

try:
    from ._common import TrainArgs, eqm_loss, train_loop, build_cifar_loader
    from . import _multi_morphism as MM
    from . import hpsm_miner as HP
except ImportError:  # top-level import (CPU ladder)
    from _common import TrainArgs, eqm_loss, train_loop, build_cifar_loader
    import _multi_morphism as MM
    import hpsm_miner as HP

_H: dict = {}


def _grab_real(args, n, device):
    loader = build_cifar_loader(args.data_dir, args.batch_size, args.num_workers)
    xs = []
    for x, _ in loader:
        xs.append(x)
        if sum(t.size(0) for t in xs) >= n:
            break
    return torch.cat(xs, 0)[:n].to(device)


def step_fn(model, x1, step, device, args: TrainArgs):
    base = eqm_loss(model, x1, device, eps=args.train_eps, a=args.a, gain=args.gain)
    lam = _H.get("lam_hpsm", 0.3); lam_c = _H.get("lam_consistency", 0.1)
    spe = _H.get("steps_per_epoch", 390); warm = _H.get("warmup_epochs", 5)
    if lam <= 0 or step < warm * spe or _H.get("scorer") is None:
        return base, {"base": base.item(), "hpsm": 0.0, "cons": 0.0, "ratio": 0.0, "decoy": 0.0}

    frozen = _H.get("frozen_Tstar")
    if frozen is not None:
        fam, magv = frozen
    else:
        if step % _H.get("mine_every", 1) == 0:
            with torch.no_grad():
                T_star, diag = HP.mine(model, x1, _H["scorer"], _H["ae"], K=_H.get("hpsm_k", 8),
                                       mode=_H.get("hpsm_mode", "loss_plus_comm"),
                                       w_gap=_H.get("w_gap", 2.0), w_decoy=_H.get("decoy_weight", 10.0),
                                       seed=step)
            _H["last_Tstar"] = T_star; _H["last_decoy"] = diag["decoy_usage"]
            # freeze after N epochs: lock the dominant family seen so far
            fa = _H.setdefault("fam_tally", {})
            fa[T_star[0]] = fa.get(T_star[0], 0) + 1
            fap = _H.get("freeze_policy_after", 0)
            if fap and step >= fap * spe:
                _H["frozen_Tstar"] = (max(fa, key=fa.get), T_star[1])
        fam, magv = _H.get("last_Tstar", ("saturate", 0.6))

    mag = torch.full((x1.size(0),), float(magv), device=device)
    Tx = MM.apply_family(fam, x1, mag)
    hpsm = eqm_loss(model, Tx.detach(), device, eps=args.train_eps, a=args.a, gain=args.gain)
    cons = HP.commutator_consistency(model, x1, fam, mag) if lam_c > 0 else torch.zeros((), device=device)
    total = base + lam * hpsm + lam_c * cons
    return total, {"base": base.item(), "hpsm": hpsm.item(), "cons": float(cons),
                   "ratio": hpsm.item() / max(base.item(), 1e-8), "decoy": _H.get("last_decoy", 0.0)}


def train(args: TrainArgs) -> float:
    e = args.extras or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _H.clear()
    _H["lam_hpsm"] = e.get("lam_hpsm", 0.3)
    _H["lam_consistency"] = e.get("lam_consistency", 0.1)
    _H["hpsm_k"] = int(e.get("hpsm_k", 8))
    _H["hpsm_mode"] = e.get("hpsm_mode", "loss_plus_comm")
    _H["warmup_epochs"] = int(e.get("warmup_epochs", 5))
    _H["freeze_policy_after"] = int(e.get("freeze_policy_after", 0))
    _H["w_gap"] = float(e.get("gap_reward", 2.0))
    _H["decoy_weight"] = float(e.get("decoy_weight", 10.0))
    _H["mine_every"] = int(e.get("mine_every", 1))
    _H["steps_per_epoch"] = max(1, 50000 // args.batch_size)
    diag = {"variant": "v18_hpsm_morph", "lam_hpsm": _H["lam_hpsm"], "lam_consistency": _H["lam_consistency"],
            "hpsm_mode": _H["hpsm_mode"]}
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
    fid = train_loop(args, step_fn, diag_keys=["base", "hpsm", "cons", "ratio", "decoy"])
    diag["final_fam_tally"] = _H.get("fam_tally", {})
    diag["frozen_Tstar"] = _H.get("frozen_Tstar")
    (out / "operator_diag.json").write_text(json.dumps(diag, indent=2))
    return fid
