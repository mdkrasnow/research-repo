"""vK_known_aug — KNOWN-symmetry augmentation (positive control for the EqM bridge).

Augment EqM with a KNOWN image symmetry transform (default: small rotation; the loader already does
random horizontal flip for all variants). This is the positive control: if a known symmetry as
augmentation moves FID at all, the harness is capable of using operator augmentation — so a discovered
operator (v12) that helps is meaningful, and one that doesn't is a real negative.

    L = eqm_loss(model, x) + lam_aug * eqm_loss(model, T_known(x).detach())

extras: known_deg (default 15.0), lam_aug (default 0.3), known_mode ("rotate" | "hflip").
"""
from __future__ import annotations
import math
import torch

from ._common import TrainArgs, eqm_loss, train_loop
from ._stable_operator import affine_warp

_K: dict = {}


def step_fn(model, x1, step, device, args: TrainArgs):
    base = eqm_loss(model, x1, device, eps=args.train_eps, a=args.a, gain=args.gain)
    lam = _K.get("lam", 0.3); mode = _K.get("mode", "rotate"); M = _K.get("M")
    if lam <= 0:
        return base, {"base": base.item(), "aug": 0.0, "ratio": 0.0}
    with torch.no_grad():
        if mode == "hflip":
            x_aug = torch.flip(x1, dims=[3])
        else:
            x_aug = affine_warp(x1, M.to(device))
    aug = eqm_loss(model, x_aug.detach(), device, eps=args.train_eps, a=args.a, gain=args.gain)
    total = base + lam * aug
    return total, {"base": base.item(), "aug": aug.item(),
                   "ratio": aug.item() / max(base.item(), 1e-8)}


def train(args: TrainArgs) -> float:
    e = args.extras or {}
    deg = e.get("known_deg", 15.0)
    th = math.radians(deg)
    _K["M"] = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    _K["lam"] = e.get("lam_aug", 0.3); _K["mode"] = e.get("known_mode", "rotate")
    print(f"[vK_known_aug] mode={_K['mode']} deg={deg} lam_aug={_K['lam']}", flush=True)
    return train_loop(args, step_fn, diag_keys=["base", "aug", "ratio"])
