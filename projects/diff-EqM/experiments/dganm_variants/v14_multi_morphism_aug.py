"""v14_multi_morphism_aug — multi-family discovered-morphism augmentation for EqM (the v17 bridge).

Replaces the old single-operator v12/v13 bridge with the v17-validated recipe (symmetry-discovery,
Phase 0-3 + MNIST/dSprites transfer): discover a FROZEN multi-family morphism policy offline against a
PCA-whitened random-conv energy-distance anchor over real CIFAR (EMA-bandit family selection over candidate
families INCLUDING decoys), then augment EqM:

    L = eqm_loss(model, x) + lam_aug * eqm_loss(model, frozen_policy(x).detach())

operator_mode: "discovered" (default) | "random" (negative control: untrained policy, uniform families) |
               "identity" (off).
extras: lam_aug (0.3), discover_steps (400), discover_seed (0), anchor_n (1536).

Discovery diagnostics (family weights, decoy_usage, rewards) -> <output_dir>/operator_diag.json. Trust the
diagnostics (decoy_usage ~ 0, valid families up) alongside FID, per the coverage/coherence confound noted
across the toy ladder.

Pre-registered CIFAR bridge gate (before any IN-1K): discovered FID < random FID AND discovered <= known-aug
FID AND decoy_usage ~ 0.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_loss, train_loop, build_cifar_loader
from . import _multi_morphism as MM

_FROZEN: dict = {}


def step_fn(model, x1, step, device, args: TrainArgs):
    base = eqm_loss(model, x1, device, eps=args.train_eps, a=args.a, gain=args.gain)
    pol = _FROZEN.get("policy"); lam = _FROZEN.get("lam", 0.3)
    if pol is None or lam <= 0:
        return base, {"base": base.item(), "aug": 0.0, "ratio": 0.0}
    with torch.no_grad():
        x_aug = pol.sample_transform(x1)
    aug = eqm_loss(model, x_aug.detach(), device, eps=args.train_eps, a=args.a, gain=args.gain)
    total = base + lam * aug
    return total, {"base": base.item(), "aug": aug.item(),
                   "ratio": aug.item() / max(base.item(), 1e-8)}


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
    lam = e.get("lam_aug", 0.3)
    steps = int(e.get("discover_steps", 400))
    dseed = int(e.get("discover_seed", args.seed))
    anchor_n = int(e.get("anchor_n", 1536))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diag = {"mode": mode, "lam_aug": lam}
    if mode == "identity" or lam <= 0:
        _FROZEN["policy"] = None
    else:
        real = _grab_real(args, anchor_n, device)
        # ANCHOR = full real CIFAR (broad position/scale). VISIBLE = a NARROWED view (center-zoom: objects
        # enlarged + centered -> reduced spatial/scale diversity). Discovery must find the morphism family
        # that bridges VISIBLE -> ANCHOR (scale/translate, the CIFAR-useful symmetries) and reject rotate/
        # decoys. Without this constructed visible->anchor shift, any on-manifold-moving op is equally
        # rewarded and discovery degenerates (e.g. picks rotate). Mirrors the v17 gym's visible/anchor gap.
        scorer = MM.AnchorScorer(real, seed=777)
        zoom = float(e.get("visible_zoom", 1.5))
        cc = int(round(32 / zoom))
        off = (32 - cc) // 2
        visible = F.interpolate(real[:, :, off:off + cc, off:off + cc], size=32,
                                mode="bilinear", align_corners=False)
        pol = MM.MorphismPolicy(MM.ALL_FAMILIES, depth=1).to(device)
        if mode == "discovered":
            d = MM.discover(pol, visible, scorer, steps=steps, seed=dseed)
            diag.update(d)
            print(f"[v14] discovered decoy_usage={d['decoy_usage']:.3f} "
                  f"weights={ {k: round(v,3) for k,v in d['family_weights'].items()} }", flush=True)
        else:  # random negative control: untrained uniform policy (no discovery)
            diag["family_weights"] = {f: 1.0 / pol.K for f in pol.families}
            diag["decoy_usage"] = len(MM.DECOY_FAMILIES) / pol.K
            print(f"[v14] RANDOM control (uniform families, no discovery)", flush=True)
        for p in pol.parameters():
            p.requires_grad_(False)
        _FROZEN["policy"] = pol
    _FROZEN["lam"] = lam

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "operator_diag.json").write_text(json.dumps(diag, indent=2))
    return train_loop(args, step_fn, diag_keys=["base", "aug", "ratio"])
