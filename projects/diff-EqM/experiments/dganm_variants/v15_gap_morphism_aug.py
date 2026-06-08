"""v15_gap_morphism_aug — CONSTRUCTED-GAP morphism-discovery bridge for EqM.

The v14 bridge trained EqM on FULL CIFAR and discovered against FULL CIFAR (no visible->anchor gap), so
the useful symmetry (crop) was already known and discovery had no headroom — known-crop won. The v17 CIFAR
gym (symmetry-discovery, 2026-06-08, gap_aware reward, 3 seeds) showed discovery REOPENS when there is a
REAL visible->anchor gap whose closing symmetry is NOT the generic one. This variant builds that gap on
EqM itself:

    TRAIN distribution = DESATURATED CIFAR (chroma x keep, default 0.25)  <- the "visible" gap split
    FID reference      = FULL CIFAR (unchanged)                          <- the "anchor"/target

So a model trained on desaturated data generates desaturated samples (high FID vs full CIFAR). The missing
factor is COLOR. The augmentation arm decides which symmetry is added on top of the desaturated base:

    L = eqm_loss(model, desat(x)) + lam_aug * eqm_loss(model, AUG(desat(x)).detach())

operator_mode:
  "discovered" — frozen gap_aware multi-family policy discovered offline on the desaturated->full gap
                 (expected to learn `saturate`; the RIGHT missing factor -> should LOWER FID most).
  "known"      — KNOWN generic CIFAR symmetry = random crop (translate). The WRONG symmetry for a color
                 gap (crop adds no color) -> should NOT close the FID gap. Negative control that pins the
                 thesis: discovery beats known-aug precisely where the useful symmetry is unknown.
  "random"     — uniform over valid families (no discovery).
  "base"       — no aug (lower bound: pure desaturated training).

Pre-registered gate (constructed-gap, human-approved 2026-06-08): discovered FID < known FID AND
discovered FID < random FID AND discovered FID < base FID, with discovery decoy_usage ~ 0. This is the
INVERSE of the full-CIFAR bridge prediction (there known-crop won) and is the discriminating experiment.

extras: keep (0.25), lam_aug (0.3), discover_steps (300), anchor_n (1536), gap_aware (true),
        use_ae (true), ae_weight (5.0 when gap_aware), crop_pad (4).
Diagnostics -> <output_dir>/operator_diag.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ._common import TrainArgs, eqm_loss, train_loop, build_cifar_loader
from . import _multi_morphism as MM

_S: dict = {}


def _desat(x, keep):
    gray = x.mean(1, keepdim=True)
    return (gray + keep * (x - gray)).clamp(-1, 1)


def _rand_crop_pad(x, pad):
    H, W = x.shape[-2], x.shape[-1]
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    dx = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    dy = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    return xp[:, :, dy:dy + H, dx:dx + W]


def step_fn(model, x1, step, device, args: TrainArgs):
    keep = _S.get("keep", 0.25)
    xv = _desat(x1, keep)  # train on the desaturated (visible) gap split
    base = eqm_loss(model, xv, device, eps=args.train_eps, a=args.a, gain=args.gain)
    mode = _S.get("mode", "discovered"); lam = _S.get("lam", 0.3); pol = _S.get("policy")
    if lam <= 0 or mode == "base":
        return base, {"base": base.item(), "aug": 0.0, "ratio": 0.0}
    with torch.no_grad():
        if mode == "known":
            x_aug = _rand_crop_pad(xv, _S.get("crop_pad", 4))
        else:
            x_aug = pol.sample_transform(xv)
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
    keep = float(e.get("keep", 0.25)); _S["keep"] = keep
    _S["lam"] = e.get("lam_aug", 0.3); _S["mode"] = mode
    _S["crop_pad"] = int(e.get("crop_pad", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diag = {"mode": mode, "keep": keep, "lam_aug": _S["lam"]}

    if mode in ("base", "known"):
        _S["policy"] = None
        print(f"[v15] mode={mode} keep={keep} lam={_S['lam']} (no discovery)", flush=True)
    else:
        steps = int(e.get("discover_steps", 300))
        dseed = int(e.get("discover_seed", args.seed))
        anchor_n = int(e.get("anchor_n", 1536))
        gap_aware = bool(e.get("gap_aware", True))
        real = _grab_real(args, anchor_n, device)           # ANCHOR = full (color) CIFAR
        visible = _desat(real, keep)                         # VISIBLE = desaturated gap split
        scorer = MM.AnchorScorer(real, seed=777)
        aw = float(e.get("ae_weight", 5.0 if gap_aware else 50.0))
        ae = MM.train_robust_ae(real, steps=int(e.get("ae_steps", 1200)), seed=dseed) \
            if bool(e.get("use_ae", True)) else None
        if mode == "discovered":
            restarts = int(e.get("discover_restarts", 3))
            best = None
            for r in range(restarts):
                cand = MM.MorphismPolicy(MM.ALL_FAMILIES, depth=1).to(device)
                cd = MM.discover(cand, visible, scorer, steps=steps, seed=dseed + 101 * r,
                                 ae=ae, ae_weight=aw, gap_aware=gap_aware)
                print(f"[v15] discover restart {r}: decoy_usage={cd['decoy_usage']:.3f} "
                      f"top={sorted(cd['family_weights'].items(), key=lambda x:-x[1])[:3]}", flush=True)
                if best is None or cd["decoy_usage"] < best[1]["decoy_usage"]:
                    best = (cand, cd)
            pol, d = best
            diag.update(d); diag["discover_restarts"] = restarts; diag["gap_aware"] = gap_aware
            print(f"[v15] SELECTED decoy_usage={d['decoy_usage']:.3f} "
                  f"weights={ {k: round(v,3) for k,v in sorted(d['family_weights'].items(), key=lambda x:-x[1])[:5]} }", flush=True)
        else:  # random negative control
            pol = MM.MorphismPolicy(MM.ALL_FAMILIES, depth=1).to(device)
            diag["family_weights"] = {f: 1.0 / pol.K for f in pol.families}
            diag["decoy_usage"] = len(MM.DECOY_FAMILIES) / pol.K
            print("[v15] RANDOM control (uniform families, no discovery)", flush=True)
        for p in pol.parameters():
            p.requires_grad_(False)
        _S["policy"] = pol
        del scorer, real, visible
        if ae is not None:
            del ae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "operator_diag.json").write_text(json.dumps(diag, indent=2))
    return train_loop(args, step_fn, diag_keys=["base", "aug", "ratio"])
