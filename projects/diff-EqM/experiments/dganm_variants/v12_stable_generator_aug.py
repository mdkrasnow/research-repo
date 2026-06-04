"""v12_stable_generator_aug — frozen-anchor stable-generator augmentation for EqM (the bridge).

Toy-ladder recipe (rungs 9-14) ported to real CIFAR EqM:
  Stage 1 (offline, before EqM): discover a frozen stable affine operator M = matrix_exp(A) against a
    FROZEN random-conv feature anchor (energy distance to real images) + stability reg (det~1, cond->1)
    + non-identity move. NOT through live EqM closure.
  Stage 2: FREEZE M; train EqM with augmentation:
       L = eqm_loss(model, x) + lam_aug * eqm_loss(model, affine_warp(x, M).detach())

extras knobs:
  operator_mode: "discovered" (default) | "random" (negative control) | "identity"
  lam_aug (default 0.3), discover_steps (default 600), ref_deg (15)
Operator diagnostics are saved to <output_dir>/operator_diag.json — trust these over FID alone
(per the toy coverage/coherence confound).

Arms for the bridge comparison: v00_vanilla (BASE), v10_hard_example (HARDNEG, negative),
vK_known_aug (KNOWN_AUG, positive), v12 here (discovered=treatment, random=negative control).
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import torch

from ._common import TrainArgs, eqm_loss, train_loop, build_cifar_loader
from ._stable_operator import discover_stable_affine, affine_warp

_FROZEN: dict = {}


def step_fn(model, x1, step, device, args: TrainArgs):
    base = eqm_loss(model, x1, device, eps=args.train_eps, a=args.a, gain=args.gain)
    A = _FROZEN.get("A")
    lam = _FROZEN.get("lam", 0.3)
    if A is None or lam <= 0:
        return base, {"base": base.item(), "aug": 0.0, "ratio": 0.0}
    # GROUP/orbit augmentation: sample t~U(-1,1), M_t = exp(t*A). A learned generator defines a one-parameter
    # GROUP, not one oriented map; augmenting with the whole orbit is the principled use (a continuous
    # symmetry family), and avoids baking in an arbitrary single orientation. (proxy-validated 2026-06-04)
    aug_mode = _FROZEN.get("aug_mode", "orbit")
    with torch.no_grad():
        A = A.to(device)
        if aug_mode == "single":
            t = 1.0
        elif aug_mode == "bidir":
            t = 1.0 if (torch.rand((), device=device) < 0.5) else -1.0
        else:  # orbit
            t = (torch.rand((), device=device) * 2.0 - 1.0).item()
        M_t = torch.matrix_exp(t * A)
        x_aug = affine_warp(x1, M_t)
    aug = eqm_loss(model, x_aug.detach(), device, eps=args.train_eps, a=args.a, gain=args.gain)
    total = base + lam * aug
    return total, {"base": base.item(), "aug": aug.item(),
                   "ratio": aug.item() / max(base.item(), 1e-8)}


def train(args: TrainArgs) -> float:
    e = args.extras or {}
    mode = e.get("operator_mode", "discovered")
    lam = e.get("lam_aug", 0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "identity":
        A_gen = torch.zeros(2, 2, device=device); diag = {"mode": "identity"}
    elif mode == "random":
        torch.manual_seed(args.seed + 777)
        A_gen = 0.4 * torch.randn(2, 2, device=device); M = torch.matrix_exp(A_gen)
        diag = {"mode": "random", "angle_deg": float(torch.atan2(M[1, 0], M[0, 0]) * 180 / math.pi),
                "det": float(torch.det(M))}
    else:  # discovered
        loader = build_cifar_loader(args.data_dir, args.batch_size, args.num_workers)
        it = iter(loader)
        def real_batch_fn(n):
            nonlocal it
            try:
                xb, _ = next(it)
            except StopIteration:
                it = iter(loader); xb, _ = next(it)
            return xb[:n]
        M, diag = discover_stable_affine(
            real_batch_fn, device,
            steps=e.get("discover_steps", 600), batch=min(args.batch_size, 128),
            lam_move=e.get("lam_move", 0.5), lam_det=e.get("lam_det", 1.0),
            lam_cond=e.get("lam_cond", 1.0), ref_deg=e.get("ref_deg", 15.0),
            init_seed=args.seed)
        diag["mode"] = "discovered"
        A_gen = torch.tensor(diag["A_gen"], device=device)

    _FROZEN["A"] = A_gen.detach(); _FROZEN["lam"] = lam; _FROZEN["aug_mode"] = e.get("aug_mode", "orbit")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "operator_diag.json").write_text(json.dumps(diag, indent=2))
    print(f"[v12] operator discovered (mode={mode}) lam_aug={lam}: "
          f"angle={diag.get('angle_deg','-')} det={diag.get('det','-')} "
          f"cond={diag.get('cond','-')} off_id={diag.get('off_identity','-')} "
          f"anchor_base/final={diag.get('anchor_baseline_real_real','-')}/{diag.get('anchor_final_T_real','-')} "
          f"feat_shift_consistency={diag.get('feature_shift_consistency','-')}", flush=True)
    return train_loop(args, step_fn, diag_keys=["base", "aug", "ratio"])
