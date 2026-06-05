"""v13_stable_se2_aug — CIFAR-appropriate frozen-anchor stable-generator augmentation for EqM.

Architecture fix over v12 (which used a 2x2 rotation/scale/shear generator with NO translation and gave
no CIFAR-FID benefit): v13 uses an SE(2)-style 3x3 HOMOGENEOUS affine generator that CAN express
translation/crop/scale — the CIFAR-relevant nuisance family (proxy-validated, results_cifar_se2_proxy.json).

  Stage 1 (offline): discover frozen stable SE(2) operator M=matrix_exp(A) (A 3x3, top-2 rows learned)
    vs a frozen grad-flowing random-conv feature anchor, MIXTURE objective + stability + broad move hinge.
    NOT through live EqM closure.
  Stage 2: FREEZE the generator A; train EqM with GROUP/orbit augmentation:
       sample t~U(-1,1); M_t = matrix_exp(t*A); L = eqm_loss(model,x) + lam*eqm_loss(model, warp(x,M_t).detach())

extras knobs:
  operator_mode: "discovered" (default) | "random" | "identity"
  aug_mode: "orbit" (default, t~U(-1,1)) | "single" (t=1) | "bidir" (t=+/-1)
  lam_aug (0.3), discover_steps (600), move_floor_px (2.0), move_cap_px (10.0)
Operator diagnostics -> <output_dir>/operator_diag.json (trust these over FID per the coverage confound).
"""
from __future__ import annotations
import json
import torch
from pathlib import Path

from ._common import TrainArgs, eqm_loss, train_loop, build_cifar_loader
from ._stable_operator_se2 import discover_stable_se2, affine_warp3, _build_M

_FROZEN: dict = {}


def step_fn(model, x1, step, device, args: TrainArgs):
    base = eqm_loss(model, x1, device, eps=args.train_eps, a=args.a, gain=args.gain)
    A2 = _FROZEN.get("A2"); lam = _FROZEN.get("lam", 0.3)
    if A2 is None or lam <= 0:
        return base, {"base": base.item(), "aug": 0.0, "ratio": 0.0}
    aug_mode = _FROZEN.get("aug_mode", "orbit")
    with torch.no_grad():
        A2 = A2.to(device)
        if aug_mode == "single":
            t = 1.0
        elif aug_mode == "bidir":
            t = 1.0 if (torch.rand((), device=device) < 0.5) else -1.0
        else:
            t = (torch.rand((), device=device) * 2.0 - 1.0).item()
        M_t = _build_M(t * A2, device)
        x_aug = affine_warp3(x1, M_t)
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
        A2 = torch.zeros(2, 3, device=device); diag = {"mode": "identity"}
    elif mode == "random":
        torch.manual_seed(args.seed + 777)
        A2 = torch.zeros(2, 3, device=device)
        A2[:, :2] = 0.15 * torch.randn(2, 2, device=device); A2[:, 2] = 0.15 * torch.randn(2, device=device)
        M = _build_M(A2, device)
        diag = {"mode": "random", "tx_px": float(M[0, 2] / (2.0/32.0)), "det": float(torch.det(M[:2, :2]))}
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
        M, diag = discover_stable_se2(
            real_batch_fn, device,
            steps=e.get("discover_steps", 600), batch=min(args.batch_size, 128),
            lam_move=e.get("lam_move", 0.5), lam_det=e.get("lam_det", 1.0),
            lam_cond=e.get("lam_cond", 1.0), move_floor_px=e.get("move_floor_px", 2.0),
            move_cap_px=e.get("move_cap_px", 10.0), init_seed=args.seed)
        diag["mode"] = "discovered"
        A2 = torch.tensor(diag["A_gen"], device=device)

    _FROZEN["A2"] = A2.detach(); _FROZEN["lam"] = lam; _FROZEN["aug_mode"] = e.get("aug_mode", "orbit")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "operator_diag.json").write_text(json.dumps(diag, indent=2))
    print(f"[v13] SE2 operator (mode={mode}) lam_aug={lam} aug_mode={_FROZEN['aug_mode']}: "
          f"tx={diag.get('tx_px','-')} ty={diag.get('ty_px','-')} det={diag.get('det','-')} "
          f"cond={diag.get('cond','-')} lin_off={diag.get('lin_off_identity','-')} "
          f"anchor_base/final={diag.get('anchor_baseline_real_real','-')}/{diag.get('anchor_final_T_real','-')}", flush=True)
    return train_loop(args, step_fn, diag_keys=["base", "aug", "ratio"])
