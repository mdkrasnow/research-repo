"""Experiment I driver: the full five-atom campaign, in pre-registered stages.

Stage order (each writes its own JSONL so a crash never loses earlier work):

  0  fd_calibration   eps ladder at init / early / mid / late training
  1  tc_sweep         tc in {0.5,0.7,0.8,0.9} on arms V and D, few seeds
                      -> FREEZE tc before the main comparison
  2  main             all arms x >=10 seeds at the frozen tc
  3  fd_grid          arms D,F x K in {1,4,8} x eps around the calibrated value
  4  gradnoise        parameter-gradient variance per arm at 4 training stages

Everything is driven from one process so shared seeds/initializations are
guaranteed identical across arms (§27: shared init, shared minibatches).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import torch

from .calibrate import DEFAULT_EPS_LADDER, calibrate_fd, pick_plateau
from .gradnoise import gradient_noise
from .interpolant import build_interpolant
from .models import build_model
from .objectives import FDConfig
from .toy5 import ToyConfig, evaluate_transport, make_batch, run, train
from .toy5 import weak_conservation_residual  # noqa: F401  (used via run)

ALL_ARMS = (
    "btm_vector",
    "btm_scalar_exact",
    "btm_scalar_action_exact",
    "btm_scalar_fd_directional",
    "btm_scalar_fd_action",
    "eqm_legacy_vector",
    "eqm_legacy_scalar",
)


def _append(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- stage 0
def stage_fd_calibration(base: ToyConfig, out, checkpoints=(0, 250, 1500, 6000),
                         n_probe=4096, K=4):
    """Calibrate eps at four points along a REAL scalar-arm training run."""
    device = torch.device(base.device)
    interp = build_interpolant("self_stopping", tc=base.tc)
    gen = torch.Generator(device=device).manual_seed(999)
    probe = make_batch(base, interp, n_probe, device, gen)

    cfg = replace(base, arm="btm_scalar_fd_directional")
    model = build_model(cfg.arm, d=2, width=cfg.width, depth=cfg.depth,
                        seed=cfg.seed).to(device)
    picked = {}
    prev = 0
    for ck in checkpoints:
        if ck > prev:
            sub = replace(cfg, steps=ck - prev)
            trained, _ = _continue_training(sub, model)
            model = trained
            prev = ck
        rows = calibrate_fd(model, probe["z"], K=K, eps_ladder=DEFAULT_EPS_LADDER,
                            seed=17)
        eps, info = pick_plateau(rows)
        picked[ck] = eps
        for r in rows:
            _append(out, {"stage": "fd_calibration", "train_step": ck, **r})
        _log(f"  calib @step {ck}: eps* = {eps:g} "
             f"(rel_rmse {info['chosen']['rel_rmse']:.3g}, "
             f"cancel {info['chosen']['cancel_ratio']:.3g})")
    return picked


def _continue_training(cfg: ToyConfig, model):
    """Train `model` in place for cfg.steps more steps (calibration helper)."""
    from .objectives import compute_loss
    from .fd import assert_no_double_backward
    from .objectives import FD_ARMS

    device = torch.device(cfg.device)
    interp = build_interpolant("self_stopping", tc=cfg.tc)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    fdcfg = FDConfig(eps_fd=cfg.eps_fd, K=cfg.K, fp32_subtract=cfg.fp32_subtract)
    gen = torch.Generator(device=device).manual_seed(10_000 + cfg.seed)
    for _ in range(cfg.steps):
        batch = make_batch(cfg, interp, cfg.batch, device, gen)
        loss = compute_loss(cfg.arm, model, batch, fdcfg, generator=gen)
        opt.zero_grad(set_to_none=True)
        if cfg.arm in FD_ARMS:
            with assert_no_double_backward():
                loss.backward()
        else:
            loss.backward()
        opt.step()
    return model, None


# ---------------------------------------------------------------- stage 1
def stage_tc_sweep(base: ToyConfig, out, tcs=(0.5, 0.7, 0.8, 0.9), seeds=(0, 1, 2)):
    best, table = None, {}
    for tc in tcs:
        maes = {}
        for arm in ("btm_vector", "btm_scalar_fd_directional"):
            vals = []
            for s in seeds:
                cfg = replace(base, arm=arm, tc=tc, seed=s)
                rec = run(cfg, verbose=False)
                rec["stage"] = "tc_sweep"
                _append(out, {k: v for k, v in rec.items() if k != "history"})
                vals.append(rec["mass_mae"])
            maes[arm] = sum(vals) / len(vals)
            _log(f"  tc={tc} {arm}: mean MAE {maes[arm]:.4f} "
                 f"({['%.4f' % v for v in vals]})")
        # robustness score: the worse of the two arms (we want a tc that is
        # good for BOTH the gold-standard vector arm and the proposed FD arm)
        table[tc] = max(maes.values())
    best = min(table, key=table.get)
    _log(f"  FROZEN tc = {best} (worst-arm MAE {table[best]:.4f}); table={table}")
    _append(out, {"stage": "tc_sweep_decision", "frozen_tc": best,
                  "worst_arm_mae": table})
    return best


# ---------------------------------------------------------------- stage 2
def stage_main(base: ToyConfig, out, tc, eps, n_seeds=10, arms=ALL_ARMS):
    for arm in arms:
        for s in range(n_seeds):
            cfg = replace(base, arm=arm, tc=tc, seed=s, eps_fd=eps)
            rec = run(cfg, verbose=False)
            rec["stage"] = "main"
            _append(out, {k: v for k, v in rec.items() if k != "history"})
            _log(f"  {arm} seed {s}: MAE {rec['mass_mae']:.4f} "
                 f"unres {rec['unresolved_frac']:.3f} "
                 f"R {rec['R_overall_median_rel']:.3f} "
                 f"stable={rec['stable']}")


# ---------------------------------------------------------------- stage 3
def stage_fd_grid(base: ToyConfig, out, tc, eps_center, seeds=(0, 1, 2, 3, 4)):
    ladder = sorted({eps_center / 3, eps_center, eps_center * 3})
    for arm in ("btm_scalar_fd_directional", "btm_scalar_fd_action"):
        for K in (1, 4, 8):
            for eps in ladder:
                for s in seeds:
                    cfg = replace(base, arm=arm, tc=tc, seed=s, K=K, eps_fd=eps)
                    rec = run(cfg, verbose=False)
                    rec["stage"] = "fd_grid"
                    _append(out, {k: v for k, v in rec.items() if k != "history"})
                _log(f"  {arm} K={K} eps={eps:g}: last MAE {rec['mass_mae']:.4f}")


# ---------------------------------------------------------------- stage 4
def stage_gradnoise(base: ToyConfig, out, tc, eps, checkpoints=(0, 250, 1500, 6000),
                    n_batches=200):
    device = torch.device(base.device)
    interp = build_interpolant("self_stopping", tc=tc)
    for arm in ("btm_scalar_exact", "btm_scalar_action_exact",
                "btm_scalar_fd_directional", "btm_scalar_fd_action"):
        for K in ((1, 4) if "fd" in arm else (1,)):
            cfg = replace(base, arm=arm, tc=tc, eps_fd=eps, K=K)
            model = build_model(arm, d=2, width=cfg.width, depth=cfg.depth,
                                seed=0).to(device)
            prev = 0
            for ck in checkpoints:
                if ck > prev:
                    model, _ = _continue_training(replace(cfg, steps=ck - prev),
                                                  model)
                    prev = ck
                gen = torch.Generator(device=device).manual_seed(4242)
                bf = lambda: make_batch(cfg, interp, cfg.batch, device, gen)
                st = gradient_noise(model, bf, arm,
                                    FDConfig(eps_fd=eps, K=K), n_batches, gen)
                st.update({"stage": "gradnoise", "train_step": ck, "K": K})
                _append(out, st)
                _log(f"  gradnoise {arm} K={K} @step {ck}: "
                     f"noise_scale {st['noise_scale']:.3g} "
                     f"cos {st['mean_pairwise_cosine']:.4f}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval-n", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--stages", default="0,1,2,3,4")
    ap.add_argument("--tc", type=float, default=None,
                    help="skip stage 1 and use this frozen tc")
    ap.add_argument("--eps-fd", type=float, default=None,
                    help="skip stage 0 and use this eps")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "toy_campaign.jsonl")
    stages = {s.strip() for s in args.stages.split(",")}

    base = ToyConfig(device=args.device, steps=args.steps, batch=args.batch,
                     eval_n=args.eval_n)
    _log(f"base config: {asdict(base)}")

    eps = args.eps_fd
    if "0" in stages:
        _log("STAGE 0: FD step-size calibration")
        picked = stage_fd_calibration(base, out)
        eps = picked[max(picked)]        # late-training plateau is the binding one
        _log(f"STAGE 0 -> eps_fd = {eps:g} (late-training plateau)")
    if eps is None:
        eps = base.eps_fd
    base = replace(base, eps_fd=eps)

    tc = args.tc
    if "1" in stages:
        _log("STAGE 1: tc sweep")
        tc = stage_tc_sweep(base, out)
    if tc is None:
        tc = base.tc

    if "2" in stages:
        _log(f"STAGE 2: main comparison, tc={tc}, eps={eps:g}, "
             f"{args.seeds} seeds x {len(ALL_ARMS)} arms")
        stage_main(base, out, tc, eps, n_seeds=args.seeds)
    if "3" in stages:
        _log("STAGE 3: FD (K, eps) grid")
        stage_fd_grid(base, out, tc, eps)
    if "4" in stages:
        _log("STAGE 4: gradient-noise diagnostic")
        stage_gradnoise(base, out, tc, eps)

    _log(f"DONE -> {out}")


if __name__ == "__main__":
    main()
