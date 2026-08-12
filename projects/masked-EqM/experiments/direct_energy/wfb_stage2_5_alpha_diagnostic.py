"""
WFB-EqM Stage 2.5: frozen-checkpoint alpha-family diagnostic (2026-08-12,
reviewer note responding to the Stage 2 v5/D factorial finding).

MOTIVATION. Stage 2's ARM D (--wfb-backward --reset-adam-state) matched
ARM A's (--exact-fwrev) delta_theta_norm almost exactly (0.240 vs 0.249,
resolving the earlier ~40x-smaller-step confound) yet produced NET WORSE
held-out probe loss (sum over 500 steps +0.0447 vs ARM A's -0.0245) and a
near-chance/slightly-negative update-direction cosine (44.2% positive vs
ARM A's 49.8% and ARM B's loaded-state 74.0%). This falsifies the simple
"stale Adam state -> WFB just needs equal-sized steps" hypothesis.

The reviewer's mathematical account: WFB (g_wfb = M^T (A+lambda I)^{-1/2} r)
fixes the PARAMETER gradient's conditioning (per-mode gain sigma_i, bounded
relative to the unbounded sigma_i^2 raw/direct gain) but the INDUCED FIELD
UPDATE delta_s = M delta_theta ~ -eta * sigma_i^2/(sigma_i^2+lambda)^{1/2} * r_i
still carries ONE power of sigma_i -- i.e. an ordinary-sized parameter step
can still produce an arbitrarily large field movement in a high-sigma mode.
The proposed fix is alpha=1 ("FBGN", full damped Gauss-Newton):
g_alpha = M^T (A+lambda I)^{-alpha} r, whose induced field gain is
sigma_i^2/(sigma_i^2+lambda) in [0,1] for every mode.

THIS SCRIPT (no training, no weight updates persisted) tests that account
DIRECTLY, on the same frozen instability-regime checkpoint Stage 1 used,
before committing to another training job:

  (1) Per real batch (spike + control, same selection as Stage 1), for
      alpha in {0, 0.5, 1}: compute g_alpha (compute_wfb_gradient's alpha
      kwarg, fb_direct/exact_hvp.py) and the induced field update
      q_alpha = M g_alpha (field_jvp_direct). Report ||q_alpha||/||r||
      (empirical field gain) and cos(q_alpha, -r) (alignment with the
      residual-reducing direction) per alpha -- the direct empirical
      analogue of the closed-form per-mode gain formula.
  (2) A REAL (non-Adam) one-step probe at eta=1 in NATIVE scale
      (theta' = theta - (2/(B*D)) * g_alpha, matching the same native-scale
      convention proven exactly in test_wfb_gradient_matches_native_fwrev_scale)
      on a held-out real-data probe batch (SAME construction as train.py's
      checklist probe): apply, measure actual field-MSE-vs-target change,
      then EXACTLY restore original parameters (float64-precision clone),
      for alpha in {0, 0.5, 1}. Confirms whether alpha=1's closed-form
      contraction prediction (r'_i ~ lambda/(sigma_i^2+lambda) r_i, i.e.
      guaranteed local improvement) survives the model's actual
      nonlinearity, vs alpha=0/0.5 which carry no such guarantee.

No GPU weight updates are persisted; frozen-checkpoint only, matching
Stage 1's zero-update causal-diagnostic design.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matched_replay_jacobian_diagnostic import (  # noqa: E402
    build_pool, load_model, probe_direct, rank_pool, select_spike_control,
)
from fb_direct.exact_hvp import compute_wfb_gradient, field_jvp_direct  # noqa: E402
from transport.utils import mean_flat  # noqa: E402

ALPHAS = (0.0, 0.5, 1.0)
ALPHA_NAME = {0.0: "direct", 0.5: "wfb", 1.0: "fbgn"}


def cosine(a, b):
    dot = float((a * b).sum())
    na, nb = float(a.norm()), float(b.norm())
    return dot / (na * nb + 1e-30)


def one_batch_alpha_sweep(model, xt, t, y, ut, rho, k, lambda_max_num_iters, seed):
    """(1) above: per-alpha g_alpha, induced field update q_alpha, gains."""
    row = {}
    for alpha in ALPHAS:
        model.zero_grad(set_to_none=True)
        t0 = time.time()
        result = compute_wfb_gradient(model, xt, t, y, ut, params=list(model.parameters()),
                                       rho=rho, k=k, lambda_max_num_iters=lambda_max_num_iters,
                                       seed=seed, alpha=alpha)
        model.zero_grad(set_to_none=True)
        q_alpha = field_jvp_direct(model, xt, t, y, result["params"], result["g_wfb"])
        r = result["r"]
        r_norm = result["r_norm"]
        field_gain = float(q_alpha.norm()) / (r_norm + 1e-30)
        cos_q_neg_r = cosine(q_alpha, -r)
        row[ALPHA_NAME[alpha]] = {
            "alpha": alpha,
            "g_alpha_norm": result["g_wfb_norm"],
            "field_update_norm": float(q_alpha.norm()),
            "field_gain_ratio_vs_r": field_gain,
            "cos_field_update_vs_neg_r": cos_q_neg_r,
            "lambda_max": result["lambda_max"], "lam": result["lam"],
            "breakdown": result["breakdown"], "breakdown_reason": result["breakdown_reason"],
            "wall_s": time.time() - t0,
        }
        model.zero_grad(set_to_none=True)
    row["r_norm"] = result["r_norm"]
    return row


@torch.no_grad()
def _clone_params(params):
    return [p.detach().clone() for p in params]


@torch.no_grad()
def _restore_params(params, saved):
    for p, s in zip(params, saved):
        p.copy_(s)


def probe_field_and_loss(model, probe_xt, probe_t, probe_y, probe_ut):
    """Held-out real-data probe field + MSE-to-target, model.eval(), no grad
    retained -- mirrors train.py's checklist probe_field()/probe_loss."""
    was_training = model.training
    model.eval()
    z = probe_xt.detach().clone().requires_grad_(True)
    E = model(z, probe_t, probe_y, energy_only=True)
    g = torch.autograd.grad(E.sum(), z, create_graph=False)[0].detach()
    field = -g
    model.train(was_training)
    loss = float(mean_flat((field - probe_ut) ** 2).mean())
    return field, loss


def one_step_probe_sweep(model, xt, t, y, ut, probe_xt, probe_t, probe_y, probe_ut,
                          rho, k, lambda_max_num_iters, seed, eta=1.0):
    """(2) above: real (non-Adam) one-step apply-then-revert at native scale
    eta=1, for alpha in {0, 0.5, 1}, measured on a SEPARATE held-out probe
    batch (not the batch g_alpha was computed from -- tests generalization
    of the step, not just fitting to its own batch)."""
    B_shape = xt.shape[0]
    D_shape = xt[0].numel()
    native_scale = 2.0 / (B_shape * D_shape)

    probe_field_before, probe_loss_before = probe_field_and_loss(model, probe_xt, probe_t, probe_y, probe_ut)

    out = {"native_scale": native_scale, "probe_loss_before": probe_loss_before}
    for alpha in ALPHAS:
        model.zero_grad(set_to_none=True)
        result = compute_wfb_gradient(model, xt, t, y, ut, params=list(model.parameters()),
                                       rho=rho, k=k, lambda_max_num_iters=lambda_max_num_iters,
                                       seed=seed, alpha=alpha)
        model.zero_grad(set_to_none=True)
        params = result["params"]
        saved = _clone_params(params)
        with torch.no_grad():
            for p, g in zip(params, result["g_wfb"]):
                p.add_(g, alpha=-eta * native_scale)
        probe_field_after, probe_loss_after = probe_field_and_loss(model, probe_xt, probe_t, probe_y, probe_ut)
        _restore_params(params, saved)
        # Sanity: restoration must be exact (float64 CPU / fp32 GPU copy_, no numerical drift).
        with torch.no_grad():
            max_restore_err = max(float((p - s).abs().max()) for p, s in zip(params, saved))
        step_norm_sq = sum(float((g.detach() * (eta * native_scale)).float().square().sum()) for g in result["g_wfb"])
        out[ALPHA_NAME[alpha]] = {
            "alpha": alpha,
            "step_norm": step_norm_sq ** 0.5,
            "probe_loss_after": probe_loss_after,
            "probe_delta_L": probe_loss_after - probe_loss_before,
            "max_restore_err": max_restore_err,
            "breakdown": result["breakdown"], "breakdown_reason": result["breakdown_reason"],
        }
        model.zero_grad(set_to_none=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-direct", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-size", type=int, default=256)
    p.add_argument("--spike-frac", type=float, default=0.06)
    p.add_argument("--num-control", type=int, default=8)
    p.add_argument("--sweep-n", type=int, default=8, help="how many spike + control batches get the full (1)+(2) sweep (expensive: 3 alphas x (compute_wfb_gradient + JVP) per batch)")
    p.add_argument("--rho", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--lambda-max-num-iters", type=int, default=20)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--vae", default="ema")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import types
    pool_args = types.SimpleNamespace(data_path=args.data_path, pool_size=args.pool_size, batch_size=args.batch_size,
                                       image_size=args.image_size, seed=args.seed, vae=args.vae)
    fixed_inputs = build_pool(pool_args, device)

    print("[stage2.5] loading direct checkpoint (instability regime)...")
    model = load_model(args.ckpt_direct, args.model, args.image_size, args.num_classes, "direct", device)

    print("[stage2.5] ranking pool via probe_direct (same ordinary criterion as Stage 1)...")
    pool_grad, pool_loss = rank_pool(probe_direct, model, fixed_inputs, "direct(late)")
    spike_idx, control_idx = select_spike_control(pool_grad, args.spike_frac, args.num_control)
    print(f"[stage2.5] pool={len(pool_grad)}, spike_n={len(spike_idx)}, control_n={len(control_idx)}")

    sweep_spike = spike_idx[:args.sweep_n]
    sweep_control = control_idx[:args.sweep_n]

    # --- (1) per-batch alpha sweep: field-gain + cosine, spike vs control ---
    rows_1 = {"spike": [], "control": []}
    for label, idx_list in (("spike", sweep_spike), ("control", sweep_control)):
        print(f"[stage2.5] === alpha field-gain sweep: {label} ({len(idx_list)} batches) ===")
        for n_done, i in enumerate(idx_list):
            xt, t, y, ut = fixed_inputs[i]
            row = one_batch_alpha_sweep(model, xt, t, y, ut, args.rho, args.k, args.lambda_max_num_iters, args.seed)
            row["batch_idx"] = i
            rows_1[label].append(row)
            print(f"  [{label} {n_done+1}/{len(idx_list)}] r_norm={row['r_norm']:.2f} " +
                  " | ".join(f"{name}: gain={row[name]['field_gain_ratio_vs_r']:.3f} "
                             f"cos={row[name]['cos_field_update_vs_neg_r']:.3f}" for name in ("direct", "wfb", "fbgn")))

    summary_1 = {}
    for label in ("spike", "control"):
        rr = rows_1[label]
        if not rr:
            continue
        summary_1[label] = {
            name: {
                "field_gain_median": statistics.median([r[name]["field_gain_ratio_vs_r"] for r in rr]),
                "cos_median": statistics.median([r[name]["cos_field_update_vs_neg_r"] for r in rr]),
                "g_alpha_norm_median": statistics.median([r[name]["g_alpha_norm"] for r in rr]),
            } for name in ("direct", "wfb", "fbgn")
        }

    # --- (2) real one-step apply/revert probe on a SEPARATE held-out batch ---
    probe_pool_idx = [i for i in range(len(fixed_inputs)) if i not in set(spike_idx) | set(control_idx)]
    if not probe_pool_idx:
        probe_pool_idx = list(range(len(fixed_inputs)))
    probe_xt, probe_t, probe_y, probe_ut = fixed_inputs[probe_pool_idx[0]]

    print(f"[stage2.5] === one-step apply/revert probe (eta={args.eta}, native scale) on "
          f"{min(args.sweep_n, len(sweep_spike)+len(sweep_control))} source batches, held-out probe batch idx={probe_pool_idx[0]} ===")
    rows_2 = []
    for label, idx_list in (("spike", sweep_spike), ("control", sweep_control)):
        for i in idx_list:
            xt, t, y, ut = fixed_inputs[i]
            out = one_step_probe_sweep(model, xt, t, y, ut, probe_xt, probe_t, probe_y, probe_ut,
                                        args.rho, args.k, args.lambda_max_num_iters, args.seed, eta=args.eta)
            out["source_batch_idx"] = i
            out["source_label"] = label
            rows_2.append(out)
            print(f"  [one-step {label} batch {i}] probe_loss_before={out['probe_loss_before']:.6f} " +
                  " | ".join(f"{name}: step_norm={out[name]['step_norm']:.4g} dL={out[name]['probe_delta_L']:+.6f}"
                             for name in ("direct", "wfb", "fbgn")))

    summary_2 = {
        name: {
            "delta_L_mean": statistics.mean([r[name]["probe_delta_L"] for r in rows_2]),
            "delta_L_frac_improved": sum(1 for r in rows_2 if r[name]["probe_delta_L"] < 0) / len(rows_2),
            "step_norm_median": statistics.median([r[name]["step_norm"] for r in rows_2]),
            "max_restore_err": max(r[name]["max_restore_err"] for r in rows_2),
        } for name in ("direct", "wfb", "fbgn")
    } if rows_2 else {}

    results = {
        "checkpoint": args.ckpt_direct, "rho": args.rho, "k": args.k, "eta": args.eta,
        "n_spike_swept": len(sweep_spike), "n_control_swept": len(sweep_control),
        "summary_1_field_gain_and_cosine": summary_1,
        "summary_2_one_step_probe": summary_2,
        "rows_1_field_gain_sweep": rows_1,
        "rows_2_one_step_probe": rows_2,
    }

    print("\n[stage2.5] SUMMARY:")
    print(json.dumps({k: v for k, v in results.items() if not k.startswith("rows_")}, indent=2,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("\n[stage2.5] FULL RESULTS:")
    print(json.dumps(results, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        except OSError as e:
            print(f"[stage2.5] WARNING: --out write failed ({e}); data is in the FULL RESULTS block above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
