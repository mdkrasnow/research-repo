"""
WFB-EqM Stage 2.6a: FBGN Lanczos residual-certificate k-convergence check
(2026-08-12, reviewer derivation reconciling the >1 induced-field-gain
measurements in documentation/wfb-eqm-stage2-5-report-2026-08-12.md).

MOTIVATION. Stage 2.5 measured alpha=1 (FBGN) induced field gains
||q_alpha||/||r|| of 2.83 (spike median) / 3.41 (control median) at k=12 --
apparently violating the provable bound ||M g_1||/||r|| <= 1 (eigenvalues of
A(A+lambda I)^{-1} lie in [0,1)). The reviewer's reconciliation: this bound is
exact only for the CONVERGED (A+lambda I)^{-1} solve; a k-step Lanczos
truncation leaks a term ell_m into the (m+1)-th Krylov direction. By the
three-term recurrence A Q_m = Q_m T_m + beta_m q_{m+1} e_m^T, the TRUE
linear-system residual rho_m := r - (A+lambda I) u_m satisfies rho_m = -ell_m
EXACTLY, and consequently

    ||q_m|| / ||r||  <=  1 + ||rho_m|| / ||r||                    (*)

i.e. a measured gain > 1 is a DIAGNOSTIC of an unconverged shifted solve
(||rho_m||/||r|| large), not a violated theorem. This has been validated on a
synthetic operator in tests/test_fb_direct_exact_hvp.py
(test_fbgn_lanczos_residual_certificate: rho_m == -ell_m confirmed exactly,
certificate never violated, k=n recovers the plain <=1 bound).

THIS SCRIPT (no training, no weight updates persisted) checks the SAME
identities on the real model, on real spike + control batches, at
k in {12, 24, 48, 96}, to confirm the discrepancy really is finite-k
truncation on this ~32k-dim field space (not a hidden scale/reduction bug --
already ruled out by code inspection: mixed_gram_mv and this script's own
field_jvp_direct call share the exact same M/M^T primitives) and to show the
field-gain trend as k grows. Per the reviewer: do NOT let this experiment
choose a production k -- once the mechanism is confirmed, alpha=1 should move
to an adaptive-tolerance CG solve (Stage 2.6b), not a fixed larger k.
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
from fb_direct.exact_hvp import (  # noqa: E402
    _lanczos_shifted_solve_residual_certificate, estimate_lambda_max, mixed_gram_mv,
)

K_VALUES = (12, 24, 48, 96)


def one_batch_k_sweep(model, xt, t, y, ut, params, rho, lambda_max_num_iters, seed):
    """For a single (severe or control) batch: r = field - ut (canonical,
    unrescaled), lambda = rho * lambda_max(A) (same convention as
    compute_wfb_gradient), then for each k in K_VALUES run the residual
    certificate and record field_gain / rho_norm/r_norm / ell_norm/r_norm /
    certificate_bound / max identity error |rho - (-ell)|."""
    model.zero_grad(set_to_none=True)
    from fb_direct.exact_hvp import compute_field_direct
    field = compute_field_direct(model, xt, t, y)
    r = (field - ut).detach()
    r_norm = float(r.norm())

    lam_result = estimate_lambda_max(model, xt, t, y, params, num_iters=lambda_max_num_iters, seed=seed)
    lambda_max = lam_result["lambda_max"]
    lam = rho * lambda_max

    row = {"r_norm": r_norm, "lambda_max": lambda_max, "lam": lam, "per_k": {}}
    for k in K_VALUES:
        t0 = time.time()
        cert = _lanczos_shifted_solve_residual_certificate(
            lambda v: mixed_gram_mv(model, xt, t, y, params, v), r, lam, k=k)
        identity_err = float((cert["rho"] + cert["ell"]).norm())  # should be ~0: rho == -ell
        row["per_k"][k] = {
            "field_gain": cert["field_gain"],
            "rho_norm_over_r_norm": cert["rho_norm"] / (r_norm + 1e-30),
            "ell_norm_over_r_norm": cert["ell_norm"] / (r_norm + 1e-30),
            "certificate_bound": cert["certificate_bound"],
            "identity_err_rho_vs_neg_ell": identity_err,
            "m": cert["m"],
            "breakdown": cert["breakdown"], "breakdown_reason": cert["breakdown_reason"],
            "wall_s": time.time() - t0,
        }
        model.zero_grad(set_to_none=True)
    return row


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
    p.add_argument("--sweep-n", type=int, default=4, help="how many spike + control batches get the full k-sweep (expensive: len(K_VALUES) certificate solves per batch, each with its own k A-applies)")
    p.add_argument("--rho", type=float, default=1e-4)
    p.add_argument("--lambda-max-num-iters", type=int, default=20)
    p.add_argument("--vae", default="ema")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import types
    pool_args = types.SimpleNamespace(data_path=args.data_path, pool_size=args.pool_size, batch_size=args.batch_size,
                                       image_size=args.image_size, seed=args.seed, vae=args.vae)
    fixed_inputs = build_pool(pool_args, device)

    print("[stage2.6a] loading direct checkpoint (instability regime)...")
    model = load_model(args.ckpt_direct, args.model, args.image_size, args.num_classes, "direct", device)
    params = [pp for pp in model.parameters() if pp.requires_grad]

    print("[stage2.6a] ranking pool via probe_direct (same criterion as Stage 1/2.5)...")
    pool_grad, pool_loss = rank_pool(probe_direct, model, fixed_inputs, "direct(late)")
    spike_idx, control_idx = select_spike_control(pool_grad, args.spike_frac, args.num_control)
    print(f"[stage2.6a] pool={len(pool_grad)}, spike_n={len(spike_idx)}, control_n={len(control_idx)}")

    sweep_spike = spike_idx[:args.sweep_n]
    sweep_control = control_idx[:args.sweep_n]

    rows = {"spike": [], "control": []}
    for label, idx_list in (("spike", sweep_spike), ("control", sweep_control)):
        print(f"[stage2.6a] === k-convergence sweep: {label} ({len(idx_list)} batches), k in {K_VALUES} ===")
        for n_done, i in enumerate(idx_list):
            xt, t, y, ut = fixed_inputs[i]
            row = one_batch_k_sweep(model, xt, t, y, ut, params, args.rho, args.lambda_max_num_iters, args.seed)
            row["batch_idx"] = i
            rows[label].append(row)
            gain_str = " | ".join(f"k={k}: gain={row['per_k'][k]['field_gain']:.3f} "
                                   f"rho/r={row['per_k'][k]['rho_norm_over_r_norm']:.2e} "
                                   f"id_err={row['per_k'][k]['identity_err_rho_vs_neg_ell']:.2e}" for k in K_VALUES)
            print(f"  [{label} {n_done+1}/{len(idx_list)}] r_norm={row['r_norm']:.2f} lam={row['lam']:.2f} | {gain_str}")

    summary = {}
    for label in ("spike", "control"):
        rr = rows[label]
        if not rr:
            continue
        summary[label] = {
            k: {
                "field_gain_median": statistics.median([r["per_k"][k]["field_gain"] for r in rr]),
                "rho_norm_over_r_norm_median": statistics.median([r["per_k"][k]["rho_norm_over_r_norm"] for r in rr]),
                "certificate_bound_median": statistics.median([r["per_k"][k]["certificate_bound"] for r in rr]),
                "max_identity_err": max(r["per_k"][k]["identity_err_rho_vs_neg_ell"] for r in rr),
                "certificate_violations": sum(1 for r in rr if r["per_k"][k]["field_gain"] > r["per_k"][k]["certificate_bound"] + 1e-6),
            } for k in K_VALUES
        }

    results = {
        "checkpoint": args.ckpt_direct, "rho": args.rho, "k_values": list(K_VALUES),
        "n_spike_swept": len(sweep_spike), "n_control_swept": len(sweep_control),
        "summary": summary,
        "rows": rows,
    }

    print("\n[stage2.6a] SUMMARY:")
    print(json.dumps(summary, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("\n[stage2.6a] FULL RESULTS:")
    print(json.dumps(results, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        except OSError as e:
            print(f"[stage2.6a] WARNING: --out write failed ({e}); data is in the FULL RESULTS block above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
