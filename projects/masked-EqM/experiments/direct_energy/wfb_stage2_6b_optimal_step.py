"""
WFB-EqM Stage 2.6b: optimal linearized step-size comparison, WFB (alpha=0.5)
vs FBGN (alpha=1), no Adam (2026-08-13, reviewer spec).

MOTIVATION. Stage 2.5's one-step probe compared alpha=0.5 and alpha=1 at a
FIXED raw step (eta=1, native scale) -- confounded, because WFB's raw
gradient/induced field update is much larger in magnitude than FBGN's at that
same eta (field gain ~217 vs ~2.8 in Stage 2.5's k=12 measurement), so the
comparison could not distinguish "better geometry" from "happened to take a
smaller step here." Stage 2.6a separately confirmed the field-gain magnitude
question is real but orthogonal (k=12 truncation, not a bound violation).

THIS SCRIPT removes the step-size confound directly, per the reviewer's
exact protocol: for each alpha in {0.5 (WFB), 1.0 (FBGN)}, on the SAME batch,
compute the canonical residual r = field - ut, the preconditioned gradient
g_alpha, and the induced field update q_alpha = M g_alpha (field_jvp_direct).
The first-order LEAST-SQUARES-OPTIMAL step coefficient (minimizing
||r - eta*q_alpha||^2, i.e. the coefficient applied directly to g_alpha as
theta' = theta - eta_star * g_alpha, in the SAME r/q unit convention
compute_wfb_gradient already uses -- no separate native_scale factor, unlike
Stage 2.5's raw eta=1 probe) is

    eta_alpha* = (r . q_alpha) / ||q_alpha||^2.

Apply/revert at eta_alpha* AND a trust-region bracket {0.5, 1.0, 1.5} x
eta_alpha*, on BOTH the source batch (same-batch loss) and a separate
held-out probe batch (generalization), and report

    rho = actual_delta_L / predicted_delta_L_linear

where predicted_delta_L_linear is the closed-form quadratic-model prediction
L(theta - eta*g_alpha) ~= (||r - eta*q_alpha||^2 - ||r||^2) / (B*D) (matching
the training loss's own mean-squared-error reduction convention). rho ~= 1
means the local Gauss-Newton/WFB linearization is trustworthy at that step
size; rho << 1 or negative means the step left the linear-model's validity
region.

k=96 (not Stage 2.5's k=12) for BOTH alphas: Stage 2.6a showed alpha=1's
Lanczos solve is materially unconverged at k=12 (median field gain 2.80-3.61,
collapsing to 0.69-0.70 by k=96) on this checkpoint's severe/high-condition-
number batches -- using k=12 here would silently re-introduce exactly the
confound this script exists to remove. alpha=0.5 pays the same k for a
fair apples-to-apples compute budget (WFB's Lanczos solve was never shown to
need k>12, but the extra cost is cheap relative to alpha=1's).

No training/no weight updates persisted -- same zero-update causal-diagnostic
design as Stage 1/2.5/2.6a.
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
from fb_direct.exact_hvp import compute_wfb_gradient, field_jvp_direct, compute_field_direct  # noqa: E402
from transport.utils import mean_flat  # noqa: E402

ALPHAS = (0.5, 1.0)
ALPHA_NAME = {0.5: "wfb", 1.0: "fbgn"}
BRACKET = (0.5, 1.0, 1.5)  # multipliers of eta_star


@torch.no_grad()
def _clone_params(params):
    return [p.detach().clone() for p in params]


@torch.no_grad()
def _restore_params(params, saved):
    for p, s in zip(params, saved):
        p.copy_(s)


def field_and_loss(model, xt, t, y, ut):
    """Same-batch field + MSE-to-target, model.eval(), no grad retained."""
    was_training = model.training
    model.eval()
    field = compute_field_direct(model, xt, t, y)
    model.train(was_training)
    loss = float(mean_flat((field - ut) ** 2).mean())
    return field, loss


def one_batch_stage2_6b(model, xt, t, y, ut, probe_xt, probe_t, probe_y, probe_ut,
                         rho, k, lambda_max_num_iters, seed):
    B = xt.shape[0]
    D = xt[0].numel()
    BD = B * D

    row = {}
    for alpha in ALPHAS:
        model.zero_grad(set_to_none=True)
        t0 = time.time()
        result = compute_wfb_gradient(model, xt, t, y, ut, params=list(model.parameters()),
                                       rho=rho, k=k, lambda_max_num_iters=lambda_max_num_iters,
                                       seed=seed, alpha=alpha)
        model.zero_grad(set_to_none=True)
        params = result["params"]
        g_alpha = result["g_wfb"]
        r = result["r"]
        r_norm = result["r_norm"]

        q_alpha = field_jvp_direct(model, xt, t, y, params, g_alpha)
        r_dot_q = float((r * q_alpha).sum())
        q_norm_sq = float((q_alpha * q_alpha).sum())
        eta_star = r_dot_q / (q_norm_sq + 1e-30)

        alpha_row = {
            "alpha": alpha, "g_alpha_norm": result["g_wfb_norm"],
            "r_norm": r_norm, "q_alpha_norm": float(q_alpha.norm()),
            "eta_star": eta_star, "wall_lanczos_s": time.time() - t0,
            "breakdown": result["breakdown"], "breakdown_reason": result["breakdown_reason"],
            "lambda_max": result["lambda_max"], "lam": result["lam"],
            "brackets": {},
        }

        _, L_before_same = field_and_loss(model, xt, t, y, ut)
        _, probe_L_before = field_and_loss(model, probe_xt, probe_t, probe_y, probe_ut)

        saved = _clone_params(params)
        for mult in BRACKET:
            eta = mult * eta_star
            # predicted (linearized) delta_L on the SAME batch, closed form:
            # L' ~= (||r - eta*q||^2)/(B*D); delta_L_pred = L' - L(r) = (-2 eta (r.q) + eta^2 ||q||^2)/(B*D)
            predicted_delta_L = (-2.0 * eta * r_dot_q + eta * eta * q_norm_sq) / BD

            with torch.no_grad():
                for p, g in zip(params, g_alpha):
                    p.add_(g, alpha=-eta)
            _, L_after_same = field_and_loss(model, xt, t, y, ut)
            _, probe_L_after = field_and_loss(model, probe_xt, probe_t, probe_y, probe_ut)
            _restore_params(params, saved)
            with torch.no_grad():
                max_restore_err = max(float((p - s).abs().max()) for p, s in zip(params, saved))

            actual_delta_L = L_after_same - L_before_same
            rho_ratio = actual_delta_L / predicted_delta_L if abs(predicted_delta_L) > 1e-30 else None

            alpha_row["brackets"][f"{mult}x"] = {
                "eta": eta, "predicted_delta_L": predicted_delta_L,
                "actual_delta_L_same_batch": actual_delta_L, "rho": rho_ratio,
                "probe_delta_L": probe_L_after - probe_L_before,
                "max_restore_err": max_restore_err,
            }
        alpha_row["L_before_same"] = L_before_same
        alpha_row["probe_L_before"] = probe_L_before
        row[ALPHA_NAME[alpha]] = alpha_row
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
    p.add_argument("--sweep-n", type=int, default=16, help="how many spike + control batches get the full sweep (reviewer spec: 32-64 total)")
    p.add_argument("--rho", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=96, help="Lanczos steps for BOTH alphas -- Stage 2.6a showed alpha=1 needs k>>12 on this checkpoint")
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

    print("[stage2.6b] loading direct checkpoint (instability regime)...")
    model = load_model(args.ckpt_direct, args.model, args.image_size, args.num_classes, "direct", device)

    print("[stage2.6b] ranking pool via probe_direct (same criterion as Stage 1/2.5/2.6a)...")
    pool_grad, pool_loss = rank_pool(probe_direct, model, fixed_inputs, "direct(late)")
    spike_idx, control_idx = select_spike_control(pool_grad, args.spike_frac, args.num_control)
    print(f"[stage2.6b] pool={len(pool_grad)}, spike_n={len(spike_idx)}, control_n={len(control_idx)}")

    sweep_spike = spike_idx[:args.sweep_n]
    sweep_control = control_idx[:args.sweep_n]

    probe_pool_idx = [i for i in range(len(fixed_inputs)) if i not in set(spike_idx) | set(control_idx)]
    if not probe_pool_idx:
        probe_pool_idx = list(range(len(fixed_inputs)))
    probe_xt, probe_t, probe_y, probe_ut = fixed_inputs[probe_pool_idx[0]]
    print(f"[stage2.6b] held-out probe batch idx={probe_pool_idx[0]}")

    rows = {"spike": [], "control": []}
    for label, idx_list in (("spike", sweep_spike), ("control", sweep_control)):
        print(f"[stage2.6b] === optimal-step-size sweep: {label} ({len(idx_list)} batches), k={args.k} ===")
        for n_done, i in enumerate(idx_list):
            xt, t, y, ut = fixed_inputs[i]
            row = one_batch_stage2_6b(model, xt, t, y, ut, probe_xt, probe_t, probe_y, probe_ut,
                                       args.rho, args.k, args.lambda_max_num_iters, args.seed)
            row["batch_idx"] = i
            rows[label].append(row)
            summary_str = " | ".join(
                f"{name}: eta*={row[name]['eta_star']:.3g} " +
                " ".join(f"[{m}x rho={row[name]['brackets'][f'{m}x']['rho']}]" for m in BRACKET)
                for name in ("wfb", "fbgn"))
            print(f"  [{label} {n_done+1}/{len(idx_list)}] {summary_str}")

    def agg(rr, name, mult, field):
        vals = [r[name]["brackets"][f"{mult}x"][field] for r in rr
                if r[name]["brackets"][f"{mult}x"][field] is not None]
        return statistics.median(vals) if vals else None

    summary = {}
    for label in ("spike", "control"):
        rr = rows[label]
        if not rr:
            continue
        summary[label] = {
            name: {
                f"{m}x": {
                    "rho_median": agg(rr, name, m, "rho"),
                    "actual_delta_L_same_batch_median": agg(rr, name, m, "actual_delta_L_same_batch"),
                    "probe_delta_L_median": agg(rr, name, m, "probe_delta_L"),
                    "frac_probe_improved": sum(1 for r in rr if r[name]["brackets"][f"{m}x"]["probe_delta_L"] < 0) / len(rr),
                } for m in BRACKET
            } for name in ("wfb", "fbgn")
        }

    results = {
        "checkpoint": args.ckpt_direct, "rho": args.rho, "k": args.k, "bracket_multipliers": list(BRACKET),
        "n_spike_swept": len(sweep_spike), "n_control_swept": len(sweep_control),
        "summary": summary,
        "rows": rows,
    }

    print("\n[stage2.6b] SUMMARY:")
    print(json.dumps(summary, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("\n[stage2.6b] FULL RESULTS:")
    print(json.dumps(results, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        except OSError as e:
            print(f"[stage2.6b] WARNING: --out write failed ({e}); data is in the FULL RESULTS block above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
