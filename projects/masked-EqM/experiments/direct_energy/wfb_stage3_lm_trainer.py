"""
WFB-EqM Stage 3: Forward-Backward Gauss-Newton training WITHOUT Adam
(2026-08-13, reviewer-directed clean progression: "WFB/GN geometry ->
principled step size -> direct parameter update, and only later decide
whether an adaptive optimizer can be layered on top").

MOTIVATION. Stage 2's 2x2 factorial showed WFB's raw-scale safety (ARM B,
loaded Adam) was a magnitude accident, not a property of the direction itself
-- resetting Adam state (ARM D) recalibrated the step back to "typical" size
and exposed WFB's still-unbounded induced-field-update conditioning. Stage
2.5/2.6a's field-gain measurements and Stage 2.6b's optimal-step-size
comparison established that (a) FBGN (alpha=1) bounds the induced field
update in [0,1] once the Lanczos solve is adequately converged (k=96 on this
checkpoint), and (b) at the LOCALLY OPTIMAL linearized step size eta_alpha* =
r.q_alpha/||q_alpha||^2, WFB and FBGN can be compared without the earlier
step-size confound. The remaining open question, per the reviewer: does
EITHER geometry actually TRAIN when the step size is chosen by the geometry's
OWN local model (not Adam's coordinate-wise, geometry-blind recalibration)?

THIS SCRIPT answers that directly: a from-scratch (no Adam, no momentum, no
per-coordinate preconditioning beyond the WFB/FBGN operator itself) trainer
using BACKTRACKING LINE SEARCH (Nocedal & Wright, "Numerical Optimization"
2006, Algorithm 3.1) as the globalization strategy -- the standard,
convergence-guaranteed mechanism for turning a locally-defined descent
direction into a safe finite step, without any adaptive per-coordinate
rescaling that could (per the reviewer's Stage 2 v5/D account) undo the
geometry's own conditioning.

ALGORITHM (per training step, single batch):
  1. r = field - ut (canonical residual).
  2. g_alpha = M^T(A+lambda I)^{-alpha} r  (compute_wfb_gradient, FIXED alpha
     for this run: 0.0=direct/negative-control, 0.5=WFB, 1.0=FBGN).
  3. q_alpha = M g_alpha (field_jvp_direct) -- induced field update.
  4. r_dot_q = r . q_alpha; q_norm_sq = ||q_alpha||^2.
     If r_dot_q <= 0 (g_alpha is not, to first order, a descent direction
     for this batch -- can happen off the idealized-linear-model regime,
     or for alpha=0's known-pathological conditioning): SKIP this step
     (log skip_reason="not_descent_direction", no update), continue.
  5. eta_star = r_dot_q / q_norm_sq (exact minimizer of the LOCAL quadratic
     model of the same-batch loss, Stage 2.6b's closed form).
  6. Armijo backtracking: eta = eta_star; predicted_slope0 = -2*r_dot_q/(B*D)
     (d/deta of the predicted delta-L at eta=0). For up to
     --max-backtracks tries: apply theta' = theta - eta*g_alpha, measure
     actual_delta_L on the SAME batch; accept if
       actual_delta_L <= c1 * eta * predicted_slope0   (c1 = --armijo-c1,
     the standard sufficient-decrease condition -- guarantees the accepted
     step decreases loss by at least a c1-fraction of what the local linear
     model predicted, the textbook mechanism that makes Newton-type methods
     globally convergent without any adaptive per-coordinate rescaling).
     If not satisfied: revert, eta *= --backtrack-factor, retry (reusing the
     SAME g_alpha/q_alpha -- no re-solve needed, cheap). If exhausted without
     acceptance: reject the step entirely (log skip_reason="backtrack_exhausted").

CONTROLS (per this project's mandatory positive/negative control rule):
  alpha=0.0 (direct/raw M^T r) run through this EXACT SAME backtracking
  harness is the negative control -- it tests whether proper globalization
  ALONE (with no WFB/FBGN preconditioning at all) can rescue the known-
  unstable raw direction. If alpha=0 also trains stably here, that would
  falsify "WFB/FBGN geometry is necessary" and point to "Adam specifically
  was the problem" instead. If alpha=0 still thrashes/stalls (expected,
  given its provably unbounded per-mode field-update gain), that confirms
  the geometry itself is doing necessary work, not just the line search.

No DDP (single-GPU, matching all prior Stage 1-2.6 diagnostic-scale
infrastructure) -- this is a mechanism-validity test at diagnostic scale, not
a paper-scale run; per this project's CIFAR/proxy-scale discipline, a
positive result here is a FILTER for a properly-scaled follow-up, not
publishable on its own.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matched_replay_jacobian_diagnostic import build_pool, load_model  # noqa: E402
from fb_direct.exact_hvp import (  # noqa: E402
    compute_wfb_gradient, compute_fbgn_gradient_cg, field_jvp_direct, compute_field_direct,
)
from transport.utils import mean_flat  # noqa: E402


def field_and_loss(model, xt, t, y, ut):
    was_training = model.training
    model.eval()
    field = compute_field_direct(model, xt, t, y)
    model.train(was_training)
    loss = float(mean_flat((field - ut) ** 2).mean())
    return field, loss


def probe_loss_avg(model, probe_batches):
    losses = []
    for xt, t, y, ut in probe_batches:
        _, l = field_and_loss(model, xt, t, y, ut)
        losses.append(l)
    return sum(losses) / len(losses)


@torch.no_grad()
def _clone_params(params):
    return [p.detach().clone() for p in params]


@torch.no_grad()
def _restore_params(params, saved):
    for p, s in zip(params, saved):
        p.copy_(s)


def train_step(model, xt, t, y, ut, rho, k, lambda_max_num_iters, alpha, seed,
               armijo_c1, backtrack_factor, max_backtracks, solver="lanczos", cg_tol=1e-2, cg_max_iters=200):
    B, D = xt.shape[0], xt[0].numel()
    BD = B * D

    model.zero_grad(set_to_none=True)
    if solver == "cg":
        assert alpha == 1.0, "solver='cg' is only implemented for alpha=1.0 (FBGN)"
        result = compute_fbgn_gradient_cg(model, xt, t, y, ut, params=list(model.parameters()),
                                           rho=rho, cg_tol=cg_tol, cg_max_iters=cg_max_iters,
                                           lambda_max_num_iters=lambda_max_num_iters, seed=seed)
    else:
        result = compute_wfb_gradient(model, xt, t, y, ut, params=list(model.parameters()),
                                       rho=rho, k=k, lambda_max_num_iters=lambda_max_num_iters,
                                       seed=seed, alpha=alpha)
    model.zero_grad(set_to_none=True)
    params = result["params"]
    g_alpha = result["g_wfb"]
    r = result["r"]

    q_alpha = field_jvp_direct(model, xt, t, y, params, g_alpha)
    r_dot_q = float((r * q_alpha).sum())
    q_norm_sq = float((q_alpha * q_alpha).sum())

    step_info = {
        "lambda_max": result["lambda_max"], "lam": result["lam"], "m_lanczos": result["m"],
        "breakdown": result["breakdown"], "breakdown_reason": result["breakdown_reason"],
        "r_norm": result["r_norm"], "g_alpha_norm": result["g_wfb_norm"],
        "q_alpha_norm": float(q_alpha.norm()), "r_dot_q": r_dot_q,
    }

    if r_dot_q <= 0.0:
        step_info.update({"accepted": False, "skip_reason": "not_descent_direction",
                           "eta_star": None, "eta_used": None, "n_backtracks": 0,
                           "actual_delta_L": None, "predicted_delta_L": None, "rho_lm": None})
        return step_info

    eta_star = r_dot_q / q_norm_sq
    predicted_slope0 = -2.0 * r_dot_q / BD

    _, L_before = field_and_loss(model, xt, t, y, ut)
    saved = _clone_params(params)
    eta = eta_star
    accepted = False
    n_backtracks = 0
    actual_delta_L = None
    predicted_delta_L = None
    for attempt in range(max_backtracks + 1):
        with torch.no_grad():
            for p, g in zip(params, g_alpha):
                p.add_(g, alpha=-eta)
        _, L_after = field_and_loss(model, xt, t, y, ut)
        actual_delta_L = L_after - L_before
        predicted_delta_L = eta * predicted_slope0 + (eta ** 2) * q_norm_sq / BD
        armijo_bound = armijo_c1 * eta * predicted_slope0  # negative number; sufficient-decrease threshold
        if actual_delta_L <= armijo_bound:
            accepted = True
            break
        _restore_params(params, saved)
        eta *= backtrack_factor
        n_backtracks = attempt + 1

    if not accepted:
        _restore_params(params, saved)

    rho_lm = (actual_delta_L / predicted_delta_L) if (predicted_delta_L is not None
              and abs(predicted_delta_L) > 1e-30) else None
    step_info.update({
        "accepted": accepted, "skip_reason": None if accepted else "backtrack_exhausted",
        "eta_star": eta_star, "eta_used": eta if accepted else None, "n_backtracks": n_backtracks,
        "actual_delta_L": actual_delta_L, "predicted_delta_L": predicted_delta_L,
        "L_before": L_before, "rho_lm": rho_lm,
    })
    model.zero_grad(set_to_none=True)
    return step_info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-direct", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, required=True, help="0.0=direct(negative control), 0.5=WFB, 1.0=FBGN")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--rho", type=float, default=1e-4, help="wfb spectral damping: lambda = rho * lambda_max(A)")
    p.add_argument("--k", type=int, default=96, help="Lanczos steps -- Stage 2.6a: alpha=1 needs k>>12 on this checkpoint")
    p.add_argument("--solver", choices=["lanczos", "cg"], default="lanczos",
                    help="'cg' (alpha=1.0 only) replaces fixed-k Lanczos with the adaptive-tolerance "
                         "CG solve (2026-08-13) -- Stage 3's alpha=1.0 smoke at k=96 found "
                         "theoretically-impossible not_descent_direction skips from truncation error; "
                         "CG resolved this on the real model (field_gain 0.983 vs 2.83-3.61, 15 iters).")
    p.add_argument("--cg-tol", type=float, default=1e-3, help="(solver=cg) target ||rho||/||r|| tolerance")
    p.add_argument("--cg-max-iters", type=int, default=500)
    p.add_argument("--lambda-max-num-iters", type=int, default=20)
    p.add_argument("--armijo-c1", type=float, default=1e-4)
    p.add_argument("--backtrack-factor", type=float, default=0.5)
    p.add_argument("--max-backtracks", type=int, default=6)
    p.add_argument("--n-probe-batches", type=int, default=8)
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--vae", default="ema")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--adaptive-damping", action="store_true",
                    help="Levenberg-Marquardt style: after each accepted step, adjust rho (hence "
                         "lambda = rho*lambda_max in the shifted solve) based on rho_lm = "
                         "actual_delta_L/predicted_delta_L at the applied eta (Stage 3 postmortem "
                         "2026-08-13: FBGN's curvature (lambda_max) grew ~36x avg/~19000x peak "
                         "uncontrolled over 300 steps at fixed rho, coinciding exactly with every "
                         "same-batch loss spike -- Armijo alone cannot see or prevent this, since it "
                         "only checks sufficient decrease relative to the CURRENT point). "
                         "rho_lm > --lm-good-threshold -> shrink rho (more aggressive); "
                         "rho_lm < --lm-bad-threshold -> grow rho (more conservative/damped).")
    p.add_argument("--lm-good-threshold", type=float, default=0.75)
    p.add_argument("--lm-bad-threshold", type=float, default=0.25)
    p.add_argument("--lm-shrink-factor", type=float, default=0.5)
    p.add_argument("--lm-grow-factor", type=float, default=2.0)
    p.add_argument("--rho-min", type=float, default=1e-6)
    p.add_argument("--rho-max", type=float, default=1e-1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)
    metrics_path = os.path.join(args.results_dir, f"{args.run_tag}_metrics.jsonl")
    metrics_f = open(metrics_path, "w")

    pool_size = args.max_steps + args.n_probe_batches + 8
    import types
    pool_args = types.SimpleNamespace(data_path=args.data_path, pool_size=pool_size, batch_size=args.batch_size,
                                       image_size=args.image_size, seed=args.seed, vae=args.vae)
    fixed_inputs = build_pool(pool_args, device)

    train_batches = fixed_inputs[:args.max_steps]
    probe_batches = fixed_inputs[args.max_steps:args.max_steps + args.n_probe_batches]
    print(f"[stage3] {len(train_batches)} train batches, {len(probe_batches)} held-out probe batches")

    print(f"[stage3] loading direct checkpoint (instability regime), alpha={args.alpha}...")
    model = load_model(args.ckpt_direct, args.model, args.image_size, args.num_classes, "direct", device)

    probe_loss_initial = probe_loss_avg(model, probe_batches)
    print(f"[stage3] initial probe loss (avg over {len(probe_batches)} batches): {probe_loss_initial:.6f}")

    n_accepted, n_skipped_not_descent, n_skipped_backtrack = 0, 0, 0
    current_rho = args.rho
    t_start = time.time()
    for step, (xt, t, y, ut) in enumerate(train_batches):
        info = train_step(model, xt, t, y, ut, current_rho, args.k, args.lambda_max_num_iters, args.alpha,
                           seed=args.seed * 100000 + step, armijo_c1=args.armijo_c1,
                           backtrack_factor=args.backtrack_factor, max_backtracks=args.max_backtracks,
                           solver=args.solver, cg_tol=args.cg_tol, cg_max_iters=args.cg_max_iters)
        info["step"] = step
        info["wall_s_cum"] = time.time() - t_start
        info["rho_used"] = current_rho
        if info["accepted"]:
            n_accepted += 1
        elif info["skip_reason"] == "not_descent_direction":
            n_skipped_not_descent += 1
        else:
            n_skipped_backtrack += 1

        if args.adaptive_damping and info.get("rho_lm") is not None:
            rho_lm = info["rho_lm"]
            if rho_lm > args.lm_good_threshold:
                current_rho = max(args.rho_min, current_rho * args.lm_shrink_factor)
            elif rho_lm < args.lm_bad_threshold:
                current_rho = min(args.rho_max, current_rho * args.lm_grow_factor)
            info["rho_next"] = current_rho

        print(f"  [step {step+1}/{len(train_batches)}] accepted={info['accepted']} "
              f"skip_reason={info['skip_reason']} eta_used={info['eta_used']} n_bt={info['n_backtracks']} "
              f"L_before={info.get('L_before')} dL={info['actual_delta_L']} rho_lm={info.get('rho_lm')} "
              f"rho_used={info['rho_used']:.3g}"
              f"{' rho_next=' + format(info['rho_next'], '.3g') if 'rho_next' in info else ''} "
              f"(accept_rate={n_accepted/(step+1):.3f}, wall={info['wall_s_cum']:.0f}s)", flush=True)
        if (step + 1) % args.probe_every == 0 or step == len(train_batches) - 1:
            info["probe_loss"] = probe_loss_avg(model, probe_batches)
            print(f"    -> probe_loss={info['probe_loss']:.6f}", flush=True)
        metrics_f.write(json.dumps(info, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        metrics_f.flush()

        if (step + 1) % args.ckpt_every == 0 or step == len(train_batches) - 1:
            ckpt_path = os.path.join(args.results_dir, f"{args.run_tag}_step{step+1}.pt")
            torch.save({"model": model.state_dict(), "step": step + 1, "alpha": args.alpha}, ckpt_path)

    probe_loss_final = probe_loss_avg(model, probe_batches)
    metrics_f.close()

    summary = {
        "alpha": args.alpha, "max_steps": args.max_steps, "k": args.k, "rho": args.rho,
        "solver": args.solver, "cg_tol": args.cg_tol,
        "adaptive_damping": args.adaptive_damping, "final_rho": current_rho,
        "n_accepted": n_accepted, "n_skipped_not_descent_direction": n_skipped_not_descent,
        "n_skipped_backtrack_exhausted": n_skipped_backtrack,
        "accept_rate": n_accepted / len(train_batches),
        "probe_loss_initial": probe_loss_initial, "probe_loss_final": probe_loss_final,
        "probe_loss_delta": probe_loss_final - probe_loss_initial,
        "wall_s_total": time.time() - t_start,
        "metrics_path": metrics_path,
    }
    print("\n[stage3] SUMMARY:")
    print(json.dumps(summary, indent=2))
    summary_path = os.path.join(args.results_dir, f"{args.run_tag}_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    except OSError as e:
        print(f"[stage3] WARNING: summary --out write failed ({e}); summary is in the block above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
