"""
WFB-EqM Stage 3A: frozen-checkpoint mechanism discriminator (2026-08-13).

============================================================
THE QUESTION
============================================================
The Stage 3 300-step runs are complete and all three arms WORSEN the fixed
held-out probe, monotonically in alpha:

    direct (alpha=0.0)   probe 10.624 -> 10.832   (+0.208)
    WFB    (alpha=0.5)   probe 10.624 -> 11.214   (+0.590)
    FBGN   (alpha=1.0)   probe 10.625 -> 14.881   (+4.257)

all at accept_rate = 1.000 with zero skips of any kind. So the certified
CG-FBGN mechanism reduces the loss on every single one of its own minibatches
and yet drives the population/held-out field objective UP.

Two qualitatively different mechanisms can produce that, and they imply
OPPOSITE repairs:

  H1  GAUSS-NEWTON LOCAL-MODEL FAILURE. FBGN solves the LINEARIZED residual
      problem r(theta+p) ~ r(theta) + M p, but the true least-squares Hessian
      is M^T M + sum_i r_i grad^2 r_i, and FBGN drops the second term. If the
      linearization is simply invalid at the step length FBGN takes, the fix
      is genuine Levenberg-Marquardt damping keyed to the actual/predicted
      REDUCTION RATIO (Martens 2010), i.e. shorter/better-damped steps.

  H2  STOCHASTIC MINIBATCH OVER-SOLVING. FBGN may be solving its own batch's
      local problem *correctly* -- and that problem is simply the wrong one.
      Every direction here is built from a fresh minibatch of only 8 images.
      If the FBGN direction is not even an INFINITESIMAL descent direction for
      an independent batch, then no step-size rule and no damping can rescue
      it, because the defect is present in the limit eta -> 0. The fix would
      instead be a larger stochastic model (bigger GN batch) and/or an
      independent-batch acceptance test.

  H3  both.

============================================================
WHY THE EXISTING 300-STEP METRICS CANNOT DECIDE THIS
============================================================
The runs already log rho_lm = actual_delta_L / predicted_delta_L, and it
separates the arms cleanly:

    direct  median R = 0.991   (eta median 3.8e-6, removes  4.5% of own residual)
    WFB     median R = 0.911   (eta median 2.5e-3, removes 13.6% of own residual)
    FBGN    median R = 0.528   (eta median 1.19,   removes 16.4% of own residual)

Read naively that says "FBGN's local model is bad" = H1. But R is CONFOUNDED
WITH STEP LENGTH: FBGN takes eta ~ 1.2 full Gauss-Newton steps while direct
takes eta ~ 4e-6 infinitesimal ones. ANY smooth model looks accurate in the
limit eta -> 0, so a lower R for the arm that takes hugely longer steps is
exactly what H1 and H2 BOTH predict. The 300-step logs never varied eta at a
FIXED direction, and never evaluated any direction against data other than
the batch that produced it. Both gaps are closed here.

A second thing the existing logs quietly falsify, and which this script
re-measures on fixed batches: the Stage 3 postmortem's claim that runaway
lambda_max growth CAUSED the divergence. FBGN's probe trajectory is

    step   25  50  75 100 125 150 175 200 225 250 275 300
    probe  13.9 15.5 16.9 14.1 13.7 12.5 13.4 13.4 12.9 15.1 13.5 14.9

i.e. the damage is essentially COMPLETE by step 25 (10.6 -> 13.9) and the rest
is a stationary noisy band with NO upward trend -- while lambda_max grew ~36x
across those same 300 steps. Curvature growth and probe damage are therefore
decoupled in time, and "lambda_max growth caused divergence" is not supported.
(That claim is corrected in the postmortem as part of this stage.)

============================================================
DESIGN
============================================================
NO training, NO optimizer, NO Adam, NO parameter persistence. Every candidate
update is apply -> evaluate -> revert, with EXACT restoration asserted
bitwise (max|theta_restored - theta_saved| == 0) after every single revert.

Deterministic data banks, all built from ONE seeded pool so that indices are
stable and directly comparable to the 300-step runs:

    pool = build_pool(seed=0, pool_size=332)     # frozen (xt, t, y, ut) tensors
    pool[  0:300]  = the EXACT training batches the 300-step runs consumed
    pool[300:308]  = P, the EXACT fixed held-out probe those runs reported
    pool[308:316]  = B, model-direction bank   (8 batches x 8 = 64 examples)
    pool[316:324]  = V, independent trust bank (8 batches x 8 = 64 examples)

B, V and P are mutually disjoint and disjoint from the training batches.
Corruption is frozen at pool-build time (images, labels, VAE latents, t, the
Gaussian eps via transport.sample, and hence xt and the target ut are all
materialized ONCE and reused byte-identically), so every evaluation of the
same bank is deterministic -- verified explicitly at startup by evaluating P
twice and asserting an exact match.

P is EVALUATION ONLY. It is never used for acceptance, damping, step-size
selection, or any other algorithmic decision.

============================================================
WHAT IS MEASURED, PER (checkpoint, batch B_i, direction)
============================================================
Directions (all expressed as the applied DISPLACEMENT p, so theta' = theta + eta*p):

    p_fbgn   = -M^T (M M^T + lambda I)^{-1} r     (CG, true-residual certified)
    p_direct = -M^T r                             (raw exact-direct control)

For p_direct we scan TWO eta parameterizations: its own native eta* grid, and
a NORM-MATCHED grid where ||eta*p_direct|| equals ||eta*p_fbgn|| at each FBGN
eta. The norm-matched control is what makes the cross-method comparison fair:
it asks whether FBGN specifically damages cross-batch transfer, or whether ANY
step of that parameter length does.

Per direction and per eta in {1, 1/2, 1/4, 1/8} x eta*:

  6.1/6.2/6.3  source-batch predicted vs actual decrease and the LM reduction
               ratio R_B = dL_actual / dL_pred, with the local model
               m_B(eta) = ||r + eta*M p||^2 / (B*D)  (repo normalization:
               L = mean_flat((field-ut)^2).mean() = ||r||^2/(B*D)).

  6.4          LINEARIZATION DEFECT
               D_B(eta) = ||r(theta+eta p) - [r(theta) + eta M p]|| / ||eta M p||
               -- the direct test of whether the linear residual model FBGN
               solved was locally valid. This is the H1 discriminator.

  6.5          INFINITESIMAL INDEPENDENT-BATCH TRANSFER (the H2 discriminator,
               and the only quantity here that is entirely eta-INDEPENDENT):
                   g_V = grad_theta L_V  over the 64-example trust bank
                   d_V = g_V . p         (< 0  => descent on V for small eta)
                   C_V = -d_V / (||g_V|| ||p||)
               If d_V > 0, then by first-order Taylor NO step size and NO
               damping of this direction can improve the trust objective --
               the direction itself is wrong for the population. That is H2 in
               its sharpest form, and it is unfalsifiable by any eta-scan.

  6.6          ACTUAL independent transfer dL_V(eta), with per-example paired
               differences (mean, s.e., fraction improved), plus dL_P(eta) on
               the global probe for ANALYSIS ONLY.

  6.7          curvature covariates on these FIXED batches: lambda_max(A),
               ||p||, ||M p||/||r||, CG iterations, TRUE CG residual ratio
               (recomputed with a fresh operator application, never the
               recursively-updated CG residual).

Optionally (--lambda-sweep, Stage 3B's H1 probe folded in to save a job
round-trip): repeat the FBGN direction at lambda in {lambda_0, 10*lambda_0,
100*lambda_0} on a subset of batches, to test whether stronger damping moves
D_B down and R_B toward 1 while PRESERVING independent-batch descent.

============================================================
CLASSIFICATION (pre-registered, decided from the output table, not here)
============================================================
  H1 if  R_B far below 1 and D_B large at the taken step, AND d_V < 0
         (direction is fine, step/model is not) -> repair = LM damping.
  H2 if  R_B ~ 1 and D_B << 1 (model is accurate) BUT d_V >= 0 frequently, or
         dL_V > 0 even at the smallest eta -> repair = larger model batch /
         independent acceptance; damping CANNOT help.
  H3 if  both.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# NOTE: build_pool/load_model are imported LAZILY inside main(), not here --
# that import chain reaches train.py -> diffusers/AutoencoderKL, which is a
# GPU-job-only dependency. Keeping it out of module scope lets the pure-math
# helpers below (and hence tests/test_stage3a_discriminator_metrics.py) be
# imported and validated on a plain CPU box with no diffusers installed.
from fb_direct.exact_hvp import (  # noqa: E402
    compute_field_direct, exact_field_vjp, field_jvp_direct,
    estimate_lambda_max, cg_solve_shifted_system, mixed_gram_mv,
)
from transport.utils import mean_flat  # noqa: E402


# ------------------------------------------------------------------ helpers

def residual_and_loss(model, xt, t, y, ut):
    """Canonical residual r = field - ut and the repo-convention loss
    L = mean_flat(r^2).mean() = ||r||^2/(B*D)."""
    field = compute_field_direct(model, xt, t, y)
    r = (field - ut).detach()
    loss = float(mean_flat((field - ut) ** 2).mean())
    return r, loss


def bank_loss(model, batches):
    """Mean loss over a bank, plus the per-example losses (for paired stats).
    Every batch has the same feature count D, so the mean of per-batch means
    equals the global mean over examples."""
    per_example = []
    for xt, t, y, ut in batches:
        field = compute_field_direct(model, xt, t, y)
        pe = mean_flat((field - ut) ** 2).detach()
        per_example.append(pe)
    pe = torch.cat(per_example)
    return float(pe.mean()), pe


@torch.no_grad()
def clone_params(params):
    return [p.detach().clone() for p in params]


@torch.no_grad()
def restore_params(params, saved):
    for p, s in zip(params, saved):
        p.copy_(s)


@torch.no_grad()
def assert_exact_restore(params, saved):
    """Exact restoration verification -- bitwise, not approximate. A frozen
    diagnostic that silently leaks parameter drift across candidate steps
    would invalidate every measurement after the first."""
    worst = 0.0
    for p, s in zip(params, saved):
        d = float((p.detach() - s).abs().max())
        worst = max(worst, d)
    if worst != 0.0:
        raise RuntimeError(f"apply/revert did not restore parameters exactly: max|diff|={worst:.3e}")
    return worst


@torch.no_grad()
def apply_step(params, p_dir, eta):
    for prm, d in zip(params, p_dir):
        prm.add_(d, alpha=eta)


def dot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)))


def pnorm(a):
    return float(sum((x * x).sum() for x in a) ** 0.5)


def grad_bank(model, batches, params):
    """Exact theta-gradient of the bank's mean loss:
        L_V = (1/(N*D)) sum_j ||r_j||^2   =>   grad = sum_j M_j^T (2/(N*D)) r_j.
    exact_field_vjp accumulates into .grad and then negates the WHOLE
    accumulated buffer, so contributions MUST be harvested one batch at a time
    and summed manually (accumulating across calls would re-negate earlier
    terms)."""
    n_batches = len(batches)
    total = [torch.zeros_like(p) for p in params]
    for xt, t, y, ut in batches:
        B, D = xt.shape[0], xt[0].numel()
        field = compute_field_direct(model, xt, t, y)
        r = (field - ut).detach()
        v = r * (2.0 / (n_batches * B * D))
        model.zero_grad(set_to_none=True)
        exact_field_vjp(model, xt, t, y, v)
        for tot, prm in zip(total, params):
            if prm.grad is not None:
                tot.add_(prm.grad.detach())
    model.zero_grad(set_to_none=True)
    return total


def true_cg_residual_ratio(model, xt, t, y, params, u, r, lam):
    """||r - (M M^T + lambda I) u|| / ||r||, recomputed with a FRESH operator
    application. Never trust CG's recursively-updated residual for a
    correctness certificate -- that is the quantity that drifts."""
    Au = mixed_gram_mv(model, xt, t, y, params, u)
    resid = r - (Au + lam * u)
    return float(resid.norm() / (r.norm() + 1e-30))


def fbgn_direction(model, xt, t, y, ut, params, rho, cg_tol, cg_max_iters,
                   lambda_max_num_iters, seed, lam_override=None):
    """p_fbgn = -M^T (M M^T + lambda I)^{-1} r, with a true-residual certificate."""
    model.zero_grad(set_to_none=True)
    r, L = residual_and_loss(model, xt, t, y, ut)

    lam_res = estimate_lambda_max(model, xt, t, y, params, num_iters=lambda_max_num_iters, seed=seed)
    lambda_max = lam_res["lambda_max"]
    lam = rho * lambda_max if lam_override is None else lam_override

    cg = cg_solve_shifted_system(model, xt, t, y, params, r, lam, tol=cg_tol, max_iters=cg_max_iters)
    u = cg["u"]
    if u is None or not torch.isfinite(u).all():
        raise RuntimeError("fbgn_direction: CG produced a non-finite/absent u")
    cg_true_resid = true_cg_residual_ratio(model, xt, t, y, params, u, r, lam)

    model.zero_grad(set_to_none=True)
    exact_field_vjp(model, xt, t, y, u)
    g = [prm.grad.detach().clone() if prm.grad is not None else torch.zeros_like(prm) for prm in params]
    model.zero_grad(set_to_none=True)
    p_dir = [-gi for gi in g]  # theta' = theta + eta*p  ==  theta - eta*g
    return {"p": p_dir, "r": r, "L": L, "lambda_max": lambda_max, "lam": lam,
            "cg_n_iters": cg["n_iters"], "cg_converged": bool(cg["converged"]),
            "cg_reported_ratio": cg["final_residual_ratio"], "cg_true_residual_ratio": cg_true_resid}


def direct_direction(model, xt, t, y, ut, params):
    """p_direct = -M^T r, the raw exact-direct control."""
    model.zero_grad(set_to_none=True)
    r, L = residual_and_loss(model, xt, t, y, ut)
    model.zero_grad(set_to_none=True)
    exact_field_vjp(model, xt, t, y, r)
    g = [prm.grad.detach().clone() if prm.grad is not None else torch.zeros_like(prm) for prm in params]
    model.zero_grad(set_to_none=True)
    return {"p": [-gi for gi in g], "r": r, "L": L}


def evaluate_direction(model, params, xt, t, y, ut, p_dir, r, L0, etas,
                       V_batches, P_batches, g_V, V0, V0_pe, P0, eval_probe=True):
    """Apply/evaluate/revert p_dir at each eta; returns one row per eta.

    Computes, per eta: source predicted vs actual decrease and R_B; the
    linearization defect D_B; and the ACTUAL independent-trust and global-probe
    deltas. The eta-independent infinitesimal transfer (d_V, C_V) is computed
    once by the caller and echoed here."""
    B, D = xt.shape[0], xt[0].numel()
    BD = B * D

    # q = M p, the induced field update for the applied displacement direction.
    q = field_jvp_direct(model, xt, t, y, params, p_dir)
    r_dot_q = float((r * q).sum())
    q_norm_sq = float((q * q).sum())
    q_norm = q_norm_sq ** 0.5
    r_norm = float(r.norm())

    # eta* = argmin_eta ||r + eta q||^2 = -r.q/||q||^2. Descent requires r.q < 0
    # for the DISPLACEMENT convention (p already carries the minus sign).
    eta_star = -r_dot_q / q_norm_sq if q_norm_sq > 0 else 0.0

    d_V = dot(g_V, p_dir)
    gV_norm = pnorm(g_V)
    p_norm = pnorm(p_dir)
    C_V = -d_V / (gV_norm * p_norm + 1e-30)

    base = {
        "eta_star": eta_star, "r_norm": r_norm, "q_norm": q_norm,
        "field_gain_ratio": q_norm / (r_norm + 1e-30), "r_dot_q": r_dot_q,
        "p_norm": p_norm, "d_V": d_V, "gV_norm": gV_norm, "C_V": C_V,
        "L0_source": L0, "L0_trust": V0, "L0_probe": P0,
    }

    rows = []
    saved = clone_params(params)
    for frac in etas:
        eta = eta_star * frac
        # local Gauss-Newton model of the SOURCE loss along p:
        #   m(eta) = ||r + eta q||^2 / BD
        m_eta = float(((r + eta * q) ** 2).sum()) / BD
        pred_delta = m_eta - L0                      # negative when the model predicts improvement

        apply_step(params, p_dir, eta)
        r_new, L_new = residual_and_loss(model, xt, t, y, ut)
        actual_delta = L_new - L0
        # linearization defect: how far the TRUE residual moved from where the
        # linear model M p said it would.
        lin_pred_r = r + eta * q
        defect = float((r_new - lin_pred_r).norm())
        D_B = defect / (abs(eta) * q_norm + 1e-30)

        V_new, V_new_pe = bank_loss(model, V_batches)
        paired = (V_new_pe - V0_pe)
        P_new = bank_loss(model, P_batches)[0] if eval_probe else None

        restore_params(params, saved)
        assert_exact_restore(params, saved)

        n = paired.numel()
        # The norm-matched direct control can be driven to eta >> eta*_direct
        # (matching FBGN's parameter step length along a direction whose own
        # optimal step is ~1e-6 of it), which can push the model somewhere the
        # loss is non-finite. That is a legitimate MEASUREMENT -- "a step of
        # that length along this direction destroys the model" -- so it is
        # recorded as such rather than crashing the job; but it is flagged and
        # NaN-sanitized so it cannot silently poison a median downstream.
        finite = all(math.isfinite(v) for v in (L_new, V_new, D_B) if v is not None)
        rows.append({
            **base,
            "eta_frac": frac, "eta": eta,
            "pred_delta_source": pred_delta, "actual_delta_source": actual_delta,
            "R_B": (actual_delta / pred_delta) if abs(pred_delta) > 1e-30 else None,
            "D_B": D_B,
            "delta_L_trust": V_new - V0,
            "trust_paired_mean": float(paired.mean()),
            "trust_paired_se": float(paired.std(unbiased=True) / (n ** 0.5)),
            "trust_frac_improved": float((paired < 0).float().mean()),
            "delta_L_probe": (P_new - P0) if P_new is not None else None,
            "finite": finite,
        })
    return [{k: (None if isinstance(v, float) and not math.isfinite(v) else v)
             for k, v in row.items()} for row in rows]


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True,
                   help="repeatable NAME=PATH, e.g. start=/path/x.pt fbgn300=/path/y.pt")
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=300,
                   help="offset of the 300-step runs' probe slice; keeps P byte-identical to theirs")
    p.add_argument("--n-probe-batches", type=int, default=8)
    p.add_argument("--n-model-batches", type=int, default=8)
    p.add_argument("--n-trust-batches", type=int, default=8)
    p.add_argument("--rho", type=float, default=1e-4)
    p.add_argument("--cg-tol", type=float, default=1e-2)
    p.add_argument("--cg-max-iters", type=int, default=500)
    p.add_argument("--lambda-max-num-iters", type=int, default=20)
    p.add_argument("--etas", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125])
    p.add_argument("--lambda-sweep", type=float, nargs="*", default=None,
                   help="Stage 3B H1 probe: extra lambda multipliers, e.g. 10 100")
    p.add_argument("--lambda-sweep-batches", type=int, default=4)
    p.add_argument("--lambda-sweep-etas", type=float, nargs="+", default=[1.0, 0.25])
    p.add_argument("--vae", default="ema")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--run-tag", default="stage3a")
    args = p.parse_args()

    from matched_replay_jacobian_diagnostic import build_pool, load_model  # lazy: pulls in diffusers

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{args.run_tag}_rows.jsonl")
    out_f = open(out_path, "w")

    # ---- deterministic banks, index-compatible with the 300-step runs -------
    n_P, n_B, n_V = args.n_probe_batches, args.n_model_batches, args.n_trust_batches
    pool_size = args.train_steps + n_P + n_B + n_V
    import types
    pool_args = types.SimpleNamespace(data_path=args.data_path, pool_size=pool_size,
                                      batch_size=args.batch_size, image_size=args.image_size,
                                      seed=args.seed, vae=args.vae)
    pool = build_pool(pool_args, device)
    o = args.train_steps
    P_batches = pool[o:o + n_P]
    B_batches = pool[o + n_P:o + n_P + n_B]
    V_batches = pool[o + n_P + n_B:o + n_P + n_B + n_V]
    print(f"[3a] banks: P={len(P_batches)} (pool[{o}:{o+n_P}], IDENTICAL to the 300-step runs' probe), "
          f"B={len(B_batches)}, V={len(V_batches)} -- mutually disjoint, disjoint from pool[0:{o}] training batches",
          flush=True)

    ckpts = []
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        ckpts.append((name, path))

    summary = {"config": vars(args), "checkpoints": [], "probe_determinism": {}}

    for ck_name, ck_path in ckpts:
        print(f"\n[3a] ===== checkpoint {ck_name} =====", flush=True)
        model = load_model(ck_path, args.model, args.image_size, args.num_classes, "direct", device)
        params = [prm for prm in model.parameters() if prm.requires_grad]

        # ---- probe determinism verification (gate: must be EXACT) ----------
        P0, _ = bank_loss(model, P_batches)
        P0b, _ = bank_loss(model, P_batches)
        print(f"[3a] probe determinism check: {P0!r} vs {P0b!r} (delta={P0b - P0:.3e})", flush=True)
        summary["probe_determinism"][ck_name] = {"first": P0, "second": P0b, "delta": P0b - P0}
        if P0 != P0b:
            raise RuntimeError(f"[3a] FIXED PROBE IS NOT DETERMINISTIC on {ck_name} "
                               f"({P0!r} vs {P0b!r}) -- STOP, repair the probe before any interpretation.")

        V0, V0_pe = bank_loss(model, V_batches)
        print(f"[3a] {ck_name}: probe L_P={P0:.6f}  trust L_V={V0:.6f}", flush=True)

        # ---- exact-direct gradient on the independent trust bank -----------
        t_g = time.time()
        g_V = grad_bank(model, V_batches, params)
        print(f"[3a] g_V computed ({time.time() - t_g:.1f}s), ||g_V||={pnorm(g_V):.6g}", flush=True)

        ck_rows = []
        for bi, (xt, t, y, ut) in enumerate(B_batches):
            t_b = time.time()

            # ---------- FBGN direction ----------
            fb = fbgn_direction(model, xt, t, y, ut, params, args.rho, args.cg_tol,
                                args.cg_max_iters, args.lambda_max_num_iters,
                                seed=args.seed * 1000 + bi)
            fb_rows = evaluate_direction(model, params, xt, t, y, ut, fb["p"], fb["r"], fb["L"],
                                         args.etas, V_batches, P_batches, g_V, V0, V0_pe, P0)
            for row in fb_rows:
                row.update({"ckpt": ck_name, "batch": bi, "method": "fbgn", "lam": fb["lam"],
                            "lam_mult": 1.0, "lambda_max": fb["lambda_max"],
                            "cg_n_iters": fb["cg_n_iters"], "cg_converged": fb["cg_converged"],
                            "cg_true_residual_ratio": fb["cg_true_residual_ratio"],
                            "cg_reported_ratio": fb["cg_reported_ratio"]})
            ck_rows += fb_rows
            # NaN-sanitization above can null out a row's eta (only possible if
            # ||M p|| == 0, i.e. a degenerate direction); such a row cannot
            # define a step norm to match, so it is dropped from the control.
            fbgn_step_norms = [abs(r["eta"]) * r["p_norm"] for r in fb_rows
                               if r["eta"] is not None and r["p_norm"] is not None]

            # ---------- raw-direct control, native eta grid ----------
            dr = direct_direction(model, xt, t, y, ut, params)
            dr_rows = evaluate_direction(model, params, xt, t, y, ut, dr["p"], dr["r"], dr["L"],
                                         args.etas, V_batches, P_batches, g_V, V0, V0_pe, P0)
            for row in dr_rows:
                row.update({"ckpt": ck_name, "batch": bi, "method": "direct_native",
                            "lam": None, "lam_mult": None, "lambda_max": fb["lambda_max"]})
            ck_rows += dr_rows

            # ---------- raw-direct control, NORM-MATCHED to FBGN's steps ----
            # eta chosen so ||eta*p_direct|| == ||eta_fbgn*p_fbgn||, isolating
            # "is it FBGN's DIRECTION" from "is it a step of that LENGTH".
            dr_p_norm = pnorm(dr["p"])
            nm_fracs, eta_star_dr = [], None
            q_dr = field_jvp_direct(model, xt, t, y, params, dr["p"])
            rq = float((dr["r"] * q_dr).sum())
            qq = float((q_dr * q_dr).sum())
            eta_star_dr = -rq / qq if qq > 0 else 0.0
            for sn in fbgn_step_norms:
                eta_nm = sn / (dr_p_norm + 1e-30)
                nm_fracs.append(eta_nm / eta_star_dr if eta_star_dr != 0 else 0.0)
            nm_rows = evaluate_direction(model, params, xt, t, y, ut, dr["p"], dr["r"], dr["L"],
                                         nm_fracs, V_batches, P_batches, g_V, V0, V0_pe, P0)
            for row, sn in zip(nm_rows, fbgn_step_norms):
                row.update({"ckpt": ck_name, "batch": bi, "method": "direct_normmatched",
                            "lam": None, "lam_mult": None, "lambda_max": fb["lambda_max"],
                            "matched_step_norm": sn})
            ck_rows += nm_rows

            # ---------- Stage 3B H1 probe: damping sweep ----------
            if args.lambda_sweep and bi < args.lambda_sweep_batches:
                for mult in args.lambda_sweep:
                    fbl = fbgn_direction(model, xt, t, y, ut, params, args.rho, args.cg_tol,
                                         args.cg_max_iters, args.lambda_max_num_iters,
                                         seed=args.seed * 1000 + bi,
                                         lam_override=mult * fb["lam"])
                    lrows = evaluate_direction(model, params, xt, t, y, ut, fbl["p"], fbl["r"], fbl["L"],
                                               args.lambda_sweep_etas, V_batches, P_batches,
                                               g_V, V0, V0_pe, P0)
                    for row in lrows:
                        row.update({"ckpt": ck_name, "batch": bi, "method": "fbgn_lamsweep",
                                    "lam": fbl["lam"], "lam_mult": mult,
                                    "lambda_max": fbl["lambda_max"],
                                    "cg_n_iters": fbl["cg_n_iters"],
                                    "cg_true_residual_ratio": fbl["cg_true_residual_ratio"]})
                    ck_rows += lrows

            r0 = fb_rows[0]
            print(f"  [batch {bi+1}/{len(B_batches)}] done in {time.time() - t_b:.0f}s "
                  f"| fbgn: eta*={r0['eta_star']:.4g} d_V={r0['d_V']:.4g} "
                  f"C_V={r0['C_V']:.4f} R@1={r0['R_B'] if r0['R_B'] is None else round(r0['R_B'], 3)} "
                  f"D@1={r0['D_B']:.4f} dLv@1={r0['delta_L_trust']:.4g} "
                  f"| direct: d_V={dr_rows[0]['d_V']:.4g} C_V={dr_rows[0]['C_V']:.4f}", flush=True)

        for row in ck_rows:
            out_f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        out_f.flush()
        summary["checkpoints"].append({"name": ck_name, "path": ck_path, "L_P": P0, "L_V": V0,
                                       "n_rows": len(ck_rows)})
        del model
        torch.cuda.empty_cache()

    out_f.close()
    with open(os.path.join(args.results_dir, f"{args.run_tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[3a] DONE ->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
