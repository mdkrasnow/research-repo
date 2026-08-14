"""
Stage 3B — the batch-size discriminator for H2 (stochastic minibatch over-solving).

Stage 3A (job 38985448) established, at the healthy `start` checkpoint that FBGN
training actually began from:

  * the certified FBGN direction has cosine C_V = 0.00108 with the independent-batch
    gradient -- essentially orthogonal -- and is an ASCENT direction on 4 of 8 batches
    (d_V >= 0, and d_V is eta-INDEPENDENT, so no step size and no damping can fix it);
  * the RAW B=8 minibatch gradient is itself only C_V = 0.0395 aligned;
  * the pre-registered H1 repair (damping) fixed the local model (R_B -10.7 -> +0.15,
    D_B 3.40 -> 0.91) while driving C_V NEGATIVE -- it failed its own falsifier.

So the mechanism is: the B=8 GN system contains almost no population signal, and
(A+lambda I)^{-1} whitening amplifies exactly the small-curvature directions that only
those 8 images constrain, costing a further 36x in alignment (0.0395 -> 0.00108).

That mechanism makes ONE sharp, cheap, falsifiable prediction, and this script tests it
BEFORE any expensive stacked CG solve is paid for:

    If B = 8 is merely signal-STARVED, C_V(B) for the PLAIN minibatch gradient must rise
    materially with B. Pure noise-averaging (g_B = g_pop + xi/sqrt(B), xi isotropic)
    gives C_V(B) ~ sqrt(B) until saturation -- 0.0395 -> ~0.079 at B=32 -> ~0.158 at
    B=128, nowhere near enough for GN whitening to survive a 36x alignment loss.

    * C_V(B) rises FASTER than sqrt(B)          -> batch size is a real lever; run the
                                                   stacked GN solve at the smallest such B.
    * C_V(B) tracks sqrt(B)                     -> minibatch and population objectives are
                                                   decoupled at this checkpoint at every
                                                   batch size FBGN can afford; the
                                                   pre-registered KILL condition fires.

The sqrt(B) law is not assumed -- it is FIT to the measured points and reported as an
exponent, so the reader sees the actual scaling rather than a pass/fail against a
constant that was picked in advance.

WHAT THIS SCRIPT DOES NOT DO
  No training. No optimizer. No Adam. No parameter persistence. The gradient scan does
  not modify theta at all (it only reads .grad). The optional stacked-GN arm uses the
  same apply/evaluate/revert-with-bitwise-verification protocol as Stage 3A.

  The deterministic global probe P is evaluated for reporting ONLY. It is never used for
  acceptance, damping, step-size selection, or retries.

BANKS (drawn from the same deterministic pool as Stage 3A, seed=0, so pool[i] is
byte-identical to the Stage 3A / 300-step-run entry at index i):

  pool[  0:300]  training batches of the 300-step runs      -- UNTOUCHED here
  pool[300:308]  P, the exact probe those runs reported     -- evaluation only
  pool[308:316]  B, Stage 3A's model bank                   -- UNTOUCHED here
  pool[316:324]  V, Stage 3A's trust bank                   -- UNTOUCHED here
  pool[324:324+n_ref]        R, the REFERENCE bank -> g_ref (the population proxy)
  pool[324+n_ref:...]        the disjoint draws consumed by the batch-size scan

g_ref is built from a LARGE bank (default 32 batches x 8 = 256 examples) precisely so
that the reference is not itself the noisy quantity under test. Stage 3A's V (64
examples) is too small to serve as the population proxy for B=128 draws.

Usage (single GPU):
  python wfb_stage3b_batchsize_alignment.py --ckpt <path> --data-path <imagenet-train> \
      --batch-sizes 8,16,32,64,128 --n-rep 8 --out-dir <dir> [--stacked-gn 32]
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

from fb_direct.exact_hvp import (  # noqa: E402
    compute_field_direct, compute_fbgn_gradient_cg,
    compute_fbgn_gradient_cg_microbatched, exact_field_vjp,
    stacked_mixed_gram_mv,
)


# ------------------------------------------------------------------ gradient plumbing

def batch_grad(model, batches, params):
    """Exact theta-gradient of the MEAN loss over `batches`, treated as ONE batch of
    size sum_j B_j:

        L = (1/(N*D)) sum_j ||r_j||^2,  N = sum_j B_j   =>   grad = sum_j M_j^T (2/(N*D)) r_j

    exact_field_vjp negates the WHOLE accumulated .grad buffer, so each microbatch's
    contribution is harvested and summed manually; accumulating across calls would
    re-negate the earlier terms. (Same convention as Stage 3A's grad_bank.)
    """
    N = sum(xt.shape[0] for xt, _, _, _ in batches)
    total = [torch.zeros_like(p) for p in params]
    for xt, t, y, ut in batches:
        D = xt[0].numel()
        field = compute_field_direct(model, xt, t, y)
        r = (field - ut).detach()
        v = r * (2.0 / (N * D))
        model.zero_grad(set_to_none=True)
        exact_field_vjp(model, xt, t, y, v)
        for tot, prm in zip(total, params):
            if prm.grad is not None:
                tot.add_(prm.grad.detach())
    model.zero_grad(set_to_none=True)
    return total


def dot(a, b):
    return sum(float((x * y).sum()) for x, y in zip(a, b))


def norm(a):
    return math.sqrt(max(0.0, sum(float((x * x).sum()) for x in a)))


def bank_loss(model, batches):
    """Mean loss over the bank, in the repo's convention: mean_flat((f-u)^2).mean().

    NOT @torch.no_grad: the field of a scalar-energy model IS an input
    gradient, so compute_field_direct must build a first-order graph in z
    (`torch.autograd.grad(E.sum(), z)`).  Decorating this function with
    no_grad made E.sum() a leaf and killed both Stage 3B submissions
    (39024136, 39030308) with "element 0 of tensors does not require grad".
    Only the ACCUMULATION is kept gradient-free, via float() on each term and
    detach inside compute_field_direct's caller; no parameter graph survives
    the loop.
    """
    tot, n = 0.0, 0
    for xt, t, y, ut in batches:
        field = compute_field_direct(model, xt, t, y).detach()
        with torch.no_grad():
            tot += float(((field - ut) ** 2).mean(
                dim=tuple(range(1, xt.dim()))).sum())
        n += xt.shape[0]
    return tot / n


@torch.no_grad()
def save_params(params):
    return [p.detach().clone() for p in params]


@torch.no_grad()
def restore_and_verify(params, saved):
    for p, s in zip(params, saved):
        p.copy_(s)
    worst = max(float((p.detach() - s).abs().max()) for p, s in zip(params, saved))
    if worst != 0.0:
        raise RuntimeError(f"revert did not restore theta exactly: max|diff|={worst:.3e}")
    return worst


@torch.no_grad()
def apply_step(params, p_dir, eta):
    for prm, d in zip(params, p_dir):
        prm.add_(d, alpha=eta)


# ------------------------------------------------------------------ scaling fit

def fit_power_law(bs, cs):
    """Least-squares fit of log C = a + b log B. b is the measured scaling exponent;
    pure isotropic noise-averaging predicts b = 0.5. Returns (b, a, r2)."""
    pts = [(math.log(b), math.log(c)) for b, c in zip(bs, cs) if c > 0]
    if len(pts) < 2:
        return float("nan"), float("nan"), float("nan")
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in pts)
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return b, a, r2


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data-path", type=str, required=True)
    ap.add_argument("--model", type=str, default="EqM-B/2")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--vae", type=str, default="ema")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8, help="pool granularity; MUST stay 8 "
                                                             "so pool[i] matches Stage 3A")
    ap.add_argument("--train-steps", type=int, default=300,
                    help="offset of the P/B/V banks. MUST match the 300-step runs.")
    ap.add_argument("--n-ref", type=int, default=32, help="reference-bank batches (x8 examples)")
    ap.add_argument("--batch-sizes", type=str, default="8,16,32,64,128")
    ap.add_argument("--n-rep", type=int, default=8, help="independent draws per batch size")
    ap.add_argument("--stacked-gn", type=int, default=0,
                    help="if >0, also run the stacked-GN arm at this model batch size "
                         "(0 = gradient scan only)")
    ap.add_argument("--n-gn-batches", type=int, default=4)
    ap.add_argument("--rho", type=float, default=1e-4)
    ap.add_argument("--cg-tol", type=float, default=5e-2)
    ap.add_argument("--cg-max-iters", type=int, default=250)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    # imported here, not at module scope: the import chain reaches train.py -> diffusers,
    # which keeps the pure-math helpers above importable on a CPU box without diffusers.
    from matched_replay_jacobian_diagnostic import build_pool, load_model

    bss = [int(x) for x in args.batch_sizes.split(",")]
    for b in bss:
        if b % args.batch_size != 0:
            raise ValueError(f"batch size {b} is not a multiple of the pool granularity "
                             f"{args.batch_size}; the scan draws whole pool entries")
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- pool layout, byte-compatible with Stage 3A ---------------------------------
    o = args.train_steps
    n_P = n_B = n_V = 8
    ref_lo = o + n_P + n_B + n_V                       # 324 at train_steps=300
    ref_hi = ref_lo + args.n_ref
    scan_entries = sum(args.n_rep * (b // args.batch_size) for b in bss)
    pool_size = ref_hi + scan_entries

    pool_args = argparse.Namespace(pool_size=pool_size, batch_size=args.batch_size,
                                   image_size=args.image_size, data_path=args.data_path,
                                   seed=args.seed, vae=args.vae)
    print(f"[3b] pool_size={pool_size} = {ref_hi} (banks+ref) + {scan_entries} (scan draws)",
          flush=True)
    pool = build_pool(pool_args, device)

    P_batches = pool[o:o + n_P]
    R_batches = pool[ref_lo:ref_hi]
    print(f"[3b] banks: P=pool[{o}:{o+n_P}] (eval only, identical to the 300-step runs), "
          f"R=pool[{ref_lo}:{ref_hi}] ({args.n_ref * args.batch_size} examples), "
          f"scan draws=pool[{ref_hi}:{pool_size}] -- all mutually disjoint", flush=True)

    model = load_model(args.ckpt, args.model, args.image_size, args.num_classes, "direct", device)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    # probe determinism (report only) -------------------------------------------------
    lp1 = bank_loss(model, P_batches)
    lp2 = bank_loss(model, P_batches)
    print(f"[3b] probe determinism: {lp1!r} vs {lp2!r} (delta={abs(lp1-lp2):.3e})", flush=True)
    if lp1 != lp2:
        raise RuntimeError("probe is NOT deterministic -- corruption is being regenerated. STOP.")

    t0 = time.time()
    g_ref = batch_grad(model, R_batches, params)
    n_ref_ex = args.n_ref * args.batch_size
    print(f"[3b] g_ref over {n_ref_ex} examples: ||g_ref||={norm(g_ref):.6f} "
          f"({time.time()-t0:.1f}s)", flush=True)

    rows = []
    cursor = ref_hi

    # --- arm 1: plain minibatch gradient alignment vs batch size ---------------------
    print("\n[3b] === arm 1: plain minibatch gradient alignment C(B) = cos(g_B, g_ref) ===",
          flush=True)
    summary = {}
    for B in bss:
        k = B // args.batch_size
        cos_list, sig_list = [], []
        for rep in range(args.n_rep):
            draw = pool[cursor:cursor + k]
            cursor += k
            g_B = batch_grad(model, draw, params)
            nB = norm(g_B)
            c = dot(g_B, g_ref) / (nB * norm(g_ref) + 1e-30)
            # projection of g_B onto g_ref, in units of ||g_ref||: the "signal" component.
            proj = dot(g_B, g_ref) / (norm(g_ref) ** 2 + 1e-30)
            cos_list.append(c)
            sig_list.append(proj)
            rows.append({"arm": "grad_scan", "B": B, "rep": rep, "cos": c,
                         "proj_on_ref": proj, "g_B_norm": nB, "g_ref_norm": norm(g_ref),
                         "pool_lo": cursor - k, "pool_hi": cursor})
        cos_list.sort()
        med = cos_list[len(cos_list) // 2]
        mean = sum(cos_list) / len(cos_list)
        se = (sum((x - mean) ** 2 for x in cos_list) / max(1, len(cos_list) - 1)) ** 0.5 \
            / max(1, len(cos_list)) ** 0.5
        summary[B] = {"median_cos": med, "mean_cos": mean, "se_cos": se,
                      "mean_proj": sum(sig_list) / len(sig_list)}
        print(f"  B={B:4d}  n={args.n_rep}  cos: median={med:+.5f} mean={mean:+.5f}"
              f" +/-{se:.5f}   mean proj on g_ref={summary[B]['mean_proj']:+.4f}", flush=True)

    bexp, _, r2 = fit_power_law(bss, [summary[b]["mean_cos"] for b in bss])
    print(f"\n[3b] measured scaling  C(B) ~ B^{bexp:.3f}   (R^2={r2:.4f})", flush=True)
    print(f"[3b] pure isotropic noise-averaging predicts exponent 0.500.", flush=True)
    if bexp > 0.65:
        print("[3b] -> exponent materially ABOVE 1/2: batch size is a real lever. "
              "The stacked-GN arm is worth paying for.", flush=True)
    elif bexp < 0.35:
        print("[3b] -> exponent materially BELOW 1/2: enlarging the batch buys even less "
              "than noise-averaging. KILL condition fires.", flush=True)
    else:
        print("[3b] -> exponent consistent with 1/2: pure noise-averaging, no structural "
              "gain from larger batches. KILL condition fires unless the absolute level "
              "at the affordable B is already usable.", flush=True)

    # --- arm 2 (optional): stacked GN direction at the larger model batch ------------
    if args.stacked_gn > 0:
        Bm = args.stacked_gn
        k = Bm // args.batch_size
        print(f"\n[3b] === arm 2: stacked GN at B_model={Bm} ({k} microbatches, ONE solve) "
              f"vs B_model={args.batch_size} ===", flush=True)
        L0_P = bank_loss(model, P_batches)
        L0_R = bank_loss(model, R_batches)
        for gi in range(args.n_gn_batches):
            draw = pool[cursor:cursor + k]
            cursor += k
            micro = [(xt, t, y, ut) for (xt, t, y, ut) in draw]
            tstart = time.time()

            # -- large-batch stacked direction: ONE solve of the stacked system.
            #    NOT an average of k separate solves -- A = M M^T is not block diagonal
            #    because M^T v = sum_j M_j^T v_j couples the microbatches through theta.
            res_big = compute_fbgn_gradient_cg_microbatched(
                model, micro, params=params, rho=args.rho, cg_tol=args.cg_tol,
                cg_max_iters=args.cg_max_iters, seed=args.seed)
            p_big = [-g for g in res_big["g_wfb"]]
            triples = [(xt, t, y) for (xt, t, y, _) in micro]
            u_big = res_big["u"]
            Au = stacked_mixed_gram_mv(model, triples, params, u_big)
            r_big = res_big["r"]
            true_res_big = float((r_big - (Au + res_big["lam"] * u_big)).norm()
                                 / (r_big.norm() + 1e-30))

            # -- small-batch reference direction on the FIRST microbatch only
            xt0, t0b, y0, ut0 = micro[0]
            res_small = compute_fbgn_gradient_cg(model, xt0, t0b, y0, ut0, params=params,
                                                 rho=args.rho, cg_tol=args.cg_tol,
                                                 cg_max_iters=args.cg_max_iters,
                                                 seed=args.seed)
            p_small = [-g for g in res_small["g_wfb"]]

            for tag, p_dir, tres in (("stacked_B%d" % Bm, p_big, true_res_big),
                                     ("single_B%d" % args.batch_size, p_small, None)):
                d_ref = dot(g_ref, p_dir)
                C = d_ref / (norm(g_ref) * norm(p_dir) + 1e-30)
                row = {"arm": "stacked_gn", "tag": tag, "gn_batch": gi,
                       "B_model": Bm if tag.startswith("stacked") else args.batch_size,
                       "d_ref": d_ref, "C_ref": C, "p_norm": norm(p_dir),
                       "cg_true_residual_ratio": tres, "L0_probe": L0_P, "L0_ref": L0_R}
                rows.append(row)
                print(f"  [gn {gi+1}/{args.n_gn_batches}] {tag:14s} "
                      f"d_ref={d_ref:+.5g} C_ref={C:+.5f} ||p||={norm(p_dir):.3f}"
                      + (f" cg_true_res={tres:.4f}" if tres is not None else "")
                      + f"  ({time.time()-tstart:.0f}s)", flush=True)

            # finite-step transfer, apply/evaluate/revert with bitwise verification
            for tag, p_dir in (("stacked_B%d" % Bm, p_big),
                               ("single_B%d" % args.batch_size, p_small)):
                for eta in (0.25, 0.125):
                    saved = save_params(params)
                    apply_step(params, p_dir, eta)
                    dR = bank_loss(model, R_batches) - L0_R
                    dP = bank_loss(model, P_batches) - L0_P
                    restore_and_verify(params, saved)
                    rows.append({"arm": "stacked_gn_step", "tag": tag, "gn_batch": gi,
                                 "eta": eta, "delta_L_ref": dR, "delta_L_probe": dP})
                    print(f"      {tag:14s} eta={eta:<6} dL_ref={dR:+.5f} dL_probe={dP:+.5f}",
                          flush=True)

    out = os.path.join(args.out_dir, "stage3b_rows.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"ckpt": args.ckpt, "batch_sizes": bss, "n_rep": args.n_rep,
            "n_ref_examples": n_ref_ex, "scaling_exponent": bexp, "scaling_r2": r2,
            "summary": {str(k): v for k, v in summary.items()},
            "L0_probe": lp1, "g_ref_norm": norm(g_ref)}
    with open(os.path.join(args.out_dir, "stage3b_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[3b] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
