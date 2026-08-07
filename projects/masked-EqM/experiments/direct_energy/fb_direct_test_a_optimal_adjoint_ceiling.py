"""
Test A of Yilun's post-decomposition decision tree (2026-08-07): the TRUE
representational ceiling of the cache-adjoint correction on the real
epoch-40 checkpoint, as opposed to the oracle-a* performance measured by
fb_direct_cache_adjoint_decomposition_test.py (job 37676460/37684746:
median cosine(g_cache_vjp, g_cache_direct) = 0.940).

For each of a small number of fixed real batches, solves

    a_best = argmin_a || J_C(theta)^T a - b ||^2,   b := g_cache_direct = g_exact - g_semi

via CGNR (fb_direct/adjoint_optimization.py), and reports

    rho_best  = cos(J_C^T a_best, b)
    rho_astar = cos(J_C^T a*,     b)   (the SAME quantity job 37676460 measured)

Interpretation (per Yilun):
    rho_best ~ rho_astar (~0.94)  -> the ~0.94 ceiling is structural to this
        cache-tensor factorization; a* is already close to optimal. Move to
        Test B (oracle-a* training continuation).
    rho_best >> rho_astar          -> a* is a suboptimal adjoint for the
        (approximate) reverse operator; a learned, deliberately-biased
        corrector could beat oracle-a* performance -- reframes what a
        learned corrector should target.

Run (single GPU, gpu_test or seas_gpu):
  python experiments/direct_energy/fb_direct_test_a_optimal_adjoint_ceiling.py \
      --ckpt <epoch40.pt> --data-path <imagenet train dir> --num-batches 20 --cg-iters 20
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import EqM_models
from transport import create_transport
from train import center_crop_arr
from fb_direct.trainer import ForwardBackwardsDirectTrainer
from fb_direct.cache_adjoint import compute_g_exact, compute_g_semi_and_a_star, compute_g_cache_vjp
from fb_direct.adjoint_optimization import vjp_cache_to_theta, cgnr_solve_optimal_adjoint

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def summarize(vals):
    vals = sorted(v for v in vals if v is not None)
    return {
        "median": percentile(vals, 50),
        "p10": percentile(vals, 10),
        "p90": percentile(vals, 90),
        "n": len(vals),
    }


def cosine_named(a, b, names):
    va = torch.cat([a[n].reshape(-1) for n in names if n in a])
    vb = torch.cat([b[n].reshape(-1) for n in names if n in b])
    return float(torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=20)
    p.add_argument("--cg-iters", type=int, default=20)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // 8

    raw = torch.load(args.ckpt, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw

    model_fb = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="forward-backwards-direct",
    ).to(device)
    missing, unexpected = model_fb.load_state_dict(state_dict, strict=False)
    print(f"[testA] fb model load: missing={missing} unexpected={unexpected}")

    fb_trainer = ForwardBackwardsDirectTrainer(model_fb, device=device)
    fb_trainer.registry.tie_from_forward_()
    sync_err = fb_trainer.registry.compute_sync_error()
    print(f"[testA] phi/theta sync error after tie: {sync_err:.3e}")

    active_pairs = [
        (e.forward_name, e.backward_name)
        for e in fb_trainer.registry.entries
        if e.category in ("reverse_active", "recomputed_conditioning")
    ]
    active_names = [fn for fn, _ in active_pairs]
    depth = len(model_fb.blocks)
    print(f"[testA] {len(active_pairs)} matched pairs, depth={depth}")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    transport = create_transport("Linear", "velocity", None, None, None)
    theta = fb_trainer.theta

    per_batch = []
    it = iter(loader)
    for b in range(args.num_batches):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)

        t, x0, x1 = transport.sample(x1)
        t = t.to(x1)
        t, xt, ut = transport.path_sampler.plan(t, x0, x1)
        ut = ut * transport.get_ct(t)[:, None, None, None]

        g_exact, loss_exact = compute_g_exact(theta, xt, t, y, ut)
        g_semi, a_star, loss_fb, _cache = compute_g_semi_and_a_star(fb_trainer, xt, t, y, ut)
        g_cache_vjp_astar, _ = compute_g_cache_vjp(theta, xt, t, y, a_star, compute_per_tensor_contribution=False)

        b_theta = {n: g_exact[n] - g_semi[n] for n in active_names if n in g_exact and n in g_semi}
        rho_astar = cosine_named(g_cache_vjp_astar, b_theta, active_names)

        # v2 (job 37692043 postmortem): WARM-START at a_star. The v1 run
        # started CG at zero and reported rho_best < rho_astar, which is
        # impossible for the converged argmin (a* is feasible in the same
        # least-squares problem) -- it measured CG non-convergence, not the
        # ceiling. Warm-starting certifies every iterate improves on a*.
        a_best, history = cgnr_solve_optimal_adjoint(
            theta, xt, t, y, b_theta, num_iters=args.cg_iters, x0=a_star,
        )
        g_cache_vjp_best = vjp_cache_to_theta(theta, xt, t, y, a_best)
        rho_best = cosine_named(g_cache_vjp_best, b_theta, active_names)
        rho_trajectory = [h["rho"] for h in history]
        rho_best_iter = max(rho_trajectory) if rho_trajectory else float("nan")

        residual_start = history[0]["residual_norm"] if history else float("nan")
        residual_end = history[-1]["residual_norm"] if history else float("nan")

        per_batch.append({
            "batch": b,
            "rho_astar": rho_astar,
            "rho_best": rho_best,
            "rho_best_iter": rho_best_iter,
            "rho_trajectory": rho_trajectory,
            "cg_residual_start": residual_start,
            "cg_residual_end": residual_end,
            "loss_exact": loss_exact,
            "loss_fb": loss_fb,
        })
        print(f"[testA] batch {b}: rho_astar={rho_astar:.4f} rho_best={rho_best:.4f} "
              f"(gain={rho_best - rho_astar:+.4f}) cg_residual {residual_start:.3e}->{residual_end:.3e} "
              f"rho_traj_first_last=({rho_trajectory[0]:.4f},{rho_trajectory[-1]:.4f})")
        if rho_best < rho_astar - 1e-3:
            print(f"[testA] WARNING batch {b}: rho_best < rho_astar despite warm start -- "
                  f"cosine-vs-residual metric mismatch or numerical drift; inspect rho_trajectory")

    def col(key):
        return [r[key] for r in per_batch]

    rho_astar_summary = summarize(col("rho_astar"))
    rho_best_summary = summarize(col("rho_best"))
    median_gain = rho_best_summary["median"] - rho_astar_summary["median"]
    if median_gain < -1e-3:
        interpretation = (
            "INVALID: rho_best < rho_astar contradicts the warm-started argmin's "
            "feasibility guarantee -- CG numerical failure, do not read as a ceiling"
        )
    elif median_gain < 0.02:
        interpretation = (
            "rho_best ~ rho_astar -> the ceiling is (at least locally) structural to this "
            "cache-tensor factorization: warm-started CG found no materially better adjoint. "
            "Caveat: still a LOWER bound if residuals show non-convergence. "
            "Move to Test B (oracle-a* continuation)"
        )
    else:
        interpretation = (
            "rho_best >> rho_astar -> a* is a suboptimal adjoint; a learned corrector "
            "could beat oracle-a* performance"
        )
    summary = {
        "checkpoint": args.ckpt,
        "num_batches": args.num_batches,
        "cg_iters": args.cg_iters,
        "warm_start": "a_star",
        "depth": depth,
        "sync_error_after_tie": sync_err,
        "rho_astar_summary": rho_astar_summary,
        "rho_best_summary": rho_best_summary,
        "rho_best_iter_summary": summarize(col("rho_best_iter")),
        "median_gain_rho_best_minus_rho_astar": median_gain,
        "interpretation": interpretation,
    }
    print(f"[testA] SUMMARY: {json.dumps(summary, indent=2)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_batch": per_batch}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
