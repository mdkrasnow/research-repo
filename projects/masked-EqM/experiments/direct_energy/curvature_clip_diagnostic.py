"""
Offline curvature-vs-clip diagnostic (2026-08-10, growing-instability
mechanism check).

Question: quartile forensics on the fwrev arms (jobs 37780076 lambda0,
37780078 GP 0.05) showed both arms reproduce direct's growing-clip-rate
pattern, and GP's mean||grad_z E||^2 has the SAME relative growth rate as
lambda0's (only a constant ~9% level shift) -- evidence that a MEAN-based
penalty cannot arrest a TAIL-driven instability. The theoretically correct
object is ||grad_z^2 E||_op (spectral norm / Lipschitz constant of the
field), estimated here via power iteration (fb_direct/exact_hvp.py,
FP64-verified against exact eigh() in tests/test_fb_direct_curvature_probe.py
before this script was written).

This script runs TWO tests against real checkpoints, on a FIXED probe batch
bank shared across all checkpoints for a clean paired comparison:

(A) Cross-checkpoint growth: does spectral_norm actually grow over training
    (directly, not via the clip-rate proxy), for lambda0 vs GP arms? Tests
    whether GP's ~9% mean-level reduction does anything to the CURVATURE
    growth trend specifically (as opposed to loss_gp, which we already know
    grows at the same relative rate in both arms).

(B) Predictive comparison: pooling all (checkpoint, batch) pairs, which
    quantity better predicts the theta-space grad_norm that actually gets
    clipped during training -- the order-1 quantity GP penalizes
    (mean||grad_z E||^2) or the order-2 quantity proposed here
    (spectral_norm)? Reports Pearson AND Spearman correlation for both,
    plus max-per-batch spectral_norm (a worst-case-in-batch aggregate,
    since a single bad sample can dominate the batch-summed theta
    gradient) alongside the mean-per-batch aggregate.

Also stratifies all metrics by t (interpolation timestep) to separate
"curvature grows with more training steps" from "certain t regions are
just ill-conditioned regardless of training step" -- an alternative
explanation this diagnostic can rule in or out for free.

Run (single GPU, seas_gpu or gpu_test):
  python experiments/direct_energy/curvature_clip_diagnostic.py \
      --ckpts-lambda0 <path1> <path2> ... --ckpts-gp <path1> <path2> ... \
      --data-path <imagenet train dir> --num-batches 20 --batch-size 8
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
from fb_direct.exact_hvp import exact_fwrev_backward, power_iteration_spectral_norm

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def build_probe_bank(data_path, num_batches, batch_size, image_size, seed):
    # 2026-08-11: progress logging added after repeated silent stalls (3 of 4
    # topk_subspace_diagnostic.py submissions hung with zero output for
    # 45min-1h35m before being cancelled as presumed bad-node/stuck-mount
    # incidents -- indistinguishable from "genuinely slow" without this).
    # ImageFolder() below walks the full ~1.28M-image tree on every call
    # (COST IS INDEPENDENT OF num_batches/pool_size -- this is a FIXED
    # per-job cost, not something scaling pool_size larger makes worse),
    # exactly matching a previously-documented "ImageFolder construction
    # bad node/stuck mount" incident from project memory.
    import time
    t0 = time.time()
    print(f"  [build_probe_bank] indexing ImageFolder at {data_path} ...")
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(data_path, transform=transform)
    print(f"  [build_probe_bank] ImageFolder indexed in {time.time() - t0:.1f}s: "
          f"{len(dataset)} images, {len(dataset.classes)} classes")
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                         drop_last=True, generator=gen)
    bank = []
    it = iter(loader)
    t1 = time.time()
    for i in range(num_batches):
        x, y = next(it)
        bank.append((x, y))
        if (i + 1) % 200 == 0 or (i + 1) == num_batches:
            print(f"  [build_probe_bank] {i + 1}/{num_batches} batches loaded "
                  f"({time.time() - t1:.1f}s elapsed)")
    return bank


def load_model(ckpt_path, model_name, image_size, num_classes, device):
    latent_size = image_size // 8
    model = EqM_models[model_name](
        input_size=latent_size, num_classes=num_classes, ebm="direct",
    ).to(device)
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [warn] load_state_dict: missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def evaluate_checkpoint(model, vae, transport, probe_bank, device, spec_iters, seed):
    """For every probe batch: theta-space grad_norm (matches production
    gradient_metrics.jsonl), mean||grad_z E||^2 (order-1, GP's quantity),
    per-sample spectral_norm (order-2), mean t. Returns list of dicts."""
    rows = []
    for bi, (x, y) in enumerate(probe_bank):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
        t, x0, x1 = transport.sample(x1)
        t = t.to(x1)
        t, xt, ut = transport.path_sampler.plan(t, x0, x1)
        ut = ut * transport.get_ct(t)[:, None, None, None]

        model.zero_grad(set_to_none=True)
        stats = exact_fwrev_backward(model, xt, t, y, ut, gp_lambda=0.0)
        grad_norm = float(torch.stack([
            (p.grad.detach().float() ** 2).sum() for p in model.parameters() if p.grad is not None
        ]).sum().sqrt())
        model.zero_grad(set_to_none=True)

        spec = power_iteration_spectral_norm(model, xt, t, y, num_iters=spec_iters, seed=seed + bi)
        spec_vals = spec["spectral_norm"].cpu()

        rows.append({
            "batch": bi,
            "grad_norm": grad_norm,
            "loss_gp": stats["loss_gp"],
            "loss_main": stats["loss_main"],
            "spectral_norm_mean": float(spec_vals.mean()),
            "spectral_norm_max": float(spec_vals.max()),
            "spectral_norm_median": float(spec_vals.median()),
            "t_mean": float(t.mean()),
            "t_min": float(t.min()),
            "t_max": float(t.max()),
        })
        print(f"    batch {bi}: grad_norm={grad_norm:.3f} loss_gp={stats['loss_gp']:.3f} "
              f"spec_norm(mean/max)={float(spec_vals.mean()):.4f}/{float(spec_vals.max()):.4f} "
              f"t_mean={float(t.mean()):.3f}")
    return rows


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 1e-30 else float("nan")


def spearman(xs, ys):
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks
    return pearson(rank(xs), rank(ys))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts-lambda0", nargs="+", required=True)
    p.add_argument("--ckpts-gp", nargs="+", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=20)
    p.add_argument("--spec-iters", type=int, default=15)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # hvp_z's double-backward (fb_direct/exact_hvp.py) has no derivative for
    # fused flash/efficient attention kernels ("derivative for
    # aten::_scaled_dot_product_efficient_attention_backward is not
    # implemented") -- this is a STANDALONE script that never goes through
    # train.py's main(), which normally forces the math SDPA backend for
    # any ebm != 'none' training/eval. Replicate that here explicitly.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    transport = create_transport("Linear", "velocity", None, None, None)

    print(f"[diag] building fixed probe bank: {args.num_batches} batches x {args.batch_size}")
    probe_bank = build_probe_bank(args.data_path, args.num_batches, args.batch_size,
                                   args.image_size, args.seed)

    all_results = {"lambda0": [], "gp005": []}
    for arm_name, ckpts in [("lambda0", args.ckpts_lambda0), ("gp005", args.ckpts_gp)]:
        for ckpt_path in ckpts:
            print(f"[diag] === {arm_name}: {ckpt_path}")
            model = load_model(ckpt_path, args.model, args.image_size, args.num_classes, device)
            rows = evaluate_checkpoint(model, vae, transport, probe_bank, device,
                                        args.spec_iters, args.seed)
            all_results[arm_name].append({"ckpt": ckpt_path, "rows": rows})
            del model
            torch.cuda.empty_cache()

    # (A) Cross-checkpoint growth trend, per arm.
    growth = {}
    for arm_name, ckpt_results in all_results.items():
        growth[arm_name] = []
        for cr in ckpt_results:
            rows = cr["rows"]
            n = len(rows)
            growth[arm_name].append({
                "ckpt": cr["ckpt"],
                "median_grad_norm": sorted(r["grad_norm"] for r in rows)[n // 2],
                "median_loss_gp": sorted(r["loss_gp"] for r in rows)[n // 2],
                "median_spectral_norm_mean": sorted(r["spectral_norm_mean"] for r in rows)[n // 2],
                "max_spectral_norm_max": max(r["spectral_norm_max"] for r in rows),
            })

    # (B) Pooled predictive comparison: does spectral_norm predict grad_norm
    # better than loss_gp does, across ALL (checkpoint, batch) pairs, pooled
    # across BOTH arms (more data, and the question is architecture-general).
    pooled_grad_norm, pooled_loss_gp, pooled_spec_mean, pooled_spec_max = [], [], [], []
    for arm_name, ckpt_results in all_results.items():
        for cr in ckpt_results:
            for r in cr["rows"]:
                pooled_grad_norm.append(r["grad_norm"])
                pooled_loss_gp.append(r["loss_gp"])
                pooled_spec_mean.append(r["spectral_norm_mean"])
                pooled_spec_max.append(r["spectral_norm_max"])

    correlations = {
        "n_pooled": len(pooled_grad_norm),
        "pearson_grad_norm_vs_loss_gp": pearson(pooled_grad_norm, pooled_loss_gp),
        "spearman_grad_norm_vs_loss_gp": spearman(pooled_grad_norm, pooled_loss_gp),
        "pearson_grad_norm_vs_spectral_norm_mean": pearson(pooled_grad_norm, pooled_spec_mean),
        "spearman_grad_norm_vs_spectral_norm_mean": spearman(pooled_grad_norm, pooled_spec_mean),
        "pearson_grad_norm_vs_spectral_norm_max": pearson(pooled_grad_norm, pooled_spec_max),
        "spearman_grad_norm_vs_spectral_norm_max": spearman(pooled_grad_norm, pooled_spec_max),
    }

    summary = {"growth_trend": growth, "correlations": correlations}
    print(f"[diag] SUMMARY: {json.dumps(summary, indent=2)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "raw": all_results}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
