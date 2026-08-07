"""
Mandatory pre-registered decomposition test (Section 5 of the learned-
cache-adjoint proposal, 2026-08-07) on the real epoch-40 checkpoint, before
any corrector network is trained.

Prior diagnostics (layerwise truncation job 37541376, blockwise calibration
job 37544288, temporal reuse job 37656610) all measured PROPERTIES of
g_cache := g_exact - g_semi without ever independently reconstructing it
from a cache-conditioned adjoint. This script tests whether

    g_cache =?= J_{C_theta}(theta)^T a*,   a* := dL_fb/dC

holds numerically on real batches (see fb_direct/cache_adjoint.py's module
docstring for the exact math and the two graph-construction bugs its own
earlier drafts hit and fixed -- verified on a tiny synthetic model in
tests/test_fb_direct_cache_adjoint.py, FP64, cosine>0.999 there).

Mandatory gate (per the proposal, Section 5/9):
    cosine(g_cache_vjp, g_cache_direct) > 0.999 and low relative norm error.
If this fails on the real checkpoint, STOP -- do not train any corrector
until the discrepancy is understood.

Also reports (Section 6): per-cache-tensor-family contribution to g_cache
(RMS a* norm, VJP contribution norm, shape), so a subsequent corrector
architecture knows which cache families actually matter.

Run (single GPU, gpu_test or seas_gpu):
  python experiments/direct_energy/fb_direct_cache_adjoint_decomposition_test.py \
      --ckpt <epoch40.pt> --data-path <imagenet train dir> --num-batches 100
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import EqM_models
from transport import create_transport
from train import center_crop_arr
from fb_direct.trainer import ForwardBackwardsDirectTrainer
from fb_direct.cache_adjoint import decomposition_test

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def block_key(name):
    if name.startswith("blocks."):
        return f"blocks.{name.split('.')[1]}"
    if name.startswith("energy_head."):
        return "energy_head"
    if name.startswith("x_embedder."):
        return "x_embedder"
    raise ValueError(f"unrecognized active parameter name for block_key: {name}")


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


def cache_tensor_block(name):
    if name == "c":
        return "conditioning"
    if name.startswith("final."):
        return "energy_head"
    if name.startswith("blocks."):
        return f"blocks.{name.split('.')[1]}"
    raise ValueError(f"unrecognized cache tensor name: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--per-tensor-batches", type=int, default=3,
                    help="Number of leading batches to run the expensive "
                         "per-cache-tensor VJP inventory on (~1 extra "
                         "forward+backward pass per cache tensor family, "
                         "~100+ for a depth-12 model) -- diagnostic only, "
                         "not part of the mandatory cosine gate.")
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
    print(f"[cache_adjoint] fb model load: missing={missing} unexpected={unexpected}")

    fb_trainer = ForwardBackwardsDirectTrainer(model_fb, device=device)
    fb_trainer.registry.tie_from_forward_()
    sync_err = fb_trainer.registry.compute_sync_error()
    print(f"[cache_adjoint] phi/theta sync error after tie: {sync_err:.3e}")

    active_pairs = [
        (e.forward_name, e.backward_name)
        for e in fb_trainer.registry.entries
        if e.category in ("reverse_active", "recomputed_conditioning")
    ]
    depth = len(model_fb.blocks)
    print(f"[cache_adjoint] {len(active_pairs)} matched pairs, depth={depth}")

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

    per_batch = []
    tensor_contrib_by_block = defaultdict(list)
    a_star_rms_by_block = defaultdict(list)

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

        result = decomposition_test(
            fb_trainer, active_pairs, xt, t, y, ut,
            compute_per_tensor_contribution=(b < args.per_tensor_batches),
        )

        for name, val in result["per_tensor_contribution"].items():
            tensor_contrib_by_block[cache_tensor_block(name)].append(val)
        for name, val in result["a_star_rms"].items():
            a_star_rms_by_block[cache_tensor_block(name)].append(val)

        per_batch.append({
            "batch": b,
            "cosine_g_cache_vjp_vs_direct": result["cosine_g_cache_vjp_vs_direct"],
            "rel_norm_error_g_cache": result["rel_norm_error_g_cache"],
            "cosine_g_hat_vs_exact": result["cosine_g_hat_vs_exact"],
            "cosine_g_semi_vs_exact": result["cosine_g_semi_vs_exact"],
            "loss_exact": result["loss_exact"],
            "loss_fb": result["loss_fb"],
        })
        if b % 10 == 0 or b == args.num_batches - 1:
            print(f"[cache_adjoint] batch {b}: "
                  f"cos(vjp,direct)={result['cosine_g_cache_vjp_vs_direct']:.4f} "
                  f"rel_err={result['rel_norm_error_g_cache']:.3e} "
                  f"cos(hat,exact)={result['cosine_g_hat_vs_exact']:.4f} "
                  f"cos(semi,exact)={result['cosine_g_semi_vs_exact']:.4f}")

    def col(key):
        return sorted(r[key] for r in per_batch)

    summary = {
        "checkpoint": args.ckpt,
        "num_batches": args.num_batches,
        "depth": depth,
        "sync_error_after_tie": sync_err,
        "cosine_g_cache_vjp_vs_direct_summary": summarize(col("cosine_g_cache_vjp_vs_direct")),
        "rel_norm_error_g_cache_summary": summarize(col("rel_norm_error_g_cache")),
        "cosine_g_hat_vs_exact_summary": summarize(col("cosine_g_hat_vs_exact")),
        "cosine_g_semi_vs_exact_summary": summarize(col("cosine_g_semi_vs_exact")),
        "tensor_contribution_by_block_summary": {
            blk: summarize(vals) for blk, vals in tensor_contrib_by_block.items()
        },
        "a_star_rms_by_block_summary": {
            blk: summarize(vals) for blk, vals in a_star_rms_by_block.items()
        },
        "gate_pass": (
            summarize(col("cosine_g_cache_vjp_vs_direct"))["median"] > 0.999
        ),
    }
    print(f"[cache_adjoint] SUMMARY: {json.dumps(summary, indent=2)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_batch": per_batch}, f, indent=2)
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
