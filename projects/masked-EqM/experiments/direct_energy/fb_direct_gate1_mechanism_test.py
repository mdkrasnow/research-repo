"""
Gate 1 (Experimental Decision Tree: Forward-Backward Scalar EqM):
offline mechanism test. At the epoch-80 direct-scalar checkpoint (the
FID-39.94 paper-scale checkpoint), compute g_exact = grad_theta L_exact
(the real double-backward direct-scalar training gradient) vs
g_FB = mapped grad_phi L_FB (the forward-backwards-direct semigradient,
mapped through Pi's identity correspondence into theta-parameter-name
space) on the SAME real batches (same xt/t/y/ut per batch, both arms).
NO optimizer steps are taken on either arm -- this is a pure gradient-field
diagnostic.

Per-batch metrics: cosine(g_FB, g_exact) [global + per-block], grad-norm
[global + per-block] for both arms. Across the run: median cosine, fraction
of batches with cosine>0, and the FB/exact gradient-norm ratio at the
p99.9 percentile (tail risk -- explosion signature) and on the top-1%
highest-exact-norm batches observed IN THIS RUN, used as an in-run proxy
for "replayed explosion batches" (the actual historical backbone-gradient-
explosion batches from the original direct-scalar training run occurred at
specific, different training steps/data orderings under Adam-optimizer
drift that this static-weight offline test cannot literally replay; the
proxy is the closest honest approximation available without re-running the
full optimizer trajectory -- see documentation/forward-backwards-direct.md
Gate 1 section for the caveat).

Advancement criteria (decision tree text, verbatim):
  - median cos > 0.2
  - >= 80% of batches have cos > 0
  - FB p99.9 gradient norm is at least 5x lower than exact's
  - On the proxy high-stress batches, FB gradients are at least 10x smaller

Run (single GPU, gpu_test or seas_gpu):
  python experiments/direct_energy/fb_direct_gate1_mechanism_test.py \
      --ckpt <epoch80.pt> --data-path <imagenet train dir> --num-batches 500
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
from fb_direct.forward_cache import forward_energy_with_cache

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def mean_flat(x):
    return x.mean(dim=list(range(1, len(x.shape))))


def block_of(name):
    # e.g. "blocks.7.attn.qkv.weight" -> "blocks.7"; everything else -> "other"
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        return f"blocks.{parts[1]}"
    return "other"


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=500)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    # Root cause of Gate 0's initial false-failure was TF32 matmul precision
    # (proven via FP64 bisection to be unrelated to the manual VJP -- see
    # documentation/forward-backwards-direct.md). Every gate script that
    # compares theta-space gradients against the FB semigradient must
    # disable TF32 globally, matching the methodology that made Gate 0 pass
    # (job 37522678), or this script will reproduce the same false signal.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // 8

    raw = torch.load(args.ckpt, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw

    model_direct = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="direct",
    ).to(device)
    missing, unexpected = model_direct.load_state_dict(state_dict, strict=False)
    print(f"[gate1] direct model load: missing={missing} unexpected={unexpected}")

    model_fb = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="forward-backwards-direct",
    ).to(device)
    missing, unexpected = model_fb.load_state_dict(state_dict, strict=False)
    print(f"[gate1] fb model load: missing={missing} unexpected={unexpected}")

    fb_trainer = ForwardBackwardsDirectTrainer(model_fb, device=device)
    fb_trainer.registry.tie_from_forward_()
    sync_err = fb_trainer.registry.compute_sync_error()
    print(f"[gate1] phi/theta sync error after tie: {sync_err:.3e}")

    # Active (reverse_active + recomputed_conditioning) forward<->backward
    # name pairs, in a fixed order, used to build matched gradient vectors
    # for both arms every batch.
    active_pairs = [
        (e.forward_name, e.backward_name)
        for e in fb_trainer.registry.entries
        if e.category in ("reverse_active", "recomputed_conditioning")
    ]
    print(f"[gate1] {len(active_pairs)} matched (theta,phi) parameter pairs "
          f"used for gradient comparison")

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

    theta_named = dict(model_direct.named_parameters())
    phi_named = dict(fb_trainer.phi.named_parameters())

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

        # --- control: g_exact, real double-backward direct training step ---
        model_direct.zero_grad(set_to_none=True)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            xt_exact = xt.detach().clone().requires_grad_(True)
            field_exact = model_direct(xt_exact, t, y, train=True)
        loss_exact = mean_flat((field_exact - ut) ** 2).mean()
        loss_exact.backward()

        # --- treatment: g_FB, single-backward semigradient, NO opt.step() ---
        fb_trainer.registry.tie_from_forward_()
        fb_trainer.optimizer.zero_grad(set_to_none=True)
        E, cache = forward_energy_with_cache(fb_trainer.theta, xt, t, y)
        grad_tilde = fb_trainer.phi(cache)
        prediction_fb = -grad_tilde
        loss_fb = mean_flat((prediction_fb - ut) ** 2).mean()
        loss_fb.backward()

        # --- matched gradient vectors, global + per-block ---
        exact_flat, fb_flat = [], []
        block_exact = defaultdict(list)
        block_fb = defaultdict(list)
        for fname, bname in active_pairs:
            g_e = theta_named[fname].grad
            g_f = phi_named[bname].grad
            if g_e is None or g_f is None:
                continue
            g_e = g_e.detach().reshape(-1).float()
            g_f = g_f.detach().reshape(-1).float()
            exact_flat.append(g_e)
            fb_flat.append(g_f)
            blk = block_of(fname)
            block_exact[blk].append(g_e)
            block_fb[blk].append(g_f)

        exact_vec = torch.cat(exact_flat)
        fb_vec = torch.cat(fb_flat)
        exact_norm = float(exact_vec.norm())
        fb_norm = float(fb_vec.norm())
        cosine = float(torch.nn.functional.cosine_similarity(exact_vec.unsqueeze(0), fb_vec.unsqueeze(0)).item())

        per_block = {}
        for blk in block_exact:
            be = torch.cat(block_exact[blk]).norm().item()
            bf = torch.cat(block_fb[blk]).norm().item()
            per_block[blk] = {"exact_norm": be, "fb_norm": bf}

        rec = {
            "batch": b,
            "exact_norm": exact_norm,
            "fb_norm": fb_norm,
            "cosine": cosine,
            "loss_exact": float(loss_exact.detach()),
            "loss_fb": float(loss_fb.detach()),
            "per_block": per_block,
        }
        per_batch.append(rec)
        if b % 50 == 0 or b == args.num_batches - 1:
            print(f"[gate1] batch {b}: cos={cosine:.4f} exact_norm={exact_norm:.4g} fb_norm={fb_norm:.4g}")

    cosines = sorted(r["cosine"] for r in per_batch)
    exact_norms = sorted(r["exact_norm"] for r in per_batch)
    fb_norms_matched = [r["fb_norm"] for r in per_batch]

    median_cos = percentile(cosines, 50)
    frac_pos = sum(1 for c in cosines if c > 0) / len(cosines)
    p999_exact = percentile(exact_norms, 99.9)
    # FB norm at the SAME batches used for p999_exact percentile index (tail of exact-norm distribution)
    order = sorted(range(len(per_batch)), key=lambda i: per_batch[i]["exact_norm"])
    p999_idx = min(len(order) - 1, max(0, int(round(0.999 * (len(order) - 1)))))
    p999_batch = per_batch[order[p999_idx]]
    p999_fb_over_exact_ratio_reduction = (
        p999_batch["exact_norm"] / p999_batch["fb_norm"] if p999_batch["fb_norm"] > 0 else float("inf")
    )

    # proxy "explosion batches": top-1% by exact grad norm within this run
    n_top = max(1, int(round(0.01 * len(per_batch))))
    top_batches = [per_batch[i] for i in order[-n_top:]]
    top_ratios = [
        (rb["exact_norm"] / rb["fb_norm"] if rb["fb_norm"] > 0 else float("inf"))
        for rb in top_batches
    ]
    proxy_min_ratio = min(top_ratios)
    proxy_mean_ratio = sum(r for r in top_ratios if r != float("inf")) / max(
        1, sum(1 for r in top_ratios if r != float("inf"))
    )

    criteria = {
        "median_cosine_gt_0.2": median_cos > 0.2,
        "frac_batches_cosine_gt_0_ge_0.8": frac_pos >= 0.8,
        "fb_p999_norm_ge_5x_lower": p999_fb_over_exact_ratio_reduction >= 5.0,
        "fb_ge_10x_smaller_on_proxy_explosion_batches": proxy_min_ratio >= 10.0,
    }
    overall_pass = all(criteria.values())

    summary = {
        "checkpoint": args.ckpt,
        "num_batches": args.num_batches,
        "median_cosine": median_cos,
        "frac_batches_cosine_positive": frac_pos,
        "p999_exact_norm": p999_exact,
        "p999_batch_fb_norm": p999_batch["fb_norm"],
        "p999_fb_over_exact_ratio_reduction": p999_fb_over_exact_ratio_reduction,
        "proxy_explosion_batch_count": n_top,
        "proxy_explosion_min_ratio_reduction": proxy_min_ratio,
        "proxy_explosion_mean_ratio_reduction": proxy_mean_ratio,
        "criteria": criteria,
        "gate1_overall_pass": overall_pass,
        "sync_error_after_tie": sync_err,
        "note": (
            "proxy_explosion_* uses the top-1% highest-exact-gradient-norm "
            "batches observed WITHIN THIS static-weight offline run as a "
            "stand-in for the historical training-time gradient-explosion "
            "batches (which occurred at different steps under live Adam "
            "drift and cannot be literally replayed here)."
        ),
    }
    print(f"[gate1] SUMMARY: {json.dumps(summary, indent=2)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_batch": per_batch}, f, indent=2)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
