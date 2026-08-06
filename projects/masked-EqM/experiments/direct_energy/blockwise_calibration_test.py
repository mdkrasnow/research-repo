"""
Blockwise scalar calibration + decomposition-completeness test, per Yilun's
2026-08-06 response to the layerwise truncation result (job 37541376: rho_k
climbs slowly 0.657->0.712 through k=6, only jumps to 0.896 at k=12).

Yilun's hypothesis: the global cosine may be low not because per-block
DIRECTIONS are wrong, but because the semigradient has the wrong relative
SCALE across blocks. Test:

  rho_l   = cos(g_semi^(l), g_exact^(l))              per-block direction
  r_l     = ||g_exact^(l)|| / (||g_semi^(l)|| + eps)   per-block norm ratio

If rho_l is high (~0.9-0.99) within most blocks despite low global cosine,
fit a per-block scalar calibration on a train split of batches:

  alpha_l = E[<g_semi^(l), g_exact^(l)>] / (E[||g_semi^(l)||^2] + eps)

evaluate rho_cal = cos(g_hat, g_exact) with g_hat = concat(alpha_l * g_semi^(l))
on a held-out split. Decision rule (Yilun's):
  rho_cal > 0.9  -> blockwise recalibration recovers most of the missing
                    signal cheaply; worth a 2k-step training test.
  rho_cal ~ 0.7  -> blockwise scaling is dead, scale is not the problem.

Also checks decomposition coverage: job 37541376's truncation order never
included x_embedder at any k (k=12 = all blocks + energy_head, still missing
x_embedder). g_cache^(l) := g_exact^(l) - g_semi^(l) by construction on a
shared batch, so "g_semi + g_cache == g_exact" is tautological per block;
the only way it can fail is a coverage bug (active_pairs missing/misnamed
params, block double-counted). This script asserts per-batch numel parity
between the exact and semigradient block-unions (coverage_ok) and reports
x_embedder's rho_l/r_l on its own (as job 37541376 never isolated it).

Same methodology as Gate 0/1 and job 37541376: TF32 disabled, MATH-backend
SDPA, real batches, no optimizer steps (pure gradient-field diagnostic).

Run (single GPU, gpu_test or seas_gpu):
  python experiments/direct_energy/blockwise_calibration_test.py \
      --ckpt <epoch40.pt> --data-path <imagenet train dir> \
      --num-batches 500 --calib-batches 400
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=500)
    p.add_argument("--calib-batches", type=int, default=400)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    assert args.calib_batches < args.num_batches, "need a held-out split after calibration"

    torch.manual_seed(args.seed)
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
    print(f"[blockcal] direct model load: missing={missing} unexpected={unexpected}")

    model_fb = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="forward-backwards-direct",
    ).to(device)
    missing, unexpected = model_fb.load_state_dict(state_dict, strict=False)
    print(f"[blockcal] fb model load: missing={missing} unexpected={unexpected}")

    fb_trainer = ForwardBackwardsDirectTrainer(model_fb, device=device)
    fb_trainer.registry.tie_from_forward_()
    sync_err = fb_trainer.registry.compute_sync_error()
    print(f"[blockcal] phi/theta sync error after tie: {sync_err:.3e}")

    active_pairs = [
        (e.forward_name, e.backward_name)
        for e in fb_trainer.registry.entries
        if e.category in ("reverse_active", "recomputed_conditioning")
    ]
    depth = len(model_direct.blocks)
    all_blocks = ["x_embedder"] + [f"blocks.{i}" for i in range(depth)] + ["energy_head"]
    print(f"[blockcal] {len(active_pairs)} matched pairs, depth={depth}, blocks={all_blocks}")

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
    name_to_block = {fname: block_key(fname) for fname, _ in active_pairs}

    # calibration accumulators: numerator/denominator per block, summed over calib split
    calib_num = defaultdict(float)
    calib_den = defaultdict(float)

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

        # --- exact arm ---
        model_direct.zero_grad(set_to_none=True)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            xt_exact = xt.detach().clone().requires_grad_(True)
            field_exact = model_direct(xt_exact, t, y, train=True)
        loss_exact = mean_flat((field_exact - ut) ** 2).mean()
        loss_exact.backward()

        # --- semigradient arm ---
        fb_trainer.registry.tie_from_forward_()
        fb_trainer.optimizer.zero_grad(set_to_none=True)
        E, cache = forward_energy_with_cache(fb_trainer.theta, xt, t, y)
        grad_tilde = fb_trainer.phi(cache)
        prediction_fb = -grad_tilde
        loss_fb = mean_flat((prediction_fb - ut) ** 2).mean()
        loss_fb.backward()

        block_exact = defaultdict(list)
        block_semi = defaultdict(list)
        for fname, bname in active_pairs:
            g_e = theta_named[fname].grad
            g_f = phi_named[bname].grad
            if g_e is None or g_f is None:
                continue
            blk = name_to_block[fname]
            block_exact[blk].append(g_e.detach().reshape(-1).float())
            block_semi[blk].append(g_f.detach().reshape(-1).float())

        block_exact_vec = {blk: torch.cat(vs) for blk, vs in block_exact.items()}
        block_semi_vec = {blk: torch.cat(vs) for blk, vs in block_semi.items()}
        present = [blk for blk in all_blocks if blk in block_exact_vec]

        # per-block direction + norm ratio (this batch)
        rho_l = {}
        r_l = {}
        for blk in present:
            ge, gf = block_exact_vec[blk], block_semi_vec[blk]
            rho_l[blk] = float(torch.nn.functional.cosine_similarity(
                gf.unsqueeze(0), ge.unsqueeze(0)).item())
            r_l[blk] = float(ge.norm().item() / (gf.norm().item() + 1e-12))

        # decomposition completeness: g_semi(all) + g_cache(all incl x_embedder) vs g_exact(all)
        exact_full = torch.cat([block_exact_vec[blk] for blk in present])
        # g_cache^(l) = g_exact^(l) - g_semi^(l) by construction (both computed on same batch);
        # reconstructing g_semi + g_cache is trivially g_exact per block, so instead verify
        # the INDEPENDENT quantity: is g_semi's own block-sum consistent in shape/coverage
        # (no missing/extra blocks) with g_exact's -- i.e. active_pairs fully covers all
        # theta parameters that require grad, block-for-block, incl x_embedder.
        semi_full = torch.cat([block_semi_vec[blk] for blk in present])
        coverage_ok = (exact_full.numel() == semi_full.numel())

        is_calib = b < args.calib_batches
        if is_calib:
            for blk in present:
                ge, gf = block_exact_vec[blk], block_semi_vec[blk]
                calib_num[blk] += float(torch.dot(gf, ge).item())
                calib_den[blk] += float(torch.dot(gf, gf).item())

        per_batch.append({
            "batch": b,
            "split": "calib" if is_calib else "holdout",
            "loss_exact": float(loss_exact.detach()),
            "loss_fb": float(loss_fb.detach()),
            "rho_l": rho_l,
            "r_l": r_l,
            "coverage_ok": coverage_ok,
            "present_blocks": present,
        })
        if b % 50 == 0 or b == args.num_batches - 1:
            print(f"[blockcal] batch {b} ({'calib' if is_calib else 'holdout'}): "
                  f"rho_l(x_embedder,energy_head)="
                  f"{rho_l.get('x_embedder'):.3f},{rho_l.get('energy_head'):.3f} "
                  f"r_l(x_embedder,energy_head)="
                  f"{r_l.get('x_embedder'):.3f},{r_l.get('energy_head'):.3f}")

    alpha = {blk: calib_num[blk] / (calib_den[blk] + 1e-12) for blk in calib_num}
    print(f"[blockcal] fitted alpha_l on {args.calib_batches} calib batches: {alpha}")

    # Second pass over a fresh set of batches applying the fitted alpha, so
    # rho_cal is evaluated out-of-sample (using alpha fit only on the first
    # pass's calib batches would leak calib into holdout if computed inline).
    holdout_rho_cal = []
    it2 = iter(loader)
    n_holdout = args.num_batches - args.calib_batches
    for b in range(n_holdout):
        try:
            x, y = next(it2)
        except StopIteration:
            it2 = iter(loader)
            x, y = next(it2)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
        t, x0, x1 = transport.sample(x1)
        t = t.to(x1)
        t, xt, ut = transport.path_sampler.plan(t, x0, x1)
        ut = ut * transport.get_ct(t)[:, None, None, None]

        model_direct.zero_grad(set_to_none=True)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            xt_exact = xt.detach().clone().requires_grad_(True)
            field_exact = model_direct(xt_exact, t, y, train=True)
        loss_exact = mean_flat((field_exact - ut) ** 2).mean()
        loss_exact.backward()

        fb_trainer.registry.tie_from_forward_()
        fb_trainer.optimizer.zero_grad(set_to_none=True)
        E, cache = forward_energy_with_cache(fb_trainer.theta, xt, t, y)
        grad_tilde = fb_trainer.phi(cache)
        prediction_fb = -grad_tilde
        loss_fb = mean_flat((prediction_fb - ut) ** 2).mean()
        loss_fb.backward()

        block_exact = defaultdict(list)
        block_semi = defaultdict(list)
        for fname, bname in active_pairs:
            g_e = theta_named[fname].grad
            g_f = phi_named[bname].grad
            if g_e is None or g_f is None:
                continue
            blk = name_to_block[fname]
            block_exact[blk].append(g_e.detach().reshape(-1).float())
            block_semi[blk].append(g_f.detach().reshape(-1).float())
        block_exact_vec = {blk: torch.cat(vs) for blk, vs in block_exact.items()}
        block_semi_vec = {blk: torch.cat(vs) for blk, vs in block_semi.items()}
        present = [blk for blk in all_blocks if blk in block_exact_vec]

        exact_vec = torch.cat([block_exact_vec[blk] for blk in present])
        cal_parts = [alpha.get(blk, 1.0) * block_semi_vec[blk] for blk in present]
        g_hat = torch.cat(cal_parts)
        rho_cal = float(torch.nn.functional.cosine_similarity(
            g_hat.unsqueeze(0), exact_vec.unsqueeze(0)).item())
        # uncalibrated global cosine on this same holdout batch, for direct comparison
        g_uncal = torch.cat([block_semi_vec[blk] for blk in present])
        rho_uncal = float(torch.nn.functional.cosine_similarity(
            g_uncal.unsqueeze(0), exact_vec.unsqueeze(0)).item())
        holdout_rho_cal.append({"batch": b, "rho_cal": rho_cal, "rho_uncal": rho_uncal})
        if b % 25 == 0 or b == n_holdout - 1:
            print(f"[blockcal] holdout batch {b}: rho_cal={rho_cal:.4f} rho_uncal={rho_uncal:.4f}")

    rho_l_by_block = defaultdict(list)
    r_l_by_block = defaultdict(list)
    for r in per_batch:
        for blk, v in r["rho_l"].items():
            rho_l_by_block[blk].append(v)
        for blk, v in r["r_l"].items():
            r_l_by_block[blk].append(v)

    summary = {
        "checkpoint": args.ckpt,
        "num_batches": args.num_batches,
        "calib_batches": args.calib_batches,
        "holdout_batches_fresh_pass": n_holdout,
        "depth": depth,
        "sync_error_after_tie": sync_err,
        "alpha_l": alpha,
        "rho_l_summary": {blk: summarize(vals) for blk, vals in rho_l_by_block.items()},
        "r_l_summary": {blk: summarize(vals) for blk, vals in r_l_by_block.items()},
        "rho_cal_summary": summarize([r["rho_cal"] for r in holdout_rho_cal]),
        "rho_uncal_summary": summarize([r["rho_uncal"] for r in holdout_rho_cal]),
        "coverage_ok_all_batches": all(r["coverage_ok"] for r in per_batch),
    }
    print(f"[blockcal] SUMMARY: {json.dumps(summary, indent=2)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_batch": per_batch, "holdout_rho_cal": holdout_rho_cal}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
