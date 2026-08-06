"""
Layerwise decomposition of the forward-backwards-direct semigradient's
missing credit-assignment term, per Yilun's proposed next experiment
(2026-08-06, in response to the Gate 2 divergence finding).

Exact parameter gradient decomposes as:

    g_exact = g_semi + g_cache

where g_semi is what forward-backwards-direct actually computes (single
backward through phi, mapped through Pi into theta-parameter-name space)
and g_cache = (dC_theta/dtheta)^T dL/dC is the credit-assignment term the
semigradient method drops by treating the forward cache C as a detached
leaf (see fb_direct/trainer.py training_step's docstring).

g_cache is itself a sum over blocks: g_cache = sum_l g_cache^(l). This
script measures, for truncated corrections

    g^(k) = g_semi + sum_{l in top-k blocks nearest the output} g_cache^(l)

the cosine rho_k = cos(g^(k), g_exact) for k in {0, 1, 2, 4, 6, 12}, to
find how much of the exact second-order graph is actually needed for good
credit assignment (Yilun's question: "how little of the exact second-order
credit-assignment pathway can we retain").

Block order (nearest-to-output first, since g_cache is hypothesized to be
concentrated near the loss): energy_head, blocks.{depth-1}, ..., blocks.0,
x_embedder. energy_head and x_embedder are reported separately from
"blocks.i" (unlike Gate 1's `block_of`, which lumped both into "other") --
that lumping is fine for a norm-only report but wrong here, since energy_head
sits right next to the loss and x_embedder sits at the very start of the
forward pass, and the truncation order depends on knowing which is which.

No optimizer steps on either arm -- pure gradient-field diagnostic, same
methodology (TF32 disabled, MATH-backend SDPA, same real batches for both
arms) as Gate 0/1/exact_gradient_audit.

Run (single GPU, gpu_test or seas_gpu):
  python experiments/direct_energy/layerwise_gradient_decomposition.py \
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

TRUNCATION_KS = [0, 1, 2, 4, 6, 12]


def mean_flat(x):
    return x.mean(dim=list(range(1, len(x.shape))))


def block_key(name):
    # "blocks.7.attn.qkv.weight" -> "blocks.7"; energy_head.* -> "energy_head";
    # x_embedder.* -> "x_embedder". Every active_pairs forward_name is one of
    # these three prefixes for this architecture (param_mapping.py).
    if name.startswith("blocks."):
        return f"blocks.{name.split('.')[1]}"
    if name.startswith("energy_head."):
        return "energy_head"
    if name.startswith("x_embedder."):
        return "x_embedder"
    raise ValueError(f"unrecognized active parameter name for block_key: {name}")


def build_truncation_order(depth):
    # nearest-output-first: energy_head, blocks.{depth-1}, ..., blocks.0, x_embedder
    order = ["energy_head"] + [f"blocks.{i}" for i in range(depth - 1, -1, -1)] + ["x_embedder"]
    return order


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
    print(f"[layerwise] direct model load: missing={missing} unexpected={unexpected}")

    model_fb = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="forward-backwards-direct",
    ).to(device)
    missing, unexpected = model_fb.load_state_dict(state_dict, strict=False)
    print(f"[layerwise] fb model load: missing={missing} unexpected={unexpected}")

    fb_trainer = ForwardBackwardsDirectTrainer(model_fb, device=device)
    fb_trainer.registry.tie_from_forward_()
    sync_err = fb_trainer.registry.compute_sync_error()
    print(f"[layerwise] phi/theta sync error after tie: {sync_err:.3e}")

    active_pairs = [
        (e.forward_name, e.backward_name)
        for e in fb_trainer.registry.entries
        if e.category in ("reverse_active", "recomputed_conditioning")
    ]
    depth = len(model_direct.blocks)
    trunc_order = build_truncation_order(depth)
    print(f"[layerwise] {len(active_pairs)} matched pairs, depth={depth}, "
          f"truncation order={trunc_order}")

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

        # --- per-block flat vectors, in theta-name space (mapped 1:1 since
        # every mapping_type here is "identity", so shapes already match) ---
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

        present_order = [blk for blk in trunc_order if blk in block_exact_vec]
        exact_vec = torch.cat([block_exact_vec[blk] for blk in present_order])

        # g^(k) built per-block: for the first k blocks (nearest output, per
        # present_order) use the EXACT block gradient (i.e. include that
        # block's g_cache correction in full); for the remaining blocks use
        # only the semigradient's own block gradient. Concatenated in the
        # same order as exact_vec so cosine is well-defined.
        rho_k = {}
        for k in TRUNCATION_KS:
            g_k_parts = [
                block_exact_vec[blk] if i < k else block_semi_vec[blk]
                for i, blk in enumerate(present_order)
            ]
            g_k = torch.cat(g_k_parts)
            rho_k[k] = float(
                torch.nn.functional.cosine_similarity(
                    g_k.unsqueeze(0), exact_vec.unsqueeze(0)
                ).item()
            )

        per_batch.append({
            "batch": b,
            "loss_exact": float(loss_exact.detach()),
            "loss_fb": float(loss_fb.detach()),
            "rho_k": rho_k,
        })
        if b % 50 == 0 or b == args.num_batches - 1:
            print(f"[layerwise] batch {b}: rho_k={rho_k}")

    agg = {}
    for k in TRUNCATION_KS:
        vals = sorted(r["rho_k"].get(k) for r in per_batch if k in r["rho_k"])
        vals = [v for v in vals if v is not None]
        agg[k] = {
            "median_rho": percentile(vals, 50),
            "p10_rho": percentile(vals, 10),
            "p90_rho": percentile(vals, 90),
            "n": len(vals),
        }
    summary = {
        "checkpoint": args.ckpt,
        "num_batches": args.num_batches,
        "depth": depth,
        "truncation_order": trunc_order,
        "truncation_ks": TRUNCATION_KS,
        "rho_k_summary": agg,
        "sync_error_after_tie": sync_err,
    }
    print(f"[layerwise] SUMMARY: {json.dumps(summary, indent=2)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_batch": per_batch}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
