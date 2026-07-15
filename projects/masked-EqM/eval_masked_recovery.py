# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Masked-recovery eval for structured start-state EqM checkpoints.

Given a checkpoint, corrupts held-out real images with Bernoulli masking
(same transport.corruption.mask_corrupt used in training), starts the
GD/NAG sampler from that corrupted latent instead of pure noise, and
measures how well the recovered image matches the original in the masked
region. Supports an optional hard-constrain mode (--hard-constrain) that
resets visible pixels to ground truth at every step -- this is the
ceiling/oracle arm; without it, a checkpoint's raw recovery is the
treatment/floor arm depending on which corruption_mode it was trained with.
See projects/masked-EqM/CLAUDE.md and .state/pipeline.json for the
3-arm (floor/ceiling/treatment) evaluation this feeds into.
"""
import argparse
import json
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder

from download import find_model
from models import EqM_models

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_ema_model(ckpt_path, model_name, latent_size, num_classes, uncond, ebm, device):
    model = EqM_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        uncond=uncond,
        ebm=ebm,
    ).to(device)
    ema = deepcopy(model).to(device)
    state_dict = find_model(ckpt_path)
    if "ema" in state_dict:
        ema.load_state_dict(state_dict["ema"])
    else:
        ema.load_state_dict(state_dict)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad = False
    return ema


def gd_recover(model_fn, z0, y, num_sampling_steps, stepsize, sampler, mu,
               hard_constrain=False, keep_mask=None, x1_true=None):
    """Run the GD/NAG sampler starting from z0 instead of pure noise.
    If hard_constrain, reset the visible (keep_mask==1) region back to
    x1_true after every step -- only the masked-out region is left to the
    model's own dynamics. keep_mask/x1_true required when hard_constrain."""
    xt = z0
    t = torch.ones((z0.shape[0],)).to(z0)
    m = torch.zeros_like(xt)
    for _ in range(num_sampling_steps - 1):
        if sampler == "gd":
            out = model_fn(xt, t, y)
            if not torch.is_tensor(out):
                out = out[0]
        else:  # ngd
            x_ = xt + stepsize * m * mu
            out = model_fn(x_, t, y)
            if not torch.is_tensor(out):
                out = out[0]
            m = out
        xt = xt + out * stepsize
        t = t + stepsize
        if hard_constrain:
            xt = keep_mask * x1_true + (1 - keep_mask) * xt
    return xt


def main(args):
    # seed the global RNG so keep_mask/eps below are identical across
    # separate processes (treatment vs ceiling runs) given the same --seed --
    # otherwise each run corrupts a DIFFERENT random mask/noise and the two
    # arms are not comparable at all.
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // 8

    ema = load_ema_model(args.ckpt, args.model, latent_size, args.num_classes,
                          args.uncond, args.ebm, device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    lpips_fn = None
    if _HAS_LPIPS and not args.no_lpips:
        lpips_fn = lpips.LPIPS(net="alex").to(device)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:args.num_images].tolist()

    errs = []
    lpips_errs = []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            xs, ys = zip(*(dataset[i] for i in batch_idx))
            x = torch.stack(xs).to(device)
            y = torch.tensor(ys).to(device)

            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
            keep_mask = (torch.rand_like(x1[:, :1]) > args.mask_prob).float()
            eps = torch.randn_like(x1)
            z0 = keep_mask * x1 + (1 - keep_mask) * eps

            if args.vae_roundtrip_oracle:
                # positive control: skip masking/model entirely, just measure
                # the VAE's own encode->decode reconstruction floor -- the
                # best achievable score through this pipeline with a
                # hypothetically perfect model.
                xt = x1
            else:
                xt = gd_recover(
                    ema, z0, y, args.num_sampling_steps, args.stepsize,
                    args.sampler, args.mu,
                    hard_constrain=args.hard_constrain, keep_mask=keep_mask, x1_true=x1,
                )

            recovered = vae.decode(xt / 0.18215).sample
            masked_region = (1 - keep_mask)
            # upsample latent-space mask to pixel space (8x downsample factor)
            masked_region_px = F.interpolate(masked_region, size=x.shape[-2:], mode="nearest")
            err = ((recovered - x) ** 2 * masked_region_px).sum(dim=[1, 2, 3]) / \
                  (masked_region_px.sum(dim=[1, 2, 3]) * x.shape[1] + 1e-8)
            errs.extend(err.cpu().tolist())

            if lpips_fn is not None:
                # full-image perceptual distance (LPIPS is not naturally
                # maskable pixelwise -- same convention as
                # eval_fourier_recovery.py / eval_downsample_recovery.py)
                lp = lpips_fn(recovered.clamp(-1, 1), x.clamp(-1, 1)).flatten()
                lpips_errs.extend(lp.cpu().tolist())

    result = {
        "ckpt": args.ckpt,
        "num_images": len(errs),
        "mask_prob": args.mask_prob,
        "hard_constrain": args.hard_constrain,
        "has_lpips": lpips_fn is not None,
        "mean_masked_mse": sum(errs) / len(errs),
        "mean_lpips": (sum(lpips_errs) / len(lpips_errs)) if lpips_errs else None,
        "per_image_mse": errs,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"mean_masked_mse={result['mean_masked_mse']:.5f} mean_lpips={result['mean_lpips']} "
          f"over {len(errs)} images -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True,
                         help="ImageFolder-format held-out data (e.g. imagenet val)")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--uncond", type=bool, default=True)
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none")
    parser.add_argument("--stepsize", type=float, default=0.0017)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--sampler", type=str, default="gd", choices=["gd", "ngd"])
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--mask-prob", type=float, default=0.5,
                         help="fraction of latent masked out for the recovery test")
    parser.add_argument("--hard-constrain", action="store_true",
                         help="ceiling/oracle mode: reset visible pixels to ground truth every step")
    parser.add_argument("--vae-roundtrip-oracle", action="store_true",
                         help="positive control: skip masking/model, just measure VAE encode->decode floor")
    parser.add_argument("--no-lpips", action="store_true", help="skip LPIPS even if installed")
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="eval_results/masked_recovery.json")
    args = parser.parse_args()
    main(args)
