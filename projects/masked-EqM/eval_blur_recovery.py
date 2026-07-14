# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Blur-recovery eval for structured start-state EqM checkpoints (blur
corruption family, added 2026-07-14 to test whether the masking result
generalizes to a second structured-start-state corruption).

Given a checkpoint, corrupts held-out real images with Gaussian blur
(same transport.corruption.blur_corrupt used in training), starts the
GD/NAG sampler from that corrupted latent instead of pure noise, and
measures full-image recovery MSE (and LPIPS if available) vs the
original. Unlike masking, blur corrupts the WHOLE image (no held-out
visible region) so there is no keep_mask/hard-constrain ceiling arm here
-- the only positive control is --vae-roundtrip-oracle (VAE encode/decode
floor). Works zero-shot on ANY checkpoint regardless of what corruption
family it was trained on (pass --sigma-grid to sweep severities).
"""
import argparse
import json
import os
from copy import deepcopy

import torch

from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder

from download import find_model
from models import EqM_models
from transport.corruption import blur_corrupt

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


def gd_recover(model_fn, z0, y, num_sampling_steps, stepsize, sampler, mu):
    """Run the GD/NAG sampler starting from z0 instead of pure noise.
    No hard-constrain option here -- blur corrupts the whole image, there
    is no held-out visible region to reset each step."""
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
    return xt


def main(args):
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

    sigmas = [float(s) for s in args.sigma_grid.split(",")] if args.sigma_grid else [args.sigma]

    per_sigma = {}
    with torch.no_grad():
        for sigma in sigmas:
            mse_errs, lpips_errs = [], []
            for start in range(0, len(indices), args.batch_size):
                batch_idx = indices[start:start + args.batch_size]
                xs, ys = zip(*(dataset[i] for i in batch_idx))
                x = torch.stack(xs).to(device)
                y = torch.tensor(ys).to(device)

                x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
                z0 = blur_corrupt(x1, sigma)

                if args.vae_roundtrip_oracle:
                    xt = x1
                else:
                    xt = gd_recover(
                        ema, z0, y, args.num_sampling_steps, args.stepsize,
                        args.sampler, args.mu,
                    )

                recovered = vae.decode(xt / 0.18215).sample
                err = ((recovered - x) ** 2).mean(dim=[1, 2, 3])
                mse_errs.extend(err.cpu().tolist())

                if lpips_fn is not None:
                    lp = lpips_fn(recovered.clamp(-1, 1), x.clamp(-1, 1)).flatten()
                    lpips_errs.extend(lp.cpu().tolist())

            per_sigma[str(sigma)] = {
                "mean_mse": sum(mse_errs) / len(mse_errs),
                "mean_lpips": (sum(lpips_errs) / len(lpips_errs)) if lpips_errs else None,
                "num_images": len(mse_errs),
            }
            print(f"sigma={sigma} mean_mse={per_sigma[str(sigma)]['mean_mse']:.5f} "
                  f"mean_lpips={per_sigma[str(sigma)]['mean_lpips']}")

    result = {
        "ckpt": args.ckpt,
        "sigmas": sigmas,
        "vae_roundtrip_oracle": args.vae_roundtrip_oracle,
        "has_lpips": lpips_fn is not None,
        "per_sigma": per_sigma,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {args.out}")


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
    parser.add_argument("--sigma", type=float, default=1.0,
                         help="single blur sigma for the recovery test (ignored if --sigma-grid set)")
    parser.add_argument("--sigma-grid", type=str, default=None,
                         help="comma-separated list of sigmas for a held-out severity grid, e.g. '0.5,1.0,2.0,4.0'")
    parser.add_argument("--vae-roundtrip-oracle", action="store_true",
                         help="positive control: skip blur/model, just measure VAE encode->decode floor")
    parser.add_argument("--no-lpips", action="store_true", help="skip LPIPS even if installed")
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="eval_results/blur_recovery.json")
    args = parser.parse_args()
    main(args)
