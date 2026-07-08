# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Cross-corruption generalization eval for structured start-state EqM
checkpoints.

Goal: does a gaussian+mask-trained model learn a genuinely more general
repair field, or has it just memorized the two exact corruption
distributions (Bernoulli mask p=0.5, pure Gaussian) it was trained on?
Tests each checkpoint against corruptions it was NOT trained on: unseen
mask severities, block masks, stroke masks, partial Gaussian noise, and a
composed noisy+masked corruption (the key test -- this is neither the
pure Gaussian task nor the pure mask task).

For each corruption, reports:
- recovery MSE (masked-region MSE for mask-like corruptions, full-image
  MSE for noise-only) via the same GD/NAG sampler as eval_masked_recovery.py
- field-norm ordering (clean vs this corruption vs pure noise), same proxy
  as eval_energy_ordering.py, to check whether the field stays calibrated
  off the exact training distribution.

Single script covering all corruption types so every checkpoint gets
identical RNG-matched inputs across arms (critical for cross-arm
comparison -- see eval_masked_recovery.py's RNG-seeding fix history).
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------
# Corruption functions (eval-time only -- NOT used in training, unlike
# transport/corruption.py). Each returns (z0, keep_mask_or_None).
# --------------------------------------------------------------------------

def bernoulli_mask_corrupt(x1, mask_prob):
    keep_mask = (torch.rand_like(x1[:, :1]) > mask_prob).float()
    eps = torch.randn_like(x1)
    z0 = keep_mask * x1 + (1 - keep_mask) * eps
    return z0, keep_mask


def block_mask_corrupt(x1, mask_prob):
    """Single contiguous square block per sample, sized to cover ~mask_prob
    fraction of the latent area, placed at a random offset -- real
    inpainting-style occlusion, not random pixel dropout."""
    b, _, h, w = x1.shape
    side = max(1, int(round((mask_prob ** 0.5) * min(h, w))))
    keep_mask = torch.ones((b, 1, h, w), device=x1.device, dtype=x1.dtype)
    for i in range(b):
        top = torch.randint(0, max(1, h - side + 1), (1,)).item()
        left = torch.randint(0, max(1, w - side + 1), (1,)).item()
        keep_mask[i, :, top:top + side, left:left + side] = 0.0
    eps = torch.randn_like(x1)
    z0 = keep_mask * x1 + (1 - keep_mask) * eps
    return z0, keep_mask


def stroke_mask_corrupt(x1, num_strokes=6, thickness=2, steps=12):
    """Irregular occlusion: per sample, a handful of short random-walk
    strokes drawn onto the mask -- closer to realistic occlusion shape
    than a single rectangular block."""
    b, _, h, w = x1.shape
    keep_mask = torch.ones((b, 1, h, w), device=x1.device, dtype=x1.dtype)
    for i in range(b):
        for _ in range(num_strokes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            for _ in range(steps):
                dy = torch.randint(-2, 3, (1,)).item()
                dx = torch.randint(-2, 3, (1,)).item()
                y = min(max(y + dy, 0), h - 1)
                x = min(max(x + dx, 0), w - 1)
                y0, y1 = max(0, y - thickness), min(h, y + thickness + 1)
                x0, x1_ = max(0, x - thickness), min(w, x + thickness + 1)
                keep_mask[i, :, y0:y1, x0:x1_] = 0.0
    eps = torch.randn_like(x1)
    z0 = keep_mask * x1 + (1 - keep_mask) * eps
    return z0, keep_mask


def gaussian_noise_corrupt(x1, sigma):
    """Partial additive Gaussian noise (denoising task at a given severity,
    not full replacement) -- no keep_mask, full-image recovery metric."""
    eps = torch.randn_like(x1)
    z0 = x1 + sigma * eps
    return z0, None


def noisy_masked_corrupt(x1, mask_prob, sigma):
    """Composed corruption: neither the pure mask task nor the pure noise
    task -- masked-out region is pure noise, VISIBLE region also gets
    additive noise. This is the key compositional-generalization test."""
    keep_mask = (torch.rand_like(x1[:, :1]) > mask_prob).float()
    eps_visible = torch.randn_like(x1)
    eps_masked = torch.randn_like(x1)
    visible_noisy = x1 + sigma * eps_visible
    z0 = keep_mask * visible_noisy + (1 - keep_mask) * eps_masked
    return z0, keep_mask


# --------------------------------------------------------------------------

def load_ema_model(ckpt_path, model_name, latent_size, num_classes, uncond, ebm, device):
    model = EqM_models[model_name](
        input_size=latent_size, num_classes=num_classes, uncond=uncond, ebm=ebm,
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


def gd_recover(model_fn, z0, y, num_sampling_steps, stepsize):
    xt = z0
    t = torch.zeros((z0.shape[0],)).to(z0)
    for _ in range(num_sampling_steps - 1):
        out = model_fn(xt, t, y)
        if not torch.is_tensor(out):
            out = out[0]
        xt = xt + out * stepsize
        t = t + stepsize
    return xt


def field_norm(model, x, y):
    t = torch.zeros((x.shape[0],), device=x.device)
    with torch.no_grad():
        out = model(x, t, y)
        if not torch.is_tensor(out):
            out = out[0]
    return out.flatten(1).norm(dim=1).mean().item()


def recovery_mse(recovered, x, keep_mask):
    if keep_mask is None:
        return ((recovered - x) ** 2).mean(dim=[1, 2, 3])
    masked_region_px = F.interpolate(1 - keep_mask, size=x.shape[-2:], mode="nearest")
    return ((recovered - x) ** 2 * masked_region_px).sum(dim=[1, 2, 3]) / \
        (masked_region_px.sum(dim=[1, 2, 3]) * x.shape[1] + 1e-8)


CORRUPTIONS = {
    "mask_p0.25": lambda x1: bernoulli_mask_corrupt(x1, 0.25),
    "mask_p0.75": lambda x1: bernoulli_mask_corrupt(x1, 0.75),
    "mask_p0.9": lambda x1: bernoulli_mask_corrupt(x1, 0.9),
    "block_mask": lambda x1: block_mask_corrupt(x1, 0.5),
    "stroke_mask": lambda x1: stroke_mask_corrupt(x1),
    "gaussian_sigma0.3": lambda x1: gaussian_noise_corrupt(x1, 0.3),
    "gaussian_sigma0.6": lambda x1: gaussian_noise_corrupt(x1, 0.6),
    "gaussian_sigma1.0": lambda x1: gaussian_noise_corrupt(x1, 1.0),
    "noisy_masked": lambda x1: noisy_masked_corrupt(x1, 0.5, 0.4),
}


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // 8

    ema = load_ema_model(args.ckpt, args.model, latent_size, args.num_classes,
                          args.uncond, args.ebm, device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:args.num_images].tolist()

    results = {name: {"mse": [], "field_clean": [], "field_corrupt": [], "field_noise": []}
               for name in CORRUPTIONS}

    for start in range(0, len(indices), args.batch_size):
        batch_idx = indices[start:start + args.batch_size]
        xs, ys = zip(*(dataset[i] for i in batch_idx))
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys).to(device)

        with torch.no_grad():
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)

        for name, corrupt_fn in CORRUPTIONS.items():
            torch.manual_seed(args.seed + hash(name) % 100000 + start)
            z0, keep_mask = corrupt_fn(x1)
            noise = torch.randn_like(x1)

            with torch.no_grad():
                recovered_latent = gd_recover(ema, z0, y, args.num_sampling_steps, args.stepsize)
                recovered = vae.decode(recovered_latent / 0.18215).sample
                mse = recovery_mse(recovered, x, keep_mask)
                results[name]["mse"].extend(mse.cpu().tolist())
                results[name]["field_clean"].append(field_norm(ema, x1, y))
                results[name]["field_corrupt"].append(field_norm(ema, z0, y))
                results[name]["field_noise"].append(field_norm(ema, noise, y))

    summary = {"ckpt": args.ckpt, "num_images": len(indices), "corruptions": {}}
    for name, r in results.items():
        clean = sum(r["field_clean"]) / len(r["field_clean"])
        corrupt = sum(r["field_corrupt"]) / len(r["field_corrupt"])
        noise = sum(r["field_noise"]) / len(r["field_noise"])
        summary["corruptions"][name] = {
            "mean_recovery_mse": sum(r["mse"]) / len(r["mse"]),
            "field_norm_clean": clean,
            "field_norm_corrupt": corrupt,
            "field_norm_noise": noise,
            "ordering_holds": clean < corrupt < noise,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    for name, r in summary["corruptions"].items():
        print(f"{name}: mse={r['mean_recovery_mse']:.5f} ordering_holds={r['ordering_holds']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--uncond", type=bool, default=True)
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none")
    parser.add_argument("--stepsize", type=float, default=0.0017)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="eval_results/generalization.json")
    args = parser.parse_args()
    main(args)
