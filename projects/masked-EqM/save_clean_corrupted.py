"""Run ON CLUSTER (needs dataset + VAE). Saves clean + corrupted PNGs for a
fixed set of manifest indices at cutoffs 0.05/0.10/0.15, reusing the exact
same encode/corrupt determinism as eval_fourier_recovery.py (imported, not
reimplemented) so these line up pixel-for-pixel with the recovered PNGs
already saved by the severity sweep jobs.
"""
import argparse
import json
import os
import sys

import torch
from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_fourier_recovery import (
    build_or_load_manifest, deterministic_fourier_corrupt,
)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    indices, labels = build_or_load_manifest(dataset, args.manifest, 1024, 0)

    idx_set = set(json.loads(args.indices))
    cutoffs = [float(c) for c in args.cutoffs.split(",")]

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        for idx in indices:
            if idx not in idx_set:
                continue
            x, _ = dataset[idx]
            x = x.unsqueeze(0).to(device)

            clean_path = os.path.join(args.out_dir, f"clean_idx{idx}.png")
            if not os.path.exists(clean_path):
                save_image(x[0].clamp(-1, 1) * 0.5 + 0.5, clean_path)

            dist = vae.encode(x).latent_dist
            gen = torch.Generator().manual_seed(1_000_000 + idx)
            x1 = dist.mean + dist.std * torch.randn(dist.mean.shape, generator=gen).to(
                device=device, dtype=dist.mean.dtype)
            x1 = x1.mul(0.18215)

            for cutoff in cutoffs:
                corr_path = os.path.join(args.out_dir, f"corrupted_cutoff{cutoff}_idx{idx}.png")
                if os.path.exists(corr_path):
                    continue
                gen2 = torch.Generator().manual_seed(2_000_000 + idx * 1000 + int(cutoff * 1000))
                z0 = deterministic_fourier_corrupt(x1[0] / 0.18215, cutoff, gen2, device)
                z0 = (z0 * 0.18215).unsqueeze(0)
                decoded = vae.decode(z0 / 0.18215).sample
                save_image(decoded[0].clamp(-1, 1) * 0.5 + 0.5, corr_path)
            print(f"done idx={idx}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--indices", type=str, required=True, help="JSON list of manifest indices")
    p.add_argument("--cutoffs", type=str, default="0.05,0.1,0.15")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()
    main(args)
