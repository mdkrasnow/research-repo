"""
Calibrate downsample_corrupt's factor and fourier_corrupt's cutoff to match
the clean-vs-corrupted difficulty of the existing p=0.5 mask task, using the
same pixel-space LPIPS-matching method as blur (see blur-calibration.md --
raw latent MSE is not comparable across corruption families with different
value distributions, e.g. bounded-smoothing vs unbounded-noise-replacement).

Downsample: binary-search factor in [1.05, 16.0] (factor=1 -> no corruption,
larger factor -> more destructive).
Fourier: binary-search cutoff in [0.02, 1.0] (cutoff=1 -> no corruption,
smaller cutoff -> more destructive, since less of the spectrum is kept).
"""
import argparse
import torch as th
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
import lpips

from transport.corruption import mask_corrupt, downsample_corrupt, fourier_corrupt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str,
                    default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val")
    p.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-images", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mask-prob", type=float, default=0.5)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--max-iters", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    xs, latents = [], []
    n = 0
    with th.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
            xs.append(x)
            latents.append(x1)
            n += x1.shape[0]
            if n >= args.num_images:
                break
    x_clean = th.cat(xs, dim=0)[:args.num_images]
    x1 = th.cat(latents, dim=0)[:args.num_images]

    decode_bs = args.batch_size

    def decode_chunked(z0):
        outs = []
        for i in range(0, z0.shape[0], decode_bs):
            outs.append(vae.decode(z0[i:i + decode_bs] / 0.18215).sample)
        return th.cat(outs, dim=0)

    def lpips_dist(z0):
        dists = []
        for i in range(0, z0.shape[0], decode_bs):
            recon = decode_chunked(z0[i:i + decode_bs]).clamp(-1, 1)
            clean = x_clean[i:i + decode_bs].clamp(-1, 1)
            dists.append(lpips_fn(recon, clean).flatten())
        return th.cat(dists).mean().item()

    def latent_mse(z0):
        return ((z0 - x1) ** 2).mean().item()

    def pixel_mse(z0):
        return ((decode_chunked(z0) - x_clean) ** 2).mean().item()

    def binary_search(corrupt_fn, lo, hi, target_lpips, higher_is_less_corrupt):
        """higher_is_less_corrupt: True if increasing the param REDUCES
        corruption (e.g. fourier cutoff); False if increasing param
        INCREASES corruption (e.g. downsample factor)."""
        lpips_lo = lpips_dist(corrupt_fn(lo))
        lpips_hi = lpips_dist(corrupt_fn(hi))
        assert (lpips_lo >= target_lpips) != (lpips_hi >= target_lpips) or abs(lpips_lo - lpips_hi) < 1e-6, (
            f"target={target_lpips:.4f} not bracketed: lo({lo})={lpips_lo:.4f} hi({hi})={lpips_hi:.4f}"
        )
        for _ in range(args.max_iters):
            mid = 0.5 * (lo + hi)
            lpips_mid = lpips_dist(corrupt_fn(mid))
            if abs(lpips_mid - target_lpips) < args.tol:
                lo = hi = mid
                break
            if higher_is_less_corrupt:
                # increasing param -> less corrupt -> lpips decreases as param increases
                if lpips_mid > target_lpips:
                    lo = mid
                else:
                    hi = mid
            else:
                if lpips_mid > target_lpips:
                    hi = mid
                else:
                    lo = mid
        return 0.5 * (lo + hi)

    with th.no_grad():
        mask_z0 = mask_corrupt(x1, args.mask_prob)
        mask_lpips = lpips_dist(mask_z0)
        mask_latent_mse = latent_mse(mask_z0)
        mask_pixel_mse = pixel_mse(mask_z0)
        print(f"mask target: lpips={mask_lpips:.6f} latent_mse={mask_latent_mse:.6f} pixel_mse={mask_pixel_mse:.6f}")

        # Downsample: factor=1 -> no corruption (lpips~0), larger factor -> more corrupt (lpips increases)
        ds_factor = binary_search(lambda f: downsample_corrupt(x1, f), 1.05, 16.0, mask_lpips, higher_is_less_corrupt=False)
        ds_z0 = downsample_corrupt(x1, ds_factor)
        ds_lpips = lpips_dist(ds_z0)
        ds_latent_mse = latent_mse(ds_z0)
        ds_pixel_mse = pixel_mse(ds_z0)
        print(f"downsample (factor={ds_factor:.4f}): lpips={ds_lpips:.6f} latent_mse={ds_latent_mse:.6f} pixel_mse={ds_pixel_mse:.6f}")

        # Fourier: cutoff=1 -> no corruption (lpips~0), smaller cutoff -> more corrupt (lpips increases)
        fc_cutoff = binary_search(lambda c: fourier_corrupt(x1, c), 0.02, 1.0, mask_lpips, higher_is_less_corrupt=True)
        fc_z0 = fourier_corrupt(x1, fc_cutoff)
        fc_lpips = lpips_dist(fc_z0)
        fc_latent_mse = latent_mse(fc_z0)
        fc_pixel_mse = pixel_mse(fc_z0)
        print(f"fourier (cutoff={fc_cutoff:.4f}): lpips={fc_lpips:.6f} latent_mse={fc_latent_mse:.6f} pixel_mse={fc_pixel_mse:.6f}")

    print(f"RESULT downsample_factor={ds_factor:.4f} fourier_cutoff={fc_cutoff:.4f} "
          f"mask_lpips={mask_lpips:.6f} ds_lpips={ds_lpips:.6f} fc_lpips={fc_lpips:.6f}")


if __name__ == "__main__":
    main()
