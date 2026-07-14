"""
Calibrate blur_corrupt's sigma to match the clean-vs-corrupted difficulty
of the existing p=0.5 mask task, so the two structured-start-state families
present the field with comparably hard corruption (per 2026-07-14 goal:
"choose a moderate training blur severity using a principled rule ...
matching the input corruption difficulty of the existing p=0.5 mask task
by clean-vs-corrupted MSE or LPIPS").

FIRST ATTEMPT (raw latent MSE) FAILED and is not used: mask_corrupt replaces
~50% of latent elements with unit-variance Gaussian noise, an unbounded-MSE
corruption; blur_corrupt is a bounded local/global smoothing operation whose
MSE saturates (as sigma grows, blur_corrupt(x1) -> per-channel spatial mean
of x1, a fixed finite value) well below mask's raw latent MSE (empirically:
mask MSE ~0.85 at p=0.5 vs blur's asymptotic ceiling ~0.47 even at sigma=8,
already large enough to average over ~the whole 32x32 latent). No sigma can
close that gap -- the two corruptions are not comparable in raw latent-MSE
units at all.

Rule actually used: match PIXEL-SPACE LPIPS distance instead (decode both
corrupted latents through the VAE, compare to the clean decoded image with
LPIPS-AlexNet). LPIPS is bounded work in perceptual-similarity units, not raw
variance, so it doesn't have this asymmetric-ceiling problem, and the task
explicitly allows MSE-or-LPIPS as the matching criterion. Also reports raw
latent MSE and pixel MSE for both corruptions for the record (asymmetry is
expected and documented, not treated as a bug).
"""
import argparse
import torch as th
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
import lpips

from transport.corruption import mask_corrupt, blur_corrupt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str,
                    default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val")
    p.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-images", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mask-prob", type=float, default=0.5)
    p.add_argument("--sigma-lo", type=float, default=0.1)
    p.add_argument("--sigma-hi", type=float, default=8.0)
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
        # decode in small chunks -- full-batch VAE decode OOMs on the
        # MIG-sliced gpu_test 20GB card (8GB single-alloc conv activation
        # for 256 images at once)
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

    with th.no_grad():
        mask_z0 = mask_corrupt(x1, args.mask_prob)
        mask_lpips = lpips_dist(mask_z0)
        mask_latent_mse = latent_mse(mask_z0)
        mask_pixel_mse = pixel_mse(mask_z0)

        lo, hi = args.sigma_lo, args.sigma_hi
        lpips_lo = lpips_dist(blur_corrupt(x1, lo))
        lpips_hi = lpips_dist(blur_corrupt(x1, hi))
        assert lpips_lo <= mask_lpips <= lpips_hi, (
            f"target mask_lpips={mask_lpips:.5f} not bracketed by sigma range "
            f"[{lo},{hi}] -> lpips=[{lpips_lo:.5f},{lpips_hi:.5f}]; widen --sigma-hi"
        )

        for _ in range(args.max_iters):
            mid = 0.5 * (lo + hi)
            lpips_mid = lpips_dist(blur_corrupt(x1, mid))
            if abs(lpips_mid - mask_lpips) < args.tol:
                lo = hi = mid
                break
            if lpips_mid < mask_lpips:
                lo = mid
            else:
                hi = mid
        sigma_star = 0.5 * (lo + hi)
        blur_z0 = blur_corrupt(x1, sigma_star)
        blur_lpips = lpips_dist(blur_z0)
        blur_latent_mse = latent_mse(blur_z0)
        blur_pixel_mse = pixel_mse(blur_z0)

    print(f"n_images={args.num_images} mask_prob={args.mask_prob}")
    print(f"mask: lpips={mask_lpips:.6f} latent_mse={mask_latent_mse:.6f} pixel_mse={mask_pixel_mse:.6f}")
    print(f"blur (calibrated sigma={sigma_star:.6f}): "
          f"lpips={blur_lpips:.6f} latent_mse={blur_latent_mse:.6f} pixel_mse={blur_pixel_mse:.6f}")
    print(f"RESULT sigma={sigma_star:.4f} mask_lpips={mask_lpips:.6f} blur_lpips={blur_lpips:.6f} "
          f"mask_latent_mse={mask_latent_mse:.6f} blur_latent_mse={blur_latent_mse:.6f}")


if __name__ == "__main__":
    main()
