"""
Calibrate blur_corrupt's sigma to match the clean-vs-corrupted difficulty
of the existing p=0.5 mask task, so the two structured-start-state families
present the field with comparably hard corruption (per 2026-07-14 goal:
"choose a moderate training blur severity using a principled rule ...
matching the input corruption difficulty of the existing p=0.5 mask task
by clean-vs-corrupted MSE").

Rule: on a held-out real-image batch, encode to VAE latents (identical
pipeline to eval_masked_recovery.py / train.py), compute
  target = mean((mask_corrupt(x1, 0.5) - x1)^2)
then binary-search blur_sigma so that
  mean((blur_corrupt(x1, sigma) - x1)^2) ~= target
Reports the exact sigma found + both MSEs for the record.
"""
import argparse
import torch as th
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL

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
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--max-iters", type=int, default=25)
    args = p.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    latents = []
    n = 0
    with th.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
            latents.append(x1)
            n += x1.shape[0]
            if n >= args.num_images:
                break
    x1 = th.cat(latents, dim=0)[:args.num_images]

    with th.no_grad():
        mask_mse = ((mask_corrupt(x1, args.mask_prob) - x1) ** 2).mean().item()

        lo, hi = args.sigma_lo, args.sigma_hi
        mse_lo = ((blur_corrupt(x1, lo) - x1) ** 2).mean().item()
        mse_hi = ((blur_corrupt(x1, hi) - x1) ** 2).mean().item()
        assert mse_lo <= mask_mse <= mse_hi, (
            f"target mask_mse={mask_mse:.5f} not bracketed by sigma range "
            f"[{lo},{hi}] -> mse=[{mse_lo:.5f},{mse_hi:.5f}]; widen --sigma-hi"
        )

        for _ in range(args.max_iters):
            mid = 0.5 * (lo + hi)
            mse_mid = ((blur_corrupt(x1, mid) - x1) ** 2).mean().item()
            if abs(mse_mid - mask_mse) < args.tol:
                lo = hi = mid
                break
            if mse_mid < mask_mse:
                lo = mid
            else:
                hi = mid
        sigma_star = 0.5 * (lo + hi)
        blur_mse = ((blur_corrupt(x1, sigma_star) - x1) ** 2).mean().item()

    print(f"n_images={args.num_images} mask_prob={args.mask_prob}")
    print(f"mask_clean_vs_corrupt_mse={mask_mse:.6f}")
    print(f"calibrated_blur_sigma={sigma_star:.6f}")
    print(f"blur_clean_vs_corrupt_mse={blur_mse:.6f}")
    print(f"RESULT sigma={sigma_star:.4f} mask_mse={mask_mse:.6f} blur_mse={blur_mse:.6f}")


if __name__ == "__main__":
    main()
