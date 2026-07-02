#!/usr/bin/env python3
"""
FID evaluation for DG-ANM-EqM: Sample images via gradient descent, compute FID.

This evaluates the *generative quality* of the model, complementing the proxy
metric (short_horizon_recovery_distance) used in autoresearch.

Sampling: Start from Gaussian noise z ~ N(0,I), run N steps of EqM gradient
descent: x_{t+1} = x_t + f(x_t) * step_size. The model operates in pixel space
(no VAE) on CIFAR-10 32x32.

FID: Computed using Inception-v3 features (2048-dim from pool3 layer).
Reference statistics are precomputed from the CIFAR-10 training set.

Usage:
    python projects/diff-EqM/experiments/evaluate_fid.py \
        --config projects/diff-EqM/configs/eval_fid.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ============================================================================
# Inception-v3 Feature Extractor
# ============================================================================

class InceptionV3Features(torch.nn.Module):
    """Inception-v3 truncated at pool3 for 2048-dim features."""

    def __init__(self, device="cuda"):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()

        # Build truncated model up to avgpool (pool3 = 2048-dim)
        self.blocks = torch.nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        ).to(device)
        self.blocks.eval()

    @torch.no_grad()
    def forward(self, x):
        # Inception expects 299x299, RGB, normalized to [-1, 1] (roughly)
        # Our images are 32x32 in [-1, 1], upsample to 299x299
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.blocks(x)
        return x.flatten(1)  # (B, 2048)


# ============================================================================
# FID Computation
# ============================================================================

def compute_statistics(features):
    """Compute mean and covariance of feature vectors."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Frechet Inception Distance."""
    diff = mu1 - mu2
    result = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    covmean = np.asarray(result[0])

    if not np.isfinite(covmean).all():
        print(f"WARNING: fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = np.asarray(linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)[0])

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"WARNING: imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


# ============================================================================
# Sample Generation via EqM Gradient Descent
# ============================================================================

@torch.no_grad()
def generate_samples(model, num_samples, config, device):
    """
    Generate images by running EqM gradient descent from Gaussian noise.

    x_0 ~ N(0, I)
    x_{t+1} = x_t + f(x_t) * step_size
    for num_steps iterations.
    """
    num_steps = config.get("num_sampling_steps", 250)
    step_size = config.get("sampling_step_size", 0.01)
    batch_size = config.get("sample_batch_size", 64)
    use_nag = config.get("use_nag", False)
    nag_mu = config.get("nag_mu", 0.3)
    num_classes = 10

    all_samples = []
    remaining = num_samples

    while remaining > 0:
        B = min(batch_size, remaining)
        # Start from noise
        x = torch.randn(B, 3, 32, 32, device=device)
        t = torch.ones(B, device=device)
        y = torch.randint(0, num_classes, (B,), device=device)

        # Momentum for NAG-GD (optional)
        m = torch.zeros_like(x)

        for _ in range(num_steps):
            if use_nag:
                x_look = x + step_size * m * nag_mu
                field = model(x_look, t, y)
                m = field
            else:
                field = model(x, t, y)
            x = x + field * step_size

        # Clamp to valid pixel range [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)
        all_samples.append(x.cpu())
        remaining -= B

        if (num_samples - remaining) % 1000 < batch_size:
            print(f"  Generated {num_samples - remaining}/{num_samples} samples")

    return torch.cat(all_samples, dim=0)[:num_samples]


# ============================================================================
# Reference Statistics
# ============================================================================

def get_or_compute_reference_stats(config, inception, device):
    """Load precomputed CIFAR-10 reference stats, or compute them."""
    stats_path = Path(config.get("reference_stats_path", "projects/diff-EqM/results/cifar10_inception_stats.npz"))

    if stats_path.exists():
        print(f"Loading precomputed reference stats from {stats_path}")
        data = np.load(stats_path)
        return data["mu"], data["sigma"]

    print("Computing CIFAR-10 reference statistics (one-time)...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    cifar_train = datasets.CIFAR10(
        config.get("data_dir", "./data"), train=True, download=True, transform=transform
    )
    loader = DataLoader(cifar_train, batch_size=64, shuffle=False, num_workers=2)

    all_feats = []
    for x, _ in loader:
        x = x.to(device)
        feats = inception(x)
        all_feats.append(feats.cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    mu, sigma = compute_statistics(all_feats)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, mu=mu, sigma=sigma)
    print(f"Saved reference stats to {stats_path}")

    return mu, sigma


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FID Evaluation for DG-ANM-EqM")
    parser.add_argument("--config", required=True, help="Path to FID eval config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    from train_dganm import EqMSmallCIFAR

    model = EqMSmallCIFAR(
        image_size=32,
        patch_size=config.get("patch_size", 4),
        hidden_size=config.get("hidden_size", 384),
        depth=config.get("depth", 12),
        num_heads=config.get("num_heads", 6),
        num_classes=10,
        uncond=config.get("uncond", True),
        ebm=config.get("ebm", "none"),
    )

    ckpt_path = config["checkpoint_path"]
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
        print("Loaded EMA weights")
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print("Loaded model weights")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Generate samples
    num_samples = config.get("num_fid_samples", 10000)
    print(f"\nGenerating {num_samples} samples...")
    t0 = time.time()
    samples = generate_samples(model, num_samples, config, device)
    gen_time = time.time() - t0
    print(f"Sample generation took {gen_time:.1f}s ({gen_time/num_samples*1000:.1f}ms/sample)")

    # Optionally save sample grid for visual inspection
    save_samples_path = config.get("save_samples_path", None)
    if save_samples_path:
        from torchvision.utils import save_image
        grid_samples = samples[:64]
        save_image(grid_samples, save_samples_path, nrow=8, normalize=True, value_range=(-1, 1))
        print(f"Saved sample grid to {save_samples_path}")

    # Compute Inception features for generated samples
    print("\nComputing Inception features for generated samples...")
    inception = InceptionV3Features(device=device)
    gen_feats = []
    gen_loader = DataLoader(TensorDataset(samples), batch_size=64)
    for (batch,) in gen_loader:
        batch = batch.to(device)
        feats = inception(batch)
        gen_feats.append(feats.cpu().numpy())
    gen_feats = np.concatenate(gen_feats, axis=0)

    # Reference statistics
    print("\nGetting CIFAR-10 reference statistics...")
    ref_mu, ref_sigma = get_or_compute_reference_stats(config, inception, device)

    # Compute FID
    gen_mu, gen_sigma = compute_statistics(gen_feats)
    fid = compute_fid(ref_mu, ref_sigma, gen_mu, gen_sigma)

    # Compute Inception Score (bonus metric — measures diversity + quality)
    # IS uses the softmax outputs, which we don't have from truncated inception.
    # Skip for now; FID is the primary metric.

    # Print results in parseable format
    print(f"\n{'='*50}")
    print(f"fid: {fid:.4f}")
    print(f"num_samples: {num_samples}")
    print(f"num_sampling_steps: {config.get('num_sampling_steps', 250)}")
    print(f"sampling_step_size: {config.get('sampling_step_size', 0.01)}")
    print(f"generation_time_seconds: {gen_time:.1f}")
    print(f"checkpoint: {ckpt_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
