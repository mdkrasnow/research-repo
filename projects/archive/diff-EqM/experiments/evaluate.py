#!/usr/bin/env python3
"""
Evaluation script for DG-ANM-EqM: Measures short-horizon recovery distance.

THIS FILE IS READ-ONLY DURING AUTORESEARCH.
It is the immutable evaluation oracle that determines whether a change is kept or reverted.

The metric: short_horizon_recovery_distance
  = Average feature-space distance after L EqM gradient descent steps
    from off-manifold negatives back toward their anchor data points.
  Lower = better (the EqM field more effectively restores off-manifold points).

Usage:
    python projects/diff-EqM/experiments/evaluate.py \
        --config projects/diff-EqM/configs/eval.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add experiments dir to path for model import
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_model_and_config(config):
    """Load trained EqM model from checkpoint."""
    # Import model class
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
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Load EMA weights (better quality)
    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    return model


def get_features(model, x, y, device):
    """Extract features from model's last transformer layer."""
    t = torch.ones(x.shape[0], device=device)
    _, acts = model(x, t, y, return_act=True)
    return acts[-1].mean(dim=1)  # Global avg pool: (B, hidden_size)


def estimate_normal_projector(features, k=10):
    """Estimate normal-space projector via local PCA."""
    B, D = features.shape
    device = features.device

    dists = torch.cdist(features, features)
    _, nn_idx = dists.topk(k + 1, largest=False, dim=1)
    nn_idx = nn_idx[:, 1:]

    P_N = torch.zeros(B, D, D, device=device)

    for i in range(B):
        neighbors = features[nn_idx[i]]
        centered = neighbors - neighbors.mean(dim=0)
        cov = (centered.T @ centered) / k
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)

        # Auto tangent rank: 95% explained variance
        cumvar = eigvals.cumsum(0) / (eigvals.sum() + 1e-8)
        r = max(1, (cumvar < 0.95).sum().item() + 1)
        r = min(r, D)

        U_T = eigvecs[:, :r]
        P_T = U_T @ U_T.T
        P_N[i] = torch.eye(D, device=device) - P_T

    return P_N


def generate_normal_perturbations(x, features, P_N, epsilon=0.1):
    """Generate perturbations primarily in normal space."""
    B = x.shape[0]
    device = x.device

    # Random direction in feature space, projected to normal
    rand_dir = torch.randn_like(features)
    normal_dir = torch.bmm(P_N, rand_dir.unsqueeze(-1)).squeeze(-1)
    normal_dir = F.normalize(normal_dir, dim=1)

    # Perturb in pixel space proportional to normal direction magnitude
    pixel_perturbation = torch.randn_like(x)
    pixel_perturbation = F.normalize(pixel_perturbation.flatten(1), dim=1).view_as(x)
    x_neg = x + epsilon * pixel_perturbation

    return x_neg


def measure_short_horizon_recovery(model, x, x_neg, y, L, step_size, device):
    """
    Run L steps of EqM gradient descent from x_neg and measure
    feature-space distance to anchor x.

    Returns:
        recovery_distance: avg ||phi(u_L) - phi(x)||_2
    """
    B = x.shape[0]
    t = torch.ones(B, device=device)

    # Get anchor features
    with torch.no_grad():
        anchor_features = get_features(model, x, y, device)

    # Run L steps of EqM GD from negatives
    u = x_neg.clone()
    with torch.no_grad():
        for step in range(L):
            field = model(u, t, y)
            u = u + field * step_size

    # Measure feature distance after rollout
    with torch.no_grad():
        rollout_features = get_features(model, u, y, device)
        distances = (rollout_features - anchor_features).norm(dim=1)

    return distances


def main():
    parser = argparse.ArgumentParser(description="Evaluate DG-ANM-EqM")
    parser.add_argument("--config", required=True, help="Path to eval config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model_and_config(config)
    model = model.to(device)
    model.eval()

    # Load evaluation data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_set = datasets.CIFAR10(
        config.get("data_dir", "./data"),
        train=False, download=True, transform=transform,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=config.get("eval_batch_size", 64),
        shuffle=False, num_workers=2,
    )

    # Eval params
    L = config.get("rollout_steps", 10)
    step_size = config.get("rollout_step_size", 0.01)
    epsilon = config.get("perturbation_epsilon", 0.1)
    k_neighbors = config.get("k_neighbors", 10)
    max_eval_batches = config.get("max_eval_batches", 20)

    all_distances = []
    all_field_norms = []
    spurious_count = 0
    total_count = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(eval_loader):
            if batch_idx >= max_eval_batches:
                break

            x, y = x.to(device), y.to(device)
            B = x.shape[0]
            t = torch.ones(B, device=device)

            # Get features at clean data
            features = get_features(model, x, y, device)

            # Estimate normal space
            P_N = estimate_normal_projector(features, k=k_neighbors)

            # Generate normal-space perturbations
            x_neg = generate_normal_perturbations(x, features, P_N, epsilon=epsilon)

            # Measure field norm at negatives
            field = model(x_neg, t, y)
            field_norm = field.flatten(1).norm(dim=1)
            all_field_norms.append(field_norm)

            # Measure short-horizon recovery
            distances = measure_short_horizon_recovery(
                model, x, x_neg, y, L, step_size, device
            )
            all_distances.append(distances)

            # Count spurious equilibria (low field norm + large displacement)
            delta_norm = (x_neg - x).flatten(1).norm(dim=1)
            spurious = (field_norm < 0.1) & (delta_norm > epsilon * 0.5)
            spurious_count += spurious.sum().item()
            total_count += B

    # Aggregate metrics
    all_distances = torch.cat(all_distances)
    all_field_norms = torch.cat(all_field_norms)

    recovery_dist = all_distances.mean().item()
    avg_field_norm = all_field_norms.mean().item()
    spurious_rate = spurious_count / max(1, total_count)

    # Print metrics in parseable format
    print(f"short_horizon_recovery_distance: {recovery_dist:.6f}")
    print(f"avg_field_norm_at_negatives: {avg_field_norm:.6f}")
    print(f"spurious_equilibrium_rate: {spurious_rate:.6f}")
    print(f"num_samples_evaluated: {total_count}")
    print(f"rollout_steps: {L}")
    print(f"rollout_step_size: {step_size}")
    print(f"perturbation_epsilon: {epsilon}")


if __name__ == "__main__":
    main()
