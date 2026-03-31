#!/usr/bin/env python3
"""
DG-ANM Training for Equilibrium Matching on CIFAR-10.

This script trains an EqM-S/2 model with optional Differential-Geometry-Guided
Adversarial Negative Mining (DG-ANM). It is the ONLY modifiable file during
autoresearch iterations.

Usage:
    python projects/diff-EqM/experiments/train_dganm.py \
        --config projects/diff-EqM/configs/baseline.json

Architecture: EqM-S/2 (depth=12, hidden=384, patch=2, heads=6)
Data: CIFAR-10 (32x32, no VAE — direct pixel space for Stage 1)
"""

import argparse
import json
import time
import math
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ===========================================================================
# Small EqM for CIFAR-10 (operates in pixel space, no VAE)
# ===========================================================================

class EqMSmallCIFAR(nn.Module):
    """
    Minimal EqM for CIFAR-10 32x32 images.
    Uses a small transformer operating directly on 3-channel pixel patches.
    No VAE encoding — direct pixel space for Stage 1 experiments.
    """
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        class_dropout_prob=0.1,
        uncond=True,
        ebm="none",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.uncond = uncond
        self.ebm = ebm

        # Patch embedding
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        # Timestep and class embeddings
        self.t_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.y_embed = nn.Embedding(num_classes + 1, hidden_size)  # +1 for null class
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # Output projection
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        # Zero-init output projection
        nn.init.zeros_(self.out_proj.weight.data)
        nn.init.zeros_(self.out_proj.bias.data)

    def _timestep_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def unpatchify(self, x):
        p = self.patch_size
        h = w = self.image_size // p
        c = self.in_channels
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x0, t, y, return_act=False, get_energy=False, train=False):
        x0.requires_grad_(True)

        if self.uncond:
            t = torch.zeros_like(t)

        # Patch embed
        x = self.patch_embed(x0)  # (B, hidden, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden)
        x = x + self.pos_embed

        # Conditioning
        t_emb = self.t_embed(self._timestep_embedding(t, self.hidden_size))
        # Class label dropout for CFG
        if self.training and self.class_dropout_prob > 0:
            drop_mask = torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob
            y = torch.where(drop_mask, self.num_classes, y)
        y_emb = self.y_embed(y)
        c = t_emb + y_emb

        # Transformer
        acts = []
        for block in self.blocks:
            x = block(x, c)
            acts.append(x)

        # Output
        x = self.norm(x)
        x = self.out_proj(x)
        x = self.unpatchify(x)

        # Explicit energy formulations
        E = 0
        if self.ebm == "dot":
            E = torch.sum(x * x0, dim=(1, 2, 3))
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()], [x0], create_graph=train)[0]
        elif self.ebm == "l2":
            E = -torch.sum(x ** 2, dim=(1, 2, 3)) / 2
            if E.requires_grad:
                x = torch.autograd.grad([E.sum()], [x0], create_graph=train)[0]

        if get_energy:
            return x, -E
        if return_act:
            return x, acts
        return x


class TransformerBlock(nn.Module):
    """Transformer block with adaptive layer norm conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        # AdaLN modulation: 6 * hidden for shift/scale/gate on attn and mlp
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c).chunk(6, dim=1)
        # Attention
        h = self.norm1(x) * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate_a.unsqueeze(1) * h
        # MLP
        h = self.norm2(x) * (1 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_m.unsqueeze(1) * h
        return x


# ===========================================================================
# EqM Training Loss (from upstream transport/transport.py)
# ===========================================================================

def eqm_training_loss(model, x1, y, device, c_type="truncated", c_a=0.8, c_lambda=4.0):
    """
    Compute EqM training loss.

    L_EqM = (f(x_gamma) - (eps - x) * c(gamma))^2

    Args:
        model: EqM model
        x1: clean images (B, C, H, W)
        y: class labels (B,)
        c_type: type of c(gamma) schedule
        c_a: truncation point for c_trunc
        c_lambda: gradient multiplier
    """
    B = x1.shape[0]
    eps = torch.randn_like(x1)  # noise
    gamma = torch.rand(B, device=device)  # interpolation factor

    # x_gamma = gamma * x + (1 - gamma) * eps
    x_gamma = gamma.view(B, 1, 1, 1) * x1 + (1 - gamma.view(B, 1, 1, 1)) * eps

    # Target gradient: (eps - x) * c(gamma)
    # c(gamma) controls gradient magnitude, vanishes at gamma=1 (data manifold)
    if c_type == "truncated":
        ct = torch.where(
            gamma <= c_a,
            torch.ones_like(gamma),
            (1 - gamma) / (1 - c_a)
        ) * c_lambda
    elif c_type == "linear":
        ct = (1 - gamma) * c_lambda
    else:
        ct = torch.ones_like(gamma) * c_lambda

    target = (eps - x1) * ct.view(B, 1, 1, 1)
    t = gamma  # EqM uses gamma as "timestep" (set to 0 if uncond)

    pred = model(x_gamma, t, y, train=True)
    loss = F.mse_loss(pred, target)
    return loss


# ===========================================================================
# DG-ANM: Geometry Estimation
# ===========================================================================

def estimate_local_geometry(features, k=10, r=None):
    """
    Estimate local tangent/normal decomposition via feature-space PCA.

    Args:
        features: (B, D) feature vectors for a minibatch
        k: number of neighbors for local PCA
        r: tangent rank (if None, auto-select via explained variance)

    Returns:
        P_T: (B, D, D) tangent projectors
        P_N: (B, D, D) normal projectors
    """
    B, D = features.shape
    device = features.device

    # Compute pairwise distances
    dists = torch.cdist(features, features)  # (B, B)
    # Get k-nearest neighbors (excluding self)
    _, nn_idx = dists.topk(k + 1, largest=False, dim=1)
    nn_idx = nn_idx[:, 1:]  # exclude self, shape (B, k)

    P_T = torch.zeros(B, D, D, device=device)
    P_N = torch.zeros(B, D, D, device=device)

    for i in range(B):
        neighbors = features[nn_idx[i]]  # (k, D)
        local_mean = neighbors.mean(dim=0)
        centered = neighbors - local_mean  # (k, D)

        # Local covariance
        cov = (centered.T @ centered) / k  # (D, D)

        # Eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending order
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)

        # Auto-select rank if not specified
        if r is None:
            cumvar = eigvals.cumsum(0) / eigvals.sum()
            r_local = max(1, (cumvar < 0.95).sum().item() + 1)
        else:
            r_local = r

        r_local = min(r_local, D)
        U_T = eigvecs[:, :r_local]  # (D, r)
        P_T[i] = U_T @ U_T.T
        P_N[i] = torch.eye(D, device=device) - P_T[i]

    return P_T, P_N


# ===========================================================================
# DG-ANM: Adversarial Negative Mining
# ===========================================================================

def mine_negatives(
    model, x, y, features, P_N, P_T,
    epsilon=0.1,
    mining_steps=5,
    mining_lr=0.01,
    rollout_steps=5,
    rollout_step_size=0.01,
    lambda_N=1.0,
    lambda_T=1.0,
    lambda_W=0.1,
    lambda_A=0.0,
    lambda_R=0.1,
    device="cuda",
):
    """
    Generate adversarial negatives via projected gradient ascent in normal space.

    Args:
        model: EqM model (eval mode, no grad for model params)
        x: anchor clean images (B, C, H, W)
        y: class labels (B,)
        features: (B, D) feature vectors at anchors
        P_N: (B, D, D) normal projectors
        P_T: (B, D, D) tangent projectors
        epsilon: perturbation budget (L2 ball radius)
        mining_steps: number of PGA steps
        mining_lr: step size for mining
        rollout_steps: L for short-horizon EqM descent
        rollout_step_size: eta for EqM rollout
        lambda_*: weights for mining objective components

    Returns:
        x_neg: (B, C, H, W) mined negatives
        mining_info: dict with diagnostics
    """
    B = x.shape[0]

    # Initialize tracking variables
    delta_N = torch.zeros(B, features.shape[1], device=device)
    delta_T = torch.zeros(B, features.shape[1], device=device)
    field_norm = torch.zeros(B, device=device)

    # Initialize perturbation in normal space direction
    # We perturb in pixel space but measure geometry in feature space
    delta = torch.randn_like(x) * 0.01
    delta.requires_grad_(True)

    t_ones = torch.ones(B, device=device)

    for _ in range(mining_steps):
        x_neg = x.detach() + delta  # delta carries grad

        # Forward pass — field at negative, grad flows through delta
        field = model(x_neg, t_ones, y, train=False)
        field_norm = field.flatten(1).norm(dim=1)

        # Feature displacement (through model, grad flows through delta)
        _, neg_acts = model(x_neg, t_ones, y, return_act=True)
        neg_features = neg_acts[-1].mean(dim=1)  # (B, hidden_size)

        delta_phi = neg_features - features.detach()  # (B, D)

        # Project to tangent/normal components
        delta_N = torch.bmm(P_N.detach(), delta_phi.unsqueeze(-1)).squeeze(-1)
        delta_T = torch.bmm(P_T.detach(), delta_phi.unsqueeze(-1)).squeeze(-1)

        # Mining objective (maximize)
        L_normal = delta_N.norm(dim=1) ** 2
        L_tan = delta_T.norm(dim=1) ** 2
        L_weak = -field_norm

        mining_obj = (
            lambda_N * L_normal
            - lambda_T * L_tan
            + lambda_W * L_weak
        ).mean()

        # Gradient ascent on delta
        grad_delta = torch.autograd.grad(mining_obj, delta, retain_graph=False)[0]

        with torch.no_grad():
            delta = delta + mining_lr * grad_delta.sign()
            # Project back to epsilon ball
            delta_norm = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            delta = delta * torch.clamp(epsilon / (delta_norm + 1e-8), max=1.0)
            delta = delta.detach().requires_grad_(True)

    # Trajectory failure diagnostic (no grad needed, just for logging)
    if lambda_R > 0 and rollout_steps > 0:
        with torch.no_grad():
            x_neg_final = x.detach() + delta
            u = x_neg_final.clone()
            for _ in range(rollout_steps):
                grad = model(u, t_ones, y)
                u = u + grad * rollout_step_size

    x_neg = (x + delta).detach()
    mining_info = {
        "avg_normal_component": delta_N.norm(dim=1).mean().item() if delta_N is not None else 0.0,
        "avg_tangent_component": delta_T.norm(dim=1).mean().item() if delta_T is not None else 0.0,
        "avg_field_norm_at_neg": field_norm.mean().item() if field_norm is not None else 0.0,
    }
    return x_neg, mining_info


# ===========================================================================
# DG-ANM: Negative Training Loss
# ===========================================================================

def dganm_negative_loss(
    model, x, x_neg, y,
    margin=1.0,
    rho=0.1,
    tau=0.0,
    rollout_steps=5,
    rollout_step_size=0.01,
    device="cuda",
):
    """
    Compute the DG-ANM negative loss.

    L_neg = max(0, m - |g_theta(x^-)|) + rho * L_traj + tau * L_align

    Args:
        model: EqM model
        x: anchor clean images
        x_neg: mined negatives
        y: class labels
        margin: minimum field norm at negatives
        rho: trajectory failure weight
        tau: alignment weight (0 for now, requires Jacobian)
    """
    B = x.shape[0]
    t_ones = torch.ones(B, device=device)

    # Field at negatives
    field_at_neg = model(x_neg, t_ones, y, train=True)
    field_norm = field_at_neg.flatten(1).norm(dim=1)

    # Margin loss: enforce nontrivial restoring force
    loss_margin = F.relu(margin - field_norm).mean()

    # Trajectory failure loss
    loss_traj = torch.tensor(0.0, device=device)
    if rho > 0 and rollout_steps > 0:
        u = x_neg.detach().clone()
        for _ in range(rollout_steps):
            with torch.no_grad():
                grad = model(u.detach(), t_ones, y)
            u = u.detach() + grad * rollout_step_size
        # Distance from anchor after rollout
        loss_traj = F.mse_loss(u, x)

    total = loss_margin + rho * loss_traj
    return total


# ===========================================================================
# Main Training Loop
# ===========================================================================

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def get_cifar10_loaders(batch_size, data_dir="./data"):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def main():
    parser = argparse.ArgumentParser(description="DG-ANM Training for EqM on CIFAR-10")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model
    model = EqMSmallCIFAR(
        image_size=32,
        patch_size=config.get("patch_size", 4),
        hidden_size=config.get("hidden_size", 384),
        depth=config.get("depth", 12),
        num_heads=config.get("num_heads", 6),
        num_classes=10,
        uncond=config.get("uncond", True),
        ebm=config.get("ebm", "none"),
    ).to(device)
    ema = deepcopy(model).to(device)
    ema.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.0),
    )

    # Data
    batch_size = config.get("batch_size", 128)
    train_loader, test_loader = get_cifar10_loaders(batch_size, data_dir=config.get("data_dir", "./data"))

    # Training config
    epochs = config.get("epochs", 1)
    use_mining = config.get("use_mining", False)
    mining_config = config.get("mining", {})
    neg_loss_config = config.get("negative_loss", {})
    gamma = config.get("gamma", 0.0)  # weight of negative loss

    # Output
    output_dir = Path(config.get("output_dir", "projects/diff-EqM/results/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # EqM loss config
    c_type = config.get("c_type", "truncated")
    c_a = config.get("c_a", 0.8)
    c_lambda = config.get("c_lambda", 4.0)

    print(f"Training for {epochs} epochs, mining={use_mining}, gamma={gamma}")
    print(f"Config: {json.dumps(config, indent=2)}")

    global_step = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_neg_loss = 0.0
        num_batches = 0

        max_batches = config.get("max_batches", None)
        for batch_idx, (x, y) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)

            # Standard EqM loss
            loss_eqm = eqm_training_loss(model, x, y, device, c_type, c_a, c_lambda)
            total_loss = loss_eqm

            # DG-ANM negative mining + loss
            if use_mining and gamma > 0:
                model.eval()
                with torch.no_grad():
                    t_ones = torch.ones(x.shape[0], device=device)
                    _, acts = model(x, t_ones, y, return_act=True)
                    features = acts[-1].mean(dim=1)  # (B, hidden_size)

                P_T, P_N = estimate_local_geometry(
                    features,
                    k=mining_config.get("k", 10),
                    r=mining_config.get("r", None),
                )

                x_neg, mining_info = mine_negatives(
                    model, x, y, features, P_N, P_T,
                    epsilon=mining_config.get("epsilon", 0.1),
                    mining_steps=mining_config.get("mining_steps", 5),
                    mining_lr=mining_config.get("mining_lr", 0.01),
                    rollout_steps=mining_config.get("rollout_steps", 5),
                    rollout_step_size=mining_config.get("rollout_step_size", 0.01),
                    lambda_N=mining_config.get("lambda_N", 1.0),
                    lambda_T=mining_config.get("lambda_T", 1.0),
                    lambda_W=mining_config.get("lambda_W", 0.1),
                    lambda_A=mining_config.get("lambda_A", 0.0),
                    lambda_R=mining_config.get("lambda_R", 0.1),
                    device=device,
                )

                model.train()
                loss_neg = dganm_negative_loss(
                    model, x, x_neg, y,
                    margin=neg_loss_config.get("margin", 1.0),
                    rho=neg_loss_config.get("rho", 0.1),
                    tau=neg_loss_config.get("tau", 0.0),
                    rollout_steps=neg_loss_config.get("rollout_steps", 5),
                    rollout_step_size=neg_loss_config.get("rollout_step_size", 0.01),
                    device=device,
                )
                total_loss = total_loss + gamma * loss_neg
                epoch_neg_loss += loss_neg.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            update_ema(ema, model)

            epoch_loss += loss_eqm.item()
            num_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / num_batches
                print(f"[step {global_step}] loss_eqm={avg_loss:.4f} "
                      f"neg_loss={epoch_neg_loss/max(1,num_batches):.4f} "
                      f"elapsed={elapsed:.1f}s")

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_epoch_loss:.4f}")

    # Save checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "global_step": global_step,
    }
    ckpt_path = output_dir / "final_model.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    elapsed = time.time() - start_time
    print(f"Training complete. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
