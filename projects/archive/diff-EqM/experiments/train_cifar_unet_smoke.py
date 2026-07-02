#!/usr/bin/env python3
"""
Stage A.5 Step B — CIFAR-10 UNet Smoke Test.

Purpose: verify our data pipeline + FID eval are correct by training a standard
UNet with a standard Flow Matching objective on CIFAR-10. Target FID < 15 in
a few hours on one A100. If we reach that, the Stage A.5 bug is isolated to
architecture (transformer vs. UNet), which unblocks Step A (port FM UNet + EqM
loss).

This script does NOT train an EqM model. It is diagnostic only.

Data pipeline: mirrors train_dganm.py (CIFAR-10, [-1,1] normalization).
FID eval: reuses experiments/evaluate_fid.py's Inception-v3 pool3 features and
reference stats path, so any eval-side bug would surface here too.

Usage:
    python projects/diff-EqM/experiments/train_cifar_unet_smoke.py \
        --output-dir projects/diff-EqM/results/cifar_unet_smoke \
        --epochs 150 --batch-size 128 --lr 2e-4
"""

import argparse
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Reuse existing eval utilities so any pipeline bug is controlled for.
# The diff-EqM package dir has a dash, so we import by file path via importlib.
_EVAL_FID_PATH = Path(__file__).resolve().parent / "evaluate_fid.py"
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("diffeqm_evaluate_fid", _EVAL_FID_PATH)
assert _spec is not None and _spec.loader is not None, f"Cannot load {_EVAL_FID_PATH}"
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
InceptionV3Features = _eval_mod.InceptionV3Features
compute_fid = _eval_mod.compute_fid
compute_statistics = _eval_mod.compute_statistics
get_or_compute_reference_stats = _eval_mod.get_or_compute_reference_stats


def ema_update(ema_model, model, decay=0.9999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def build_unet():
    """Standard-size DDPM-style UNet from diffusers, sized for CIFAR-10 32x32."""
    from diffusers import UNet2DModel
    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    )


def flow_matching_loss(model, x1, device):
    """Rectified flow / FM loss: model predicts velocity v = x1 - x0 at x_t."""
    B = x1.size(0)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)
    t_expand = t.view(B, 1, 1, 1)
    xt = (1.0 - t_expand) * x0 + t_expand * x1
    target = x1 - x0
    # diffusers UNet2DModel expects timesteps in [0, 1000] (noise step indexing);
    # we pass a continuous time scaled to that range for its sinusoidal embedding.
    t_model = (t * 999).long()
    pred = model(xt, t_model).sample
    return F.mse_loss(pred, target)


@torch.no_grad()
def sample_euler(model, num_samples, batch_size, num_steps, device):
    """Euler integration from x0 ~ N(0,I) to x1 along the learned velocity field."""
    all_samples = []
    remaining = num_samples
    while remaining > 0:
        B = min(batch_size, remaining)
        x = torch.randn(B, 3, 32, 32, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            t_model = (t * 999).long()
            v = model(x, t_model).sample
            x = x + v * dt
        x = torch.clamp(x, -1.0, 1.0)
        all_samples.append(x.cpu())
        remaining -= B
    return torch.cat(all_samples, dim=0)[:num_samples]


def compute_model_fid(model, inception, ref_mu, ref_sigma, num_samples, batch_size,
                      num_steps, device):
    samples = sample_euler(model, num_samples, batch_size, num_steps, device)
    feats = []
    for i in range(0, samples.size(0), 64):
        batch = samples[i:i + 64].to(device)
        feats.append(inception(batch).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu, sigma = compute_statistics(feats)
    return compute_fid(ref_mu, ref_sigma, mu, sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--eval-every-epochs", type=int, default=25)
    ap.add_argument("--final-fid-samples", type=int, default=10000)
    ap.add_argument("--eval-fid-samples", type=int, default=2000)
    ap.add_argument("--sample-steps", type=int, default=100)
    ap.add_argument("--sample-batch-size", type=int, default=256)
    ap.add_argument(
        "--reference-stats-path",
        default="projects/diff-EqM/results/cifar10_inception_stats.npz",
    )
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    model = build_unet().to(device)
    ema_model = deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params/1e6:.2f}M", flush=True)

    inception = InceptionV3Features().to(device).eval()
    ref_cfg = {"reference_stats_path": args.reference_stats_path, "data_dir": args.data_dir}
    ref_mu, ref_sigma = get_or_compute_reference_stats(ref_cfg, inception, device)
    print("Reference stats loaded.", flush=True)

    log_lines = []
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            loss = flow_matching_loss(model, x, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema_update(ema_model, model, decay=args.ema_decay)
            ep_loss += loss.item()
            n_batches += 1
            step += 1
        avg = ep_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} loss {avg:.4f} step {step} elapsed {elapsed:.0f}s", flush=True)
        log_lines.append(f"{epoch+1}\t{step}\t{avg:.6f}\t{elapsed:.1f}")

        if (epoch + 1) % args.eval_every_epochs == 0 or (epoch + 1) == args.epochs:
            ema_model.eval()
            fid = compute_model_fid(
                ema_model, inception, ref_mu, ref_sigma,
                num_samples=args.eval_fid_samples,
                batch_size=args.sample_batch_size,
                num_steps=args.sample_steps, device=device,
            )
            print(f"  [eval] epoch {epoch+1} FID(ema, {args.eval_fid_samples} samples) = {fid:.4f}", flush=True)
            log_lines.append(f"eval\t{epoch+1}\t{fid:.6f}")

    # Final larger-sample FID
    ema_model.eval()
    final_fid = compute_model_fid(
        ema_model, inception, ref_mu, ref_sigma,
        num_samples=args.final_fid_samples,
        batch_size=args.sample_batch_size,
        num_steps=args.sample_steps, device=device,
    )
    # Emit grep-friendly final line (matches our autoresearch pattern style)
    print(f"cifar10_smoke_fid: {final_fid:.4f}", flush=True)
    log_lines.append(f"final\t{args.final_fid_samples}\t{final_fid:.6f}")

    torch.save({
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "final_fid": final_fid,
        "args": vars(args),
    }, out / "final.pt")
    (out / "train_log.tsv").write_text("\n".join(log_lines) + "\n")
    print(f"Saved checkpoint and log to {out}", flush=True)


if __name__ == "__main__":
    main()
