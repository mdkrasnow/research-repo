#!/usr/bin/env python3
"""
Stage A.5 Step A — Vanilla EqM on CIFAR-10 via Flow Matching's UNet.

Combines:
  - Flow Matching's CIFAR-10 UNet (projects/diff-EqM/fm-upstream/models/unet.py),
    matching the paper's Appendix B.1 "non-transformer" architecture: 128
    model channels, 4 res blocks, attention at res=2, channel_mult=[2,2,2],
    dropout=0.3.
  - The EqM c(t)-weighted linear-path velocity target (from
    eqm-upstream/transport/transport.py: get_ct(t) with a=0.8, gain=4).
  - Same CIFAR-10 data pipeline + FID evaluator used by the Step B smoke,
    so we're varying only the loss objective (FM -> EqM) on top of a
    known-working backbone.

Exit criterion: vanilla EqM FID <= 3.66 on CIFAR-10, 50K samples
(paper reports 3.36; we allow +0.3 for seed variance).

This script trains ONLY vanilla EqM — no DG-ANM mining. After it reproduces
the paper, a follow-up script (train_cifar_dganm_unet.py) will add mining.

Usage:
    python projects/diff-EqM/experiments/train_cifar_eqm_unet.py \\
        --output-dir projects/diff-EqM/results/step_a_vanilla_eqm_cifar \\
        --epochs 1800 --batch-size 64 --lr 2e-4 --seed 0
"""

import argparse
import importlib.util
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Put fm-upstream on sys.path so `import models.unet` resolves to our vendored
# copy of Flow Matching's CIFAR UNet.
REPO_ROOT = Path(__file__).resolve().parents[3]
FM_UPSTREAM = REPO_ROOT / "projects" / "diff-EqM" / "fm-upstream"
sys.path.insert(0, str(FM_UPSTREAM))
from models.unet import UNetModel  # noqa: E402

# Reuse the existing FID pipeline by path-import (dir has a dash).
_EVAL_FID_PATH = Path(__file__).resolve().parent / "evaluate_fid.py"
_spec = importlib.util.spec_from_file_location("diffeqm_evaluate_fid", _EVAL_FID_PATH)
assert _spec is not None and _spec.loader is not None
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
InceptionV3Features = _eval_mod.InceptionV3Features
compute_fid = _eval_mod.compute_fid
compute_statistics = _eval_mod.compute_statistics
get_or_compute_reference_stats = _eval_mod.get_or_compute_reference_stats


# ----------------------------- Model -----------------------------------------

CIFAR10_UNET_CONFIG = dict(
    in_channels=3,
    model_channels=128,
    out_channels=3,
    num_res_blocks=4,
    attention_resolutions=[2],
    dropout=0.3,
    channel_mult=[2, 2, 2],
    conv_resample=False,
    dims=2,
    num_classes=None,
    use_checkpoint=False,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_new_attention_order=True,
    with_fourier_features=False,
)


def build_unet():
    return UNetModel(**CIFAR10_UNET_CONFIG)


# ------------------------- EqM loss ------------------------------------------

def eqm_ct(t, a=0.8, gain=4.0):
    """EqM c(t) weighting per eqm-upstream/transport/transport.py:get_ct.

    c(t) = gain * min( 1.0 - (1 - 1)/a * t,  1/(1-a) - 1/(1-a) * t )
         = gain * min( 1, (1 - t) / (1 - a) )

    For t <= a: c(t) = gain (constant).
    For t > a : c(t) decays linearly to 0 at t=1.
    Default a=0.8, gain=4 matches the paper and our upstream.
    """
    start = 1.0
    c = torch.minimum(start - (start - 1.0) / a * t, 1.0 / (1.0 - a) - 1.0 / (1.0 - a) * t)
    return c * gain


def eqm_loss(model, x1, device, eps=1e-3, a=0.8, gain=4.0):
    """Vanilla EqM loss on a linear path (ICPlan).

    Sample t ~ U(eps, 1-eps), x0 ~ N(0, I).
    x_t = (1 - t) * x0 + t * x1     (linear interpolation path)
    u_t = x1 - x0                   (constant velocity along the path)
    Target = c(t) * u_t             (energy-compatible scaling)
    Loss = mean ||model(x_t, t) - c(t) * u_t||^2
    """
    B = x1.size(0)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps) + eps  # uniform on [eps, 1-eps]
    t_ = t.view(B, 1, 1, 1)
    xt = (1.0 - t_) * x0 + t_ * x1
    ut = x1 - x0
    ct = eqm_ct(t, a=a, gain=gain).view(B, 1, 1, 1)
    target = ct * ut
    # FM's UNet expects timesteps in [0, 1000] (int-like); pass scaled t.
    t_model = (t * 999.0).clamp_min(0.0)
    pred = model(xt, t_model)
    return F.mse_loss(pred, target)


# ------------------------- Samplers ------------------------------------------

@torch.no_grad()
def sample_gd(model, num_samples, batch_size, num_steps, step_size, device,
              a=0.8, gain=4.0):
    """Gradient-descent-style sampling on the learned EqM field.

    Matches the EqM paper's optimization-based sampling: start from noise at
    t=0 (pure prior), take small steps against the predicted field with a
    fixed step size. We pass the model's current position as both input and
    t=0 (equilibrium state); in practice the paper uses a schedule, but for
    a vanilla baseline a plain GD is a reasonable default and matches the
    smoke-test pattern. See projects/diff-EqM/experiments/evaluate_fid.py
    for the original EqM GD sampler our eval pipeline was built around.

    x_{k+1} = x_k + step_size * model(x_k, t=0)
    """
    all_samples = []
    remaining = num_samples
    while remaining > 0:
        B = min(batch_size, remaining)
        x = torch.randn(B, 3, 32, 32, device=device)
        t = torch.zeros(B, device=device)
        for _ in range(num_steps):
            field = model(x, t)
            x = x + step_size * field
        x = torch.clamp(x, -1.0, 1.0)
        all_samples.append(x.cpu())
        remaining -= B
    return torch.cat(all_samples, dim=0)[:num_samples]


@torch.no_grad()
def sample_euler(model, num_samples, batch_size, num_steps, device,
                 a=0.8, gain=4.0):
    """Euler integration along the learned velocity field with c(t)-rescaling.

    Because we trained model(xt, t) to predict c(t)*(x1 - x0), the velocity
    that integrates from noise to data is model_output / c(t). We use this
    as a baseline sampler alternative to GD. Integrates from t=0 to t=1 with
    step size dt=1/num_steps.
    """
    all_samples = []
    remaining = num_samples
    while remaining > 0:
        B = min(batch_size, remaining)
        x = torch.randn(B, 3, 32, 32, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            t_model = (t * 999.0).clamp_min(0.0)
            pred = model(x, t_model)
            ct = eqm_ct(t, a=a, gain=gain).view(B, 1, 1, 1).clamp_min(1e-3)
            v = pred / ct
            x = x + v * dt
        x = torch.clamp(x, -1.0, 1.0)
        all_samples.append(x.cpu())
        remaining -= B
    return torch.cat(all_samples, dim=0)[:num_samples]


def compute_model_fid(model, inception, ref_mu, ref_sigma, num_samples,
                      batch_size, sampler_fn, device):
    samples = sampler_fn(model, num_samples, batch_size, device)
    feats = []
    for i in range(0, samples.size(0), 64):
        batch = samples[i:i + 64].to(device)
        feats.append(inception(batch).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu, sigma = compute_statistics(feats)
    return compute_fid(ref_mu, ref_sigma, mu, sigma)


# ----------------------------- EMA -------------------------------------------

def ema_update(ema_model, model, decay):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ----------------------------- Main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--epochs", type=int, default=1800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every-epochs", type=int, default=100)
    ap.add_argument("--eval-fid-samples", type=int, default=5000)
    ap.add_argument("--final-fid-samples", type=int, default=50000)
    ap.add_argument("--sampler", choices=["gd", "euler"], default="euler",
                    help="GD = optimization-based (paper style); euler = "
                         "integrate rescaled velocity field")
    ap.add_argument("--gd-num-steps", type=int, default=500)
    ap.add_argument("--gd-step-size", type=float, default=1.0)
    ap.add_argument("--euler-num-steps", type=int, default=50)
    ap.add_argument("--sample-batch-size", type=int, default=256)
    ap.add_argument("--a", type=float, default=0.8, help="EqM c(t) interp point")
    ap.add_argument("--gain", type=float, default=4.0, help="EqM c(t) gain")
    ap.add_argument("--train-eps", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--reference-stats-path",
        default="projects/diff-EqM/results/cifar10_inception_stats.npz",
    )
    ap.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint to resume from")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed: {args.seed}  sampler: {args.sampler}", flush=True)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = build_unet().to(device)
    ema_model = deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        ema_model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params/1e6:.2f}M", flush=True)

    inception = InceptionV3Features().to(device).eval()
    ref_cfg = {"reference_stats_path": args.reference_stats_path,
               "data_dir": args.data_dir}
    ref_mu, ref_sigma = get_or_compute_reference_stats(ref_cfg, inception, device)
    print("Reference stats loaded.", flush=True)

    def make_sampler_fn():
        if args.sampler == "gd":
            return lambda m, n, b, d: sample_gd(
                m, n, b, args.gd_num_steps, args.gd_step_size, d,
                a=args.a, gain=args.gain)
        return lambda m, n, b, d: sample_euler(
            m, n, b, args.euler_num_steps, d, a=args.a, gain=args.gain)

    log_lines = []
    t0 = time.time()
    step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_loss = 0.0
        nb = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            loss = eqm_loss(model, x, device, eps=args.train_eps,
                            a=args.a, gain=args.gain)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema_update(ema_model, model, decay=args.ema_decay)
            ep_loss += loss.item()
            nb += 1
            step += 1
        avg = ep_loss / max(nb, 1)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} loss {avg:.4f} step {step} "
              f"elapsed {elapsed:.0f}s", flush=True)
        log_lines.append(f"{epoch+1}\t{step}\t{avg:.6f}\t{elapsed:.1f}")

        # periodic checkpoint + small FID
        if (epoch + 1) % args.eval_every_epochs == 0 or (epoch + 1) == args.epochs:
            ema_model.eval()
            fid = compute_model_fid(
                ema_model, inception, ref_mu, ref_sigma,
                num_samples=args.eval_fid_samples,
                batch_size=args.sample_batch_size,
                sampler_fn=make_sampler_fn(), device=device,
            )
            print(f"  [eval] epoch {epoch+1} FID(ema, {args.eval_fid_samples} samples, "
                  f"{args.sampler}) = {fid:.4f}", flush=True)
            log_lines.append(f"eval\t{epoch+1}\t{fid:.6f}")
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
            }, out / "checkpoint.pt")

    # Final 50K-sample FID
    ema_model.eval()
    final_fid = compute_model_fid(
        ema_model, inception, ref_mu, ref_sigma,
        num_samples=args.final_fid_samples,
        batch_size=args.sample_batch_size,
        sampler_fn=make_sampler_fn(), device=device,
    )
    # grep-friendly line
    print(f"cifar10_eqm_fid: {final_fid:.4f}", flush=True)
    log_lines.append(f"final\t{args.final_fid_samples}\t{final_fid:.6f}")

    torch.save({
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": args.epochs,
        "final_fid": final_fid,
        "args": vars(args),
    }, out / "final.pt")
    (out / "train_log.tsv").write_text("\n".join(log_lines) + "\n")
    print(f"Saved final checkpoint and log to {out}", flush=True)


if __name__ == "__main__":
    main()
