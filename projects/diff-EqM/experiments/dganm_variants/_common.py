"""Shared infra for DG-ANM variant trainers.

Every variant in this folder gets the same backbone (FM UNet), EqM loss,
Euler sampler, CIFAR-10 loader, FID evaluator, EMA, and training harness.
Variants differ ONLY in how they construct the per-step loss (the `step_fn`
they hand to `train_loop`).

This keeps variant files small and makes diffs between variants actually
readable — if two variants disagree on UNet config, loader, or FID pipeline,
that's a bug, not a design choice.
"""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[4]
FM_UPSTREAM = REPO_ROOT / "projects" / "diff-EqM" / "fm-upstream"
if str(FM_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(FM_UPSTREAM))
from models.unet import UNetModel  # noqa: E402

_EVAL_FID_PATH = (
    REPO_ROOT / "projects" / "diff-EqM" / "experiments" / "evaluate_fid.py"
)
_spec = importlib.util.spec_from_file_location("diffeqm_evaluate_fid", _EVAL_FID_PATH)
assert _spec is not None and _spec.loader is not None
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
InceptionV3Features = _eval_mod.InceptionV3Features
compute_fid = _eval_mod.compute_fid
compute_statistics = _eval_mod.compute_statistics
get_or_compute_reference_stats = _eval_mod.get_or_compute_reference_stats


CIFAR10_UNET_CONFIG = dict(
    in_channels=3, model_channels=128, out_channels=3,
    num_res_blocks=4, attention_resolutions=[2], dropout=0.3,
    channel_mult=[2, 2, 2], conv_resample=False, dims=2,
    num_classes=None, use_checkpoint=False, num_heads=1,
    num_head_channels=-1, num_heads_upsample=-1,
    use_scale_shift_norm=True, resblock_updown=False,
    use_new_attention_order=True, with_fourier_features=False,
)


class UNetWrapper(nn.Module):
    def __init__(self, unet: UNetModel):
        super().__init__()
        self.unet = unet
        self._features = None

        def _hook(_mod, _inp, out):
            self._features = out
        self.unet.middle_block.register_forward_hook(_hook)

    def forward(self, x, timesteps, return_features: bool = False,
                extra: Optional[dict] = None):
        if extra is None:
            extra = {}
        self._features = None
        out = self.unet(x, timesteps, extra)
        if return_features:
            return out, self._features
        return out


def build_unet() -> UNetWrapper:
    return UNetWrapper(UNetModel(**CIFAR10_UNET_CONFIG))


def eqm_ct(t: torch.Tensor, a: float = 0.8, gain: float = 4.0) -> torch.Tensor:
    start = 1.0
    c = torch.minimum(
        start - (start - 1.0) / a * t,
        1.0 / (1.0 - a) - 1.0 / (1.0 - a) * t,
    )
    return c * gain


def eqm_loss(model: nn.Module, x1: torch.Tensor, device: torch.device,
             eps: float = 1e-3, a: float = 0.8, gain: float = 4.0) -> torch.Tensor:
    B = x1.size(0)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps) + eps
    t_ = t.view(B, 1, 1, 1)
    xt = (1.0 - t_) * x0 + t_ * x1
    ut = x1 - x0
    ct = eqm_ct(t, a=a, gain=gain).view(B, 1, 1, 1)
    target = ct * ut
    t_model = (t * 999.0).clamp_min(0.0)
    pred = model(xt, t_model)
    return F.mse_loss(pred, target)


@torch.no_grad()
def sample_euler(model: nn.Module, num_samples: int, batch_size: int,
                 num_steps: int, device: torch.device,
                 a: float = 0.8, gain: float = 4.0) -> torch.Tensor:
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
                      batch_size, num_steps, device, a=0.8, gain=4.0):
    samples = sample_euler(model, num_samples, batch_size, num_steps, device,
                           a=a, gain=gain)
    feats = []
    for i in range(0, samples.size(0), 64):
        batch = samples[i:i + 64].to(device)
        feats.append(inception(batch).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu, sigma = compute_statistics(feats)
    return compute_fid(ref_mu, ref_sigma, mu, sigma)


def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def build_cifar_loader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


@dataclass
class TrainArgs:
    output_dir: str
    variant: str
    data_dir: str = "./data"
    epochs: int = 25
    batch_size: int = 128
    lr: float = 2e-4
    ema_decay: float = 0.9999
    seed: int = 0
    eval_every_epochs: int = 25
    eval_fid_samples: int = 5000
    final_fid_samples: int = 10000
    sample_batch_size: int = 256
    euler_num_steps: int = 50
    a: float = 0.8
    gain: float = 4.0
    train_eps: float = 1e-3
    num_workers: int = 4
    reference_stats_path: str = "projects/diff-EqM/results/cifar10_inception_stats.npz"
    resume: Optional[str] = None
    # Variant-specific knobs live in `extras` so new variants add fields
    # without touching CLI plumbing.
    extras: Optional[dict] = None


# step_fn signature: (model, x_batch, step_idx, device, args) -> (total_loss, diag_dict)
StepFn = Callable[[nn.Module, torch.Tensor, int, torch.device, TrainArgs], tuple]


def train_loop(args: TrainArgs, step_fn: StepFn, diag_keys: list[str]) -> float:
    """Generic training loop. Variant supplies `step_fn` building the loss.

    Returns final FID (also prints grep-friendly `cifar10_dganm_fid: <value>`).
    """
    import time

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} seed: {args.seed} variant: {args.variant}", flush=True)

    loader = build_cifar_loader(args.data_dir, args.batch_size, args.num_workers)

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

    log_lines = []
    t0 = time.time()
    step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_total = 0.0
        ep_diag = {k: 0.0 for k in diag_keys}
        nb = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)

            total_loss, diag = step_fn(model, x, step, device, args)
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()
            ema_update(ema_model, model, decay=args.ema_decay)

            ep_total += total_loss.item()
            for k in diag_keys:
                ep_diag[k] += float(diag.get(k, 0.0))
            nb += 1
            step += 1

        avg_total = ep_total / max(nb, 1)
        diag_str = " ".join(f"{k} {ep_diag[k]/max(nb,1):.4f}" for k in diag_keys)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} total {avg_total:.4f} {diag_str} "
              f"step {step} elapsed {elapsed:.0f}s", flush=True)
        log_lines.append(
            "\t".join([str(epoch+1), str(step), f"{avg_total:.6f}"] +
                      [f"{ep_diag[k]/max(nb,1):.6f}" for k in diag_keys] +
                      [f"{elapsed:.1f}"]))

        if (epoch + 1) % args.eval_every_epochs == 0 or (epoch + 1) == args.epochs:
            ema_model.eval()
            fid = compute_model_fid(
                ema_model, inception, ref_mu, ref_sigma,
                num_samples=args.eval_fid_samples,
                batch_size=args.sample_batch_size,
                num_steps=args.euler_num_steps,
                device=device, a=args.a, gain=args.gain,
            )
            print(f"  [eval] epoch {epoch+1} FID(ema, {args.eval_fid_samples} "
                  f"samples) = {fid:.4f}", flush=True)
            log_lines.append(f"eval\t{epoch+1}\t{fid:.6f}")
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args) if hasattr(args, "__dict__") else dict(args.__dict__),
                "variant": args.variant,
            }, out / "checkpoint.pt")

    ema_model.eval()
    final_fid = compute_model_fid(
        ema_model, inception, ref_mu, ref_sigma,
        num_samples=args.final_fid_samples,
        batch_size=args.sample_batch_size,
        num_steps=args.euler_num_steps,
        device=device, a=args.a, gain=args.gain,
    )
    print(f"cifar10_dganm_fid: {final_fid:.4f}", flush=True)
    print(f"cifar10_variant_fid[{args.variant}]: {final_fid:.4f}", flush=True)
    log_lines.append(f"final\t{args.final_fid_samples}\t{final_fid:.6f}")

    torch.save({
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": args.epochs,
        "final_fid": final_fid,
        "variant": args.variant,
    }, out / "final.pt")
    (out / "train_log.tsv").write_text("\n".join(log_lines) + "\n")
    print(f"Saved final checkpoint and log to {out}", flush=True)
    return final_fid
