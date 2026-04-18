#!/usr/bin/env python3
"""
Stage A.5 Step A (DG-ANM variant) — DG-ANM on CIFAR-10 via FM UNet.

Parallel to train_cifar_eqm_unet.py (vanilla). Same backbone, same EqM
c(t)-weighted loss, same sampler, same eval pipeline. Adds differential-
geometry-guided adversarial negative mining on top, using the current
best-so-far hyperparameters from the IN-100 autoresearch proxy:

  gamma=6.0, mining_epsilon=0.8, mining_steps=3, mine_every=5,
  neg_margin=5.0, lr=2e-4

Important caveat: these hyperparameters were tuned on a completely
different regime (2-epoch IN-100, bs=16, transformer backbone). They
may not transfer to 1800-epoch CIFAR bs=64 UNet. This run is exploratory.

Exit criterion: FID <= vanilla's FID by >=1.0 on 50K samples (or see
where we land vs. the concurrent vanilla run).

Emits "cifar10_dganm_fid: <value>" on the final line.
"""

import argparse
import importlib.util
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
FM_UPSTREAM = REPO_ROOT / "projects" / "diff-EqM" / "fm-upstream"
sys.path.insert(0, str(FM_UPSTREAM))
from models.unet import UNetModel  # noqa: E402

_EVAL_FID_PATH = Path(__file__).resolve().parent / "evaluate_fid.py"
_spec = importlib.util.spec_from_file_location("diffeqm_evaluate_fid", _EVAL_FID_PATH)
assert _spec is not None and _spec.loader is not None
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
InceptionV3Features = _eval_mod.InceptionV3Features
compute_fid = _eval_mod.compute_fid
compute_statistics = _eval_mod.compute_statistics
get_or_compute_reference_stats = _eval_mod.get_or_compute_reference_stats


# ---------------- Model wrapper (with feature hook) --------------------------

class UNetWrapper(nn.Module):
    """Wraps FM UNet with extra={} default and middle-block feature hook."""

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


CIFAR10_UNET_CONFIG = dict(
    in_channels=3, model_channels=128, out_channels=3,
    num_res_blocks=4, attention_resolutions=[2], dropout=0.3,
    channel_mult=[2, 2, 2], conv_resample=False, dims=2,
    num_classes=None, use_checkpoint=False, num_heads=1,
    num_head_channels=-1, num_heads_upsample=-1,
    use_scale_shift_norm=True, resblock_updown=False,
    use_new_attention_order=True, with_fourier_features=False,
)


def build_unet():
    return UNetWrapper(UNetModel(**CIFAR10_UNET_CONFIG))


# ---------------- EqM loss + samplers ----------------------------------------

def eqm_ct(t, a=0.8, gain=4.0):
    start = 1.0
    c = torch.minimum(start - (start - 1.0) / a * t,
                      1.0 / (1.0 - a) - 1.0 / (1.0 - a) * t)
    return c * gain


def eqm_loss(model, x1, device, eps=1e-3, a=0.8, gain=4.0):
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
def sample_euler(model, num_samples, batch_size, num_steps, device,
                 a=0.8, gain=4.0):
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


# ---------------- DG-ANM geometry + mining -----------------------------------

def estimate_local_geometry(features, k=10):
    """Feature-space kNN -> tangent / normal projectors (per-sample).

    features: (B, D)
    Returns P_T, P_N each (B, D, D).
    """
    B, D = features.shape
    device = features.device
    k = min(k, B - 1)
    dists = torch.cdist(features, features)  # (B, B)
    nn_idx = dists.topk(k=k + 1, largest=False).indices[:, 1:]  # drop self

    P_T = torch.zeros(B, D, D, device=device)
    for i in range(B):
        nbrs = features[nn_idx[i]]                 # (k, D)
        centered = nbrs - features[i:i+1]          # (k, D)
        # tangent = column space of centered^T, via SVD
        U, S, _ = torch.linalg.svd(centered.T, full_matrices=False)  # U: (D, k)
        # keep top components above a small threshold
        rank = (S > 1e-6).sum().item()
        U_T = U[:, :rank]
        P_T[i] = U_T @ U_T.T
    P_N = torch.eye(D, device=device).unsqueeze(0) - P_T
    return P_T, P_N


def mine_negatives(
    model, x, features, P_N, P_T,
    epsilon=0.8, mining_steps=3, mining_lr=0.01,
    lambda_N=1.0, lambda_T=1.0, lambda_W=0.1,
    device=None,
):
    """PGA in pixel space maximizing feature-normal displacement &
    (-field norm), projected onto L2 epsilon-ball per sample.
    """
    B = x.shape[0]
    delta = torch.randn_like(x) * 0.01
    delta.requires_grad_(True)

    t_ones = torch.ones(B, device=device)
    field_norm = torch.zeros(B, device=device)
    delta_N = torch.zeros(B, features.shape[1], device=device)
    delta_T = torch.zeros(B, features.shape[1], device=device)

    for _ in range(mining_steps):
        x_neg = x.detach() + delta
        t_model = (t_ones * 999.0)
        field, neg_feats_spatial = model(x_neg, t_model, return_features=True)
        field_norm = field.flatten(1).norm(dim=1)

        # mean-pool middle-block features over spatial dims -> (B, D)
        neg_features = neg_feats_spatial.flatten(2).mean(dim=2)
        delta_phi = neg_features - features.detach()

        delta_N = torch.bmm(P_N.detach(), delta_phi.unsqueeze(-1)).squeeze(-1)
        delta_T = torch.bmm(P_T.detach(), delta_phi.unsqueeze(-1)).squeeze(-1)

        L_normal = delta_N.norm(dim=1) ** 2
        L_tan = delta_T.norm(dim=1) ** 2
        L_weak = -field_norm

        obj = (lambda_N * L_normal - lambda_T * L_tan + lambda_W * L_weak).mean()

        grad_delta = torch.autograd.grad(obj, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta = delta + mining_lr * grad_delta.sign()
            flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            delta = delta * torch.clamp(epsilon / (flat + 1e-8), max=1.0)
            delta = delta.detach().requires_grad_(True)

    x_neg = (x + delta).detach()
    info = {
        "avg_normal": delta_N.norm(dim=1).mean().item(),
        "avg_tangent": delta_T.norm(dim=1).mean().item(),
        "avg_field_norm_neg": field_norm.mean().item(),
    }
    return x_neg, info


def dganm_negative_loss(model, x_neg, margin=5.0, device=None):
    """Simple DG-ANM margin loss: hinge on (margin - |field|)."""
    B = x_neg.size(0)
    t_ones = torch.ones(B, device=device) * 999.0
    field = model(x_neg, t_ones)
    field_norm = field.flatten(1).norm(dim=1)
    return F.relu(margin - field_norm).mean()


# ---------------- EMA --------------------------------------------------------

def ema_update(ema_model, model, decay):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ---------------- Main -------------------------------------------------------

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
    ap.add_argument("--sample-batch-size", type=int, default=256)
    ap.add_argument("--euler-num-steps", type=int, default=50)
    ap.add_argument("--a", type=float, default=0.8)
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--train-eps", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    # DG-ANM specific
    ap.add_argument("--gamma", type=float, default=6.0,
                    help="weight of negative loss")
    ap.add_argument("--mining-epsilon", type=float, default=0.8)
    ap.add_argument("--mining-steps", type=int, default=3)
    ap.add_argument("--mining-lr", type=float, default=0.01)
    ap.add_argument("--mine-every", type=int, default=5)
    ap.add_argument("--neg-margin", type=float, default=5.0)
    ap.add_argument("--geometry-k", type=int, default=10)
    ap.add_argument(
        "--reference-stats-path",
        default="projects/diff-EqM/results/cifar10_inception_stats.npz",
    )
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} seed: {args.seed} gamma: {args.gamma}", flush=True)

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

    def sampler_fn(m, n, b, d):
        return sample_euler(m, n, b, args.euler_num_steps, d,
                            a=args.a, gain=args.gain)

    log_lines = []
    t0 = time.time()
    step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_loss = 0.0
        ep_base = 0.0
        ep_neg = 0.0
        nb = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)

            base_loss = eqm_loss(model, x, device, eps=args.train_eps,
                                 a=args.a, gain=args.gain)
            total_loss = base_loss
            neg_val = 0.0

            if args.gamma > 0 and args.mine_every > 0 and (step % args.mine_every == 0):
                # Geometry: features at anchors (no grad through model).
                with torch.no_grad():
                    t_ones = torch.ones(x.size(0), device=device) * 999.0
                    _, feats_spatial = model(x, t_ones, return_features=True)
                    anchor_features = feats_spatial.flatten(2).mean(dim=2)
                    P_T, P_N = estimate_local_geometry(
                        anchor_features.detach(), k=args.geometry_k)

                x_neg, mining_info = mine_negatives(
                    model, x, anchor_features, P_N, P_T,
                    epsilon=args.mining_epsilon,
                    mining_steps=args.mining_steps,
                    mining_lr=args.mining_lr,
                    device=device,
                )
                neg_loss = dganm_negative_loss(
                    model, x_neg, margin=args.neg_margin, device=device)
                total_loss = total_loss + args.gamma * neg_loss
                neg_val = neg_loss.item()

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()
            ema_update(ema_model, model, decay=args.ema_decay)

            ep_loss += total_loss.item()
            ep_base += base_loss.item()
            ep_neg += neg_val
            nb += 1
            step += 1

        avg_total = ep_loss / max(nb, 1)
        avg_base = ep_base / max(nb, 1)
        avg_neg = ep_neg / max(nb, 1)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} total {avg_total:.4f} "
              f"base {avg_base:.4f} neg {avg_neg:.4f} step {step} "
              f"elapsed {elapsed:.0f}s", flush=True)
        log_lines.append(
            f"{epoch+1}\t{step}\t{avg_total:.6f}\t{avg_base:.6f}\t{avg_neg:.6f}\t{elapsed:.1f}")

        if (epoch + 1) % args.eval_every_epochs == 0 or (epoch + 1) == args.epochs:
            ema_model.eval()
            fid = compute_model_fid(
                ema_model, inception, ref_mu, ref_sigma,
                num_samples=args.eval_fid_samples,
                batch_size=args.sample_batch_size,
                sampler_fn=sampler_fn, device=device,
            )
            print(f"  [eval] epoch {epoch+1} FID(ema, {args.eval_fid_samples} samples) "
                  f"= {fid:.4f}", flush=True)
            log_lines.append(f"eval\t{epoch+1}\t{fid:.6f}")
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
            }, out / "checkpoint.pt")

    ema_model.eval()
    final_fid = compute_model_fid(
        ema_model, inception, ref_mu, ref_sigma,
        num_samples=args.final_fid_samples,
        batch_size=args.sample_batch_size,
        sampler_fn=sampler_fn, device=device,
    )
    print(f"cifar10_dganm_fid: {final_fid:.4f}", flush=True)
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
