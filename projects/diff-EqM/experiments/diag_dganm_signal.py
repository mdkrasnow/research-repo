#!/usr/bin/env python3
"""
Minimal DG-ANM signal diagnostic (fast — <30s on CPU).

The ONE question: at the hyperparameters that "won" the IN-100 autoresearch
(neg_margin=5), does ReLU(margin - ||field||) at mined negatives actually
produce a nonzero gradient to the model parameters? Or is it saturated to 0
and therefore inert?

Approach:
  1. Build fresh FM CIFAR UNet.
  2. One tiny batch (bs=8) — we're not training, just probing.
  3. Forward at t=1 on (a) real images and (b) gaussian noise + small delta
     (a proxy for what mined negatives look like, without actually running PGA).
  4. Measure ||field|| distribution.
  5. For margin in [5, 20, 50, 100, p50, p90]: compute neg_loss and the
     gradient norm wrt model params.
  6. Verdict.

Usage: python projects/diff-EqM/experiments/diag_dganm_signal.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
FM_UPSTREAM = REPO_ROOT / "projects" / "diff-EqM" / "fm-upstream"
sys.path.insert(0, str(FM_UPSTREAM))
from models.unet import UNetModel  # noqa: E402

CFG = dict(
    in_channels=3, model_channels=128, out_channels=3,
    num_res_blocks=4, attention_resolutions=[2], dropout=0.3,
    channel_mult=[2, 2, 2], conv_resample=False, dims=2,
    num_classes=None, use_checkpoint=False, num_heads=1,
    num_head_channels=-1, num_heads_upsample=-1,
    use_scale_shift_norm=True, resblock_updown=False,
    use_new_attention_order=True, with_fourier_features=False,
)


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += g.pow(2).sum().item()
    return float(np.sqrt(sq))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cpu"
    t0 = time.time()
    print(f"[{time.time()-t0:5.1f}s] device={device}", flush=True)

    print(f"[{time.time()-t0:5.1f}s] Building UNet...", flush=True)
    net = UNetModel(**CFG).to(device)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    print(f"[{time.time()-t0:5.1f}s]   params={sum(p.numel() for p in params)/1e6:.2f}M",
          flush=True)

    print(f"[{time.time()-t0:5.1f}s] Loading CIFAR batches (bs=8)...", flush=True)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    ds = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    # Stack a few batches for a short training warmup to break zero_module init
    batches = [torch.stack([ds[i + 8*b][0] for i in range(8)]).to(device)
               for b in range(20)]
    x = batches[0]
    print(f"[{time.time()-t0:5.1f}s]   got {len(batches)} batches, "
          f"x.shape={tuple(x.shape)}", flush=True)

    # Break the zero_module init with a tiny warmup: N steps of plain
    # flow-matching loss. Without this, |field|=0 identically, which is the
    # state at step 0 but NOT the state by the time DG-ANM's margin kicks in.
    WARMUP = 30
    print(f"[{time.time()-t0:5.1f}s] Warmup: {WARMUP} steps of plain "
          f"FM loss to move out of zero_module init...", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    for step in range(WARMUP):
        xb = batches[step % len(batches)]
        B = xb.size(0)
        x0 = torch.randn_like(xb)
        t = torch.rand(B, device=device) * 0.998 + 0.001
        t_ = t.view(B, 1, 1, 1)
        xt = (1 - t_) * x0 + t_ * xb
        target = xb - x0  # plain FM target (no c(t) weighting for speed)
        t_model = (t * 999).clamp_min(0)
        pred = net(xt, t_model, {})
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == WARMUP - 1:
            print(f"[{time.time()-t0:5.1f}s]    warmup step {step}: "
                  f"loss={loss.item():.4f}", flush=True)

    # Build "negative-ish" inputs: real + small random perturbation (proxy for
    # the PGA output; what matters is the field-norm magnitude at near-data
    # perturbed points).
    delta = torch.randn_like(x) * 0.8  # eps=0.8 matches our mining budget
    x_neg = x + delta
    t_ones = torch.ones(x.size(0), device=device) * 999.0

    print(f"[{time.time()-t0:5.1f}s] Forward pass at (x + delta, t=1)...", flush=True)
    field = net(x_neg, t_ones, {})
    field_norm = field.flatten(1).norm(dim=1)
    fn = field_norm.detach()
    print(f"[{time.time()-t0:5.1f}s]   done", flush=True)

    print("\n=== ||field|| at near-data perturbed inputs (fresh UNet) ===",
          flush=True)
    print(f"  shape = {tuple(field.shape)}", flush=True)
    print(f"  per-sample |field|:", flush=True)
    for v in fn.tolist():
        print(f"    {v:10.4f}", flush=True)
    stats = {
        "min":  fn.min().item(),
        "p10":  fn.quantile(0.10).item(),
        "p50":  fn.quantile(0.50).item(),
        "mean": fn.mean().item(),
        "p90":  fn.quantile(0.90).item(),
        "max":  fn.max().item(),
    }
    print(f"  summary: min={stats['min']:.2f} p10={stats['p10']:.2f} "
          f"p50={stats['p50']:.2f} mean={stats['mean']:.2f} "
          f"p90={stats['p90']:.2f} max={stats['max']:.2f}", flush=True)

    print("\n=== margin sweep: is ReLU(m - |field|) producing gradient? ===",
          flush=True)
    print(f"  {'margin':>8} | {'frac<m':>7} | {'neg_loss':>10} | "
          f"{'||grad wrt model params||':>28}", flush=True)
    print("  " + "-" * 64, flush=True)
    margin_candidates = [1.0, 5.0, 20.0, 50.0, 100.0,
                         stats["p50"] * 0.9, stats["p50"], stats["p90"]]
    for margin in margin_candidates:
        frac = (fn < margin).float().mean().item()
        neg_loss = F.relu(margin - field_norm).mean()
        if frac > 0:
            gn = grad_norm(neg_loss, params)
        else:
            # neg_loss is exactly 0 — no grad graph above zeros.
            gn = 0.0
        flag = " <-- ours" if abs(margin - 5.0) < 1e-6 else ""
        print(f"  {margin:8.3f} | {frac*100:5.1f}% | "
              f"{neg_loss.item():10.6f} | {gn:28.10f}{flag}", flush=True)

    print(f"\n[{time.time()-t0:5.1f}s] done", flush=True)

    print("\n=== verdict ===", flush=True)
    if stats["min"] > 5.0:
        print("  EVERY sample has |field| > 5. At margin=5, neg_loss = 0 exactly.",
              flush=True)
        print("  Gradient to model params is 0. DG-ANM is a no-op at this margin.",
              flush=True)
        print(f"  To get signal, use margin >= ~{stats['p50']:.1f} "
              f"(p50 on this sample).", flush=True)
    else:
        print(f"  Some samples ({((fn<5).float().mean().item())*100:.0f}%) have "
              "|field| < 5. neg_loss is small-but-nonzero.", flush=True)
        print("  Check ||grad|| row at margin=5 above for actual signal magnitude.",
              flush=True)


if __name__ == "__main__":
    main()
