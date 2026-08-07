"""
Temporal stability of the credit-assignment correction g_cache, per Yilun's
round-2 proposal (2026-08-06/07), after the blockwise calibration test
(job 37544288) killed the scale-mismatch hypothesis (rho_cal 0.6508 <=
rho_uncal 0.6580, per-block direction cosine low, 0.24-0.51, inside every
transformer block).

This is the last cheap offline test before committing to periodic exact
correction as a training method, or abandoning "avoid double backward"
entirely per Yilun's decision tree.

Definitions (g_cache(theta;B) := g_exact(theta;B) - g_semi(theta;B), both
full concatenated active-parameter vectors in fixed registry order):

  T_param(Delta)   = cos( g_cache(theta_0;B), g_cache(theta_Delta;B) )
                      -- same probe batch B, held fixed; measures whether
                      the CORRECTION ITSELF drifts as theta moves.
  rho_stale(Delta) = cos( g_semi(theta_Delta;B) + g_cache(theta_0;B),
                           g_exact(theta_Delta;B) )
                      -- same batch B; is a stale (same-batch) correction
                      still useful Delta steps later?
  rho_cross(Delta) = cos( g_semi(theta_Delta;B') + g_cache(theta_0;B),
                           g_exact(theta_Delta;B') )
                      -- B' != B, a FRESH batch; this is the real periodic-
                      correction scenario (batch changes every step too).
  rho_uncal(Delta) = cos( g_semi(theta_Delta;B), g_exact(theta_Delta;B) )
                      -- no correction at all, for reference (same batch B).

theta_0..theta_64 come from ONE continuous real training trajectory: start
at the epoch-40 checkpoint (theta_0), train model_direct with the EXACT
double-backward method (same loss/optimizer/grad-clip as Gate 2's control
arm, job 37535954: AdamW lr=1e-4 wd=0, max_grad_norm=6.87141) for up to
max(deltas) steps, snapshotting theta's state_dict at each Delta in the
sweep. g_cache(theta_0;B) is computed once per probe batch before training
starts and held fixed for every Delta -- this directly tests "if I compute
the exact correction once, how many steps can I reuse it for."

Decision rule (Yilun's): rho_cross(K) >~ 0.9 for a useful K>=8 -> periodic
exact correction every K steps is worth a training test. Only Delta=1 or 2
working -> compute savings not compelling. Same-batch stable but cross-batch
collapses -> correction is sample-specific, EMA of raw correction is dead.

Same methodology as prior diagnostics: TF32 disabled, MATH-backend SDPA,
real ImageNet batches.

Run (single GPU, gpu_test or seas_gpu):
  python experiments/direct_energy/temporal_correction_stability.py \
      --ckpt <epoch40.pt> --data-path <imagenet train dir> \
      --num-probe 10 --num-cross 5 --deltas 1,2,4,8,16,32,64
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import EqM_models
from transport import create_transport
from train import center_crop_arr
from fb_direct.trainer import ForwardBackwardsDirectTrainer
from fb_direct.forward_cache import forward_energy_with_cache

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def mean_flat(x):
    return x.mean(dim=list(range(1, len(x.shape))))


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def summarize(vals):
    vals = sorted(v for v in vals if v is not None)
    return {"median": percentile(vals, 50), "p10": percentile(vals, 10),
            "p90": percentile(vals, 90), "n": len(vals)}


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def sample_transport_batch(loader_iter, loader, vae, transport, device):
    try:
        x, y = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        x, y = next(loader_iter)
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
    t, x0, x1 = transport.sample(x1)
    t = t.to(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    ut = ut * transport.get_ct(t)[:, None, None, None]
    return loader_iter, (xt.detach(), t.detach(), y.detach(), ut.detach())


def exact_grad_vec(model_direct, active_pairs, theta_named, batch):
    xt, t, y, ut = batch
    model_direct.zero_grad(set_to_none=True)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        xt_exact = xt.detach().clone().requires_grad_(True)
        field = model_direct(xt_exact, t, y, train=True)
    loss = mean_flat((field - ut) ** 2).mean()
    loss.backward()
    parts = [theta_named[fname].grad.detach().reshape(-1).float() for fname, _ in active_pairs]
    return torch.cat(parts), float(loss.detach())


def semi_grad_vec(fb_trainer, active_pairs, phi_named, batch):
    xt, t, y, ut = batch
    fb_trainer.registry.tie_from_forward_()
    fb_trainer.optimizer.zero_grad(set_to_none=True)
    _, cache = forward_energy_with_cache(fb_trainer.theta, xt, t, y)
    grad_tilde = fb_trainer.phi(cache)
    prediction_fb = -grad_tilde
    loss = mean_flat((prediction_fb - ut) ** 2).mean()
    loss.backward()
    parts = [phi_named[bname].grad.detach().reshape(-1).float() for _, bname in active_pairs]
    return torch.cat(parts), float(loss.detach())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-probe", type=int, default=10)
    p.add_argument("--num-cross", type=int, default=5)
    p.add_argument("--deltas", default="1,2,4,8,16,32,64")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=6.87141)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    deltas = sorted(int(d) for d in args.deltas.split(","))

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // 8

    raw = torch.load(args.ckpt, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw

    model_direct = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="direct",
    ).to(device)
    missing, unexpected = model_direct.load_state_dict(state_dict, strict=False)
    print(f"[temporal] direct model load: missing={missing} unexpected={unexpected}")

    model_fb = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="forward-backwards-direct",
    ).to(device)
    model_fb.load_state_dict(state_dict, strict=False)
    fb_trainer = ForwardBackwardsDirectTrainer(model_fb, device=device)
    fb_trainer.registry.tie_from_forward_()
    print(f"[temporal] phi/theta sync error after tie: {fb_trainer.registry.compute_sync_error():.3e}")

    active_pairs = [
        (e.forward_name, e.backward_name)
        for e in fb_trainer.registry.entries
        if e.category in ("reverse_active", "recomputed_conditioning")
    ]
    theta_named = dict(model_direct.named_parameters())
    phi_named = dict(fb_trainer.phi.named_parameters())
    print(f"[temporal] {len(active_pairs)} matched pairs; deltas={deltas}")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    transport = create_transport("Linear", "velocity", None, None, None)

    train_it = iter(loader)
    probe_it = iter(loader)
    cross_it = iter(loader)

    # --- fixed probe bank: materialized (xt,t,y,ut), held fixed for every Delta ---
    probe_batches = []
    for _ in range(args.num_probe):
        probe_it, batch = sample_transport_batch(probe_it, loader, vae, transport, device)
        probe_batches.append(batch)

    # --- g_cache(theta_0; B) computed once, before any training step ---
    g_cache_0 = []
    for batch in probe_batches:
        g_exact_0, _ = exact_grad_vec(model_direct, active_pairs, theta_named, batch)
        g_semi_0, _ = semi_grad_vec(fb_trainer, active_pairs, phi_named, batch)
        g_cache_0.append((g_exact_0 - g_semi_0).clone())
    print(f"[temporal] computed g_cache(theta_0;B) for {len(g_cache_0)} probe batches")

    # --- real training trajectory: exact double-backward, AdamW, grad clip ---
    optimizer = torch.optim.AdamW(model_direct.parameters(), lr=args.lr, weight_decay=0)
    results_by_delta = {}
    step = 0
    for target_step in deltas:
        while step < target_step:
            train_it, batch = sample_transport_batch(train_it, loader, vae, transport, device)
            xt, t, y, ut = batch
            model_direct.train()
            model_direct.zero_grad(set_to_none=True)
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                field = model_direct(xt, t, y, train=True)
            loss = mean_flat((field - ut) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_direct.parameters(), args.max_grad_norm)
            optimizer.step()
            step += 1
        model_direct.eval()

        # snapshot theta_step into the fb model for diagnostics
        model_fb.load_state_dict(model_direct.state_dict(), strict=False)
        fb_trainer.registry.tie_from_forward_()
        sync_err = fb_trainer.registry.compute_sync_error()

        T_param, rho_stale, rho_uncal = [], [], []
        for i, batch in enumerate(probe_batches):
            g_exact_s, _ = exact_grad_vec(model_direct, active_pairs, theta_named, batch)
            g_semi_s, _ = semi_grad_vec(fb_trainer, active_pairs, phi_named, batch)
            g_cache_s = g_exact_s - g_semi_s
            T_param.append(cos(g_cache_0[i], g_cache_s))
            g_hat_stale = g_semi_s + g_cache_0[i]
            rho_stale.append(cos(g_hat_stale, g_exact_s))
            rho_uncal.append(cos(g_semi_s, g_exact_s))

        rho_cross = []
        for k in range(args.num_cross):
            cross_it, batch_fresh = sample_transport_batch(cross_it, loader, vae, transport, device)
            g_exact_f, _ = exact_grad_vec(model_direct, active_pairs, theta_named, batch_fresh)
            g_semi_f, _ = semi_grad_vec(fb_trainer, active_pairs, phi_named, batch_fresh)
            probe_idx = k % len(g_cache_0)
            g_hat_cross = g_semi_f + g_cache_0[probe_idx]
            rho_cross.append(cos(g_hat_cross, g_exact_f))

        results_by_delta[target_step] = {
            "sync_error_after_tie": sync_err,
            "T_param": summarize(T_param),
            "rho_stale": summarize(rho_stale),
            "rho_uncal": summarize(rho_uncal),
            "rho_cross": summarize(rho_cross),
        }
        print(f"[temporal] Delta={target_step} (step={step}): "
              f"T_param={results_by_delta[target_step]['T_param']['median']:.4f} "
              f"rho_stale={results_by_delta[target_step]['rho_stale']['median']:.4f} "
              f"rho_cross={results_by_delta[target_step]['rho_cross']['median']:.4f} "
              f"rho_uncal={results_by_delta[target_step]['rho_uncal']['median']:.4f}")

    summary = {
        "checkpoint": args.ckpt,
        "deltas": deltas,
        "num_probe": args.num_probe,
        "num_cross": args.num_cross,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "results_by_delta": results_by_delta,
    }
    print(f"[temporal] SUMMARY: {json.dumps(summary, indent=2)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
