"""FD calibration + gradient-noise + weak-conservation probes at ImageNet scale.

Runs on FROZEN checkpoints, takes no optimizer steps, and never writes weights.
Its purpose is to de-risk the FD estimator BEFORE committing GPU-weeks:

  --do calibrate   the eps ladder of section 6 on a real B/2 energy model, with
                   exact autograd directional derivatives as ground truth
  --do gradnoise   per-arm stochastic parameter-gradient variance (section 19)
  --do weak        weak conservation residual R_psi on frozen linear/quadratic
                   probes (section 20)

Because it can be pointed at ANY checkpoint, running it on the existing
late-training `--ebm direct` checkpoint answers the question that actually
gates the campaign: is a central difference on this architecture's scalar
energy numerically usable at the energy scale training ends up at?
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from download import find_model
from models import EqM_models
from transport import create_transport

from .calibrate import DEFAULT_EPS_LADDER, calibrate_fd, pick_plateau
from .fd import assert_no_double_backward, rademacher_directions
from .image_losses import (
    BTMConfig,
    btm_loss,
    btm_sample,
    build_image_interpolant,
    frozen_label_dropout,
    phi_closure,
)


def center_crop_arr(pil_image, image_size):
    import numpy as np
    from PIL import Image
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size),
                                     resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size),
                                 resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy:cy + image_size, cx:cx + image_size])


def build(args, device):
    latent = args.image_size // 8
    model = EqM_models[args.model](input_size=latent, num_classes=1000,
                                   uncond=True, ebm=args.ebm).to(device)
    if args.ckpt:
        sd = find_model(args.ckpt)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-ema").to(device)
    tf = transforms.Compose([
        transforms.Lambda(lambda im: center_crop_arr(im, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True)])
    ds = ImageFolder(args.data_path, transform=tf)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True)
    return model, vae, dl


def encode(vae, x):
    with torch.no_grad():
        return vae.encode(x).latent_dist.sample().mul_(0.18215)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default="EqM-B/2")
    ap.add_argument("--ebm", default="direct")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--do", default="calibrate,gradnoise,weak")
    ap.add_argument("--fd-k", type=int, default=4)
    ap.add_argument("--tc", type=float, default=0.8)
    ap.add_argument("--eps-fd", type=float, default=1e-3)
    ap.add_argument("--n-grad-batches", type=int, default=64)
    ap.add_argument("--n-weak-batches", type=int, default=32)
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False   # TF32 destroyed a previous
    torch.backends.cudnn.allow_tf32 = False         # gate; never on for FD work
    if args.ebm != "none":
        # train.py does exactly this for every scalar-energy run: the fused
        # SDPA kernels have no double-backward implementation, so Arm G /
        # Arm A (which differentiate through grad_z E) raise
        # "derivative for aten::_scaled_dot_product_efficient_attention_backward
        # is not implemented" without it.  The FD arms do not need it, but the
        # arms must share a backend or their gradients are not comparable.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vae, dl = build(args, device)
    transport = create_transport("Linear", "velocity", "None", 0, 0)
    interp = build_image_interpolant(BTMConfig(tc=args.tc))
    todo = {s.strip() for s in args.do.split(",")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {"args": vars(args), "device": device, "results": {}}
    it = iter(dl)

    def next_batch():
        nonlocal it
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        return encode(vae, x.to(device)), y.to(device)

    # ------------------------------------------------------------ calibrate
    if "calibrate" in todo:
        x1, y = next_batch()
        t, x0, x1, z, zdot = btm_sample(transport, interp, x1)
        with frozen_label_dropout(model, y) as yy:
            phi = phi_closure(model, t, yy)
            rows = calibrate_fd(phi, z.detach(), K=args.fd_k,
                                eps_ladder=DEFAULT_EPS_LADDER, seed=args.seed,
                                chunk=args.batch * 4)
        eps, info = pick_plateau(rows)
        out["results"]["calibration"] = {"rows": rows, "eps_star": eps,
                                         "info": info}
        print(f"calibration eps* = {eps:g}", flush=True)
        for r in rows:
            print(f"  eps {r['eps_fd']:.0e} h {r['h_mean']:.3g} "
                  f"relRMSE {r['rel_rmse']:.4g} corr {r.get('corr', 0):.5f} "
                  f"cos {r.get('cosine', 0):.5f} cancel {r['cancel_ratio']:.3g} "
                  f"nonfin {r['nonfinite_frac']:.2g}", flush=True)

    # ------------------------------------------------------------ gradnoise
    if "gradnoise" in todo:
        arms = [("btm_scalar_exact", 1), ("btm_scalar_action_exact", 1),
                ("btm_scalar_fd_directional", 1),
                ("btm_scalar_fd_directional", 4),
                ("btm_scalar_fd_action", 1)]
        res = {}
        for mode, K in arms:
            cfg = BTMConfig(mode=mode, tc=args.tc, fd_eps=args.eps_fd, fd_k=K,
                            fd_chunk=args.batch * 4)
            grads, t0, peak = [], time.time(), 0
            torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
            for _ in range(args.n_grad_batches):
                x1, y = next_batch()
                model.zero_grad(set_to_none=True)
                loss, _ = btm_loss(model, model, cfg, transport, x1, y)
                if mode.startswith("btm_scalar_fd"):
                    with assert_no_double_backward():
                        loss.backward()
                else:
                    loss.backward()
                grads.append(torch.cat([
                    (p.grad.detach().reshape(-1) if p.grad is not None
                     else torch.zeros(p.numel(), device=device))
                    for p in model.parameters()]).float().cpu())
            if device == "cuda":
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
            model.zero_grad(set_to_none=True)
            G = torch.stack(grads).double()
            gbar = G.mean(0)
            dev = G - gbar
            Gn = G / (G.norm(dim=1, keepdim=True) + 1e-30)
            n = Gn.shape[0]
            idx = torch.triu_indices(n, n, offset=1)
            cos = (Gn[idx[0]] * Gn[idx[1]]).sum(1)
            rec = {
                "mode": mode, "K": K, "n_batches": args.n_grad_batches,
                "E_norm_sq": float((G ** 2).sum(1).mean()),
                "mean_norm_sq": float((gbar ** 2).sum()),
                "noise_norm_sq": float((dev ** 2).sum(1).mean()),
                "mean_pairwise_cosine": float(cos.mean()),
                "median_pairwise_cosine": float(cos.median()),
                "seconds_per_step": (time.time() - t0) / args.n_grad_batches,
                "peak_mem_GiB": peak,
            }
            rec["noise_scale"] = rec["noise_norm_sq"] / (rec["mean_norm_sq"] + 1e-30)
            rec["snr"] = rec["mean_norm_sq"] / (rec["noise_norm_sq"] + 1e-30)
            res[f"{mode}|K{K}"] = rec
            print(f"gradnoise {mode} K={K}: noise_scale {rec['noise_scale']:.4g} "
                  f"cos {rec['mean_pairwise_cosine']:.4f} "
                  f"{rec['seconds_per_step']:.2f}s/step "
                  f"peak {peak:.1f}GiB", flush=True)
            del G, grads
        out["results"]["gradnoise"] = res

    # ----------------------------------------------------------------- weak
    if "weak" in todo:
        # R_psi = E_nu[grad psi . grad phi] - (E_mu1 psi - E_mu0 psi)
        d = None
        gen = torch.Generator(device=device).manual_seed(4242)
        A = None
        lhs_lin = lhs_quad = None
        r1_lin = r0_lin = r1_quad = r0_quad = None
        cnt = 0
        for _ in range(args.n_weak_batches):
            x1, y = next_batch()
            t, x0, x1, z, zdot = btm_sample(transport, interp, x1)
            if A is None:
                d = z[0].numel()
                A = torch.randn(args.n_probes, d, device=device, generator=gen)
                A = A / A.norm(dim=1, keepdim=True)
                lhs_lin = torch.zeros(args.n_probes, device=device, dtype=torch.float64)
                lhs_quad = torch.zeros_like(lhs_lin)
                r1_lin = torch.zeros_like(lhs_lin); r0_lin = torch.zeros_like(lhs_lin)
                r1_quad = torch.zeros_like(lhs_lin); r0_quad = torch.zeros_like(lhs_lin)
            zz = z.detach().requires_grad_(True)
            with frozen_label_dropout(model, y) as yy:
                E = model(zz, t, yy, energy_only=True)
            g = torch.autograd.grad(E.sum(), zz, create_graph=False)[0].detach()
            gphi = (-g).reshape(z.shape[0], -1).double()
            with torch.no_grad():
                Ad = A.double()
                zf = z.reshape(z.shape[0], -1).double()
                x0f = x0.reshape(x0.shape[0], -1).double()
                x1f = x1.reshape(x1.shape[0], -1).double()
                # linear psi_a(x) = a.x  ->  grad psi = a
                lhs_lin += (gphi @ Ad.T).sum(0)
                r1_lin += (x1f @ Ad.T).sum(0)
                r0_lin += (x0f @ Ad.T).sum(0)
                # quadratic psi_a(x) = 0.5 (a.x)^2  ->  grad psi = (a.x) a
                proj_z = zf @ Ad.T                       # [B, P]
                lhs_quad += (proj_z * (gphi @ Ad.T)).sum(0)
                r1_quad += 0.5 * ((x1f @ Ad.T) ** 2).sum(0)
                r0_quad += 0.5 * ((x0f @ Ad.T) ** 2).sum(0)
                cnt += z.shape[0]
        res = {}
        for name, lhs, r1, r0 in (("linear", lhs_lin, r1_lin, r0_lin),
                                  ("quadratic", lhs_quad, r1_quad, r0_quad)):
            L = lhs / cnt
            R = (r1 - r0) / cnt
            scale = 0.5 * (L.abs() + R.abs()) + 1e-12
            rel = ((L - R).abs() / scale)
            res[name] = {"median_rel_residual": float(rel.median()),
                         "mean_rel_residual": float(rel.mean()),
                         "median_lhs": float(L.median()),
                         "median_rhs": float(R.median()),
                         "n_samples": cnt, "n_probes": args.n_probes}
            print(f"weak {name}: median rel residual "
                  f"{res[name]['median_rel_residual']:.4g}", flush=True)
        out["results"]["weak_conservation"] = res

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
