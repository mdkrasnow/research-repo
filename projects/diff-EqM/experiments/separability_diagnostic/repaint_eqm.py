"""RePaint-on-IN-1K-EqM — image-scale confirmation of the inpainting metacognition law.

The MNIST rung (mnist_difficulty_sweep.py) found: inpaint metacognition is NULL for
confident-wrong (semantic) failures but POSITIVE for structural/instability failures, which
extreme masks induce. This runs the SAME dual-oracle test on the REAL IN-1K EqM-B/2 checkpoint.

Pipeline (single-GPU; option-B latent clamp per REPAINT_EQM_SPEC.md):
  - Load N real ImageNet images -> VAE-encode to z_known (4x32x32).
  - Pixel mask (256) -> latent mask (32, ÷8). For each mask size:
      R draws of latent-clamp RePaint: init masked region = noise, observed = z_known;
      each GD step clamp observed latent to z_known; LOG per-step norm/dot over the
      MASKED latent region only (the inpaint dynamics).
  - Two oracles (mirror MNIST):
      LPIPS-to-GT (semantic / perceptual correctness; failure = high LPIPS) and
      TV-structural (no-reference coherence of the inpainted pixel region; failure =
      abnormal total-variation = broken/blobby = instability).
  - Fresh held-out de-confounded probe on masked-region dynamics predicts each oracle's
    failure. Action arms, equal NFE: vanilla / random / probe-restart / oracle.
  - Lead metric = detection AUROC + probe-restart−random gap (the metacognition delta).
    FID is NOT computed here (human-gated); LPIPS + AUROC are the reported numbers.

Smoke: python repaint_eqm.py --ckpt <...> --n 32 --fracs 0.5 0.9 --R 2 --num-sampling-steps 60
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

VAE_SCALE = 0.18215


def _deps():
    sys_paths = [str(Path(__file__).resolve().parents[2] / "eqm-upstream")]
    import sys
    for p in sys_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    from models import EqM_models
    from download import find_model
    from diffusers.models import AutoencoderKL
    return EqM_models, find_model, AutoencoderKL


def load_gt_images(real_dir, n, image_size, dev, seed=0):
    from compute_quality_labels import list_real_images
    from PIL import Image
    import torchvision.transforms as T
    files = list_real_images(real_dir, n, seed=seed)
    tf = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
    imgs = []
    for f in files:
        try:
            im = Image.open(f).convert("RGB")
            imgs.append(tf(im) * 2 - 1)                 # [-1,1]
        except Exception:
            continue
    x = torch.stack(imgs).to(dev)                       # (B,3,H,W)
    return x


def make_latent_mask(B, ls, frac, kind, dev, rng):
    """mask 1=inpaint/hidden, 0=observed, on the latent grid (ls x ls)."""
    m = torch.zeros(B, 1, ls, ls, device=dev)
    s = max(2, int(round(frac * ls)))
    if kind == "center":
        a = (ls - s) // 2
        m[:, :, a:a + s, a:a + s] = 1.0
    elif kind == "expand":                              # border observed, center grows
        m[:, :, :s, :] = 1.0; m[:, :, -s:, :] = 1.0
    else:                                               # random block
        for i in range(B):
            y0 = int(rng.integers(0, ls - s + 1)); x0 = int(rng.integers(0, ls - s + 1))
            m[i, :, y0:y0 + s, x0:x0 + s] = 1.0
    return m


@torch.no_grad()
def gd_repaint(model, z_known, mask_lat, y, eta, steps, log=True):
    """Latent-clamp RePaint. mask_lat:(B,1,ls,ls) 1=inpaint. Log norm/dot over masked region."""
    obs = (mask_lat < 0.5)
    xt = torch.where(obs, z_known, torch.randn_like(z_known))
    t = torch.zeros((xt.shape[0],), device=xt.device)
    norms, dots = [], []
    for _ in range(steps):
        out = model(xt, t, y)
        if not torch.is_tensor(out):
            out = out[0]
        out = out.detach()
        if log:
            fm = out * mask_lat                          # masked-region dynamics only
            norms.append(fm.flatten(1).norm(dim=1))
            dots.append((fm * xt).flatten(1).sum(dim=1))
        xt = xt + out * eta
        xt = torch.where(obs, z_known, xt)               # clamp observed
    if log:
        return xt.detach(), torch.stack(norms, 1).float().cpu().numpy(), torch.stack(dots, 1).float().cpu().numpy()
    return xt.detach()


def tv_structural(img, mask_px):
    """No-reference coherence: total-variation energy inside the inpainted pixel region.
    Natural completions are smooth-ish; broken/blobby ones spike TV. Returns per-image TV."""
    x = img
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    mh = mask_px[:, :, 1:, :]; mw = mask_px[:, :, :, 1:]
    tv = (dh * mh).flatten(1).sum(1) / (mh.flatten(1).sum(1) + 1e-6) + \
         (dw * mw).flatten(1).sum(1) / (mw.flatten(1).sum(1) + 1e-6)
    return tv.cpu().numpy()


def heldout_auc(X, y, ne, frac=0.3, seeds=5, l2=1.0):
    from learned_probe import fit_logreg, auc, within_norm_auc
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if not (0 < np.mean(y) < 1):
        return float("nan"), float("nan")
    raws, wns = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s); idx = rng.permutation(len(y)); nte = int(len(y) * frac)
        te, tr = idx[:nte], idx[nte:]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        w, b = fit_logreg((X[tr] - mu) / sd, y[tr], l2=l2)
        p = 1 / (1 + np.exp(-np.clip((X[te] - mu) / sd @ w + b, -30, 30)))
        raws.append(auc(y[te], p)); wns.append(within_norm_auc(p, y[te], ne[te]))
    return (float(np.mean(raws)) if raws else float("nan"),
            float(np.mean(wns)) if wns else float("nan"))


def action(feats_shape, dn, dd, valid, R, seed):
    """probe-restart vs random/oracle at equal NFE given a (R,B) validity matrix."""
    from probe_validate import feature_groups
    from learned_probe import fit_logreg
    def feats(r):
        return np.nan_to_num(feature_groups(dn[r], dd[r])["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0)
    X = np.concatenate([feats(r) for r in range(R)]); y = np.concatenate([(~valid[r]).astype(float) for r in range(R)])
    B = valid.shape[1]; bi = np.arange(B)
    if not (0 < y.mean() < 1):
        v = float(valid[0].mean())
        return {"vanilla": v, "random_restart": v, "probe_restart": v, "oracle_restart": float(valid.any(0).mean())}
    mu = X.mean(0); sd = X.std(0) + 1e-8; w, b = fit_logreg((X - mu) / sd, y, l2=1.0)
    P = np.stack([1 / (1 + np.exp(-np.clip((feats(r) - mu) / sd @ w + b, -30, 30))) for r in range(R)])
    rng = np.random.default_rng(seed)
    return {"vanilla": float(valid[0].mean()),
            "random_restart": float(valid[rng.integers(0, R, B), bi].mean()),
            "probe_restart": float(valid[np.argmin(P, 0), bi].mean()),
            "oracle_restart": float(valid.any(0).mean())}


def main(args):
    import lpips as lpips_lib
    EqM_models, find_model, AutoencoderKL = _deps()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ls = args.image_size // 8
    model = EqM_models[args.model](input_size=ls, num_classes=args.num_classes,
                                   uncond=True, ebm="none").to(dev)
    st = find_model(args.ckpt)
    model.load_state_dict(st["ema"] if "ema" in st else st.get("model", st))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(dev).eval()
    lpips_fn = lpips_lib.LPIPS(net="alex").to(dev)

    gt = load_gt_images(args.real_dir, args.n, args.image_size, dev, seed=0)
    B = gt.shape[0]
    with torch.no_grad():
        z_known = vae.encode(gt).latent_dist.mean * VAE_SCALE          # (B,4,ls,ls)
    y = torch.tensor([(i * 1315423911) % args.num_classes for i in range(B)], device=dev)

    @torch.no_grad()
    def decode(z):
        im = vae.decode(z / VAE_SCALE).sample
        return (im / 2 + 0.5).clamp(0, 1) * 2 - 1                      # [-1,1]

    rows = []
    for frac in args.fracs:
        rng = np.random.default_rng(args.seed)
        mask_lat = make_latent_mask(B, ls, frac, args.mask, dev, rng)
        mask_px = torch.nn.functional.interpolate(mask_lat, size=(args.image_size, args.image_size), mode="nearest")
        dn, dd, lp, tv = [], [], [], []
        for r in range(args.R):
            torch.manual_seed(args.seed * 100 + r)
            xt, norm, dot = gd_repaint(model, z_known, mask_lat, y, args.stepsize, args.num_sampling_steps, log=True)
            img = decode(xt)
            with torch.no_grad():
                d = lpips_fn(img, gt).flatten().cpu().numpy()          # LPIPS to GT (lower=better)
            dn.append(norm); dd.append(dot); lp.append(d)
            tv.append(tv_structural((img / 2 + 0.5), mask_px))
            print(f"  frac={frac} draw{r}: LPIPS={d.mean():.3f} TV={tv[-1].mean():.3f}", flush=True)
        lp = np.stack(lp); tv = np.stack(tv)                           # (R,B)
        # binarize each oracle at its median over all draws -> "invalid" = worse half
        lp_thr = np.median(lp); tv_thr = np.median(tv)
        v_lp = lp <= lp_thr; v_tv = tv <= tv_thr                       # valid = good half
        ne = np.concatenate([dn[r][:, -1] for r in range(args.R)])
        def feats_all():
            from probe_validate import feature_groups
            return np.concatenate([np.nan_to_num(feature_groups(dn[r], dd[r])["ALL-shape"], nan=0.0, posinf=0.0, neginf=0.0) for r in range(args.R)])
        X = feats_all()
        _, auc_lp = heldout_auc(X, np.concatenate([(~v_lp[r]).astype(float) for r in range(args.R)]), ne)
        _, auc_tv = heldout_auc(X, np.concatenate([(~v_tv[r]).astype(float) for r in range(args.R)]), ne)
        a_lp = action(None, dn, dd, v_lp, args.R, args.seed)
        a_tv = action(None, dn, dd, v_tv, args.R, args.seed)
        # LPIPS-as-continuous oracle: restart picks min-LPIPS draw -> report mean LPIPS per arm
        bi = np.arange(B)
        from probe_validate import feature_groups
        from learned_probe import fit_logreg
        Xc = X; yc = np.concatenate([(~v_lp[r]).astype(float) for r in range(args.R)])
        lpips_arms = {"vanilla": float(lp[0].mean()), "oracle": float(lp.min(0).mean())}
        if 0 < yc.mean() < 1:
            mu = Xc.mean(0); sd = Xc.std(0) + 1e-8; w, b = fit_logreg((Xc - mu) / sd, yc, l2=1.0)
            Pr = np.stack([1 / (1 + np.exp(-np.clip((np.nan_to_num(feature_groups(dn[r], dd[r])["ALL-shape"]) - mu) / sd @ w + b, -30, 30))) for r in range(args.R)])
            rng2 = np.random.default_rng(args.seed)
            lpips_arms["random"] = float(lp[rng2.integers(0, args.R, B), bi].mean())
            lpips_arms["probe"] = float(lp[np.argmin(Pr, 0), bi].mean())
        row = {"frac": frac, "mask": args.mask,
               "lpips_oracle": {"auc_deconf": round(auc_lp, 4), "probe_minus_random": round(a_lp["probe_restart"] - a_lp["random_restart"], 4), **{k: round(v, 4) for k, v in a_lp.items()}},
               "tv_structural_oracle": {"auc_deconf": round(auc_tv, 4), "probe_minus_random": round(a_tv["probe_restart"] - a_tv["random_restart"], 4), **{k: round(v, 4) for k, v in a_tv.items()}},
               "lpips_per_arm": {k: round(v, 4) for k, v in lpips_arms.items()}}
        rows.append(row)
        print(f"frac={frac}: LPIPS-orc auc={auc_lp:.3f} gap={row['lpips_oracle']['probe_minus_random']:+.3f} | "
              f"TV-struct auc={auc_tv:.3f} gap={row['tv_structural_oracle']['probe_minus_random']:+.3f} | "
              f"LPIPS arms {row['lpips_per_arm']}", flush=True)

    work = [r for r in rows if 0.05 < r["tv_structural_oracle"].get("vanilla", 0) < 0.95]
    summary = {"ckpt": args.ckpt, "n": int(B), "mask": args.mask, "option": "B-latent-clamp",
               "steps": args.num_sampling_steps, "eta": args.stepsize, "R": args.R, "rows": rows,
               "note": "LPIPS+detection-AUROC are the reported metrics; FID human-gated (not computed)."}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "repaint_sweep.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EqM-B/2")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--real-dir", default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--mask", default="center", choices=["center", "expand", "random"])
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--vae", default="ema")
    ap.add_argument("--stepsize", type=float, default=0.003)
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
