"""Airtight inference-time comparison: FID at EQUAL COMPUTE across restart selectors.

Best-of-R restart differs across selectors ONLY in which of the R draws is kept per slot.
So ONE sampling run (draw R per slot, log full descent trajectory + Inception feat + oracle
dist) yields every selector's FID from IDENTICAL draws at IDENTICAL NFE. This makes the
"equal compute / no cherry-picking" claim true by construction.

Selectors (all keep exactly 1 of R draws per slot; same total NFE = R·steps·num_slots):
  vanilla      keep draw 0                        (floor — no selection)
  random       keep a random draw                 (null selector)
  energy_dot   keep best by final f·x energy       (trivial EqM scalar; both directions tried)
  energy_path  keep best by path-integral Σ‖f‖     (trivial; both directions)
  gradnorm     keep best by final ‖f‖              (trivial magnitude; both directions)
  probe        keep argmin P(garbage), full probe  (TREATMENT — descent-shape probe)
  oracle       keep argmin Inception-NN dist       (ceiling)
  probe@k      keep argmin partial-probe at step k (EARLY intervention; one arm per k)

energy/gradnorm are steel-manned: emit both argmin and argmax picks; the agg reports the
better (lower-FID) direction so the baseline gets its best shot.

Multi-GPU via torch.distributed.run; per-rank per-selector Inception-feature shards.
selector_fid_agg.py computes per-selector FID + the comparison table.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
VAE_SCALE = 0.18215


def _deps():
    up = str(Path(__file__).resolve().parents[2] / "eqm-upstream")
    e3 = str(Path(__file__).resolve().parent.parent / "exp3_fidelity_diversity")
    for p in (str(Path(__file__).resolve().parent), up, e3):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models import EqM_models
    from download import find_model
    from diffusers.models import AutoencoderKL
    from features import _build_inception
    return EqM_models, find_model, AutoencoderKL, _build_inception


def gd_logged(model, z, y, eta, steps):
    xt = z
    t = torch.zeros((xt.shape[0],), device=xt.device)
    norms, dots = [], []
    for _ in range(steps - 1):
        out = model(xt, t, y)
        if not torch.is_tensor(out):
            out = out[0]
        out = out.detach(); xk = xt.detach()
        norms.append(out.flatten(1).norm(dim=1))
        dots.append((out * xk).flatten(1).sum(dim=1))
        xt = xk + out * eta
    norm = torch.stack(norms, 1).float().cpu().numpy()
    dot = torch.stack(dots, 1).float().cpu().numpy()
    return xt.detach(), norm, dot


def _probe_score(norm, dot, art, k=None):
    from probe_validate import feature_groups
    if k is not None:
        norm, dot = norm[:, :k], dot[:, :k]
    X = feature_groups(norm, dot)["ALL-shape"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    z = ((X - art["mu"]) / art["sd"]) @ art["w"] + float(art["b"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def main(args):
    EqM_models, find_model, AutoencoderKL, build_incep = _deps()
    from features import inception_features
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    device = int(os.environ.get("LOCAL_RANK", "0")); torch.cuda.set_device(device)
    dev = f"cuda:{device}"

    ls = args.image_size // 8
    model = EqM_models[args.model](input_size=ls, num_classes=args.num_classes, uncond=True, ebm="none").to(dev)
    st = find_model(args.ckpt); model.load_state_dict(st["ema"] if "ema" in st else st.get("model", st))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(dev).eval()
    incep = build_incep(dev)
    art = {k: np.load(args.probe_artifact, allow_pickle=True)[k] for k in ["w", "b", "mu", "sd"]}
    # partial probes for early-intervention arms: {k_dec: artifact}
    partials = {}
    for pp in sorted(glob.glob(str(Path(args.probe_artifact).parent / "results" / "partial_probe" / "partial_probe_k*.npz"))):
        d = np.load(pp, allow_pickle=True)
        partials[int(d["k_dec"])] = {k: d[k] for k in ["w", "b", "mu", "sd"]}
    kdecs = sorted(partials.keys())
    if rank == 0:
        print(f"[sel] partial probes at k={kdecs}", flush=True)

    bank_path = Path(args.out) / "real_bank.npy"
    if rank == 0 and not bank_path.exists():
        from compute_quality_labels import list_real_images
        rf, _ = inception_features("", device=dev, batch_size=args.batch_size,
                                   files=list_real_images(args.real_dir, args.num_real_bank, seed=0))
        os.makedirs(args.out, exist_ok=True); np.save(bank_path, rf.astype(np.float32))
    for _ in range(600):
        if bank_path.exists():
            break
        time.sleep(1)
    real_bank = torch.tensor(np.load(bank_path), device=dev)

    def incep_feat(latents):
        imgs = vae.decode(latents / VAE_SCALE).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        f = incep(imgs)[0]
        if f.dim() == 4:
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)
        return f

    arms = ["vanilla", "random", "energy_dot_lo", "energy_dot_hi", "energy_path_lo", "energy_path_hi",
            "gradnorm_lo", "gradnorm_hi", "probe", "oracle"] + [f"probe_k{k}" for k in kdecs]
    feats = {a: [] for a in arms}
    slots = list(range(rank, args.num_slots, world)); bs = args.batch_size
    nfe_per_slot = args.R * args.num_sampling_steps
    t0 = time.time()
    for s0 in range(0, len(slots), bs):
        chunk = slots[s0:s0 + bs]; B = len(chunk); bi = np.arange(B)
        df = np.zeros((args.R, B, 2048), np.float32)
        d_pP = np.zeros((args.R, B)); d_od = np.zeros((args.R, B))
        d_edot = np.zeros((args.R, B)); d_epath = np.zeros((args.R, B)); d_gn = np.zeros((args.R, B))
        d_pk = {k: np.zeros((args.R, B)) for k in kdecs}
        for r in range(args.R):
            so = args.seed_offset * (args.num_slots * args.R + 7)
            gens = [torch.Generator(device=dev).manual_seed(so + int(i) * args.R + r) for i in chunk]
            z = torch.stack([torch.randn(4, ls, ls, generator=g, device=dev) for g in gens])
            y = torch.tensor([(int(i) * 1315423911) % args.num_classes for i in chunk], device=dev)
            with torch.no_grad():
                xt, norm, dot = gd_logged(model, z, y, args.stepsize, args.num_sampling_steps)
                f = incep_feat(xt)
                od = torch.cdist(f, real_bank).topk(3, largest=False).values.mean(1)
            df[r] = f.cpu().numpy()
            d_pP[r] = _probe_score(norm, dot, art)
            d_od[r] = od.cpu().numpy()
            d_edot[r] = dot[:, -1]                 # final f·x  (trivial energy)
            d_epath[r] = norm.sum(1)               # path integral Σ‖f‖
            d_gn[r] = norm[:, -1]                  # final ‖f‖
            for k in kdecs:
                d_pk[k][r] = _probe_score(norm, dot, partials[k], k=k)
        rng = np.random.default_rng(args.seed_offset * 99991 + s0)
        feats["vanilla"].append(df[0])
        feats["random"].append(df[rng.integers(0, args.R, B), bi])
        feats["energy_dot_lo"].append(df[np.argmin(d_edot, 0), bi])
        feats["energy_dot_hi"].append(df[np.argmax(d_edot, 0), bi])
        feats["energy_path_lo"].append(df[np.argmin(d_epath, 0), bi])
        feats["energy_path_hi"].append(df[np.argmax(d_epath, 0), bi])
        feats["gradnorm_lo"].append(df[np.argmin(d_gn, 0), bi])
        feats["gradnorm_hi"].append(df[np.argmax(d_gn, 0), bi])
        feats["probe"].append(df[np.argmin(d_pP, 0), bi])
        feats["oracle"].append(df[np.argmin(d_od, 0), bi])
        for k in kdecs:
            feats[f"probe_k{k}"].append(df[np.argmin(d_pk[k], 0), bi])
        if rank == 0 and (s0 // bs) % 5 == 0:
            done = s0 + B
            print(f"[sel] rank0 {done}/{len(slots)} slots  {done/max(1e-9,time.time()-t0):.1f} slot/s", flush=True)

    out = Path(args.out); os.makedirs(out, exist_ok=True)
    for a in arms:
        arr = np.concatenate(feats[a], 0).astype(np.float32) if feats[a] else np.zeros((0, 2048), np.float32)
        np.save(out / f"feat_{a}_rank{rank}.npy", arr)
    if rank == 0:
        (out / "nfe.txt").write_text(f"NFE per slot (all arms identical) = R*steps = {args.R}*{args.num_sampling_steps} = {nfe_per_slot}\n"
                                     f"num_slots={args.num_slots}  total NFE per arm = {nfe_per_slot*args.num_slots}\n"
                                     f"kept samples per arm = {args.num_slots} (fixed; no post-hoc count change)\n")
    print(f"[sel] rank{rank} DONE slots={len(slots)} R={args.R} arms={len(arms)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EqM-B/2")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probe-artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-dir", default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train")
    ap.add_argument("--num-real-bank", type=int, default=10000)
    ap.add_argument("--num-slots", type=int, default=50000)
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--vae", default="ema")
    ap.add_argument("--stepsize", type=float, default=0.003)
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed-offset", type=int, default=0)
    main(ap.parse_args())
