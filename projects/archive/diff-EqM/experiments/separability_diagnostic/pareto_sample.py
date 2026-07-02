"""Pareto / null test (PI exp #1): is restart a better use of compute than spending the
same NFE on longer/more vanilla sampling?

INCREMENTAL FID — accumulate Inception feature sum (s1) + outer-product sum (s2) + count;
NEVER store per-sample feats (avoids the 39GB dump that filled the disk). Each rank writes
tiny s1/s2/n shards; pareto_agg.py sums them and computes FID.

Modes (one run = one point on a Pareto curve, labeled by total NFE = images·steps·draws):
  long      : 1 draw, --steps S. NFE/img = S.                        (spend compute on depth)
  restart   : --R draws of --steps S, keep 1 by --select. NFE/img = R·S.  (spend on restarts)
    --select vanilla    keep draw 0          (= no selection; pays R·S but uses 1 -> control)
    --select random     keep a random draw
    --select energy_path keep argmax Σ‖f‖    (best trivial selector from the lockdown)
    --select probe_k    keep argmin partial-probe@--k-dec  (the early metacognition selector)

Headline comparison at matched NFE: e.g. NFE=750 -> {long S=750} vs {restart R=3,S=250,select=*}.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
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


def gd(model, z, y, eta, steps, log=False):
    xt = z; t = torch.zeros((xt.shape[0],), device=xt.device)
    norms, dots = [], []
    for _ in range(steps - 1):
        out = model(xt, t, y)
        if not torch.is_tensor(out):
            out = out[0]
        out = out.detach(); xk = xt.detach()
        if log:
            norms.append(out.flatten(1).norm(dim=1)); dots.append((out * xk).flatten(1).sum(1))
        xt = xk + out * eta
    if log:
        return xt.detach(), torch.stack(norms, 1).float().cpu().numpy(), torch.stack(dots, 1).float().cpu().numpy()
    return xt.detach()


def _probe(norm, dot, art, k):
    from probe_validate import feature_groups
    X = feature_groups(norm[:, :k], dot[:, :k])["ALL-shape"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    z = ((X - art["mu"]) / art["sd"]) @ art["w"] + float(art["b"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def main(args):
    EqM_models, find_model, AutoencoderKL, build_incep = _deps()
    rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1"))
    device = int(os.environ.get("LOCAL_RANK", "0")); torch.cuda.set_device(device); dev = f"cuda:{device}"
    ls = args.image_size // 8
    model = EqM_models[args.model](input_size=ls, num_classes=args.num_classes, uncond=True, ebm="none").to(dev)
    st = find_model(args.ckpt); model.load_state_dict(st["ema"] if "ema" in st else st.get("model", st)); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(dev).eval()
    incep = build_incep(dev)
    need_log = args.mode == "restart" and args.select in ("probe_k", "energy_path")
    art = None
    if args.mode == "restart" and args.select == "probe_k":
        pp = Path(args.probe_artifact).parent / "results" / "partial_probe" / f"partial_probe_k{args.k_dec}.npz"
        d = np.load(pp, allow_pickle=True); art = {k: d[k] for k in ["w", "b", "mu", "sd"]}

    def feat(latents):
        im = vae.decode(latents / VAE_SCALE).sample
        im = (im / 2 + 0.5).clamp(0, 1)
        im = torch.nn.functional.interpolate(im, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        f = incep(im)[0]
        if f.dim() == 4:
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)
        return f.float().cpu().numpy()

    s1 = np.zeros(2048); s2 = np.zeros((2048, 2048)); n = 0
    slots = list(range(rank, args.num_slots, world)); bs = args.batch_size; t0 = time.time()
    for j in range(0, len(slots), bs):
        chunk = slots[j:j + bs]; B = len(chunk); bi = np.arange(B)
        so = args.seed_offset * (args.num_slots * 8 + 7)
        if args.mode == "long":
            gens = [torch.Generator(device=dev).manual_seed(so + int(i)) for i in chunk]
            z = torch.stack([torch.randn(4, ls, ls, generator=g, device=dev) for g in gens])
            y = torch.tensor([(int(i) * 1315423911) % args.num_classes for i in chunk], device=dev)
            with torch.no_grad():
                f = feat(gd(model, z, y, args.stepsize, args.steps))
        else:  # restart
            fr = np.zeros((args.R, B, 2048), np.float32); score = np.zeros((args.R, B))
            for r in range(args.R):
                gens = [torch.Generator(device=dev).manual_seed(so + int(i) * args.R + r) for i in chunk]
                z = torch.stack([torch.randn(4, ls, ls, generator=g, device=dev) for g in gens])
                y = torch.tensor([(int(i) * 1315423911) % args.num_classes for i in chunk], device=dev)
                with torch.no_grad():
                    if need_log:
                        xt, norm, dot = gd(model, z, y, args.stepsize, args.steps, log=True)
                        if args.select == "probe_k":
                            score[r] = _probe(norm, dot, art, args.k_dec)
                        else:  # energy_path: argmax Σ‖f‖  ==  argmin(-Σ)
                            score[r] = -norm.sum(1)
                    else:
                        xt = gd(model, z, y, args.stepsize, args.steps)
                    fr[r] = feat(xt)
            if args.select == "vanilla":
                pick = np.zeros(B, int)
            elif args.select == "random":
                pick = np.random.default_rng(args.seed_offset * 7919 + j).integers(0, args.R, B)
            else:
                pick = np.argmin(score, 0)
            f = fr[pick, bi]
        s1 += f.sum(0); s2 += f.T @ f; n += len(f)
        if rank == 0 and (j // bs) % 5 == 0:
            print(f"[pareto:{args.mode}:{args.select}] {j+B}/{len(slots)} {(j+B)/max(1e-9,time.time()-t0):.1f}/s", flush=True)
    out = Path(args.out); os.makedirs(out, exist_ok=True)
    np.savez(out / f"stats_rank{rank}.npz", s1=s1, s2=s2, n=np.int64(n))
    print(f"[pareto] rank{rank} DONE n={n} NFE/img={args.R*args.steps if args.mode=='restart' else args.steps}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EqM-B/2"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probe-artifact", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["long", "restart"], required=True)
    ap.add_argument("--select", choices=["vanilla", "random", "energy_path", "probe_k"], default="vanilla")
    ap.add_argument("--R", type=int, default=3); ap.add_argument("--k-dec", type=int, default=50)
    ap.add_argument("--num-slots", type=int, default=50000); ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--stepsize", type=float, default=0.003); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=256); ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--vae", default="ema"); ap.add_argument("--seed-offset", type=int, default=0)
    main(ap.parse_args())
