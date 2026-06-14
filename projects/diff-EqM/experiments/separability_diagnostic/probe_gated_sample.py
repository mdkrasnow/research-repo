"""Stage 7: in-line restart sampler at 50k scale, with controls.

The "metacognition sampler" as probe-guided best-of-R: for each output slot, draw
R independent samples (a restart = fresh noise), score each by the trajectory-shape
probe P(garbage), and KEEP the best. Three arms from the SAME R draws (fair):

  - vanilla : keep draw 0            (NEGATIVE control — no selection == 1 draw)
  - probe   : keep argmin P(garbage) (TREATMENT — trajectory-shape probe, no image)
  - oracle  : keep argmin inception-NN dist to real (POSITIVE control — ceiling)

FID is computed from Inception features only (no images stored). The vanilla arm
must reproduce the ~31.41 B/2 baseline -> built-in pipeline sanity check.

Multi-GPU via torch.distributed.run: each rank renders a disjoint slice of slots
and writes per-rank per-arm Inception-feature shards. fid_gated_agg.py computes FID.
"""
import argparse
import os
import sys
import time
import traceback
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
    """EqM GD from noise; return final latent + (norm, dot) trajectories on GPU.
    Lean: keep per-step scalars as GPU tensors, one host copy at the end."""
    xt = z
    t = torch.zeros((xt.shape[0],), device=xt.device)
    norms, dots = [], []
    for _ in range(steps - 1):
        out = model(xt, t, y)
        if not torch.is_tensor(out):
            out = out[0]
        out = out.detach()
        xk = xt.detach()
        norms.append(out.flatten(1).norm(dim=1))
        dots.append((out * xk).flatten(1).sum(dim=1))
        xt = xk + out * eta
    norm = torch.stack(norms, 1).float().cpu().numpy()   # (B, steps-1)
    dot = torch.stack(dots, 1).float().cpu().numpy()
    return xt.detach(), norm, dot


def probe_p(norm, dot, art):
    """P(garbage) per sample from saved probe artifact + trajectory shape."""
    from probe_validate import feature_groups
    X = feature_groups(norm, dot)["ALL-shape"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    z = ((X - art["mu"]) / art["sd"]) @ art["w"] + float(art["b"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def main(args):
    EqM_models, find_model, AutoencoderKL, build_incep = _deps()
    from features import inception_features  # for real bank
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)
    dev = f"cuda:{device}"

    ls = args.image_size // 8
    model = EqM_models[args.model](input_size=ls, num_classes=args.num_classes,
                                   uncond=True, ebm="none").to(dev)
    st = find_model(args.ckpt)
    model.load_state_dict(st["ema"] if "ema" in st else st.get("model", st))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(dev).eval()
    incep = build_incep(dev)
    art = {k: np.load(args.probe_artifact, allow_pickle=True)[k]
           for k in ["w", "b", "mu", "sd"]}

    # real feature bank (for oracle knn dist). One rank builds, all reload via file.
    bank_path = Path(args.out) / "real_bank.npy"
    if rank == 0 and not bank_path.exists():
        from compute_quality_labels import list_real_images
        rf, _ = inception_features("", device=dev, batch_size=args.batch_size,
                                   files=list_real_images(args.real_dir, args.num_real_bank, seed=0))
        os.makedirs(args.out, exist_ok=True)
        np.save(bank_path, rf.astype(np.float32))
    for _ in range(600):
        if bank_path.exists():
            break
        time.sleep(1)
    real_bank = torch.tensor(np.load(bank_path), device=dev)        # (M,2048)

    def incep_feat(latents):
        imgs = vae.decode(latents / VAE_SCALE).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode="bilinear",
                                               align_corners=False, antialias=True)
        f = incep(imgs)[0]
        if f.dim() == 4:
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)
        return f                                                    # (B,2048) on GPU

    slots = list(range(rank, args.num_slots, world))
    bs = args.batch_size
    feats = {"vanilla": [], "probe": [], "oracle": []}
    t0 = time.time()
    for s0 in range(0, len(slots), bs):
        chunk = slots[s0:s0 + bs]
        B = len(chunk)
        # R draws per slot; gather features + scores
        draw_feat = np.zeros((args.R, B, 2048), np.float32)
        draw_pP = np.zeros((args.R, B), np.float64)
        draw_odist = np.zeros((args.R, B), np.float64)
        for r in range(args.R):
            # restart = same target class (depends on slot only), fresh noise (varies with r)
            gens = [torch.Generator(device=dev).manual_seed(int(i) * args.R + r) for i in chunk]
            z = torch.stack([torch.randn(4, ls, ls, generator=g, device=dev) for g in gens])
            y = torch.tensor([(int(i) * 1315423911) % args.num_classes for i in chunk], device=dev)
            with torch.no_grad():
                xt, norm, dot = gd_logged(model, z, y, args.stepsize, args.num_sampling_steps)
                f = incep_feat(xt)
                d = torch.cdist(f, real_bank).topk(3, largest=False).values.mean(1)
            draw_feat[r] = f.cpu().numpy()
            draw_pP[r] = probe_p(norm, dot, art)
            draw_odist[r] = d.cpu().numpy()
        bi = np.arange(B)
        feats["vanilla"].append(draw_feat[0])
        feats["probe"].append(draw_feat[np.argmin(draw_pP, 0), bi])
        feats["oracle"].append(draw_feat[np.argmin(draw_odist, 0), bi])
        if rank == 0 and (s0 // bs) % 5 == 0:
            done = s0 + B
            rate = done / max(1e-9, time.time() - t0)
            print(f"[gated] rank0 {done}/{len(slots)} slots  {rate:.1f} slot/s", flush=True)

    out = Path(args.out)
    os.makedirs(out, exist_ok=True)
    for arm in feats:
        np.save(out / f"feat_{arm}_rank{rank}.npy",
                np.concatenate(feats[arm], 0).astype(np.float32) if feats[arm] else np.zeros((0, 2048), np.float32))
    print(f"[gated] rank{rank} DONE slots={len(slots)} R={args.R}", flush=True)


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
    args = ap.parse_args()
    try:
        main(args)
    except Exception:
        r = os.environ.get("RANK", "0")
        tb = traceback.format_exc()
        try:
            os.makedirs(args.out, exist_ok=True)
            Path(args.out, f"ERROR_rank{r}.txt").write_text(tb)
        except Exception:
            pass
        print(f"[gated rank{r}] FATAL:\n{tb}", flush=True)
        raise
