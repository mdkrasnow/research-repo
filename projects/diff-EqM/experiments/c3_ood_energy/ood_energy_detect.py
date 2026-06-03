#!/usr/bin/env python3
"""
C3 — OOD detection via EqM energy / field-residual.

Pre-registered: documentation/c3-ood-energy-detection-proposal.md.

EqM samples by descending a field f(x,y); at the data equilibrium f -> 0, off the
manifold f is large (the model wants to move the point home). So the per-sample
field magnitude ||f(x,y)|| is a natural OOD score: small in-distribution, large OOD.

Mechanism claim (Exp 2 CONFIRMED): v10 (ANM) trains f to be accurate off-manifold,
so its field magnitude should separate in-dist from OOD BETTER than vanilla's
uncalibrated off-manifold field -> higher AUROC, dose-ordered by lambda.

Checkpoints are loaded with ebm='none' (velocity-prediction EqM, matches training):
there is no explicit scalar energy E, so the score is the field residual ||f||.
If a checkpoint exposes a scalar energy (ebm!=none), -E is also recorded.

OOD sets are DEPENDENCY-FREE (built from the same IN-1K val data + noise), so the
job needs no external dataset download:
  - gaussian    : N(0,1) latents scaled to in-dist per-sample latent norm (far-OOD)
  - uniform     : uniform-noise latents matched to in-dist norm (far-OOD)
  - wrong_label : real val latents scored with a random WRONG class (semantic near-OOD)
  - patch_shuffle: real val latents with spatial patches permuted (structure-OOD)

Score direction (pre-registered): higher ||f|| => more OOD. AUROC computed with
in-dist as negative class, OOD as positive. Reported per (arm x ood_type).
Pure forward passes under no_grad — no sampler, no decode. Cheap.

NB: never torch.inference_mode() — EqM.forward calls x.requires_grad_(True)
internally. Use torch.no_grad() (params frozen) for clean eval.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_DIFF_EQM = _HERE.parents[2]                 # projects/diff-EqM
_UPSTREAM = _DIFF_EQM / "eqm-upstream"
sys.path.insert(0, str(_UPSTREAM))

from models import EqM_models                # noqa: E402
from download import find_model              # noqa: E402

VAE_SCALE = 0.18215


# --------------------------------------------------------------------------- #
# Model loading (mirrors experiments/diagnostics/offtraj_field_robustness.py)
# --------------------------------------------------------------------------- #
def load_eqm(ckpt_path, *, use_ema, model_name, num_classes, ebm, uncond,
             image_size, device):
    latent_size = image_size // 8
    model = EqM_models[model_name](
        input_size=latent_size, num_classes=num_classes, uncond=uncond, ebm=ebm
    ).to(device)
    state = find_model(ckpt_path)
    key = None
    if isinstance(state, dict) and "ema" in state and use_ema:
        key = "ema"
    elif isinstance(state, dict) and "model" in state:
        key = "model"
    if key is not None:
        model.load_state_dict(state[key]); loaded = key
    else:
        model.load_state_dict(state); loaded = "raw"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, loaded


def read_ckpt_args(ckpt_path):
    state = find_model(ckpt_path)
    a = state.get("args") if isinstance(state, dict) else None
    if a is None:
        return {}
    return vars(a) if hasattr(a, "__dict__") else dict(a)


# --------------------------------------------------------------------------- #
# Deterministic val loader (sorted-synset round-robin -> stable, class-balanced)
# --------------------------------------------------------------------------- #
def build_val_index(val_path, n_needed):
    synsets = sorted(d for d in os.listdir(val_path)
                     if os.path.isdir(os.path.join(val_path, d)))
    per_class = []
    for ci, syn in enumerate(synsets):
        cdir = os.path.join(val_path, syn)
        files = sorted(os.listdir(cdir))
        per_class.append((ci, syn, cdir, files))
    index, k = [], 0
    while len(index) < n_needed:
        progressed = False
        for ci, syn, cdir, files in per_class:
            if k < len(files):
                index.append((f"{syn}/{files[k]}", ci, os.path.join(cdir, files[k])))
                progressed = True
                if len(index) >= n_needed:
                    break
        if not progressed:
            break
        k += 1
    return index


def load_pixels(paths, image_size, device):
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs).to(device)


# --------------------------------------------------------------------------- #
# Field score
# --------------------------------------------------------------------------- #
def field_score(model, x, y, *, want_energy):
    """Return (field_rms_per_sample, energy_per_sample_or_None).

    field_rms = ||f(x,y)||_2 per sample (the OOD score; higher => more OOD).
    """
    t = torch.ones((x.shape[0],), device=x.device)  # uncond -> t zeroed internally
    with torch.no_grad():
        if want_energy:
            out = model(x, t, y, get_energy=True, train=False)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                f, negE = out[0], out[1]
                f = f.detach(); energy = (-negE).detach()
            else:
                f = (out[0] if not torch.is_tensor(out) else out).detach()
                energy = None
        else:
            out = model(x, t, y, train=False)
            f = (out if torch.is_tensor(out) else out[0]).detach()
            energy = None
    rms = f.reshape(f.shape[0], -1).norm(dim=1)
    return rms.float().cpu(), (energy.float().cpu() if energy is not None else None)


# --------------------------------------------------------------------------- #
# OOD constructors (latent space, dependency-free)
# --------------------------------------------------------------------------- #
def make_ood(kind, x_in, y_in, gen):
    """x_in: (N,C,h,w) in-dist latents; y_in: (N,) true labels.
    Returns (x_ood, y_ood)."""
    N = x_in.shape[0]
    norms = x_in.reshape(N, -1).norm(dim=1, keepdim=True)  # (N,1)
    if kind == "gaussian":
        z = torch.randn(x_in.shape, generator=gen, device=x_in.device)
        z = z / z.reshape(N, -1).norm(dim=1, keepdim=True).view(N, 1, 1, 1) \
            * norms.view(N, 1, 1, 1)
        return z, y_in
    if kind == "uniform":
        z = (torch.rand(x_in.shape, generator=gen, device=x_in.device) * 2 - 1)
        z = z / z.reshape(N, -1).norm(dim=1, keepdim=True).view(N, 1, 1, 1) \
            * norms.view(N, 1, 1, 1)
        return z, y_in
    if kind == "wrong_label":
        offset = torch.randint(1, 1000, (N,), generator=gen, device=x_in.device)
        y_wrong = (y_in + offset) % 1000
        return x_in.clone(), y_wrong
    if kind == "patch_shuffle":
        # permute 8x8 spatial blocks within each latent (32x32 -> 16 blocks)
        x = x_in.clone()
        h = x.shape[-1]; bs = max(1, h // 4)
        blocks = []
        for i in range(0, h, bs):
            for j in range(0, h, bs):
                blocks.append((i, j))
        perm = torch.randperm(len(blocks), generator=gen)
        out = x.clone()
        for dst, src in enumerate(perm.tolist()):
            di, dj = blocks[dst]; si, sj = blocks[src]
            out[..., di:di+bs, dj:dj+bs] = x[..., si:si+bs, sj:sj+bs]
        return out, y_in
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# AUROC via Mann-Whitney U (no sklearn dependency)
# --------------------------------------------------------------------------- #
def auroc(neg_scores, pos_scores):
    """P(score(pos) > score(neg)); higher score = positive (OOD)."""
    neg = np.asarray(neg_scores, float); pos = np.asarray(pos_scores, float)
    n_neg, n_pos = len(neg), len(pos)
    if n_neg == 0 or n_pos == 0:
        return float("nan")
    alls = np.concatenate([neg, pos])
    ranks = alls.argsort().argsort().astype(float) + 1.0  # average ties ~ ok at scale
    r_pos = ranks[n_neg:].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", required=True,
                    help="comma list name:path, e.g. vanilla:/a.pt,v10_l01:/b.pt,v10_l03:/c.pt")
    ap.add_argument("--val-path", default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ood-types", default="gaussian,uniform,wrong_label,patch_shuffle")
    ap.add_argument("--model", default="EqM-B/2")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--ebm", default="none")
    ap.add_argument("--uncond", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--vae", default="ema", choices=["ema", "mse"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    arms = []
    for tok in args.checkpoints.split(","):
        name, path = tok.split(":", 1)
        arms.append((name.strip(), path.strip()))
    ood_types = [s.strip() for s in args.ood_types.split(",") if s.strip()]

    # ---- VAE ----
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # ---- in-dist val latents (deterministic mode encode) ----
    index = build_val_index(args.val_path, args.n_samples)
    print(f"[c3] built val index: {len(index)} samples over "
          f"{len({i[1] for i in index})} classes", flush=True)
    x_in_chunks, y_in_chunks = [], []
    for s in range(0, len(index), args.batch_size):
        chunk = index[s:s + args.batch_size]
        pix = load_pixels([c[2] for c in chunk], args.image_size, device)
        with torch.no_grad():
            lat = vae.encode(pix).latent_dist.mode().mul_(VAE_SCALE)
        x_in_chunks.append(lat)
        y_in_chunks.append(torch.tensor([c[1] for c in chunk], device=device))
    x_in = torch.cat(x_in_chunks); y_in = torch.cat(y_in_chunks)
    print(f"[c3] encoded in-dist latents {tuple(x_in.shape)} "
          f"mean||x||={x_in.reshape(len(x_in),-1).norm(dim=1).mean():.2f}", flush=True)

    # ---- per arm: in-dist + each OOD field scores ----
    want_energy = (args.ebm != "none")
    rows = []
    raw = {}  # arm -> {set_name: scores}
    for name, path in arms:
        a = read_ckpt_args(path)
        print(f"[c3] arm={name} ckpt_args model={a.get('model')} ebm={a.get('ebm')} "
              f"uncond={a.get('uncond')}", flush=True)
        model, src = load_eqm(path, use_ema=True, model_name=args.model,
                              num_classes=args.num_classes, ebm=args.ebm,
                              uncond=bool(args.uncond), image_size=args.image_size,
                              device=device)
        # in-dist scores
        s_in = []
        for s in range(0, len(x_in), args.batch_size):
            rms, _ = field_score(model, x_in[s:s+args.batch_size],
                                 y_in[s:s+args.batch_size], want_energy=want_energy)
            s_in.append(rms)
        s_in = torch.cat(s_in).numpy()
        raw.setdefault(name, {})["in_dist"] = s_in
        # ood scores
        for kind in ood_types:
            x_ood, y_ood = make_ood(kind, x_in, y_in, gen)
            s_ood = []
            for s in range(0, len(x_ood), args.batch_size):
                rms, _ = field_score(model, x_ood[s:s+args.batch_size],
                                     y_ood[s:s+args.batch_size], want_energy=want_energy)
                s_ood.append(rms)
            s_ood = torch.cat(s_ood).numpy()
            raw[name][kind] = s_ood
            a_roc = auroc(s_in, s_ood)
            rows.append({"arm": name, "ood_type": kind, "auroc": round(a_roc, 4),
                         "in_mean": round(float(s_in.mean()), 3),
                         "ood_mean": round(float(s_ood.mean()), 3),
                         "n_in": len(s_in), "n_ood": len(s_ood),
                         "ckpt_src": src})
            print(f"[c3] {name:10s} {kind:14s} AUROC={a_roc:.4f} "
                  f"in={s_in.mean():.2f} ood={s_ood.mean():.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- write results + verdict ----
    import csv
    with open(out / "auroc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez(out / "raw_scores.npz",
             **{f"{arm}__{s}": v for arm, d in raw.items() for s, v in d.items()})

    # delta AUROC vs vanilla (pre-registered: v10 should beat vanilla, dose-ordered)
    by = {}
    for r in rows:
        by.setdefault(r["ood_type"], {})[r["arm"]] = r["auroc"]
    verdict = {"per_ood_delta_vs_vanilla": {}, "arms": [a for a, _ in arms]}
    print("\n=== C3 VERDICT: AUROC and delta vs vanilla (positive = v10 better) ===")
    for kind, d in by.items():
        base = d.get("vanilla")
        if base is not None:
            line_delta = {a: round(d[a] - base, 4) for a in d if a != "vanilla"}
            verdict["per_ood_delta_vs_vanilla"][kind] = line_delta
            print(f"  {kind:14s} " + "  ".join(f"{a}={d[a]:.4f}" for a in d) +
                  "   delta " + "  ".join(f"{a}={line_delta[a]:+.4f}" for a in line_delta))
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"\n[c3] wrote {out/'auroc.csv'}, raw_scores.npz, verdict.json")


if __name__ == "__main__":
    main()
