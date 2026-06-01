#!/usr/bin/env python
"""Experiment 2: off-trajectory local field robustness — vanilla EqM vs ANM (v10) EqM.

Compares the two checkpoints AS FIELD PREDICTORS (not samplers). For held-out
validation data latent x1 and Gaussian noise x0, builds the EqM interpolation
point and regression target using the REPO'S OWN transport code, then evaluates
each model at perturbed points x_t + delta. Measures how field MSE / cosine /
norm calibration degrade as the perturbation radius grows.

Mechanistic claim under test: ANM keeps the EqM field correct in a local L2 tube
around the vanilla trajectory, so ANM should match vanilla at radius 0 and
degrade LESS off-trajectory (esp. at radii matching real v10 mining drift,
||delta||=eps_radius=0.3 absolute -> rel radius ~0.005 in (4,32,32) latent space).

AUTHORITATIVE TARGET CONVENTION (eqm-upstream/transport/transport.py):
    x1 = DATA latent, x0 = Gaussian noise (randn_like)
    xt     = path_sampler.plan(t, x0, x1)        -> t*x1 + (1-t)*x0   (Linear/ICPlan)
    target = (x1 - x0) * get_ct(t)
Both checkpoints were trained by experiments/train_imagenet.py via
transport.training_losses, so this is the convention that matches the weights.
The v10 mining perturbation is replicated from train_imagenet.py
:_v10_pgd_hard_example_step (identical PGA on the same target).

NB: NEVER use torch.inference_mode() here — EqM.forward calls
x0.requires_grad_(True) internally, which raises on inference tensors. Use
torch.no_grad() for metric eval; torch.enable_grad() only for mining.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Repo imports (reuse, do NOT reimplement target geometry)
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve()
# experiments/diagnostics/<this>.py -> parents[2] = projects/diff-EqM
_DIFF_EQM = _HERE.parents[2]
_UPSTREAM = _DIFF_EQM / "eqm-upstream"
sys.path.insert(0, str(_UPSTREAM))

from models import EqM_models                       # noqa: E402
from transport import create_transport              # noqa: E402
from download import find_model                     # noqa: E402

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Per-sample tensor helpers
# --------------------------------------------------------------------------- #
def _flat(x):
    return x.reshape(x.shape[0], -1)


def _l2(x):
    return _flat(x).norm(dim=1)


def _bcast(v, x):
    return v.view(v.shape[0], *([1] * (x.ndim - 1)))


def _unit(x):
    return x / _bcast(_l2(x).clamp_min(EPS), x)


def _project_off_path(delta, path_dir):
    """Remove the component of delta along path_dir = x1 - x0 (per sample)."""
    df, pf = _flat(delta), _flat(path_dir)
    coeff = (df * pf).sum(dim=1) / (pf.pow(2).sum(dim=1) + EPS)
    return delta - _bcast(coeff, delta) * path_dir


def compute_metrics(pred, target):
    """Per-sample field metrics. Returns dict of CPU 1-D tensors length B."""
    D = pred[0].numel()
    diff = pred - target
    mse = _flat(diff).pow(2).mean(dim=1)
    pred_n = _l2(pred)
    targ_n = _l2(target)
    pf, tf = _flat(pred), _flat(target)
    cosine = (pf * tf).sum(dim=1) / (pred_n * targ_n).clamp_min(EPS)
    rel_norm_err = (pred_n - targ_n).abs() / targ_n.clamp_min(EPS)
    abs_norm_err = (pred_n - targ_n).abs() / (D ** 0.5)
    field_norm_rms = pred_n / (D ** 0.5)
    norm_ratio = pred_n / targ_n.clamp_min(EPS)
    return {
        "mse": mse.cpu(),
        "cosine": cosine.cpu(),
        "rel_norm_err": rel_norm_err.cpu(),
        "abs_norm_err": abs_norm_err.cpu(),
        "field_norm_rms": field_norm_rms.cpu(),
        "norm_ratio": norm_ratio.cpu(),
        "pred_norm": pred_n.cpu(),
        "target_norm": targ_n.cpu(),
    }


def _tensor_hash(x):
    return hashlib.md5(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Transport target (REPO API — single source of truth)
# --------------------------------------------------------------------------- #
def build_eqm_point_and_target(transport, x0_noise, x1_data, t):
    """xt, target via repo transport. x1=data, x0=noise (transport convention)."""
    _, xt, ut = transport.path_sampler.plan(t, x0_noise, x1_data)
    ut = ut * transport.get_ct(t)[:, None, None, None]
    return xt, ut


# --------------------------------------------------------------------------- #
# v10 / ANM mining perturbation — replicated from
# experiments/train_imagenet.py:_v10_pgd_hard_example_step (verbatim mechanism)
# --------------------------------------------------------------------------- #
def mine_delta(probe_forward, xt, target, *, K, eps_radius, lr, gen):
    """PGA hard-example delta: argmax_{||delta||<=eps} MSE(f(xt+delta), target).

    probe_forward: callable(x_input) -> field (FROZEN shared probe model).
    gen: torch.Generator for deterministic init (per-sample reproducibility).
    Returns detached delta on xt's device.
    """
    B = xt.shape[0]
    delta = torch.empty_like(xt).normal_(0.0, eps_radius / 2.0, generator=gen)
    flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
    delta = delta * torch.clamp(eps_radius / (flat + 1e-8), max=1.0)
    with torch.enable_grad():
        for _ in range(K):
            delta = delta.detach().requires_grad_(True)
            pred = probe_forward(xt.detach() + delta)
            loss_adv = ((pred - target) ** 2).mean()
            g = torch.autograd.grad(loss_adv, delta, retain_graph=False)[0]
            with torch.no_grad():
                delta = delta + lr * g.sign()
                flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
                delta = delta * torch.clamp(eps_radius / (flat + 1e-8), max=1.0)
    return delta.detach()


def gd_drift_delta(probe_forward, xt, *, steps, step_size, momentum, kind, path_dir):
    """Frozen-sampler local drift from xt (GD/NAG-GD), orthogonalized vs path.

    Mirrors eqm-upstream/sample_gd.py update: z = z + step_size * field.
    Returns detached delta = (z_K - xt) projected off the path direction.
    """
    with torch.no_grad():
        z = xt.detach().clone()
        m = torch.zeros_like(z)
        for i in range(steps):
            if kind == "ngd" and i > 0:
                out = probe_forward(z + step_size * m * momentum)
                m = out
            else:
                out = probe_forward(z)
            z = z + out * step_size
        delta = z - xt
    return _project_off_path(delta, path_dir).detach()


# --------------------------------------------------------------------------- #
# Model loading
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
        model.load_state_dict(state[key])
        loaded = key
    else:
        model.load_state_dict(state)
        loaded = "raw"
    model.eval()
    # No training graph through params: mining only needs grad wrt delta, and
    # metric eval is under no_grad. Freezing params guarantees param.grad stays
    # None and saves the mining forward's activation graph over weights.
    for p in model.parameters():
        p.requires_grad_(False)
    return model, loaded


def read_ckpt_args(ckpt_path):
    """Pull the training Namespace from a checkpoint so we instantiate the
    matching architecture / transport. Returns dict or {}."""
    state = find_model(ckpt_path)
    a = state.get("args") if isinstance(state, dict) else None
    if a is None:
        return {}
    d = vars(a) if hasattr(a, "__dict__") else dict(a)
    return d


# --------------------------------------------------------------------------- #
# Deterministic validation loader (sorted-synset round-robin -> stable ids)
# --------------------------------------------------------------------------- #
def build_val_index(val_path, n_needed):
    synsets = sorted(
        d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))
    )
    per_class = []
    for ci, syn in enumerate(synsets):
        cdir = os.path.join(val_path, syn)
        files = sorted(os.listdir(cdir))
        per_class.append((ci, syn, cdir, files))
    index = []  # (sample_id_str, label, path)
    k = 0
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
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
    ])
    imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs).to(device)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _radius_key(r):
    return "nat" if r is None else f"{float(r):.4g}"


def aggregate(rows, run_id):
    """rows: list of per-sample dicts. Returns aggregate + paired-diff lists."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["checkpoint_type"], r["perturbation_type"],
               _radius_key(r["radius_rel_requested"]), r["t_bin"])
        for m in ("mse", "cosine", "rel_norm_err", "field_norm_rms", "norm_ratio",
                  "pred_norm", "target_norm"):
            groups[key][m].append(r[m])

    agg = []
    for (ct, pt, rad, tb), met in sorted(groups.items()):
        n = len(met["mse"])
        row = {"run_id": run_id, "checkpoint_type": ct, "perturbation_type": pt,
               "radius_rel_value": rad, "t_bin": tb, "n": n, "sample_count": n}
        for m in ("mse", "cosine", "rel_norm_err"):
            a = np.asarray(met[m], dtype=np.float64)
            row[f"{m}_mean"] = float(a.mean())
            row[f"{m}_se"] = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        for m in ("pred_norm", "target_norm"):
            row[f"{m}_mean"] = float(np.asarray(met[m], dtype=np.float64).mean())
        for m in ("field_norm_rms", "norm_ratio"):
            a = np.asarray(met[m], dtype=np.float64)
            row[f"{m}_mean"] = float(a.mean())
            row[f"{m}_p50"] = float(np.percentile(a, 50))
            row[f"{m}_p95"] = float(np.percentile(a, 95))
        agg.append(row)
    return agg


def paired_differences(rows, run_id, n_boot=1000, seed=0):
    """Paired ANM-vanilla differences, bootstrap CI over sample_id."""
    by_key = defaultdict(dict)  # (pt,rad,tb) -> {sid: {ct: metrics}}
    for r in rows:
        k = (r["perturbation_type"], _radius_key(r["radius_rel_requested"]), r["t_bin"])
        by_key[k].setdefault(r["sample_id"], {})[r["checkpoint_type"]] = r
    rng = np.random.default_rng(seed)
    out = []
    for (pt, rad, tb), sid_map in sorted(by_key.items()):
        paired = [(v["vanilla"], v["anm"]) for v in sid_map.values()
                  if "vanilla" in v and "anm" in v]
        if not paired:
            continue
        row = {"run_id": run_id, "perturbation_type": pt,
               "radius_rel_value": rad, "t_bin": tb, "n_pairs": len(paired)}
        for m in ("mse", "cosine", "rel_norm_err"):
            d = np.array([a[m] - vv[m] for vv, a in paired], dtype=np.float64)
            row[f"{m}_anm_minus_vanilla"] = float(d.mean())
            if len(d) > 1:
                idx = rng.integers(0, len(d), size=(n_boot, len(d)))
                means = d[idx].mean(axis=1)
                row[f"{m}_ci_low"] = float(np.percentile(means, 2.5))
                row[f"{m}_ci_high"] = float(np.percentile(means, 97.5))
            else:
                row[f"{m}_ci_low"] = row[f"{m}_ci_high"] = float(d.mean())
        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def make_plots(rows, agg, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable ({e}); skipping plots", file=sys.stderr)
        return
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    def numeric_radii(pt):
        rs = sorted({r["radius_rel_value"] for r in agg
                     if r["perturbation_type"] == pt and r["radius_rel_value"] != "nat"},
                    key=lambda s: float(s))
        return rs

    ptypes = sorted({r["perturbation_type"] for r in agg})

    # A/B: metric vs radius (mean over t-bins), curves vanilla/anm, panel per ptype
    for metric, fname, ylab in [("mse", "mse_vs_radius.png", "field MSE"),
                                ("cosine", "cosine_vs_radius.png", "cosine"),
                                ("rel_norm_err", "rel_norm_err_vs_radius.png", "rel norm err")]:
        npt = max(len(ptypes), 1)
        fig, axes = plt.subplots(1, npt, figsize=(5 * npt, 4), squeeze=False)
        for j, pt in enumerate(ptypes):
            ax = axes[0][j]
            for ct, color in [("vanilla", "C0"), ("anm", "C1")]:
                rs = numeric_radii(pt)
                xs, ys, es = [], [], []
                for rad in rs:
                    sel = [r for r in agg if r["checkpoint_type"] == ct
                           and r["perturbation_type"] == pt
                           and r["radius_rel_value"] == rad]
                    if not sel:
                        continue
                    ns = sum(r["n"] for r in sel)
                    mean = sum(r[f"{metric}_mean"] * r["n"] for r in sel) / max(ns, 1)
                    xs.append(float(rad)); ys.append(mean)
                    es.append(np.mean([r.get(f"{metric}_se", 0.0) for r in sel]))
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", label=ct, color=color, capsize=2)
            ax.set_title(pt); ax.set_xlabel("rel radius ||d||/||x1-x0||"); ax.set_ylabel(ylab)
            ax.legend()
        fig.tight_layout(); fig.savefig(plots / fname, dpi=110); plt.close(fig)

    # C: t-bin x radius heatmaps of ANM - vanilla
    pd_rows = paired_differences(rows, "plot")
    for metric, fname in [("mse", "tbin_heatmap_mse_diff.png"),
                          ("cosine", "tbin_heatmap_cosine_diff.png"),
                          ("rel_norm_err", "tbin_heatmap_relnorm_diff.png")]:
        for pt in ptypes:
            sub = [r for r in pd_rows if r["perturbation_type"] == pt
                   and r["radius_rel_value"] != "nat"]
            if not sub:
                continue
            tbins = sorted({r["t_bin"] for r in sub})
            rads = sorted({r["radius_rel_value"] for r in sub}, key=lambda s: float(s))
            M = np.full((len(tbins), len(rads)), np.nan)
            for r in sub:
                M[tbins.index(r["t_bin"]), rads.index(r["radius_rel_value"])] = \
                    r[f"{metric}_anm_minus_vanilla"]
            fig, ax = plt.subplots(figsize=(1.2 * len(rads) + 2, 0.6 * len(tbins) + 2))
            vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(rads))); ax.set_xticklabels(rads, rotation=45)
            ax.set_yticks(range(len(tbins))); ax.set_yticklabels(tbins)
            ax.set_xlabel("rel radius"); ax.set_ylabel("t_bin")
            ax.set_title(f"{pt}: ANM-vanilla {metric}")
            fig.colorbar(im, ax=ax); fig.tight_layout()
            fig.savefig(plots / f"{pt}__{fname}", dpi=110); plt.close(fig)

    # D: field-norm histograms at radius 0 and largest off-traj radius
    for rad_label, fname in [("0", "field_norm_hist_radius0.png"),
                             ("offtraj", "field_norm_hist_offtraj.png")]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for ct, color in [("vanilla", "C0"), ("anm", "C1")]:
            if rad_label == "0":
                sel = [r for r in rows if r["checkpoint_type"] == ct
                       and _radius_key(r["radius_rel_requested"]) == "0"]
            else:
                offr = [r["radius_rel_requested"] for r in rows
                        if r["radius_rel_requested"] not in (None, 0)]
                if not offr:
                    continue
                rmax = max(offr)
                sel = [r for r in rows if r["checkpoint_type"] == ct
                       and r["radius_rel_requested"] == rmax]
            if not sel:
                continue
            ax1.hist([r["field_norm_rms"] for r in sel], bins=40, alpha=0.5,
                     label=ct, color=color, density=True)
            ax2.hist([r["norm_ratio"] for r in sel], bins=40, alpha=0.5,
                     label=ct, color=color, density=True, range=(0, 3))
        ax1.set_title(f"field_norm_rms (radius={rad_label})"); ax1.legend()
        ax2.set_title(f"norm_ratio ||pred||/||target|| (radius={rad_label})"); ax2.legend()
        fig.tight_layout(); fig.savefig(plots / fname, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_floats(s):
    return [float(x) for x in str(s).split(",") if x != ""]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vanilla-checkpoint", required=True)
    p.add_argument("--anm-checkpoint", required=True)
    p.add_argument("--probe-checkpoint", default=None,
                   help="Frozen shared miner/sampler. Default: vanilla checkpoint.")
    p.add_argument("--val-path", default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/val")
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--ebm", default="none")
    p.add_argument("--uncond", type=int, default=1)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--vae", default="ema", choices=["ema", "mse"])
    p.add_argument("--path-type", default="Linear")
    p.add_argument("--prediction", default="velocity")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-batches", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--t-sampling-mode", default="grid", choices=["grid", "uniform"])
    p.add_argument("--t-values", default="0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95")
    p.add_argument("--perturbation-radii", default="0,0.0025,0.005,0.01,0.02,0.05,0.10")
    p.add_argument("--perturbation-type", default="random_l2_orthogonal",
                   choices=["random_l2", "random_l2_orthogonal", "sampler_drift", "all"])
    p.add_argument("--mining-K", type=int, default=1)
    p.add_argument("--mining-eps-radius", type=float, default=0.3)
    p.add_argument("--mining-lr", type=float, default=0.05)
    p.add_argument("--sampler-steps", type=int, default=4)
    p.add_argument("--sampler-step-size", type=float, default=0.0017)
    p.add_argument("--sampler-momentum", type=float, default=0.3)
    p.add_argument("--sampler-kind", default="ngd", choices=["gd", "ngd"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--direction-seed", type=int, default=456)
    p.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    p.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="fp32", choices=["fp32", "autocast"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-id", default="offtraj_v10")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sanity").mkdir(exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    probe_ckpt = args.probe_checkpoint or args.vanilla_checkpoint
    uncond = bool(args.uncond)
    if args.precision == "fp32":
        torch.set_default_dtype(torch.float32)

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    # ---- which perturbation families ----
    if args.perturbation_type == "all":
        families = ["random_l2_orthogonal", "sampler_drift"]
    else:
        families = [args.perturbation_type]
    radii = parse_floats(args.perturbation_radii)
    t_values = parse_floats(args.t_values)

    # ---- sanity: checkpoint args agreement (warn-only) ----
    va = read_ckpt_args(args.vanilla_checkpoint)
    aa = read_ckpt_args(args.anm_checkpoint)
    ckpt_meta = {"vanilla_args": {k: va.get(k) for k in
                                  ("model", "path_type", "prediction", "ebm",
                                   "image_size", "num_classes", "uncond")},
                 "anm_args": {k: aa.get(k) for k in
                              ("model", "path_type", "prediction", "ebm",
                               "image_size", "num_classes", "uncond")}}
    for k in ("model", "path_type", "prediction", "ebm", "image_size"):
        if va.get(k) is not None and aa.get(k) is not None and va[k] != aa[k]:
            print(f"[WARN] vanilla/anm differ on {k}: {va[k]} vs {aa[k]}", file=sys.stderr)

    # ---- transport (repo target geometry) ----
    transport = create_transport(args.path_type, args.prediction, None, None, None)

    # ---- models (all resident: identical inputs for free) ----
    mk = dict(use_ema=args.use_ema, model_name=args.model, num_classes=args.num_classes,
              ebm=args.ebm, uncond=uncond, image_size=args.image_size, device=device)
    vanilla, v_src = load_eqm(args.vanilla_checkpoint, **mk)
    anm, a_src = load_eqm(args.anm_checkpoint, **mk)
    probe, _ = load_eqm(probe_ckpt, **mk)
    models = {"vanilla": vanilla, "anm": anm}
    ckpt_paths = {"vanilla": args.vanilla_checkpoint, "anm": args.anm_checkpoint}
    print(f"[info] loaded vanilla({v_src}), anm({a_src}), probe on {device}")

    # ---- VAE ----
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    def fwd(model):
        return lambda xin: model(xin, _t_holder["t"], _t_holder["y"], train=False)
    _t_holder = {}

    probe_fwd = fwd(probe)

    # ---- validation index ----
    n_needed = args.num_batches * args.batch_size
    index = build_val_index(args.val_path, n_needed)
    if len(index) < n_needed:
        print(f"[warn] only {len(index)} val images available (<{n_needed})", file=sys.stderr)

    # ---- resume bookkeeping ----
    jsonl_path = out_dir / "per_sample_metrics.jsonl"
    done_batches = set()
    rows = []
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done_batches.add(r["batch_idx"])
        print(f"[resume] {len(rows)} rows, {len(done_batches)} batches done", file=sys.stderr)
    jf = open(jsonl_path, "a")

    sanity = {"ckpt_meta": ckpt_meta, "vanilla_src": v_src, "anm_src": a_src,
              "first_batch": {}, "checks": {}}
    t_bin_edges = np.linspace(0, 1, 11)

    def to_bin(tv):
        return int(np.clip(np.digitize([tv], t_bin_edges)[0] - 1, 0, 9))

    rng_master = torch.Generator(device="cpu")

    nb = min(args.num_batches, (len(index) + args.batch_size - 1) // args.batch_size)
    for b in range(nb):
        if b in done_batches:
            continue
        sl = index[b * args.batch_size:(b + 1) * args.batch_size]
        if not sl:
            break
        sids = [s[0] for s in sl]
        labels = torch.tensor([s[1] for s in sl], device=device)
        paths = [s[2] for s in sl]
        B = len(sl)

        pix = load_pixels(paths, args.image_size, device)
        with torch.no_grad():
            x1 = vae.encode(pix).latent_dist.mode().mul_(0.18215)   # deterministic

        # deterministic x0 per batch
        rng_master.manual_seed(args.seed * 1_000_003 + b)
        x0 = torch.randn(x1.shape, generator=rng_master).to(device)
        path_dir = x1 - x0
        path_norm = _l2(path_dir).clamp_min(EPS)

        for ti, tv in enumerate(t_values):
            t = torch.full((B,), float(tv), device=device)
            xt, target = build_eqm_point_and_target(transport, x0, x1, t)
            _t_holder["t"] = t
            _t_holder["y"] = labels

            # build perturbation list: (ptype, radius_req, delta)
            perturbs = []
            for fam in families:
                if fam in ("random_l2", "random_l2_orthogonal"):
                    for ri, rr in enumerate(radii):
                        if rr == 0.0:
                            perturbs.append((fam, 0.0, torch.zeros_like(xt)))
                            continue
                        g = torch.Generator(device="cpu")
                        g.manual_seed(args.direction_seed * 7919 + b * 131 + ti * 17 + ri)
                        u = torch.randn(xt.shape, generator=g).to(device)
                        if fam == "random_l2_orthogonal":
                            u = _project_off_path(u, path_dir)
                        delta = _unit(u) * _bcast(rr * path_norm, xt)
                        perturbs.append((fam, rr, delta))
                elif fam == "sampler_drift":
                    gm = torch.Generator(device=device if device.type == "cuda" else "cpu")
                    gm.manual_seed(args.direction_seed * 31 + b * 131 + ti * 17)
                    d_mine = mine_delta(probe_fwd, xt, target, K=args.mining_K,
                                        eps_radius=args.mining_eps_radius,
                                        lr=args.mining_lr, gen=gm)
                    perturbs.append(("sampler_endpoint_mined", None, d_mine))
                    d_drift = gd_drift_delta(probe_fwd, xt, steps=args.sampler_steps,
                                             step_size=args.sampler_step_size,
                                             momentum=args.sampler_momentum,
                                             kind=args.sampler_kind, path_dir=path_dir)
                    perturbs.append(("sampler_local_drift", None, d_drift))

            for ptype, rr, delta in perturbs:
                xtd = xt + delta
                if rr == 0.0:
                    assert torch.equal(xtd, xt), "radius-0 must leave xt unchanged"
                radius_act = (_l2(delta) / path_norm).cpu()
                for ct, model in models.items():
                    with torch.no_grad():
                        pred = model(xtd, t, labels, train=False)
                    met = compute_metrics(pred, target)
                    if not torch.isfinite(met["mse"]).all():
                        raise FloatingPointError(f"non-finite mse: {ct} {ptype} t={tv}")
                    for i in range(B):
                        row = {
                            "run_id": args.run_id, "checkpoint_type": ct,
                            "checkpoint_path": ckpt_paths[ct], "use_ema": args.use_ema,
                            "sample_id": sids[i], "batch_idx": b, "item_idx": i,
                            "label": int(labels[i].item()), "perturbation_type": ptype,
                            "radius_rel_requested": rr,
                            "radius_rel_actual": float(radius_act[i]),
                            "delta_norm_abs": float(_l2(delta)[i].cpu()),
                            "sampler_steps": args.sampler_steps if "drift" in ptype else None,
                            "sampler_step_size": args.sampler_step_size if "drift" in ptype else None,
                            "sampler_momentum": args.sampler_momentum if "drift" in ptype else None,
                            "t": float(tv), "t_bin": to_bin(tv),
                            "target_norm": float(met["target_norm"][i]),
                            "pred_norm": float(met["pred_norm"][i]),
                            "field_norm_rms": float(met["field_norm_rms"][i]),
                            "norm_ratio": float(met["norm_ratio"][i]),
                            "mse": float(met["mse"][i]), "cosine": float(met["cosine"][i]),
                            "rel_norm_err": float(met["rel_norm_err"][i]),
                            "abs_norm_err": float(met["abs_norm_err"][i]),
                            "seed": args.seed, "direction_seed": args.direction_seed,
                        }
                        rows.append(row)
                        jf.write(json.dumps(row) + "\n")

            # ---- first-batch sanity (radius-0, sign, pairing) ----
            if b == 0 and ti == 0 and not sanity["first_batch"]:
                sanity["first_batch"] = {
                    "x0_hash": _tensor_hash(x0), "t_hash": _tensor_hash(t),
                    "target_hash": _tensor_hash(target), "xt_hash": _tensor_hash(xt),
                    "shapes": {"x1": list(x1.shape), "x0": list(x0.shape),
                               "xt": list(xt.shape), "target": list(target.shape)},
                }
                with torch.no_grad():
                    c0 = {ct: float(compute_metrics(models[ct](xt, t, labels, train=False),
                                                    target)["cosine"].mean())
                          for ct in models}
                sanity["checks"]["radius0_cosine_mean"] = c0
                sanity["checks"]["sign_ok"] = all(v > 0 for v in c0.values())
                if not sanity["checks"]["sign_ok"]:
                    sanity["checks"]["FATAL"] = "radius-0 cosine <=0 -> target sign mismatch"
                    (out_dir / "sanity" / "first_batch_checks.json").write_text(
                        json.dumps(sanity, indent=2))
                    jf.close()
                    raise SystemExit("FATAL: radius-0 cosine <= 0; fix target sign. "
                                     f"cosines={c0}")
        print(f"[batch {b+1}/{nb}] {B} samples, {len(rows)} rows total", file=sys.stderr)

    jf.close()
    (out_dir / "sanity" / "first_batch_checks.json").write_text(json.dumps(sanity, indent=2))

    agg = aggregate(rows, args.run_id)
    write_csv(out_dir / "aggregate_metrics.csv", agg)
    write_csv(out_dir / "paired_differences.csv", paired_differences(rows, args.run_id))
    make_plots(rows, agg, out_dir)

    # ---- console summary ----
    print("\n=== SUMMARY (mean over t-bins) ===")
    print(f"{'ptype':28s} {'radius':>8s} {'van_mse':>10s} {'anm_mse':>10s} "
          f"{'van_cos':>8s} {'anm_cos':>8s}")
    keyset = sorted({(r["perturbation_type"], r["radius_rel_value"]) for r in agg})
    for pt, rad in keyset:
        def grab(ct, m):
            sel = [r for r in agg if r["checkpoint_type"] == ct
                   and r["perturbation_type"] == pt and r["radius_rel_value"] == rad]
            ns = sum(r["n"] for r in sel)
            return sum(r[f"{m}_mean"] * r["n"] for r in sel) / ns if ns else float("nan")
        print(f"{pt:28s} {rad:>8s} {grab('vanilla','mse'):>10.4f} "
              f"{grab('anm','mse'):>10.4f} {grab('vanilla','cosine'):>8.4f} "
              f"{grab('anm','cosine'):>8.4f}")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
