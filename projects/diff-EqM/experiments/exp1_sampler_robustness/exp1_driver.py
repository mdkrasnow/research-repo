#!/usr/bin/env python3
"""
Experiment 1 driver -- NFE / sampler robustness: vanilla EqM vs ANM EqM (v10).
EVALUATION ONLY. Reuses sample_gd_fixed.py for generation and compute_metrics.py
for FID/KID. Deterministic: fixed init latents + fixed balanced labels + a frozen
reference set are precomputed once and reused by all conditions.

Grid (80 cells): checkpoint_type{vanilla,anm} x sampler{gd,ngd}
                 x nfe{10,25,50,100,250} x step_mult{0.5,1.0,1.5,2.0}

Modes: --dry-run, --smoke {A,B}, --resume, --plots-only.

All sampling is GPU-only and is intended to run on the cluster via
slurm/jobs/exp1_sweep.sbatch. Run locally only for --dry-run / --plots-only.
"""
import argparse
import csv
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SAMPLE_SCRIPT = HERE / "sample_gd_fixed.py"

CSV_FIELDS = [
    "run_id", "checkpoint_type", "checkpoint_path", "checkpoint_sha", "model_arch",
    "dataset", "resolution", "sampler", "nfe", "step_mult", "base_step_size",
    "step_size", "nag_mu", "num_samples", "batch_size", "seed_schedule_id",
    "label_schedule_id", "init_noise_schedule_id", "ema_enabled", "cfg_scale",
    "vae_id", "reference_stats_path", "fid", "fid_ci_low", "fid_ci_high",
    "kid_mean", "kid_std", "kid_ci_low", "kid_ci_high", "sfid",
    "inception_score_mean", "inception_score_std", "wall_clock_sec",
    "images_per_sec", "nfe_field", "nfe_raw_forward", "nan_count",
    "divergence_count", "clip_fraction", "mean_init_latent_norm",
    "mean_final_latent_norm", "std_final_latent_norm", "mean_first_grad_norm",
    "mean_final_grad_norm", "feature_path", "sample_path", "num_gpus",
    "precision", "git_commit", "timestamp", "notes",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla-ckpt", required=True)
    p.add_argument("--anm-ckpt", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--outdir", required=True)
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--ref-samples", type=int, default=None,
                   help="reference image count (default = num_samples)")
    p.add_argument("--sample-batch-size", type=int, default=64)
    p.add_argument("--base-step-size", type=float, default=0.003)
    p.add_argument("--nag-mu", type=float, default=0.3)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--vae", default="ema")
    p.add_argument("--imagenet-ref", default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train")
    p.add_argument("--global-seed", type=int, default=123)
    p.add_argument("--nfe-list", default="10,25,50,100,250")
    p.add_argument("--step-mults", default="0.5,1.0,1.5,2.0")
    p.add_argument("--samplers", default="gd,ngd")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--kid-subsets", type=int, default=100)
    p.add_argument("--kid-subset-size", type=int, default=1000)
    p.add_argument("--smoke", choices=["A", "B"], default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-generate even if a cell's outputs already exist")
    return p.parse_args()


def sha256_file(path, limit_mb=512):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk); read += len(chunk)
            if read >= limit_mb * (1 << 20):
                break
    return h.hexdigest()[:16]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(HERE)).decode().strip()
    except Exception:
        return "unknown"


def grid_from_args(a):
    if a.smoke == "A":
        return ["gd"], [10], [1.0]
    if a.smoke == "B":
        return ["gd", "ngd"], [10, 25], [1.0, 2.0]
    return (a.samplers.split(","),
            [int(x) for x in a.nfe_list.split(",")],
            [float(x) for x in a.step_mults.split(",")])


# --------------------------------------------------------------------------- #
# Fixed inputs
# --------------------------------------------------------------------------- #
def precompute_fixed_inputs(a, eval_dir):
    import torch
    eval_dir.mkdir(parents=True, exist_ok=True)
    latent = a.image_size // 8
    seeds_p = eval_dir / "seeds.npy"
    labels_p = eval_dir / "labels.npy"
    latents_p = eval_dir / "init_latents.pt"
    meta_p = eval_dir / "metadata.json"
    # Pad to a whole number of batches: sample_gd_fixed iterates over
    # total_samples = ceil(num_samples/batch)*batch and indexes the bank by global
    # id, so the bank must cover every index that gets sampled.
    bs = a.sample_batch_size
    N = int(np.ceil(a.num_samples / bs) * bs)

    if all(p.exists() for p in [seeds_p, labels_p, latents_p, meta_p]):
        return latents_p, labels_p, json.loads(meta_p.read_text())

    rng = np.random.default_rng(a.global_seed)
    seeds = rng.integers(0, 2**31 - 1, size=N).astype(np.int64)

    # balanced labels: ceil(N/C) per class, trim to N, fixed shuffle
    C = a.num_classes
    per = int(np.ceil(N / C))
    labels = np.tile(np.arange(C), per)[:N]
    rng.shuffle(labels)
    labels = labels.astype(np.int64)

    g = torch.Generator().manual_seed(a.global_seed)
    init_latents = torch.randn(N, 4, latent, latent, generator=g)

    np.save(seeds_p, seeds)
    np.save(labels_p, labels)
    torch.save(init_latents, latents_p)
    meta = {
        "num_samples": N, "global_seed": a.global_seed, "latent": latent,
        "seed_schedule_id": f"seed{a.global_seed}_n{N}",
        "label_schedule_id": f"balanced_c{C}_seed{a.global_seed}_n{N}",
        "init_noise_schedule_id": f"randn_seed{a.global_seed}_n{N}",
        "label_hist_first10": np.bincount(labels, minlength=C)[:10].tolist(),
        "latent_hash_first16": hashlib.sha256(
            init_latents[:16].numpy().tobytes()).hexdigest()[:16],
    }
    meta_p.write_text(json.dumps(meta, indent=2))
    print(f"[inputs] wrote fixed seeds/labels/latents for N={N}")
    return latents_p, labels_p, meta


def _center_crop_256(pil, size=256):
    from PIL import Image
    while min(pil.size) >= 2 * size:
        pil = pil.resize(tuple(x // 2 for x in pil.size), resample=Image.BOX)
    scale = size / min(pil.size)
    pil = pil.resize(tuple(round(x * scale) for x in pil.size), resample=Image.BICUBIC)
    arr = np.array(pil)
    cy = (arr.shape[0] - size) // 2
    cx = (arr.shape[1] - size) // 2
    return Image.fromarray(arr[cy:cy + size, cx:cx + size])


def precompute_reference(a, eval_dir):
    """Build a FROZEN, deterministically-ordered 256x256 reference set ONCE and
    cache its Inception features + (mu, sigma). No shuf."""
    from PIL import Image
    import compute_metrics as cm

    ref_count = a.ref_samples or a.num_samples
    ref_dir = eval_dir / "ref_flat"
    stats_p = eval_dir / "ref_stats.npz"
    feats_p = eval_dir / "ref_feats.npz"

    if stats_p.exists() and feats_p.exists():
        d = np.load(stats_p)
        return d["mu"], d["sigma"], np.load(feats_p)["feats"], str(stats_p)

    root = Path(a.imagenet_ref)
    if not root.exists():
        raise FileNotFoundError(f"ImageNet reference not found: {root}")
    exts = {".JPEG", ".jpg", ".jpeg", ".png"}
    all_files = sorted(str(f) for f in root.rglob("*") if f.suffix in exts)
    if not all_files:
        raise RuntimeError(f"no images under {root}")
    rng = np.random.default_rng(a.global_seed)  # deterministic subset, NOT shuf
    pick = rng.choice(len(all_files), min(ref_count, len(all_files)), replace=False)
    pick.sort()
    chosen = [all_files[i] for i in pick]

    ref_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, f in enumerate(chosen):
        try:
            _center_crop_256(Image.open(f).convert("RGB")).save(ref_dir / f"{i:06d}.png")
            n += 1
        except Exception as e:
            print("skip ref", f, e)
    print(f"[ref] wrote {n} reference images")

    feats = cm.extract_features(str(ref_dir), cache_path=str(feats_p),
                                batch_size=a.sample_batch_size)
    mu, sigma = cm.stats_from_feats(feats)
    np.savez(stats_p, mu=mu, sigma=sigma)
    return mu, sigma, feats, str(stats_p)


# --------------------------------------------------------------------------- #
# Per-condition
# --------------------------------------------------------------------------- #
def run_id(ct, sampler, nfe, sm):
    return f"{ct}_{sampler}_nfe{nfe}_step{sm}"


def have_outputs(samples_dir, feat_path, expected_min):
    """Complete iff samples folder has >= expected_min PNGs AND the feature
    cache exists. Count check (not mere existence) prevents resume from skipping
    a cell that was interrupted mid-generation."""
    return (Path(samples_dir).exists()
            and len(list(Path(samples_dir).glob("*.png"))) >= expected_min
            and Path(feat_path).exists())


def _free_port():
    """OS-assigned free TCP port. Avoids the master-port collisions (EADDRINUSE)
    a fixed hash%1000 scheme hits across 80 sequential torchrun launches."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def generate(a, ckpt, sampler, nfe, sm, samples_dir, latents_p, labels_p):
    base = [
        str(SAMPLE_SCRIPT),
        "--model", a.model, "--image-size", str(a.image_size),
        "--num-classes", str(a.num_classes), "--ckpt", ckpt,
        "--sampler", sampler, "--num-sampling-steps", str(nfe),
        "--stepsize", str(a.base_step_size), "--step-mult", str(sm),
        "--mu", str(a.nag_mu), "--cfg-scale", str(a.cfg_scale), "--vae", a.vae,
        "--global-batch-size", str(a.sample_batch_size),
        "--num-fid-samples", str(a.num_samples),
        "--folder", str(samples_dir),
        "--init-latents", str(latents_p), "--labels", str(labels_p),
        "--global-seed", str(a.global_seed),
    ]
    t0 = time.time()
    rc = 1
    for attempt in range(3):
        port = _free_port()
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
               f"--nproc_per_node={a.num_gpus}", f"--master_port={port}"] + base
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            break
        print(f"  [retry] generate rc={rc} (attempt {attempt+1}/3), new port next")
    return rc, time.time() - t0


def main():
    a = parse_args()
    sys.path.insert(0, str(HERE))
    out = Path(a.outdir)
    eval_dir = out / "eval_inputs"
    samples_root = out / "samples"
    feats_root = out / "features"
    metrics_dir = out / "metrics"
    plots_dir = out / "plots"
    for d in [metrics_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    results_csv = metrics_dir / "results.csv"
    manifest = metrics_dir / "run_manifest.jsonl"

    import analysis
    import plots as plt_mod

    if a.plots_only:
        analysis.run_all(str(results_csv), str(metrics_dir))
        plt_mod.plot_all(str(results_csv), str(metrics_dir), str(plots_dir))
        print(f"[plots-only] wrote plots to {plots_dir}")
        return

    samplers, nfes, step_mults = grid_from_args(a)
    ckpts = {"vanilla": a.vanilla_ckpt, "anm": a.anm_ckpt}
    cells = [(ct, s, n, m) for ct in ["vanilla", "anm"]
             for s in samplers for n in nfes for m in step_mults]

    # validate checkpoints exist
    for ct, ck in ckpts.items():
        if not Path(ck).exists():
            print(f"ERROR: {ct} checkpoint not found: {ck}", file=sys.stderr)
            if not a.dry_run:
                sys.exit(2)

    if a.dry_run:
        print(f"[dry-run] {len(cells)} cells, num_samples={a.num_samples}, gpus={a.num_gpus}")
        for ct, s, n, m in cells:
            print(f"  {run_id(ct, s, n, m)}  step={a.base_step_size * m:.5f}")
        print(f"[dry-run] imagenet ref: {a.imagenet_ref} "
              f"(exists={Path(a.imagenet_ref).exists()})")
        return

    # precompute fixed inputs + frozen reference (once)
    latents_p, labels_p, meta = precompute_fixed_inputs(a, eval_dir)
    ref_mu, ref_sigma, ref_feats, ref_stats_path = precompute_reference(a, eval_dir)

    import compute_metrics as cm

    # init csv header if needed
    if not results_csv.exists():
        with open(results_csv, "w", newline="") as f:
            csv.DictWriter(f, CSV_FIELDS).writeheader()
    # done_valid = cells already in the CSV with a non-empty FID. On --resume we
    # trust a recorded metric and skip the cell even if its local samples/features
    # are gone (e.g. seeded from a recovered job.log after a crash).
    done_valid = set()
    if results_csv.exists():
        import pandas as pd
        try:
            prev = pd.read_csv(results_csv)
            done_valid = set(prev[prev.fid.notna()].run_id.astype(str))
        except Exception:
            done_valid = set()

    gsha = git_commit()
    # a completed cell has >= num_samples PNGs (full run) or >= 16 (smoke)
    expected_min = min(a.num_samples, 16) if a.num_samples < 1000 else a.num_samples
    for ct, sampler, nfe, sm in cells:
        rid = run_id(ct, sampler, nfe, sm)
        samples_dir = samples_root / ct / sampler / f"nfe_{nfe}" / f"step_{sm}"
        feat_path = feats_root / f"{rid}_inception.npz"
        notes = ""

        if a.resume and (rid in done_valid or have_outputs(samples_dir, feat_path, expected_min)):
            print(f"[skip] {rid}")
            continue
        if have_outputs(samples_dir, feat_path, expected_min) and not a.force and not a.resume:
            print(f"[exists] {rid} (use --resume to skip, --force to redo)")
        samples_dir.mkdir(parents=True, exist_ok=True)
        # invalidate any stale feature cache before regenerating this cell, else
        # extract_features() would silently return the previous run's features.
        Path(feat_path).unlink(missing_ok=True)

        rc, wall = generate(a, ckpts[ct], sampler, nfe, sm, samples_dir, latents_p, labels_p)
        png_n = len(list(samples_dir.glob("*.png")))
        gen_stats = {}
        gsp = samples_dir / "gen_stats.json"
        if gsp.exists():
            gen_stats = json.loads(gsp.read_text())

        fid = kid_m = kid_s = kid_lo = kid_hi = fid_lo = fid_hi = ""
        if rc != 0:
            notes = f"generate_rc={rc}"
        elif png_n < min(a.num_samples, 16):
            notes = f"too_few_samples={png_n}"
        else:
            try:
                feats = cm.extract_features(str(samples_dir), cache_path=str(feat_path),
                                            batch_size=a.sample_batch_size)
                if np.var(feats) < 1e-8:
                    notes = "degenerate_features(near-identical samples)"
                if a.num_samples < 5000 or a.smoke:
                    notes = (notes + ";" if notes else "") + "UNRELIABLE_small_n"
                fid = cm.compute_fid(feats, ref_mu, ref_sigma)
                fid_lo, fid_hi = cm.bootstrap_fid_ci(feats, ref_mu, ref_sigma)
                kid_m, kid_s, kid_lo, kid_hi = cm.compute_kid(
                    feats, ref_feats, a.kid_subsets, a.kid_subset_size)
            except Exception as e:
                notes = f"metric_error:{type(e).__name__}:{e}"

        row: dict[str, object] = {k: "" for k in CSV_FIELDS}
        row.update({
            "run_id": rid, "checkpoint_type": ct, "checkpoint_path": ckpts[ct],
            "checkpoint_sha": sha256_file(ckpts[ct]) if Path(ckpts[ct]).exists() else "",
            "model_arch": a.model, "dataset": "imagenet1k", "resolution": a.image_size,
            "sampler": sampler, "nfe": nfe, "step_mult": sm,
            "base_step_size": a.base_step_size, "step_size": a.base_step_size * sm,
            "nag_mu": a.nag_mu if sampler == "ngd" else "",
            "num_samples": a.num_samples, "batch_size": a.sample_batch_size,
            "seed_schedule_id": meta["seed_schedule_id"],
            "label_schedule_id": meta["label_schedule_id"],
            "init_noise_schedule_id": meta["init_noise_schedule_id"],
            "ema_enabled": True, "cfg_scale": a.cfg_scale, "vae_id": f"sd-vae-ft-{a.vae}",
            "reference_stats_path": ref_stats_path,
            "fid": fid, "fid_ci_low": fid_lo, "fid_ci_high": fid_hi,
            "kid_mean": kid_m, "kid_std": kid_s, "kid_ci_low": kid_lo, "kid_ci_high": kid_hi,
            "wall_clock_sec": round(wall, 1),
            "images_per_sec": round(png_n / wall, 3) if wall > 0 else "",
            "nfe_field": gen_stats.get("nfe_field", nfe - 1),
            "nfe_raw_forward": (nfe - 1) * (2 if a.cfg_scale > 1.0 else 1),
            "nan_count": gen_stats.get("nan_count", ""),
            "divergence_count": gen_stats.get("divergence_count", ""),
            "clip_fraction": gen_stats.get("clip_fraction", ""),
            "mean_init_latent_norm": gen_stats.get("mean_init_latent_norm", ""),
            "mean_final_latent_norm": gen_stats.get("mean_final_latent_norm", ""),
            "std_final_latent_norm": gen_stats.get("std_final_latent_norm", ""),
            "mean_first_grad_norm": gen_stats.get("mean_first_grad_norm", ""),
            "mean_final_grad_norm": gen_stats.get("mean_final_grad_norm", ""),
            "feature_path": str(feat_path), "sample_path": str(samples_dir),
            "num_gpus": a.num_gpus, "precision": "tf32",
            "git_commit": gsha, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": notes,
        })
        with open(results_csv, "a", newline="") as f:
            csv.DictWriter(f, CSV_FIELDS).writerow(row)
        with open(manifest, "a") as f:
            f.write(json.dumps({
                "run_id": rid, "git_commit": gsha, "checkpoint_sha": row["checkpoint_sha"],
                "config_hash": hashlib.sha256(json.dumps(vars(a), sort_keys=True).encode()).hexdigest()[:16],
                "latent_hash_first16": meta["latent_hash_first16"],
                "step_size": row["step_size"], "png_count": png_n, "notes": notes,
            }) + "\n")
        print(f"[done] {rid}  fid={fid}  kid={kid_m}  n={png_n}  {wall:.0f}s  {notes}")

    # completeness check
    expected = len(cells)
    import pandas as pd
    got = len(pd.read_csv(results_csv))
    if got < expected:
        print(f"WARNING: {got}/{expected} rows present", file=sys.stderr)

    analysis.run_all(str(results_csv), str(metrics_dir))
    plt_mod.plot_all(str(results_csv), str(metrics_dir), str(plots_dir))
    print(f"[exp1] complete. metrics -> {metrics_dir}, plots -> {plots_dir}")


if __name__ == "__main__":
    main()
