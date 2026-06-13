"""Stage 2 of the EqM Separability Diagnostic.

Assign each generated sample an INDEPENDENT good/garbage label that uses NO
signal from the gradient field (no circularity with the candidate scores).

PRIMARY label  = Inception-feature (pool3, 2048-d, the FID feature space)
                 nearest-neighbour distance from each generated image to a bank
                 of REAL ImageNet images. Garbage sits far from every real
                 feature; good sits close. Thresholds are fixed from the
                 distance distribution BEFORE any score is looked at:
                     good    = nn_dist < quantile(q_good)
                     garbage = nn_dist > quantile(q_garb)
                     drop the middle (ambiguous).
SECONDARY check = resnet50 IMAGENET1K_V2 max-softmax (garbage -> low). Reported
                 for direction-agreement only; NOT blended into the label.

Side product   = a bank of REAL latents (VAE-encoded real images), saved for
                 Stage 3's s4 (latent-space NN distance, a label-independent
                 dumb baseline / positive control on the label pipeline).

Reuses experiments/exp3_fidelity_diversity/features.py (the trusted FID feature
extractor) so the feature space is identical to the project's FID.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

VAE_SCALE = 0.18215
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".JPEG"}


def list_real_images(real_dir, n, seed):
    """Sample n real image paths uniformly from a directory tree (ImageFolder-
    style: real_dir/<class>/<img>.JPEG)."""
    root = Path(real_dir)
    files = []
    # walk one or two levels (class subdirs); fall back to recursive glob
    for f in root.rglob("*"):
        if f.suffix in IMAGE_EXTS:
            files.append(f)
        if len(files) >= n * 20:   # cap the walk; plenty to sample from
            break
    rng = np.random.default_rng(seed)
    if len(files) > n:
        idx = rng.choice(len(files), size=n, replace=False)
        files = [files[i] for i in idx]
    return files


def encode_latents(image_paths, device, image_size=256, batch_size=32):
    """VAE-encode real images to EqM latent space (mean * VAE_SCALE)."""
    import torch
    from PIL import Image
    from torchvision import transforms
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
    ])
    lat = []
    for i in range(0, len(image_paths), batch_size):
        imgs = []
        for f in image_paths[i:i + batch_size]:
            try:
                imgs.append(tf(Image.open(f).convert("RGB")))
            except Exception:
                continue
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean * VAE_SCALE
        lat.append(z.to("cpu", torch.float32).numpy())
    return np.concatenate(lat, axis=0) if lat else np.zeros((0, 4, image_size // 8, image_size // 8))


def knn_dist(gen_feats, real_feats, k):
    """For each gen feature, mean L2 distance to its k nearest real features.
    Chunked to bound memory."""
    import torch
    g = torch.tensor(gen_feats, dtype=torch.float32)
    r = torch.tensor(real_feats, dtype=torch.float32)
    out = np.zeros(len(g), dtype=np.float64)
    chunk = 256
    for i in range(0, len(g), chunk):
        d = torch.cdist(g[i:i + chunk], r)        # (c, M)
        kk = min(k, d.shape[1])
        nn = d.topk(kk, dim=1, largest=False).values
        out[i:i + chunk] = nn.mean(dim=1).numpy()
    return out


def main(args):
    import torch
    # exp3 feature extractor (trusted FID feature space) on sys.path
    _HERE = Path(__file__).resolve().parent
    for _p in (str(_HERE), str(_HERE.parent / "exp3_fidelity_diversity"),
               str(_HERE.parents[1] / "eqm-upstream")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from features import inception_features, classifier_predictions

    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder = Path(args.folder)
    gen_files = sorted([f for f in folder.iterdir() if f.suffix == ".png"])
    assert gen_files, f"no PNGs in {folder}"
    gen_ids = [int(f.stem) for f in gen_files]
    print(f"[sep-diag/labels] gen images: {len(gen_files)}", flush=True)

    # --- generated inception features ---
    gen_feats, gen_stems = inception_features(str(folder), device=device,
                                              batch_size=args.batch_size, files=gen_files)
    gen_ids = [int(s) for s in gen_stems]

    # --- real reference inception features ---
    real_files = list_real_images(args.real_dir, args.num_real, args.seed)
    assert real_files, f"no real images under {args.real_dir}"
    print(f"[sep-diag/labels] real reference images: {len(real_files)}", flush=True)
    real_feats, _ = inception_features([], device=device, batch_size=args.batch_size,
                                       files=real_files)

    # --- nn distance gen -> real, in FID feature space ---
    nn = knn_dist(gen_feats, real_feats, args.knn_k)

    # --- thresholds fixed from the distribution (pre-registered quantiles) ---
    tau_low = float(np.quantile(nn, args.q_good))
    tau_high = float(np.quantile(nn, 1.0 - args.q_garb))
    labels = np.where(nn < tau_low, "good",
                      np.where(nn > tau_high, "garbage", "ambiguous"))

    # --- secondary: resnet50 max-softmax (direction check only) ---
    clf = classifier_predictions(str(folder), device=device,
                                 batch_size=args.batch_size, files=gen_files)
    # align clf stems -> gen_ids order
    clf_map = {int(s): p for s, p in zip(clf["stems"], clf["prob_top1"])}
    max_softmax = np.asarray([clf_map.get(i, np.nan) for i in gen_ids])

    # agreement: garbage should have lower max-softmax than good
    good_ms = max_softmax[labels == "good"]
    garb_ms = max_softmax[labels == "garbage"]
    agree = float(np.nanmean(good_ms)) - float(np.nanmean(garb_ms))

    # --- real latent bank for Stage 3 s4 ---
    if args.build_latent_bank:
        real_lat = encode_latents(real_files, device, args.image_size, args.batch_size)
        np.savez_compressed(folder / "real_latents.npz", latents=real_lat)
        print(f"[sep-diag/labels] real latent bank: {real_lat.shape}", flush=True)

    # --- write outputs ---
    import csv
    with open(folder / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "nn_dist", "label", "max_softmax"])
        for i, d, lab, ms in zip(gen_ids, nn, labels, max_softmax):
            w.writerow([i, f"{d:.6f}", lab, f"{ms:.6f}"])

    thr = {
        "q_good": args.q_good, "q_garb": args.q_garb,
        "tau_low": tau_low, "tau_high": tau_high,
        "n_good": int((labels == "good").sum()),
        "n_garbage": int((labels == "garbage").sum()),
        "n_ambiguous": int((labels == "ambiguous").sum()),
        "knn_k": args.knn_k, "num_real": len(real_files),
        "secondary_max_softmax_good_minus_garbage": agree,
    }
    (folder / "thresholds.json").write_text(json.dumps(thr, indent=2))
    print(f"[sep-diag/labels] {json.dumps(thr)}", flush=True)
    if agree <= 0:
        print("[sep-diag/labels] WARNING: secondary max-softmax does NOT agree "
              "with label direction (good<=garbage). Labels may be suspect.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Stage 1 output dir (PNGs)")
    ap.add_argument("--real-dir", required=True, help="real ImageNet image tree")
    ap.add_argument("--num-real", type=int, default=10000)
    ap.add_argument("--knn-k", type=int, default=3)
    ap.add_argument("--q-good", type=float, default=0.40, help="good = bottom this frac of nn_dist")
    ap.add_argument("--q-garb", type=float, default=0.30, help="garbage = top this frac of nn_dist")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-latent-bank", action="store_true",
                    help="VAE-encode real images -> real_latents.npz for Stage 3 s4")
    args = ap.parse_args()
    main(args)
