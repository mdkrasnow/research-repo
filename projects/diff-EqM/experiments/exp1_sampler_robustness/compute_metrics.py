#!/usr/bin/env python3
"""
FID + KID from cached Inception-v3 (pool3, 2048-d) features. Evaluation only.

FID uses pytorch_fid's Frechet distance on a FROZEN reference (mu, sigma).
KID is the unbiased polynomial-kernel (degree 3) MMD^2 between generated and
reference features, estimated over random subsets -> (mean, std). Both metrics
read the SAME cached feature matrix, so a cell is scored once and reused on
resume / plots-only.

The reference feature set is built ONCE (deterministic ordering, no shuf) and
reused by all 80 conditions.
"""
import pathlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".JPEG"}


def _inception(device):
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model


def extract_features(image_dir, cache_path=None, batch_size=64, device="cuda",
                     max_images=None):
    """Return [N,2048] Inception features for all images in image_dir (sorted)."""
    if cache_path is not None and pathlib.Path(cache_path).exists():
        return np.load(cache_path)["feats"]

    model = _inception(device)
    tf = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    path = pathlib.Path(image_dir)
    files = sorted(f for f in path.iterdir() if f.suffix in IMAGE_EXTS)
    if max_images is not None:
        files = files[:max_images]
    if not files:
        raise RuntimeError(f"no images found in {image_dir}")

    acts = []
    for i in range(0, len(files), batch_size):
        imgs = []
        for f in files[i:i + batch_size]:
            try:
                imgs.append(tf(Image.open(f).convert("RGB")))
            except Exception:
                continue
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
        acts.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    feats = np.concatenate(acts, axis=0).astype(np.float64)

    if cache_path is not None:
        pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, feats=feats)
    return feats


def stats_from_feats(feats):
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def compute_fid(feats, ref_mu, ref_sigma):
    from pytorch_fid.fid_score import calculate_frechet_distance
    mu, sigma = stats_from_feats(feats)
    return float(calculate_frechet_distance(mu, sigma, ref_mu, ref_sigma))


def _poly_mmd2(x, y):
    """Unbiased polynomial-kernel (degree 3) MMD^2 estimate. x,y: [n,d],[m,d]."""
    d = x.shape[1]
    g = (x @ x.T) / d + 1.0
    g = g ** 3
    k = (y @ y.T) / d + 1.0
    k = k ** 3
    h = (x @ y.T) / d + 1.0
    h = h ** 3
    n, m = x.shape[0], y.shape[0]
    np.fill_diagonal(g, 0.0)
    np.fill_diagonal(k, 0.0)
    return (g.sum() / (n * (n - 1)) + k.sum() / (m * (m - 1))
            - 2.0 * h.mean())


def compute_kid(feats, ref_feats, n_subsets=100, subset_size=1000, seed=0):
    """KID mean/std + 95% CI over random subsets of size subset_size."""
    rng = np.random.default_rng(seed)
    sub = min(subset_size, feats.shape[0], ref_feats.shape[0])
    vals = []
    for _ in range(n_subsets):
        xi = rng.choice(feats.shape[0], sub, replace=False)
        yi = rng.choice(ref_feats.shape[0], sub, replace=False)
        vals.append(_poly_mmd2(feats[xi], ref_feats[yi]))
    vals = np.array(vals)
    return (float(vals.mean()), float(vals.std()),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def bootstrap_fid_ci(feats, ref_mu, ref_sigma, n_boot=50, seed=0):
    """Bootstrap FID CI by resampling generated features."""
    rng = np.random.default_rng(seed)
    n = feats.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(compute_fid(feats[idx], ref_mu, ref_sigma))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
