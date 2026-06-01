"""Feature extraction for Experiment 3.

- Inception features: pytorch_fid InceptionV3 (block for 2048-d pool3), with the
  SAME Resize((299,299)) + ToTensor preprocessing the trusted FID sbatch uses.
  This is the single feature space for FID, KID and PRDC.
- Classifier predictions: torchvision resnet50 IMAGENET1K_V2 (same model
  eval_capabilities.py uses), with the weights' canonical preprocessing.

Both read PNG folders whose filenames are zero-padded global indices
({i:06d}.png), returning arrays ordered by that index so arm-vs-arm rows align.
"""
import pathlib

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".JPEG"}


def _list_images(image_dir):
    p = pathlib.Path(image_dir)
    return sorted([f for f in p.iterdir() if f.suffix in IMAGE_EXTS])


# --------------------------------------------------------------------------- #
# Inception (pytorch_fid) -- 2048-d pool3
# --------------------------------------------------------------------------- #
def _build_inception(device):
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()
    return model


def inception_features(image_dir, device="cuda", batch_size=64, files=None):
    """Return (features (N,2048) float64, ordered file stems)."""
    model = _build_inception(device)
    tf = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    files = files if files is not None else _list_images(image_dir)
    feats, stems = [], []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        imgs = []
        for f in batch_files:
            try:
                imgs.append(tf(Image.open(f).convert("RGB")))
                stems.append(pathlib.Path(f).stem)
            except Exception:
                continue
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        with torch.no_grad():
            pred = model(x)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
        feats.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    feats = np.concatenate(feats, axis=0).astype(np.float64)
    return feats, stems


def inception_features_from_tensors(images, device="cuda", batch_size=64):
    """images: (N,3,H,W) float in [0,1] already at native resolution."""
    model = _build_inception(device)
    resize = transforms.Resize((299, 299))
    feats = []
    for i in range(0, len(images), batch_size):
        x = resize(images[i:i + batch_size]).to(device)
        with torch.no_grad():
            pred = model(x)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
        feats.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)


# --------------------------------------------------------------------------- #
# Classifier (resnet50 IMAGENET1K_V2)
# --------------------------------------------------------------------------- #
def _build_classifier(device):
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device).eval()
    return model, weights.transforms()      # canonical preprocessing


def classifier_predictions(image_dir, device="cuda", batch_size=64, files=None):
    """Return dict: top1 (N,), top5 (N,5), prob_top1 (N,), stems list.

    Uses the weights' own transform so preprocessing matches the model exactly
    (preprocessing-mismatch is a documented failure mode)."""
    model, tf = _build_classifier(device)
    files = files if files is not None else _list_images(image_dir)
    top1, top5, p1, stems = [], [], [], []
    for i in range(0, len(files), batch_size):
        imgs = []
        for f in files[i:i + batch_size]:
            try:
                imgs.append(tf(Image.open(f).convert("RGB")))
                stems.append(pathlib.Path(f).stem)
            except Exception:
                continue
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            tp = probs.topk(5, dim=1)
        top1.extend(tp.indices[:, 0].cpu().tolist())
        top5.extend(tp.indices.cpu().tolist())
        p1.extend(tp.values[:, 0].cpu().tolist())
    return {"top1": np.asarray(top1), "top5": np.asarray(top5),
            "prob_top1": np.asarray(p1), "stems": stems}


def inception_score(prob_top1=None, image_dir=None, device="cuda",
                    batch_size=64, splits=10):
    """Inception Score from the resnet50 classifier (secondary metric).
    Computed from full softmax over the generated set."""
    model, tf = _build_classifier(device)
    files = _list_images(image_dir)
    all_probs = []
    for i in range(0, len(files), batch_size):
        imgs = [tf(Image.open(f).convert("RGB")) for f in files[i:i + batch_size]]
        x = torch.stack(imgs).to(device)
        with torch.no_grad():
            all_probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
    p = np.concatenate(all_probs, axis=0)
    n = len(p)
    scores = []
    for k in range(splits):
        part = p[k * n // splits:(k + 1) * n // splits]
        py = part.mean(axis=0, keepdims=True)
        kl = (part * (np.log(part + 1e-12) - np.log(py + 1e-12))).sum(axis=1)
        scores.append(np.exp(kl.mean()))
    return float(np.mean(scores)), float(np.std(scores))
