#!/usr/bin/env python3
"""Build Experiment 4 reference banks (fixes B1: project had no val-reference FID).

Produces, with ONE Inception extractor and ONE unified .npz schema {mu,sigma}:
  <out>/in1k_val_ref_stats.npz       inception mu/sigma over 50K val images
  <out>/in1k_train_ref_stats.npz     inception mu/sigma over a FIXED class-balanced 50K train subset
  <out>/in1k_val_inception_feats.npy   raw inception feats (val)   [for KID]
  <out>/in1k_train_inception_feats.npy raw inception feats (train) [for KID]
  <out>/in1k_val_dino[.npy,_labels.npy,_ids.npy,_images.npz]      DINO NN bank (val)
  <out>/in1k_train_dino[...]                                       DINO NN bank (train)

Equal counts for train and val (no sample-size bias). Train subset is selected
with a fixed seed (reproducible, rule-based). _images.npz holds 128x128 uint8
thumbnails parallel to features, for NN panels (omit with --no-thumbs to save disk).

Usage (cluster, GPU):
  python build_references.py --train-root .../imagenet/train --val-root .../imagenet/val \
      --out projects/diff-EqM/results --count 50000 --per-class 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as feat
import metrics as met

_EXTS = (".JPEG", ".jpeg", ".jpg", ".png", ".JPG")
_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def list_class_balanced(root: str, per_class: int, seed: int):
    """Return (paths, labels, ids) with up to per_class images per synset, fixed order."""
    rng = np.random.default_rng(seed)
    classes = sorted([d for d in Path(root).iterdir() if d.is_dir()])
    paths, labels, ids = [], [], []
    for ci, cdir in enumerate(classes):
        files = sorted([f for f in cdir.iterdir() if f.suffix in _EXTS])
        if len(files) > per_class:
            sel = rng.choice(len(files), per_class, replace=False)
            files = [files[i] for i in sorted(sel)]
        for f in files:
            paths.append(str(f)); labels.append(ci); ids.append(f"{cdir.name}/{f.name}")
    return paths, np.array(labels, np.int64), np.array(ids)


def load_images(paths, size, thumb=None):
    big = np.empty((len(paths), size, size, 3), np.uint8)
    thumbs = None if thumb is None else np.empty((len(paths), thumb, thumb, 3), np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        # ADM-style center crop to square then resize
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        big[i] = np.asarray(im.resize((size, size), _BICUBIC), np.uint8)
        if thumbs is not None:
            thumbs[i] = np.asarray(im.resize((thumb, thumb), _BICUBIC), np.uint8)
    return big, thumbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=50000)
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--thumb", type=int, default=128)
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--fid-backbone", default="inception_pool3")
    ap.add_argument("--nn-backbone", default="dinov2_vitl14")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    inc = feat.build_extractor(args.fid_backbone, device=device)
    dino = feat.build_extractor(args.nn_backbone, device=device)
    thumb = None if args.no_thumbs else args.thumb

    for split, root in [("val", args.val_root), ("train", args.train_root)]:
        paths, labels, ids = list_class_balanced(root, args.per_class, args.seed)
        if len(paths) > args.count:
            paths, labels, ids = paths[: args.count], labels[: args.count], ids[: args.count]
        print(f"[{split}] {len(paths)} images -> features", flush=True)
        imgs, thumbs = load_images(paths, args.image_size, thumb)

        ifeat = inc.extract(imgs, batch_size=args.batch_size)
        mu, sigma = met.compute_statistics(ifeat)
        np.savez(out / f"in1k_{split}_ref_stats.npz", mu=mu, sigma=sigma, num_images=len(paths))
        np.save(out / f"in1k_{split}_inception_feats.npy", ifeat.astype(np.float32))

        dfeat = dino.extract(imgs, batch_size=args.batch_size)
        np.save(out / f"in1k_{split}_dino.npy", dfeat.astype(np.float32))
        np.save(out / f"in1k_{split}_dino_labels.npy", labels)
        np.save(out / f"in1k_{split}_dino_ids.npy", ids)
        if thumbs is not None:
            np.savez(out / f"in1k_{split}_dino_images.npz", arr_0=thumbs)
        # Bank metadata -> the audit asserts its NN extractor matches this
        # (catches feature-extractor / preprocessing mismatch, failure mode #4).
        import json as _json
        (out / f"in1k_{split}_dino_meta.json").write_text(_json.dumps({
            "nn_backbone": args.nn_backbone,
            "nn_preprocessing_hash": dino.preprocessing_hash(),
            "fid_backbone": args.fid_backbone,
            "fid_preprocessing_hash": inc.preprocessing_hash(),
            "count": len(paths), "image_size": args.image_size,
            "per_class": args.per_class, "subset_seed": args.seed, "split": split,
        }, indent=2))
        print(f"[{split}] wrote ref_stats + inception_feats + dino bank "
              f"(prep: {inc.preprocessing_hash()} | {dino.preprocessing_hash()})")

    print("DONE. References use unified {mu,sigma} schema; equal train/val counts.")


if __name__ == "__main__":
    main()
