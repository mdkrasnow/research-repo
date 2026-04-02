#!/usr/bin/env python3
"""
Download ImageNet-100 (100 classes, full resolution) from HuggingFace
and convert to ImageFolder format for EqM training.

No authentication required — uses ilee0022/ImageNet100 (public dataset).

Usage:
    python projects/diff-EqM/scripts/download_imagenet100.py \
        --output-dir /n/home03/mkrasnow/imagenet100

Output structure:
    imagenet100/
        train/
            class_000/
                000000.jpg
                000001.jpg
                ...
            class_001/
                ...
        validation/
            class_000/
                ...
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download ImageNet-100")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to save the dataset")
    parser.add_argument("--skip-resize", action="store_true",
                        help="Skip resizing to 256x256 (images will be variable size)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Downloading ImageNet-100 from ilee0022/ImageNet100 ===")
    print(f"Output directory: {output_dir}")

    from datasets import load_dataset
    from PIL import Image

    # Download (no token needed)
    print("Loading dataset (this will download ~17 GB)...")
    ds = load_dataset("ilee0022/ImageNet100")

    # Map split names
    split_map = {
        "train": "train",
        "validation": "validation",
    }

    for hf_split, folder_name in split_map.items():
        if hf_split not in ds:
            print(f"Split '{hf_split}' not found, skipping")
            continue

        split = ds[hf_split]
        out_dir = output_dir / folder_name
        print(f"\nConverting {hf_split} split ({len(split)} images) to {out_dir}/")

        for i, example in enumerate(split):
            label = example["label"]
            class_dir = out_dir / f"class_{label:03d}"
            class_dir.mkdir(parents=True, exist_ok=True)

            img = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize for EqM training (256x256 center crop handled by training script,
            # but images need to be at least 256px on shortest side)
            if not args.skip_resize:
                # Resize shortest side to 256, maintaining aspect ratio
                w, h = img.size
                if min(w, h) < 256:
                    scale = 256 / min(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            img.save(class_dir / f"{i:06d}.jpg", quality=95)

            if (i + 1) % 5000 == 0:
                print(f"  {hf_split}: {i + 1}/{len(split)} images saved")

        print(f"  {hf_split}: done ({len(split)} images)")

    # Count classes and images
    train_dir = output_dir / "train"
    if train_dir.exists():
        num_classes = len(list(train_dir.iterdir()))
        num_images = sum(1 for _ in train_dir.rglob("*.jpg"))
        print(f"\n=== Dataset ready ===")
        print(f"Location: {output_dir}")
        print(f"Classes: {num_classes}")
        print(f"Training images: {num_images}")
        print(f"Disk usage: {sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1e9:.1f} GB")
    else:
        print("WARNING: train directory not created")

    print(f"\nTo train EqM-B/2:")
    print(f"  --data-path {output_dir}/train --num-classes {num_classes}")


if __name__ == "__main__":
    main()
