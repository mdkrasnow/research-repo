"""Build & cache the FIXED reference feature set for Experiment 3 (one-time).

Confound controlled: the trusted FID sbatch re-shuffles its 50K reference every
run (`find | shuf -n`). Exp3 fixes ONE seeded reference subsample and caches its
features so vanilla and ANM are scored against identical statistics.

Outputs (under --out-dir):
  inception_feats.npy           (aggregate-n, 2048) features for FID/KID/PRDC ref
  inception_mu_sigma.npz        mu, sigma of the aggregate set
  inception_by_class.npz        per-class mu[c], cov[c], count[c]  (full-train based)
  inception_feats_by_class.npz  per-class raw feats (for per-class FID; ref_per_class each)
  real_classifier_hist.npy      resnet50 top1 histogram over the aggregate set (prob)
  meta.json                     seed, counts, source paths

Class index convention: sorted synset dir order == training ImageFolder order
== eval_capabilities.load_val_images convention.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import features as feat_mod


def list_class_files(root, num_classes, seed):
    synsets = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))[:num_classes]
    rng = np.random.default_rng(seed)
    files_by_class = {}
    for ci, syn in enumerate(synsets):
        d = os.path.join(root, syn)
        fs = sorted(os.listdir(d))
        rng.shuffle(fs)
        files_by_class[ci] = [os.path.join(d, f) for f in fs]
    return synsets, files_by_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-train", default="/n/holylabs/ydu_lab/Lab/raywang4/imagenet/train")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--aggregate-n", type=int, default=50000)
    ap.add_argument("--ref-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _, files_by_class = list_class_files(args.imagenet_train,
                                         args.num_classes, args.seed)

    # ---- aggregate reference (balanced seeded subsample) ----
    per_cls_agg = int(np.ceil(args.aggregate_n / args.num_classes))
    agg_files = []
    for ci in range(args.num_classes):
        agg_files.extend(files_by_class[ci][:per_cls_agg])
    agg_files = agg_files[:args.aggregate_n]
    print(f"[ref] aggregate set: {len(agg_files)} images", flush=True)

    agg_feats, _ = feat_mod.inception_features(None, device=device,
                                               batch_size=args.batch_size,
                                               files=agg_files)
    np.save(out / "inception_feats.npy", agg_feats)
    mu = agg_feats.mean(axis=0)
    sigma = np.cov(agg_feats, rowvar=False)
    np.savez(out / "inception_mu_sigma.npz", mu=mu, sigma=sigma)

    # ---- real classifier histogram over the aggregate set ----
    preds = feat_mod.classifier_predictions(None, device=device,
                                            batch_size=args.batch_size,
                                            files=agg_files)
    # resnet50 is 1000-way regardless of num_classes -> always 1000 bins
    counts = np.bincount(preds["top1"], minlength=1000).astype(np.float64)
    np.save(out / "real_classifier_hist.npy", counts / counts.sum())

    # ---- per-class reference stats (full-train based, ref_per_class each) ----
    by_mu, by_cov, by_cnt = {}, {}, {}
    raw_by_class = {}
    for ci in range(args.num_classes):
        cfiles = files_by_class[ci][:args.ref_per_class]
        cf, _ = feat_mod.inception_features(None, device=device,
                                            batch_size=args.batch_size, files=cfiles)
        by_mu[str(ci)] = cf.mean(axis=0)
        by_cov[str(ci)] = np.cov(cf, rowvar=False)
        by_cnt[str(ci)] = len(cf)
        raw_by_class[str(ci)] = cf.astype(np.float32)
        if ci % 100 == 0:
            print(f"[ref] per-class {ci}/{args.num_classes}", flush=True)
    np.savez(out / "inception_by_class.npz",
             **{f"mu_{k}": v for k, v in by_mu.items()},
             **{f"cov_{k}": v for k, v in by_cov.items()},
             **{f"cnt_{k}": np.array(v) for k, v in by_cnt.items()})
    np.savez(out / "inception_feats_by_class.npz", **raw_by_class)

    (out / "meta.json").write_text(json.dumps({
        "imagenet_train": args.imagenet_train,
        "num_classes": args.num_classes,
        "aggregate_n": len(agg_files),
        "ref_per_class": args.ref_per_class,
        "seed": args.seed,
        "synset_order_is_class_index": True,
    }, indent=2))
    print(f"[ref] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
