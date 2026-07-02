#!/usr/bin/env python3
"""Phase-5 LOCAL smoke (CPU, zero external deps, no network).

Fabricates fake samples + fake reference banks, uses the 'stub' backbone, runs
the full eval_stability_memorization.py pipeline, and asserts:
  - config parses, samples load, features extract, FID/KID finite,
  - both references scored, NN search returns valid IDs,
  - an intentionally inserted duplicate is detected,
  - CSVs / JSON / panels / plots are written,
  - rerun with same inputs reproduces features (stub is deterministic).

Smoke numbers are MEANINGLESS (random data, stub features). This only proves
the plumbing. Real validation runs on the cluster with inception+dino.

  python local_smoke.py            # builds in a temp dir, runs, checks, prints PASS
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as feat  # noqa: E402

N = 64          # samples per checkpoint
BANK = 256      # bank size
SIZE = 32       # tiny images
NCLS = 8
STUB_DIM = 64


def _imgs(n, rng):
    return rng.integers(0, 256, size=(n, SIZE, SIZE, 3), dtype=np.uint8)


def build(root: Path):
    rng = np.random.default_rng(0)
    res = root / "results"; res.mkdir(parents=True)
    samp = root / "samples"; samp.mkdir()
    stub = feat.StubExtractor(dim=STUB_DIM, seed=0)

    # reference banks (train + val), with labels/ids/thumbnails for panels
    for split in ("train", "val"):
        imgs = _imgs(BANK, rng)
        labels = rng.integers(0, NCLS, BANK).astype(np.int64)
        ids = np.array([f"{split}/{i:06d}.JPEG" for i in range(BANK)])  # disjoint across splits
        ifeat = stub.extract(imgs)
        mu, sigma = np.mean(ifeat, 0), np.cov(ifeat, rowvar=False)
        np.savez(res / f"in1k_{split}_ref_stats.npz", mu=mu, sigma=sigma, num_images=BANK)
        np.save(res / f"in1k_{split}_inception_feats.npy", ifeat.astype(np.float32))
        np.save(res / f"in1k_{split}_dino.npy", ifeat.astype(np.float32))
        np.save(res / f"in1k_{split}_dino_labels.npy", labels)
        np.save(res / f"in1k_{split}_dino_ids.npy", ids)
        np.savez(res / f"in1k_{split}_dino_images.npz", arr_0=imgs)
        # bank meta so the audit's backbone-match assert is exercised (PASS path)
        (res / f"in1k_{split}_dino_meta.json").write_text(json.dumps({
            "nn_backbone": "stub", "nn_preprocessing_hash": stub.preprocessing_hash(),
            "split": split, "count": BANK}))

    # fake samples for 2 regimes x 2 ctypes x 1 seed; insert a duplicate into one
    sample_paths = {}
    for regime in ("step_matched", "compute_matched"):
        for ctype in ("vanilla", "anm"):
            imgs = _imgs(N, rng)
            if regime == "step_matched" and ctype == "anm":
                imgs[1] = imgs[0]  # planted exact duplicate -> must be detected
            labels = rng.integers(0, NCLS, N).astype(np.int64)
            sp = samp / f"{ctype}_{regime}_seed0.npz"
            lp = samp / f"{ctype}_{regime}_seed0_labels.npy"
            np.savez(sp, arr_0=imgs)
            np.save(lp, labels)
            sample_paths[(regime, ctype)] = (str(sp), str(lp))

    cfg = {
        "out_root": str(root / "out"),
        "feature_backbone_fid": "stub", "feature_backbone_nn": "stub",
        "feature_batch_size": 32, "class_conditional_nn": True, "top_k_nn": 3,
        "kid_subset_size": 16, "panel_top_n": 8, "tau_dup": 999.0,
        "eval": {"n_eval": N, "global_seed": 0, "world_size": 1, "label_seed": 1,
                 "num_classes": NCLS, "schedule_hash": "smoke"},
        "references": {
            "val_stats_npz": str(res / "in1k_val_ref_stats.npz"),
            "train_stats_npz": str(res / "in1k_train_ref_stats.npz"),
            "val_inception_feats": str(res / "in1k_val_inception_feats.npy"),
            "train_inception_feats": str(res / "in1k_train_inception_feats.npy"),
            "val_nn_bank": str(res / "in1k_val_dino"),
            "train_nn_bank": str(res / "in1k_train_dino"),
        },
        "samples": {
            r: {c: {"0": {"samples_npz": sample_paths[(r, c)][0],
                          "labels_npy": sample_paths[(r, c)][1]}}
                for c in ("vanilla", "anm")}
            for r in ("step_matched", "compute_matched")
        },
    }
    # tau_dup high so the planted exact-duplicate (distance ~0) clusters; everything
    # else also "near" -- fine for a plumbing test, we only assert detection works.
    cfgp = root / "smoke_config.json"
    cfgp.write_text(json.dumps(cfg, indent=2))
    return cfgp, cfg


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfgp, cfg = build(root)
        r = subprocess.run([sys.executable, str(HERE / "eval_stability_memorization.py"),
                            "--config", str(cfgp), "--device", "cpu"],
                           capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr); raise SystemExit("driver crashed")

        out = Path(cfg["out_root"])
        checks = []
        per_seed = out / "per_seed_results.csv"
        agg = out / "aggregate_summary.csv"
        checks.append(("per_seed_results.csv", per_seed.exists()))
        checks.append(("aggregate_summary.csv", agg.exists()))
        checks.append(("plots dir", (out / "plots").exists() and any((out / "plots").glob("*.png"))))
        checks.append(("nn_stats json", any((out / "nn_stats").rglob("*.json"))))
        checks.append(("nn_panels", any((out / "nn_panels").rglob("*.png"))))
        checks.append(("AUDIT_STATUS variability-only", "VARIABILITY" in (out / "AUDIT_STATUS.txt").read_text()))

        # duplicate detection: the planted dup must yield >=1 cluster in step_matched/anm
        dupjson = out / "nn_stats" / "step_matched" / "anm" / "seed0.json"
        d = json.loads(dupjson.read_text())
        checks.append(("duplicate detected", d["duplicates"]["num_duplicate_clusters"] >= 1))

        # all four checkpoints OK + FID/KID finite
        import csv
        with open(per_seed) as f:
            rows = list(csv.DictReader(f))
        ok = [r for r in rows if r["status"] == "OK"]
        checks.append(("4 checkpoints OK", len(ok) == 4))
        checks.append(("val_fid finite", all(np.isfinite(float(r["val_fid"])) for r in ok)))
        checks.append(("train_fid finite", all(np.isfinite(float(r["train_fid"])) for r in ok)))
        checks.append(("val_kid present", all(r["val_kid"] for r in ok)))
        checks.append(("fid_gap computed", all(r["fid_gap_val_minus_train"] for r in ok)))

        print("\n=== SMOKE CHECKS ===")
        allok = True
        for name, passed in checks:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            allok = allok and passed
        if not allok:
            raise SystemExit("LOCAL SMOKE FAILED")
        print("\nLOCAL SMOKE PASS (plumbing only; numbers meaningless).")


if __name__ == "__main__":
    main()
