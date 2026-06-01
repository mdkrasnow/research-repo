#!/usr/bin/env python3
"""Experiment 4 driver: seed stability, train-val gap, memorization audit.

Vanilla EqM vs ANM-EqM (v10). Two regimes (step_matched, compute_matched), two
references (val + train), full memorization/duplicate audit.

This script CONSUMES pre-generated samples (one .npz of uint8 NHWC per
checkpoint, key 'arr_0' -- the format eqm-upstream/sample_gd.py emits) and
precomputed reference banks (built by build_references.py). It does NOT train
or sample; generation is a separate cluster step (see slurm/jobs/exp4_*.sbatch).

Usage:
  python eval_stability_memorization.py --config configs/full_example.json
  python eval_stability_memorization.py --config configs/smoke.json   # tiny, CPU-ok

Config schema: see configs/full_example.json. Outputs: see --out-root.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as feat
import metrics as met
import nn_audit as nn
import plots as plotlib

# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_samples_npz(path: str) -> np.ndarray:
    """Return uint8 NHWC. Accepts sample_gd.py npz (key arr_0) or a png folder."""
    p = Path(path)
    if p.is_dir():
        from PIL import Image
        files = sorted(p.glob("*.png"))
        if not files:
            raise FileNotFoundError(f"no .png in sample folder {p}")
        return np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in files])
    data = np.load(p)
    key = "arr_0" if "arr_0" in data else list(data.keys())[0]
    arr = np.asarray(data[key])
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"samples {path}: expected uint8 NHWC RGB, got {arr.shape}")
    return arr


def load_bank(prefix: str):
    """Bank = <prefix>.npy (features Nx D). Optional <prefix>_labels.npy,
    <prefix>_images.npz (key arr_0, uint8 NHWC, for panels), <prefix>_ids.npy."""
    p = Path(prefix)
    feats = np.load(str(p) + ".npy") if not str(p).endswith(".npy") else np.load(str(p))
    base = str(p)[:-4] if str(p).endswith(".npy") else str(p)
    labels = np.load(base + "_labels.npy") if Path(base + "_labels.npy").exists() else None
    ids = np.load(base + "_ids.npy") if Path(base + "_ids.npy").exists() else None
    images = None
    if Path(base + "_images.npz").exists():
        images = np.load(base + "_images.npz")["arr_0"]
    meta = None
    if Path(base + "_meta.json").exists():
        meta = json.loads(Path(base + "_meta.json").read_text())
    return {"feats": feats, "labels": labels, "ids": ids, "images": images, "meta": meta}


# ----------------------------------------------------------------------------
# controls
# ----------------------------------------------------------------------------

def assert_no_leakage(train_bank, val_bank):
    if train_bank.get("ids") is not None and val_bank.get("ids") is not None:
        inter = np.intersect1d(train_bank["ids"], val_bank["ids"])
        if inter.size:
            raise AssertionError(f"train/val leakage: {inter.size} shared bank IDs")


def assert_eval_consistency(state: dict, schedule_hash: str, prep_hashes: dict):
    """Fail loud if any checkpoint used a different eval schedule or preprocessing."""
    if state.get("schedule_hash") is None:
        state["schedule_hash"] = schedule_hash
    elif state["schedule_hash"] != schedule_hash:
        raise AssertionError(
            f"inconsistent eval seeds/label schedule: {schedule_hash} != {state['schedule_hash']}"
        )
    if state.get("prep_hashes") is None:
        state["prep_hashes"] = prep_hashes
    elif state["prep_hashes"] != prep_hashes:
        raise AssertionError(
            f"feature-extractor preprocessing mismatch: {prep_hashes} != {state['prep_hashes']}"
        )


# ----------------------------------------------------------------------------
# per-checkpoint audit
# ----------------------------------------------------------------------------

def audit_checkpoint(entry, cfg, extractors, refs, banks, device, out_root, state):
    regime, ctype, seed = entry["regime"], entry["checkpoint_type"], entry["seed"]
    tag = f"{regime}/{ctype}/seed{seed}"
    row = {
        "regime": regime, "checkpoint_type": ctype, "seed": seed,
        "checkpoint_path": entry.get("checkpoint_path", ""),
        "outer_step": entry.get("outer_step"), "effective_feu": entry.get("effective_feu"),
        "sample_count": 0, "status": "OK",
        "val_fid": None, "val_kid": None, "train_fid": None, "train_kid": None,
        "fid_gap_val_minus_train": None, "kid_gap_val_minus_train": None,
        "mean_nn_train_dist": None, "mean_nn_val_dist": None, "nn_train_val_ratio": None,
        "nn_frac_ratio_lt_0_9": None, "duplicate_rate": None, "near_duplicate_rate": None,
        "n_seeds_in_arm": state["n_seeds"].get((regime, ctype), 0),
    }
    spath = entry.get("samples_npz")
    if not spath or not (Path(spath).exists() or Path(spath).is_dir()):
        row["status"] = "BLOCKED_NEED_SAMPLES" if entry.get("checkpoint_path") else "BLOCKED_NEED_CKPT"
        print(f"[{tag}] {row['status']}: samples missing ({spath}). "
              f"Generate via sample_gd.py for {entry.get('checkpoint_path')}.")
        return row

    imgs = load_samples_npz(spath)
    row["sample_count"] = int(imgs.shape[0])
    expected = cfg["eval"]["n_eval"]
    if not cfg.get("allow_count_mismatch") and row["sample_count"] != expected:
        raise AssertionError(f"[{tag}] sample count {row['sample_count']} != n_eval {expected}")
    print(f"[{tag}] {row['sample_count']} samples -> features", flush=True)

    # consistency: schedule + preprocessing
    prep_hashes = {e.name: e.preprocessing_hash() for e in extractors.values() if hasattr(e, "preprocessing_hash")}
    assert_eval_consistency(state, cfg["eval"].get("schedule_hash", "unset"), prep_hashes)

    # --- FID / KID (inception) vs both references ---
    inc = extractors["fid"]
    gfeat = inc.extract(imgs, batch_size=cfg.get("feature_batch_size", 64))
    row["val_fid"] = met.fid_against_reference(gfeat, *refs["val"])
    row["train_fid"] = met.fid_against_reference(gfeat, *refs["train"])
    ss = cfg.get("kid_subset_size", 1000)
    row["val_kid"] = met.kid(gfeat, refs["val_feats"], subset_size=ss)[0] if refs.get("val_feats") is not None else None
    row["train_kid"] = met.kid(gfeat, refs["train_feats"], subset_size=ss)[0] if refs.get("train_feats") is not None else None
    if row["val_fid"] is not None and row["train_fid"] is not None:
        row["fid_gap_val_minus_train"] = row["val_fid"] - row["train_fid"]
    if row["val_kid"] is not None and row["train_kid"] is not None:
        row["kid_gap_val_minus_train"] = row["val_kid"] - row["train_kid"]

    # --- NN memorization (dino, cosine) ---
    nnx = extractors["nn"]
    gnn = nnx.extract(imgs, batch_size=cfg.get("feature_batch_size", 64))
    glabels = None
    lp = entry.get("labels_npy")
    if lp and Path(lp).exists():
        glabels = np.load(lp)
    sc = bool(cfg.get("class_conditional_nn", False)) and glabels is not None
    d_tr, i_tr = nn.cosine_topk(gnn, banks["train"]["feats"], k=cfg.get("top_k_nn", 3),
                                query_labels=glabels, bank_labels=banks["train"]["labels"],
                                same_class=sc, device=device)
    d_va, i_va = nn.cosine_topk(gnn, banks["val"]["feats"], k=cfg.get("top_k_nn", 3),
                                query_labels=glabels, bank_labels=banks["val"]["labels"],
                                same_class=sc, device=device)
    tau_mem = cfg.get("tau_mem", float(np.quantile(d_tr[:, 0][np.isfinite(d_tr[:, 0])], 0.01)))
    mem = nn.memorization_stats(d_tr[:, 0], d_va[:, 0], tau_mem=tau_mem)
    row["mean_nn_train_dist"] = mem["mean_d_train"]
    row["mean_nn_val_dist"] = mem["mean_d_val"]
    row["nn_train_val_ratio"] = mem["mean_ratio"]
    row["nn_frac_ratio_lt_0_9"] = mem["frac_ratio_lt_0_9"]

    # --- duplicates (dino self-NN) ---
    tau_dup = state.get("tau_dup")
    if tau_dup is None:
        tau_dup = nn.calibrate_tau_dup(banks["val"]["feats"], banks["train"]["feats"], device=device)
        state["tau_dup"] = tau_dup
    dup = nn.duplicate_stats(gnn, tau_dup=tau_dup, device=device)
    row["duplicate_rate"] = dup["near_duplicate_rate"]  # exact-dup tracked separately if hashes added
    row["near_duplicate_rate"] = dup["near_duplicate_rate"]

    # --- artifacts: suspicious indices + panels ---
    susp = nn.rank_suspicious(d_tr[:, 0], d_va[:, 0], top_n=cfg.get("panel_top_n", 32))
    ck_out = out_root / "nn_stats" / regime / ctype
    ck_out.mkdir(parents=True, exist_ok=True)
    (ck_out / f"seed{seed}.json").write_text(json.dumps({
        "tag": tag, "memorization": mem, "duplicates": {k: v for k, v in dup.items() if not k.startswith("_")},
        "tau_mem": tau_mem, "tau_dup": tau_dup, "suspicious_indices": susp,
        "distance_metric": "cosine(1 - <a,b>) on L2-normalized features",
        "feature_extractor_nn": nnx.name, "nn_preprocessing_hash": nnx.preprocessing_hash(),
        "class_conditional_nn": sc,
        "interpretation": "QUANTITATIVE MEMORIZATION PROXY ONLY. Low train-NN distance / "
                          "ratio<1 is suspicious similarity, NOT proof of memorization. "
                          "Confirm with nn_panels/ qualitative inspection; declare memorization "
                          "only on exact/near-exact duplicates with visual confirmation.",
    }, indent=2))
    state.setdefault("ratio_groups", {})[f"{regime}/{ctype}"] = (d_tr[:, 0] / (d_va[:, 0] + 1e-8)).tolist()

    if banks["train"].get("images") is not None and banks["val"].get("images") is not None:
        _render_panels(imgs, susp, i_tr, i_va, d_tr, d_va, banks, out_root, regime, ctype, seed)
    else:
        print(f"[{tag}] panels skipped (no bank images provided)")
    return row


def _render_panels(gen_imgs, susp, i_tr, i_va, d_tr, d_va, banks, out_root, regime, ctype, seed):
    pan_dir = out_root / "nn_panels" / regime / ctype
    for name, idxs in susp.items():
        idxs = idxs[:8]
        if not idxs:
            continue
        g = gen_imgs[idxs]
        tnn = banks["train"]["images"][i_tr[idxs]]  # (R,k,H,W,3)
        vnn = banks["val"]["images"][i_va[idxs]]
        plotlib.render_nn_panel(g, tnn, vnn, d_tr[idxs], d_va[idxs],
                                pan_dir / f"seed{seed}_{name}.png")


# ----------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------

def aggregate(rows):
    agg = []
    groups = sorted({(r["regime"], r["checkpoint_type"]) for r in rows})
    metrics = ["val_fid", "val_kid", "fid_gap_val_minus_train", "kid_gap_val_minus_train",
               "nn_frac_ratio_lt_0_9", "near_duplicate_rate"]
    for regime, ctype in groups:
        sub = [r for r in rows if r["regime"] == regime and r["checkpoint_type"] == ctype and r["status"] == "OK"]
        n = len(sub)
        arow = {"regime": regime, "checkpoint_type": ctype, "n_seeds": n,
                "seed_audit_valid": n >= 3}
        for m in metrics:
            vals = np.array([r[m] for r in sub if r.get(m) is not None and np.isfinite(r[m])], dtype=float)
            if vals.size:
                lo, hi = met.bootstrap_ci(vals)
                arow[f"mean_{m}"] = float(vals.mean())
                arow[f"std_{m}"] = float(vals.std()) if vals.size > 1 else 0.0
                arow[f"sem_{m}"] = float(vals.std() / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                arow[f"median_{m}"] = float(np.median(vals))
                arow[f"ci95_lo_{m}"] = lo
                arow[f"ci95_hi_{m}"] = hi
            else:
                for suff in ("mean", "std", "sem", "median", "ci95_lo", "ci95_hi"):
                    arow[f"{suff}_{m}"] = None
        agg.append(arow)
    # paired deltas (anm - vanilla) per regime
    for regime in sorted({r["regime"] for r in rows}):
        seeds = sorted({r["seed"] for r in rows if r["regime"] == regime})
        drow = {"regime": regime, "checkpoint_type": "paired_delta_anm_minus_vanilla"}
        for m in metrics:
            deltas = []
            for s in seeds:
                v = next((r[m] for r in rows if r["regime"] == regime and r["checkpoint_type"] == "vanilla" and r["seed"] == s and r["status"] == "OK"), None)
                a = next((r[m] for r in rows if r["regime"] == regime and r["checkpoint_type"] == "anm" and r["seed"] == s and r["status"] == "OK"), None)
                if v is not None and a is not None and np.isfinite(v) and np.isfinite(a):
                    deltas.append(a - v)
            drow[f"mean_{m}"] = float(np.mean(deltas)) if deltas else None
            drow[f"n_pairs_{m}"] = len(deltas)
        agg.append(drow)
    return agg


def write_csv(rows, path, fieldnames=None):
    if not rows:
        Path(path).write_text("")
        return
    fieldnames = fieldnames or sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def build_checkpoint_entries(cfg):
    entries = []
    sources = cfg.get("samples", {})  # regime->ctype->seed-> {samples_npz, checkpoint_path, ...}
    ckpts = cfg.get("checkpoints", {})
    regimes = sorted(set(list(sources) + list(ckpts)))
    for regime in regimes:
        for ctype in ["vanilla", "anm"]:
            seeds = sorted(set(list(sources.get(regime, {}).get(ctype, {}))
                               + list(ckpts.get(regime, {}).get(ctype, {}))))
            for seed in seeds:
                e = {"regime": regime, "checkpoint_type": ctype, "seed": str(seed)}
                e.update(sources.get(regime, {}).get(ctype, {}).get(str(seed), {})
                         if isinstance(sources.get(regime, {}).get(ctype, {}).get(str(seed), {}), dict) else {})
                cp = ckpts.get(regime, {}).get(ctype, {}).get(str(seed))
                if isinstance(cp, str):
                    e.setdefault("checkpoint_path", cp)
                elif isinstance(cp, dict):
                    e.update(cp)
                entries.append(e)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite an existing audit in --out-root (refused by default to "
                         "avoid silently mixing results from a different config/extractor)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    device = args.device or ("cuda" if _cuda() else "cpu")
    out_root = Path(args.out_root or cfg.get("out_root", "projects/diff-EqM/results/experiment4"))
    if (out_root / "per_seed_results.csv").exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing audit at {out_root} (pass --overwrite). "
            "Prevents silently mixing outputs from a different config / feature extractor."
        )
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # extractors
    extractors = {
        "fid": feat.build_extractor(cfg.get("feature_backbone_fid", "inception_pool3"), device=device),
        "nn": feat.build_extractor(cfg.get("feature_backbone_nn", "dinov2_vitl14"), device=device),
    }

    # references (inception stats) + optional raw inception feats for KID
    refs = {
        "val": met.load_reference_stats(cfg["references"]["val_stats_npz"]),
        "train": met.load_reference_stats(cfg["references"]["train_stats_npz"]),
    }
    refs["val_feats"] = np.load(cfg["references"]["val_inception_feats"]) if cfg["references"].get("val_inception_feats") else None
    refs["train_feats"] = np.load(cfg["references"]["train_inception_feats"]) if cfg["references"].get("train_inception_feats") else None

    # NN banks (dino feats + optional labels/ids/images)
    banks = {"train": load_bank(cfg["references"]["train_nn_bank"]),
             "val": load_bank(cfg["references"]["val_nn_bank"])}
    assert_no_leakage(banks["train"], banks["val"])

    # Gap B fix: NN bank must have been embedded with the SAME backbone the audit
    # uses for generated samples, else gen/bank features are not comparable.
    nn_hash = extractors["nn"].preprocessing_hash()
    for split in ("train", "val"):
        meta = banks[split].get("meta")
        if meta is None:
            print(f"WARNING: {split} NN bank has no _meta.json -> backbone match UNKNOWN "
                  f"(cannot verify it was built with '{extractors['nn'].name}').")
        elif meta.get("nn_preprocessing_hash") != nn_hash:
            raise AssertionError(
                f"{split} NN bank backbone mismatch: bank={meta.get('nn_preprocessing_hash')} "
                f"!= audit={nn_hash}. Rebuild bank with build_references.py --nn-backbone "
                f"{cfg.get('feature_backbone_nn')}.")

    entries = build_checkpoint_entries(cfg)
    # count seeds per arm for the n_seeds / seed_audit_valid caveat
    state = {"schedule_hash": None, "prep_hashes": None, "tau_dup": None, "n_seeds": {}}
    for e in entries:
        key = (e["regime"], e["checkpoint_type"])
        state["n_seeds"][key] = state["n_seeds"].get(key, 0) + 1

    # Gap A fix: print the checkpoint grouping table BEFORE running. Missing
    # metadata shows UNKNOWN (never invented). Also makes regime/seed mis-grouping
    # visible at a glance.
    print("\n=== Experiment 4 checkpoint grouping ===")
    print(f"{'regime':<16}{'type':<9}{'seed':<6}{'outer_step':<12}{'eff_FEU':<14}samples/ckpt")
    for e in sorted(entries, key=lambda r: (r["regime"], r["checkpoint_type"], str(r["seed"]))):
        step = e.get("outer_step", "UNKNOWN")
        feu = e.get("effective_feu", "UNKNOWN")
        src = e.get("samples_npz") or e.get("checkpoint_path") or "UNKNOWN"
        print(f"{e['regime']:<16}{e['checkpoint_type']:<9}{str(e['seed']):<6}"
              f"{str(step):<12}{str(feu):<14}{src}")
    for (regime, ctype), n in sorted(state["n_seeds"].items()):
        print(f"  arm {regime}/{ctype}: {n} seed(s)" + ("  <-- <3, variability-only" if n < 3 else ""))
    # unequal-seed warning per regime (failure indicator #1)
    for regime in sorted({k[0] for k in state["n_seeds"]}):
        nv = state["n_seeds"].get((regime, "vanilla"), 0)
        na = state["n_seeds"].get((regime, "anm"), 0)
        if nv != na:
            print(f"  WARNING [{regime}]: unequal seeds vanilla={nv} anm={na}; "
                  "paired deltas use intersecting seeds only.")
    print("=== end grouping ===\n", flush=True)

    # Gap C fix: run manifest embeds extractors + reference paths + eval config.
    (out_root / "run_manifest.json").write_text(json.dumps({
        "feature_backbone_fid": extractors["fid"].name,
        "feature_backbone_nn": extractors["nn"].name,
        "fid_preprocessing_hash": extractors["fid"].preprocessing_hash(),
        "nn_preprocessing_hash": nn_hash,
        "references": cfg["references"],
        "eval": cfg["eval"],
        "sampler": cfg.get("sampler"),
        "class_conditional_nn": bool(cfg.get("class_conditional_nn", False)),
        "device": device,
        "n_seeds_per_arm": {f"{k[0]}/{k[1]}": v for k, v in state["n_seeds"].items()},
    }, indent=2))

    rows = []
    for e in entries:
        try:
            rows.append(audit_checkpoint(e, cfg, extractors, refs, banks, device, out_root, state))
        except Exception as ex:  # noqa: BLE001 - record + continue so one bad ckpt doesn't kill the run
            print(f"[{e['regime']}/{e['checkpoint_type']}/seed{e['seed']}] ERROR: {ex}")
            rows.append({"regime": e["regime"], "checkpoint_type": e["checkpoint_type"],
                         "seed": str(e["seed"]), "status": f"ERROR: {ex}"})

    per_seed_cols = ["regime", "checkpoint_type", "seed", "checkpoint_path", "sample_count",
                     "outer_step", "effective_feu", "val_fid", "val_kid", "train_fid", "train_kid",
                     "fid_gap_val_minus_train", "kid_gap_val_minus_train", "mean_nn_train_dist",
                     "mean_nn_val_dist", "nn_train_val_ratio", "nn_frac_ratio_lt_0_9",
                     "duplicate_rate", "near_duplicate_rate", "status", "n_seeds_in_arm"]
    write_csv(rows, out_root / "per_seed_results.csv", per_seed_cols)
    agg = aggregate(rows)
    write_csv(agg, out_root / "aggregate_summary.csv")

    # plots
    p = out_root / "plots"
    ok = [r for r in rows if r.get("status") == "OK"]
    if ok:
        plotlib.seed_error_bars(ok, "val_fid", "Val FID", p / "seed_error_bars_fid.png")
        plotlib.seed_error_bars(ok, "val_kid", "Val KID", p / "seed_error_bars_kid.png")
        plotlib.train_val_gap(ok, "fid_gap_val_minus_train", p / "train_val_gap_fid.png", "FID gap (val-train)")
        plotlib.train_val_gap(ok, "kid_gap_val_minus_train", p / "train_val_gap_kid.png", "KID gap (val-train)")
        plotlib.nn_ratio_histogram(state.get("ratio_groups", {}), p / "nn_ratio_distribution.png")
        plotlib.duplicate_rate_by_seed(ok, p / "duplicate_rate_by_seed.png")
        plotlib.step_vs_compute_summary(ok, p / "step_vs_compute_summary.png")

    # caveat banner
    valid = all(a.get("seed_audit_valid") for a in agg if a["checkpoint_type"] in ("vanilla", "anm"))
    banner = ("SEED-STABILITY AUDIT VALID (>=3 seeds per arm)" if valid else
              "*** CHECKPOINT VARIABILITY AUDIT ONLY: <3 seeds per arm. "
              "Do NOT claim seed stability (B2 caveat). ***")
    print(f"\n{banner}\nWrote {out_root} in {time.time()-t0:.1f}s. "
          f"{sum(r.get('status')=='OK' for r in rows)}/{len(rows)} checkpoints OK.")
    (out_root / "AUDIT_STATUS.txt").write_text(banner + "\n")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
