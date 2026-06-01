"""Experiment 3 orchestrator: aggregate + class-wise fidelity/diversity metrics.

Assumes generation already produced PNG folders (one per arm) via
sample_scheduled.py, and the fixed reference cache exists (build_reference.py).
Loads/caches Inception features per arm, computes every metric on the SAME
feature space, writes CSV/JSON, deltas, plots, README, and prints the
success/failure verdict.

FID alone is NOT the verdict -- the printed conclusion states the
diversity/coverage outcome.

Usage (metrics phase, single GPU):
  python eval_fidelity_diversity.py \
    --gen-root results/exp3/full_lambda03_vs_vanilla/gen \
    --schedule results/exp3/full_lambda03_vs_vanilla/schedule.json \
    --reference-dir results/exp3/reference \
    --out results/exp3/full_lambda03_vs_vanilla \
    --vanilla-ckpt <p> --anm-ckpt <p> \
    --sampler gd --nfe 250 --step-size 0.003 --cfg-scale 1.0
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as feat_mod        # noqa: E402
import metrics as M                # noqa: E402
from schedule import load_schedule, schedule_hash  # noqa: E402

EXTRACTOR = "pytorch_fid_inception_v3_pool3_2048"


def load_reference(ref_dir, num_classes):
    ref_dir = Path(ref_dir)
    feats = np.load(ref_dir / "inception_feats.npy")
    ms = np.load(ref_dir / "inception_mu_sigma.npz")
    real_hist = np.load(ref_dir / "real_classifier_hist.npy")
    bc = np.load(ref_dir / "inception_by_class.npz")
    mu_by = {int(k.split("_")[1]): bc[k] for k in bc.files if k.startswith("mu_")}
    cov_by = {int(k.split("_")[1]): bc[k] for k in bc.files if k.startswith("cov_")}
    cnt_by = {int(k.split("_")[1]): int(bc[k]) for k in bc.files if k.startswith("cnt_")}
    fbc_path = ref_dir / "inception_feats_by_class.npz"
    feats_by = None
    if fbc_path.exists():
        fbc = np.load(fbc_path)
        feats_by = {int(k): fbc[k].astype(np.float64) for k in fbc.files}
    return {"feats": feats, "mu": ms["mu"], "sigma": ms["sigma"],
            "real_hist": real_hist, "mu_by": mu_by, "cov_by": cov_by,
            "cnt_by": cnt_by, "feats_by": feats_by}


def cached_features(gen_dir, cache_npy, device, batch_size):
    cache_npy = Path(cache_npy)
    files = feat_mod._list_images(gen_dir)
    stems = [int(Path(f).stem) for f in files]
    if cache_npy.exists():
        feats = np.load(cache_npy)
        if len(feats) == len(files):
            return feats, np.asarray(stems)
    feats, _ = feat_mod.inception_features(gen_dir, device=device, batch_size=batch_size,
                                           files=files)
    np.save(cache_npy, feats)
    return feats, np.asarray(stems)


def cached_preds(gen_dir, cache_csv, device, batch_size):
    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        rows = list(csv.DictReader(open(cache_csv)))
        top1 = np.array([int(r["top1"]) for r in rows])
        top5 = np.array([[int(x) for x in r["top5"].split("|")] for r in rows])
        stems = np.array([int(r["sample_id"]) for r in rows])
        return {"top1": top1, "top5": top5, "stems": stems}
    files = feat_mod._list_images(gen_dir)
    p = feat_mod.classifier_predictions(gen_dir, device=device, batch_size=batch_size,
                                        files=files)
    stems = np.array([int(s) for s in p["stems"]])
    with open(cache_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "top1", "top5", "prob_top1"])
        for i in range(len(p["top1"])):
            w.writerow([int(stems[i]), int(p["top1"][i]),
                        "|".join(str(int(x)) for x in p["top5"][i]),
                        float(p["prob_top1"][i])])
    return {"top1": p["top1"], "top5": p["top5"], "stems": stems}


def sanity_checks(arm, feats, stems, schedule, preds):
    N = schedule["num_samples"]
    assert len(feats) > 0, f"[{arm}] no features"
    assert np.isfinite(feats).all(), f"[{arm}] non-finite features"
    if len(feats) != N:
        print(f"  [WARN {arm}] feature count {len(feats)} != schedule N {N}")
    # per-class count check
    labels = schedule["labels"]
    got = np.bincount(labels[np.clip(stems, 0, N - 1)],
                      minlength=schedule["num_classes"])
    thin = int((got < schedule["samples_per_class"]).sum())
    if thin:
        print(f"  [WARN {arm}] {thin} classes have < samples_per_class generated")
    assert len(np.unique(stems)) == len(stems), f"[{arm}] duplicate sample ids"


def run_arm(arm, gen_dir, ckpt, schedule, ref, device, batch_size, out, prdc_k):
    feats, fstems = cached_features(gen_dir, Path(out) / f"features_{arm}.npy",
                                    device, batch_size)
    preds = cached_preds(gen_dir, Path(out) / f"classifier_{arm}.csv", device, batch_size)
    sanity_checks(arm, feats, fstems, schedule, preds)

    labels_full = schedule["labels"]
    gen_labels = labels_full[fstems]
    # align classifier preds to feature order by sample id
    order = {int(s): i for i, s in enumerate(preds["stems"])}
    pi = np.array([order[int(s)] for s in fstems])
    top1 = preds["top1"][pi]
    top5 = preds["top5"][pi]

    agg = M.aggregate_metrics(feats, ref["feats"], prdc_k=prdc_k)
    agg["fid_ci"] = M.bootstrap_fid(feats, ref["mu"], ref["sigma"], n_boot=100)
    div_ci = M.bootstrap_diversity(feats, gen_labels, ref["feats"], prdc_k=prdc_k,
                                   n_boot=200, subset_per_class=None)
    is_mean, is_std = (float("nan"), float("nan"))
    try:
        is_mean, is_std = feat_mod.inception_score(image_dir=gen_dir, device=device,
                                                   batch_size=batch_size)
    except Exception as e:
        print(f"  [warn {arm}] IS skipped: {e}")

    clf = M.classifier_hist_metrics(top1, top5, gen_labels,
                                    schedule["num_classes"], ref["real_hist"])
    cls_rows = M.per_class_metrics(feats, gen_labels, ref["mu_by"], ref["cov_by"],
                                   ref["cnt_by"], ref_feats_by_class=ref["feats_by"])
    # per-class conditional top1
    cond_by = {}
    for c in np.unique(gen_labels):
        m = gen_labels == c
        cond_by[int(c)] = float((top1[m] == c).mean())
    for r in cls_rows:
        r["conditional_top1_accuracy"] = cond_by.get(r["class_id"], 0.0)

    return {"feats": feats, "gen_labels": gen_labels, "agg": agg, "div_ci": div_ci,
            "is": (is_mean, is_std), "clf": clf, "cls_rows": cls_rows,
            "cond_by": cond_by, "ckpt": ckpt, "n": len(feats)}


def write_aggregate_csv(path, arm_results, schedule, cfg):
    cols = ["checkpoint_type", "checkpoint_path", "sample_count", "num_classes",
            "samples_per_class", "reference_split", "feature_extractor", "sampler",
            "nfe", "step_size", "mu", "guidance_scale", "fid", "fid_ci_low",
            "fid_ci_high", "kid_mean", "kid_std", "precision", "recall", "density",
            "coverage", "prdc_k", "inception_score", "sfid", "classifier_entropy",
            "classifier_kl_to_requested", "classifier_tv_to_requested",
            "classifier_missing_classes", "conditional_top1_accuracy",
            "conditional_top5_accuracy", "recall_se", "coverage_se"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, R in arm_results.items():
            a, clf = R["agg"], R["clf"]
            w.writerow({
                "checkpoint_type": arm, "checkpoint_path": R["ckpt"],
                "sample_count": R["n"], "num_classes": schedule["num_classes"],
                "samples_per_class": schedule["samples_per_class"],
                "reference_split": cfg["reference_split"], "feature_extractor": EXTRACTOR,
                "sampler": cfg["sampler"], "nfe": cfg["nfe"], "step_size": cfg["step_size"],
                "mu": cfg["mu"], "guidance_scale": cfg["cfg_scale"],
                "fid": round(a["fid"], 4), "fid_ci_low": round(a["fid_ci"]["ci_low"], 4),
                "fid_ci_high": round(a["fid_ci"]["ci_high"], 4),
                "kid_mean": a["kid_mean"], "kid_std": a["kid_std"],
                "precision": round(a["precision"], 4), "recall": round(a["recall"], 4),
                "density": round(a["density"], 4), "coverage": round(a["coverage"], 4),
                "prdc_k": a["prdc_k"], "inception_score": round(R["is"][0], 4),
                "sfid": "NA",
                "classifier_entropy": round(clf["classifier_entropy"], 4),
                "classifier_kl_to_requested": round(clf["classifier_kl_to_requested"], 4),
                "classifier_tv_to_requested": round(clf["classifier_tv_to_requested"], 4),
                "classifier_missing_classes": clf["classifier_missing_classes"],
                "conditional_top1_accuracy": round(clf["conditional_top1_accuracy"], 4),
                "conditional_top5_accuracy": round(clf["conditional_top5_accuracy"], 4),
                "recall_se": round(R["div_ci"]["recall"]["se"], 5),
                "coverage_se": round(R["div_ci"]["coverage"]["se"], 5),
            })


def write_class_csv(path, arm_results, quartile_map, schedule):
    cols = ["checkpoint_type", "class_id", "class_name", "num_generated", "n_ref",
            "fid_class", "fid_class_noisy_flag", "feature_distance_class",
            "feature_distance_class_normalized", "precision_class", "recall_class",
            "conditional_top1_accuracy", "weak_class_score", "class_quartile"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, R in arm_results.items():
            for r in R["cls_rows"]:
                c = r["class_id"]
                qm = quartile_map.get(c, {})
                w.writerow({
                    "checkpoint_type": arm, "class_id": c, "class_name": c,
                    "num_generated": r["n_gen"], "n_ref": r["n_ref"],
                    "fid_class": ("" if np.isnan(r["fid_class"]) else round(r["fid_class"], 3)),
                    "fid_class_noisy_flag": r["fid_class_noisy_flag"],
                    "feature_distance_class": round(r["feature_distance_class"], 4),
                    "feature_distance_class_normalized":
                        ("" if np.isnan(r["feature_distance_class_normalized"])
                         else round(r["feature_distance_class_normalized"], 4)),
                    "precision_class": "", "recall_class": "",
                    "conditional_top1_accuracy": round(r["conditional_top1_accuracy"], 4),
                    "weak_class_score": round(qm.get("weak_class_score", float("nan")), 4),
                    "class_quartile": qm.get("class_quartile", ""),
                })


def write_delta_class_csv(path, vanilla, anm, quartile_map):
    vmap = {r["class_id"]: r for r in vanilla["cls_rows"]}
    amap = {r["class_id"]: r for r in anm["cls_rows"]}
    cols = ["class_id", "class_name", "class_quartile", "delta_per_class_fid",
            "delta_feature_mean_distance", "delta_conditional_top1_accuracy",
            "anm_better_feature_distance", "anm_better_conditional_top1"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in sorted(vmap):
            v, a = vmap[c], amap[c]
            dfid = ((a["fid_class"] - v["fid_class"])
                    if not (np.isnan(a["fid_class"]) or np.isnan(v["fid_class"])) else float("nan"))
            dfd = a["feature_distance_class"] - v["feature_distance_class"]
            dacc = a["conditional_top1_accuracy"] - v["conditional_top1_accuracy"]
            w.writerow({
                "class_id": c, "class_name": c,
                "class_quartile": quartile_map.get(c, {}).get("class_quartile", ""),
                "delta_per_class_fid": ("" if np.isnan(dfid) else round(dfid, 3)),
                "delta_feature_mean_distance": round(dfd, 4),
                "delta_conditional_top1_accuracy": round(dacc, 4),
                "anm_better_feature_distance": bool(dfd < 0),
                "anm_better_conditional_top1": bool(dacc > 0),
            })
    # fraction of classes where ANM has smaller feature distance
    deltas = [amap[c]["feature_distance_class"] - vmap[c]["feature_distance_class"]
              for c in vmap]
    return float(np.mean(np.array(deltas) < 0))


def write_classifier_hist_csv(path, arm_results, real_hist, schedule):
    cols = ["checkpoint_type", "predicted_class_id", "count", "probability",
            "requested_probability", "real_reference_probability",
            "delta_vs_requested", "delta_vs_real_reference"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, R in arm_results.items():
            ph = R["clf"]["pred_hist"]
            rq = R["clf"]["requested_hist"]
            for cid in range(schedule["num_classes"]):
                w.writerow({
                    "checkpoint_type": arm, "predicted_class_id": cid,
                    "count": int(round(ph[cid] * R["n"])),
                    "probability": round(float(ph[cid]), 6),
                    "requested_probability": round(float(rq[cid]), 6),
                    "real_reference_probability": round(float(real_hist[cid]), 6),
                    "delta_vs_requested": round(float(ph[cid] - rq[cid]), 6),
                    "delta_vs_real_reference": round(float(ph[cid] - real_hist[cid]), 6),
                })


def merge_manifest(gen_dirs, out_path):
    rows = []
    for arm, d in gen_dirs.items():
        for part in Path(d).glob("manifest_part_*.csv"):
            for r in csv.DictReader(open(part)):
                r["checkpoint_type"] = arm
                rows.append(r)
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint_type", "sample_id", "seed",
                                          "requested_label"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in
                        ["checkpoint_type", "sample_id", "seed", "requested_label"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True, help="dir containing vanilla/ and anm/")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vanilla-ckpt", default="")
    ap.add_argument("--anm-ckpt", default="")
    ap.add_argument("--sampler", default="gd")
    ap.add_argument("--nfe", type=int, default=250)
    ap.add_argument("--step-size", type=float, default=0.003)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--prdc-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--reference-split", default="imagenet_train_fixed_seed0")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    schedule = load_schedule(args.schedule)
    print(f"[exp3] schedule hash={schedule_hash(schedule)} N={schedule['num_samples']}")
    ref = load_reference(args.reference_dir, schedule["num_classes"])

    cfg = {"sampler": args.sampler, "nfe": args.nfe, "step_size": args.step_size,
           "mu": "", "cfg_scale": args.cfg_scale, "reference_split": args.reference_split}

    gen_dirs = {"vanilla": str(Path(args.gen_root) / "vanilla"),
                "anm": str(Path(args.gen_root) / "anm")}
    ckpts = {"vanilla": args.vanilla_ckpt, "anm": args.anm_ckpt}

    arm_results = {}
    for arm in ("vanilla", "anm"):
        print(f"[exp3] === arm: {arm} ===")
        arm_results[arm] = run_arm(arm, gen_dirs[arm], ckpts[arm], schedule, ref,
                                   device, args.batch_size, out, args.prdc_k)

    # weak-class quartiles from VANILLA only
    quartile_map = M.weak_class_scores(arm_results["vanilla"]["cls_rows"],
                                       arm_results["vanilla"]["cond_by"])

    # pooled bottom-quartile per arm
    bq = {}
    for arm in ("vanilla", "anm"):
        bq[arm] = M.pooled_bottom_quartile(arm_results[arm]["feats"],
                                           arm_results[arm]["gen_labels"],
                                           ref["feats"], quartile_map,
                                           which="bottom", prdc_k=args.prdc_k)

    # write tables
    write_aggregate_csv(out / "aggregate_metrics.csv", arm_results, schedule, cfg)
    write_class_csv(out / "class_metrics.csv", arm_results, quartile_map, schedule)
    frac_improved = write_delta_class_csv(out / "delta_class_metrics.csv",
                                          arm_results["vanilla"], arm_results["anm"],
                                          quartile_map)
    write_classifier_hist_csv(out / "classifier_histogram.csv", arm_results,
                              ref["real_hist"], schedule)
    merge_manifest(gen_dirs, out / "samples_manifest.csv")

    # verdict
    verdict, reasons = M.decide(
        arm_results["vanilla"]["agg"], arm_results["anm"]["agg"],
        arm_results["vanilla"]["clf"], arm_results["anm"]["clf"],
        recall_se=arm_results["anm"]["div_ci"]["recall"]["se"],
        coverage_se=arm_results["anm"]["div_ci"]["coverage"]["se"],
        bq_vanilla=bq["vanilla"], bq_anm=bq["anm"],
        frac_classes_improved=frac_improved)

    summary = {
        "verdict": verdict, "reasons": reasons,
        "frac_classes_anm_better_feature_distance": frac_improved,
        "vanilla": {k: arm_results["vanilla"]["agg"][k]
                    for k in ["fid", "kid_mean", "precision", "recall", "density", "coverage"]},
        "anm": {k: arm_results["anm"]["agg"][k]
                for k in ["fid", "kid_mean", "precision", "recall", "density", "coverage"]},
        "bottom_quartile_fid": {"vanilla": (bq["vanilla"] or {}).get("fid"),
                                "anm": (bq["anm"] or {}).get("fid")},
        "classifier": {
            "vanilla_tv": arm_results["vanilla"]["clf"]["classifier_tv_to_requested"],
            "anm_tv": arm_results["anm"]["clf"]["classifier_tv_to_requested"],
            "vanilla_cond_top1": arm_results["vanilla"]["clf"]["conditional_top1_accuracy"],
            "anm_cond_top1": arm_results["anm"]["clf"]["conditional_top1_accuracy"],
        },
    }
    (out / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2))

    if not args.no_plots:
        try:
            import plots
            plots.make_all(out)
        except Exception as e:
            print(f"[warn] plotting failed: {e}")

    write_readme(out, summary)
    print_decision(summary)


def write_readme(out, summary):
    (Path(out) / "README.md").write_text(f"""# Experiment 3 — Fidelity-Diversity & Mode Coverage

Vanilla EqM vs ANM EqM (IN-1K-256, EqM-B/2, 80ep). Feature extractor:
`{EXTRACTOR}` (same as the trusted FID run). PRDC vendored (Naeem et al. 2020).

## Verdict: **{summary['verdict'].upper()}**

| metric | vanilla | anm | better? |
|---|---|---|---|
| FID ↓ | {summary['vanilla']['fid']:.3f} | {summary['anm']['fid']:.3f} | {summary['reasons']['fid_better']} |
| KID ↓ | {summary['vanilla']['kid_mean']:.5f} | {summary['anm']['kid_mean']:.5f} | {summary['reasons']['kid_better']} |
| precision ↑ | {summary['vanilla']['precision']:.3f} | {summary['anm']['precision']:.3f} | |
| recall ↑ | {summary['vanilla']['recall']:.3f} | {summary['anm']['recall']:.3f} | {summary['reasons']['recall_ok']} |
| density ↑ | {summary['vanilla']['density']:.3f} | {summary['anm']['density']:.3f} | |
| coverage ↑ | {summary['vanilla']['coverage']:.3f} | {summary['anm']['coverage']:.3f} | {summary['reasons']['coverage_ok']} |

frac classes ANM improves (feature distance): {summary['frac_classes_anm_better_feature_distance']:.3f}

## Metric reliability
- FID/KID/PRDC require large N. At 50K samples these are reliable; the smoke run
  (1K samples) numbers are PLUMBING ONLY — do not interpret them.
- Per-class FID at 50 gen/class is NOISY (flagged in class_metrics.csv). Lead with
  `feature_distance_class` and the pooled bottom-quartile FID instead.
- Classifier histogram is interpreted against the real-reference classifier
  histogram (resnet50 has its own bias).
- No sample filtering / rejection sampling is applied.

## Files
aggregate_metrics.csv/json, class_metrics.csv, delta_class_metrics.csv,
classifier_histogram.csv, samples_manifest.csv, features_*.npy, plots/*.png

## Success logic
success: FID↓ AND KID↓ AND recall ≥ vanilla−max(0.005,SE) AND coverage ≥ vanilla−max(0.005,SE)
AND classifier TV not worse by >0.02 AND conditional top1 not down >0.01.
strong_success: success + ≥2 of {{recall↑, coverage↑, bottom-quartile FID↓, ≥55% classes improve}}.
failure: FID↓ but a diversity/coverage/class guard fails.
ambiguous: FID and KID disagree, or partial.
""")


def print_decision(s):
    print("\n" + "=" * 60)
    print(f"EXP3 VERDICT: {s['verdict'].upper()}")
    print(f"  FID:      vanilla {s['vanilla']['fid']:.3f} -> anm {s['anm']['fid']:.3f}  (better={s['reasons']['fid_better']})")
    print(f"  KID:      vanilla {s['vanilla']['kid_mean']:.5f} -> anm {s['anm']['kid_mean']:.5f}  (better={s['reasons']['kid_better']})")
    print(f"  recall:   vanilla {s['vanilla']['recall']:.3f} -> anm {s['anm']['recall']:.3f}  (ok={s['reasons']['recall_ok']}, tol={s['reasons']['recall_tol']:.4f})")
    print(f"  coverage: vanilla {s['vanilla']['coverage']:.3f} -> anm {s['anm']['coverage']:.3f}  (ok={s['reasons']['coverage_ok']}, tol={s['reasons']['coverage_tol']:.4f})")
    print(f"  clf TV:   ok={s['reasons']['tv_ok']}   cond_top1 ok={s['reasons']['cond_top1_ok']}")
    print(f"  bottom-quartile FID: vanilla {s['bottom_quartile_fid']['vanilla']} -> anm {s['bottom_quartile_fid']['anm']}")
    print(f"  frac classes ANM better (feat dist): {s['frac_classes_anm_better_feature_distance']:.3f}")
    print("=" * 60)
    print("NOTE: FID alone is NOT the conclusion. Verdict above weighs diversity")
    print("      (recall/coverage) and mode coverage (class histogram, bottom-quartile).")


if __name__ == "__main__":
    main()
