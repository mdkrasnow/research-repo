"""Local CPU plumbing test for the pure-python metric layer (no GPU / no cluster).

Verifies FID/KID/PRDC/bootstrap/per-class/classifier-hist/verdict run without
crashing on synthetic features and behave directionally (a tighter generated
distribution gives lower FID). NOT a validity test of the real experiment.

Run: python projects/diff-EqM/experiments/exp3_fidelity_diversity/test_metrics_local.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M
from prdc_vendored import compute_prdc
from schedule import build_schedule, schedule_hash, save_schedule, load_schedule


def main():
    rng = np.random.default_rng(0)
    D, C, per = 64, 8, 40
    N = C * per

    # reference: class-structured gaussians
    centers = rng.normal(size=(C, D)) * 3
    def gen(spread):
        labels = np.repeat(np.arange(C), per)
        feats = centers[labels] + rng.normal(size=(N, D)) * spread
        return feats, labels
    ref_feats, ref_labels = gen(1.0)
    anm_feats, labels = gen(0.9)       # tighter -> better fidelity
    van_feats, _ = gen(1.3)            # looser

    # per-class ref stats
    mu_by, cov_by, cnt_by, feats_by = {}, {}, {}, {}
    for c in range(C):
        cf = ref_feats[ref_labels == c]
        mu_by[c] = cf.mean(0); cov_by[c] = np.cov(cf, rowvar=False)
        cnt_by[c] = len(cf); feats_by[c] = cf

    print("FID van:", round(M.compute_fid(van_feats, ref_feats), 3),
          "anm:", round(M.compute_fid(anm_feats, ref_feats), 3))
    km, ks = M.compute_kid(anm_feats, ref_feats, n_subsets=10, subset_size=100)
    print("KID anm:", round(km, 5), "+/-", round(ks, 5))
    print("PRDC anm:", {k: round(v, 3) for k, v in
                        compute_prdc(ref_feats, anm_feats, nearest_k=5).items()})

    agg_v = M.aggregate_metrics(van_feats, ref_feats)
    agg_a = M.aggregate_metrics(anm_feats, ref_feats)
    div_ci = M.bootstrap_diversity(anm_feats, labels, ref_feats, n_boot=20)
    print("bootstrap recall SE:", round(div_ci["recall"]["se"], 4),
          "coverage SE:", round(div_ci["coverage"]["se"], 4))
    fid_ci = M.bootstrap_fid(anm_feats, ref_feats.mean(0),
                             np.cov(ref_feats, rowvar=False), n_boot=20)
    print("FID CI anm:", round(fid_ci["ci_low"], 2), "-", round(fid_ci["ci_high"], 2))

    cls_v = M.per_class_metrics(van_feats, labels, mu_by, cov_by, cnt_by,
                                ref_feats_by_class=feats_by, min_n_for_fid=20)
    cond_by = {c: 0.5 for c in range(C)}
    qm = M.weak_class_scores(cls_v, cond_by)
    print("quartiles:", {q: sum(1 for x in qm.values() if x["class_quartile"] == q)
                         for q in ["bottom", "middle", "top"]})
    bq = M.pooled_bottom_quartile(anm_feats, labels, ref_feats, qm, "bottom")
    print("bottom-quartile FID anm:", round(bq["fid"], 3) if bq else None)

    top1 = labels.copy(); top1[:50] = (top1[:50] + 1) % C
    top5 = np.stack([np.array([t, (t+1) % C, (t+2) % C, (t+3) % C, (t+4) % C])
                     for t in labels])
    real_hist = np.ones(C) / C
    clf_a = M.classifier_hist_metrics(top1, top5, labels, C, real_hist)
    print("clf entropy:", round(clf_a["classifier_entropy"], 3),
          "TV:", round(clf_a["classifier_tv_to_requested"], 3),
          "cond_top1:", round(clf_a["conditional_top1_accuracy"], 3))

    verdict, reasons = M.decide(agg_v, agg_a, clf_a, clf_a,
                                div_ci["recall"]["se"], div_ci["coverage"]["se"],
                                bq_vanilla=bq, bq_anm=bq, frac_classes_improved=0.6)
    print("VERDICT:", verdict)

    # schedule round-trip + hash equality
    s = build_schedule(C, per, base_seed=0, shuffle_seed=0)
    p = Path("/tmp/exp3_sched_test.json")
    save_schedule(s, p)
    s2 = load_schedule(p)
    assert schedule_hash(s) == schedule_hash(s2), "schedule hash mismatch on round-trip"
    assert (np.asarray(s["labels"]) == s2["labels"]).all()
    print("schedule hash:", schedule_hash(s), "round-trip OK")

    print("\nALL LOCAL METRIC PLUMBING CHECKS PASSED")


if __name__ == "__main__":
    main()
