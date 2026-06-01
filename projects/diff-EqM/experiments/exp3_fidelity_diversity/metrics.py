"""Metrics for Experiment 3, all computed on shared pytorch_fid 2048-d features.

FID uses the exact pytorch_fid formula (continuity with the trusted 27.09/31.41
numbers). KID and PRDC are computed on the same features. Bootstrap CIs are
stratified by requested class. Per-class leans on feature_mean_distance (stable
at 50/class); per-class FID is emitted but flagged noisy.
"""
import numpy as np
from scipy import linalg

from prdc_vendored import compute_prdc


# --------------------------------------------------------------------------- #
# FID (pytorch_fid formula)
# --------------------------------------------------------------------------- #
def _stats(feats):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def compute_fid(gen_feats, ref_feats):
    mu_g, sig_g = _stats(gen_feats)
    mu_r, sig_r = _stats(ref_feats)
    return frechet_distance(mu_r, sig_r, mu_g, sig_g)


def compute_fid_from_stats(gen_feats, ref_mu, ref_sigma):
    mu_g, sig_g = _stats(gen_feats)
    return frechet_distance(ref_mu, ref_sigma, mu_g, sig_g)


# --------------------------------------------------------------------------- #
# KID (polynomial-kernel MMD, unbiased, subset-averaged)
# --------------------------------------------------------------------------- #
def _poly_kernel(X, Y, degree=3, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * (X @ Y.T) + coef0) ** degree


def _mmd2_unbiased(X, Y):
    m, n = X.shape[0], Y.shape[0]
    kxx = _poly_kernel(X, X)
    kyy = _poly_kernel(Y, Y)
    kxy = _poly_kernel(X, Y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return (kxx.sum() / (m * (m - 1))
            + kyy.sum() / (n * (n - 1))
            - 2.0 * kxy.mean())


def compute_kid(gen_feats, ref_feats, n_subsets=100, subset_size=1000, seed=0):
    """Returns (mean, std) of subset MMD^2 estimates."""
    gen_feats = np.asarray(gen_feats, dtype=np.float64)
    ref_feats = np.asarray(ref_feats, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s = min(subset_size, len(gen_feats), len(ref_feats))
    vals = []
    for _ in range(n_subsets):
        gi = rng.choice(len(gen_feats), s, replace=False)
        ri = rng.choice(len(ref_feats), s, replace=False)
        vals.append(_mmd2_unbiased(gen_feats[gi], ref_feats[ri]))
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.std())


# --------------------------------------------------------------------------- #
# Aggregate (FID + KID + PRDC) on full features
# --------------------------------------------------------------------------- #
def aggregate_metrics(gen_feats, ref_feats, prdc_k=5):
    fid = compute_fid(gen_feats, ref_feats)
    kid_mean, kid_std = compute_kid(gen_feats, ref_feats)
    prdc = compute_prdc(ref_feats, gen_feats, nearest_k=prdc_k)
    return {"fid": fid, "kid_mean": kid_mean, "kid_std": kid_std,
            "prdc_k": prdc_k, **prdc}


# --------------------------------------------------------------------------- #
# Bootstrap CIs -- stratified by requested class.
# Focus on recall & coverage (the gating diversity metrics). FID/KID bootstrap
# is expensive (sqrtm per replicate); keep n_fid small and document.
# --------------------------------------------------------------------------- #
def bootstrap_diversity(gen_feats, gen_labels, ref_feats, prdc_k=5,
                        n_boot=200, subset_per_class=None, seed=0):
    """Stratified bootstrap SE/CI for precision/recall/density/coverage.

    Resamples requested-class groups with replacement, preserving class balance,
    on a per-class subset for tractability (kNN is O(n^2)).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(gen_labels)
    classes = np.unique(labels)
    idx_by_class = {c: np.where(labels == c)[0] for c in classes}
    # subsample reference once for speed
    ref_n = min(len(ref_feats), 5000)
    ref_sub = ref_feats[rng.choice(len(ref_feats), ref_n, replace=False)]

    keys = ["precision", "recall", "density", "coverage"]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        picks = []
        for c in classes:
            pool = idx_by_class[c]
            take = subset_per_class or len(pool)
            picks.append(rng.choice(pool, take, replace=True))
        gi = np.concatenate(picks)
        # cap generated subsample to keep kNN tractable
        if len(gi) > 5000:
            gi = rng.choice(gi, 5000, replace=False)
        d = compute_prdc(ref_sub, gen_feats[gi], nearest_k=prdc_k)
        for k in keys:
            acc[k].append(d[k])
    out = {}
    for k in keys:
        a = np.asarray(acc[k])
        out[k] = {"mean": float(a.mean()), "se": float(a.std()),
                  "ci_low": float(np.percentile(a, 2.5)),
                  "ci_high": float(np.percentile(a, 97.5))}
    return out


def bootstrap_fid(gen_feats, ref_mu, ref_sigma, n_boot=100, seed=0):
    """Bootstrap CI for FID by resampling generated features with replacement."""
    rng = np.random.default_rng(seed)
    n = len(gen_feats)
    vals = []
    for _ in range(n_boot):
        gi = rng.choice(n, n, replace=True)
        vals.append(compute_fid_from_stats(gen_feats[gi], ref_mu, ref_sigma))
    vals = np.asarray(vals)
    return {"mean": float(vals.mean()), "se": float(vals.std()),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5))}


# --------------------------------------------------------------------------- #
# Per-class metrics
# --------------------------------------------------------------------------- #
def per_class_metrics(gen_feats, gen_labels, ref_mu_by_class, ref_cov_by_class,
                      ref_count_by_class, ref_feats_by_class=None,
                      min_n_for_fid=50, mahalanobis_lambda=1e-3):
    """Per-class feature_mean_distance (+normalized), per-class FID (flagged
    noisy), counts. ref_*_by_class are dicts keyed by int class id."""
    labels = np.asarray(gen_labels)
    rows = []
    for c in np.unique(labels):
        gi = np.where(labels == c)[0]
        gf = gen_feats[gi]
        n_gen = len(gf)
        mu_g = gf.mean(axis=0)
        mu_r = ref_mu_by_class[int(c)]
        d = float(np.linalg.norm(mu_g - mu_r))
        # normalized (Mahalanobis) distance using ref class cov
        cov_r = ref_cov_by_class[int(c)]
        cov_reg = cov_r + mahalanobis_lambda * np.eye(cov_r.shape[0])
        diff = mu_g - mu_r
        try:
            d_norm = float(np.sqrt(diff @ np.linalg.solve(cov_reg, diff)))
        except np.linalg.LinAlgError:
            d_norm = float("nan")
        n_ref = int(ref_count_by_class[int(c)])
        row = {"class_id": int(c), "n_gen": n_gen, "n_ref": n_ref,
               "feature_distance_class": d,
               "feature_distance_class_normalized": d_norm,
               "fid_class": float("nan"), "fid_class_noisy_flag": True}
        if ref_feats_by_class is not None and n_gen >= min_n_for_fid \
                and len(ref_feats_by_class[int(c)]) >= min_n_for_fid:
            row["fid_class"] = compute_fid(gf, ref_feats_by_class[int(c)])
        rows.append(row)
    return rows


def weak_class_scores(vanilla_class_rows, cond_top1_by_class):
    """weak_score = z(feature_distance) + z(1 - cond_top1). Vanilla-only.
    Returns dict class_id -> (score, quartile) with worst 25% = 'bottom'."""
    cids = [r["class_id"] for r in vanilla_class_rows]
    fdist = np.array([r["feature_distance_class"] for r in vanilla_class_rows])
    inacc = np.array([1.0 - cond_top1_by_class.get(c, 0.0) for c in cids])

    def z(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else np.zeros_like(x)

    score = z(fdist) + z(inacc)
    order = np.argsort(-score)  # worst (highest) first
    n = len(cids)
    bottom = set(np.array(cids)[order[: n // 4]])
    top = set(np.array(cids)[order[-(n // 4):]]) if n >= 4 else set()
    out = {}
    for i, c in enumerate(cids):
        q = "bottom" if c in bottom else ("top" if c in top else "middle")
        out[c] = {"weak_class_score": float(score[i]), "class_quartile": q}
    return out


def pooled_bottom_quartile(gen_feats, gen_labels, ref_feats, quartile_map,
                           which="bottom", prdc_k=5):
    """Pool all generated samples whose requested class is in `which` quartile
    (~N/4 samples -> reliable) and compute aggregate metrics vs full reference."""
    labels = np.asarray(gen_labels)
    sel = np.array([quartile_map.get(int(c), {}).get("class_quartile") == which
                    for c in labels])
    if sel.sum() == 0:
        return None
    return aggregate_metrics(gen_feats[sel], ref_feats, prdc_k=prdc_k)


# --------------------------------------------------------------------------- #
# Classifier histogram metrics
# --------------------------------------------------------------------------- #
# resnet50 IMAGENET1K_V2 always emits 1000-way predictions, independent of how
# many classes the schedule conditions on (smoke uses 100). All classifier
# histograms MUST be length CLF_VOCAB or pred/requested/real arrays mismatch.
CLF_VOCAB = 1000


def classifier_hist_metrics(pred_top1, pred_top5, requested_labels,
                            num_classes, real_hist=None, vocab=CLF_VOCAB):
    pred_top1 = np.asarray(pred_top1)
    requested = np.asarray(requested_labels)
    vocab = max(vocab, int(pred_top1.max()) + 1, int(requested.max()) + 1)
    counts = np.bincount(pred_top1, minlength=vocab).astype(np.float64)
    pred_hist = counts / counts.sum()
    req_counts = np.bincount(requested, minlength=vocab).astype(np.float64)
    req_hist = req_counts / req_counts.sum()

    nz = pred_hist > 0
    entropy = float(-(pred_hist[nz] * np.log(pred_hist[nz])).sum())
    eps = 1e-12
    kl = float((pred_hist * np.log((pred_hist + eps) / (req_hist + eps))).sum())
    tv = float(0.5 * np.abs(pred_hist - req_hist).sum())
    missing = int((counts == 0).sum())

    cond_top1 = float((pred_top1 == requested).mean())
    top5 = np.asarray(pred_top5)
    cond_top5 = float(np.mean([requested[i] in top5[i] for i in range(len(requested))]))

    out = {"classifier_entropy": entropy, "classifier_kl_to_requested": kl,
           "classifier_tv_to_requested": tv, "classifier_missing_classes": missing,
           "conditional_top1_accuracy": cond_top1,
           "conditional_top5_accuracy": cond_top5,
           "pred_hist": pred_hist, "requested_hist": req_hist}
    if real_hist is not None:
        rh = np.zeros(vocab); rh[:min(vocab, len(real_hist))] = real_hist[:vocab]
        out["classifier_tv_to_real"] = float(0.5 * np.abs(pred_hist - rh).sum())
    return out


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def decide(vanilla_agg, anm_agg, vanilla_clf, anm_clf,
           recall_se, coverage_se, bq_vanilla=None, bq_anm=None,
           frac_classes_improved=None, noise_floor=None):
    """success / strong_success / failure / ambiguous, with reasons."""
    recall_tol = max(0.005, recall_se)
    cov_tol = max(0.005, coverage_se)

    fid_better = anm_agg["fid"] < vanilla_agg["fid"]
    kid_better = anm_agg["kid_mean"] < vanilla_agg["kid_mean"]
    recall_ok = anm_agg["recall"] >= vanilla_agg["recall"] - recall_tol
    cov_ok = anm_agg["coverage"] >= vanilla_agg["coverage"] - cov_tol
    tv_ok = (anm_clf["classifier_tv_to_requested"]
             <= vanilla_clf["classifier_tv_to_requested"] + 0.02)
    acc_ok = (anm_clf["conditional_top1_accuracy"]
              >= vanilla_clf["conditional_top1_accuracy"] - 0.01)

    reasons = {
        "fid_better": fid_better, "kid_better": kid_better,
        "recall_ok": recall_ok, "coverage_ok": cov_ok,
        "tv_ok": tv_ok, "cond_top1_ok": acc_ok,
        "recall_tol": recall_tol, "coverage_tol": cov_tol,
    }

    primary_success = all([fid_better, kid_better, recall_ok, cov_ok, tv_ok, acc_ok])

    # ambiguous: fidelity metrics disagree
    if fid_better != kid_better:
        return "ambiguous", reasons

    if not primary_success:
        # FID improved but a diversity/coverage/class guard failed -> failure
        if fid_better and not (recall_ok and cov_ok and tv_ok and acc_ok):
            return "failure", reasons
        return "ambiguous", reasons

    # strong success checks
    strong_signals = 0
    if anm_agg["recall"] > vanilla_agg["recall"]:
        strong_signals += 1
    if anm_agg["coverage"] > vanilla_agg["coverage"]:
        strong_signals += 1
    if bq_vanilla and bq_anm and bq_anm["fid"] < bq_vanilla["fid"]:
        strong_signals += 1
    if frac_classes_improved is not None and frac_classes_improved >= 0.55:
        strong_signals += 1
    reasons["strong_signals"] = strong_signals
    if strong_signals >= 2:
        return "strong_success", reasons
    return "success", reasons
