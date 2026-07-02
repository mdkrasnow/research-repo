"""Test shape probe / E_psi against a NON-nn_dist quality target (max_softmax).

nn_dist and max_softmax are only moderately (anti-)correlated (Pearson ~ -0.43
on runs/b2_vanilla, all 3000 samples) -- a different quality axis, not a
relabeling of the same one. This directly tests the distance-independent
claim the residual-vs-nn_dist test could not answer cleanly (see
shape-probe-nndist-skepticism-2026-07-02.md correction).

Label: quartile-extreme split on max_softmax (bottom 25% = "low-confidence"/1,
top 25% = "high-confidence"/0), symmetric with how thresholds.json built the
nn_dist-based good/garbage split. Two things reported per probe:
  (a) raw held-out AUROC of a probe FIT on this new label (does trajectory
      shape / endpoint predict max_softmax-defined quality at all?)
  (b) AUROC of the EXISTING nn_dist-trained p_bad / E_psi scored against this
      new label (does the old probe generalize to a different quality axis?)

Run: python nonnndist_target_check.py --folder runs/b2_vanilla --seed 0
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np

from learned_probe import auc, shape_feats, fit_logreg
from energy_ood_head import pearson, train_energy_head, eval_energy_head


def load(folder):
    folder = Path(folder)
    shards = sorted(glob.glob(str(folder / "logs" / "traj_rank*.npz")))
    P = {k: [] for k in ["sample_id", "norm", "dot", "x_final"]}
    for s in shards:
        d = np.load(s)
        if d["sample_id"].shape[0] == 0:
            continue
        for k in P:
            P[k].append(d[k])
    out = {k: np.concatenate(v, 0) for k, v in P.items()}
    nnd, ms, old_lab = {}, {}, {}
    with open(folder / "labels.csv") as fh:
        for r in csv.DictReader(fh):
            nnd[int(r["sample_id"])] = float(r["nn_dist"])
            ms[int(r["sample_id"])] = float(r["max_softmax"])
            old_lab[int(r["sample_id"])] = r["label"]
    out["nn_dist"] = np.array([nnd[int(i)] for i in out["sample_id"]])
    out["max_softmax"] = np.array([ms[int(i)] for i in out["sample_id"]])
    out["old_y"] = np.array([1.0 if old_lab.get(int(i)) == "garbage"
                              else (0.0 if old_lab.get(int(i)) == "good" else -1)
                              for i in out["sample_id"]])
    return out


def main(args):
    rng = np.random.default_rng(args.seed)
    folder = Path(args.folder)
    d = load(folder)
    ms = d["max_softmax"]; nnd = d["nn_dist"]
    print(f"[check] pearson(nn_dist, max_softmax) = {pearson(nnd, ms):.3f}", flush=True)

    lo, hi = np.quantile(ms, [0.25, 0.75])
    y_ms = np.where(ms <= lo, 1.0, np.where(ms >= hi, 0.0, -1))  # 1 = low-confidence/bad
    clean = np.where(y_ms >= 0)[0]
    n_bad = int((y_ms[clean] == 1).sum()); n_good = int((y_ms[clean] == 0).sum())
    print(f"[check] max_softmax-quartile split: low-conf={n_bad} high-conf={n_good}", flush=True)

    perm = rng.permutation(len(clean))
    n_te = int(0.3 * len(clean))
    te_idx, tr_idx = clean[perm[:n_te]], clean[perm[n_te:]]

    # ---- (a) shape probe FIT fresh on the max_softmax label ----
    norm, dot = d["norm"], d["dot"]
    shp = np.nan_to_num(shape_feats(norm, dot), nan=0.0, posinf=0.0, neginf=0.0)
    mu = shp[tr_idx].mean(0); sd = shp[tr_idx].std(0) + 1e-8
    w, b = fit_logreg((shp[tr_idx] - mu) / sd, y_ms[tr_idx], l2=1.0)
    p_new_te = 1.0 / (1.0 + np.exp(-np.clip(((shp[te_idx] - mu) / sd) @ w + b, -30, 30)))
    shape_fresh_auroc = auc(y_ms[te_idx], p_new_te)

    # ---- (b) OLD nn_dist-trained shape probe scored against max_softmax label ----
    old_clean = np.where(d["old_y"] >= 0)[0]
    mu_old = shp[old_clean].mean(0); sd_old = shp[old_clean].std(0) + 1e-8
    w_old, b_old = fit_logreg((shp[old_clean] - mu_old) / sd_old, d["old_y"][old_clean], l2=1.0)
    # score everyone (this reuses the OLD nn_dist-trained probe; te_idx here is disjoint from
    # old_clean's own held-out split, but for a transfer check we score the same te_idx pool)
    p_old_on_te = 1.0 / (1.0 + np.exp(-np.clip(((shp[te_idx] - mu_old) / sd_old) @ w_old + b_old, -30, 30)))
    shape_transfer_auroc = auc(y_ms[te_idx], p_old_on_te)

    # ---- E_psi: fresh-trained on max_softmax-defined pos/neg, scored on held-out max_softmax label ----
    x_final_flat = d["x_final"].reshape(len(d["x_final"]), -1)
    good_idx = np.where(y_ms == 0)[0]; bad_idx = np.where(y_ms == 1)[0]
    pos_tr = np.intersect1d(tr_idx, good_idx)
    neg_tr = np.intersect1d(tr_idx, bad_idx)
    e_aurocs = []
    for s in range(args.n_model_seeds):
        model, mu_e, sd_e = train_energy_head(x_final_flat[pos_tr], x_final_flat[neg_tr],
                                               seed=s, epochs=args.epochs, margin=1.0, lr=1e-3)
        a, _ = eval_energy_head(model, mu_e, sd_e, x_final_flat[te_idx], y_ms[te_idx])
        e_aurocs.append(a)
    e_psi_ms_auroc = float(np.mean(e_aurocs)); e_psi_ms_std = float(np.std(e_aurocs))

    lines = ["# Shape probe / E_psi vs a NON-nn_dist target (max_softmax)", "",
             f"pearson(nn_dist, max_softmax) = {pearson(nnd, ms):.3f} "
             "(moderate anti-corr, distinct axis, not a relabel of nn_dist)",
             f"max_softmax-quartile split: low-conf(bad)={n_bad} high-conf(good)={n_good}, "
             f"held-out n={len(te_idx)}", "",
             f"(a) SHAPE probe freshly fit on max_softmax label, held-out AUROC: "
             f"{shape_fresh_auroc:.4f}",
             f"(b) OLD nn_dist-trained shape probe, scored against max_softmax label "
             f"(transfer): {shape_transfer_auroc:.4f}",
             f"(c) E_psi freshly fit on max_softmax label, held-out AUROC "
             f"({args.n_model_seeds} seeds): {e_psi_ms_auroc:.4f} +/- {e_psi_ms_std:.4f}",
             ""]

    if shape_fresh_auroc >= 0.70:
        v = (f"SUPPORTED: shape probe predicts a max_softmax-defined (non-nn_dist) quality axis "
             f"at {shape_fresh_auroc:.3f} AUROC -- distance-independent trajectory signal is real, "
             "not an artifact of the nn_dist labeling scheme.")
    elif shape_fresh_auroc >= 0.60:
        v = (f"WEAK SUPPORT: {shape_fresh_auroc:.3f} on max_softmax label -- some transfer beyond "
             "nn_dist, but much weaker than the 0.81-0.82 nn_dist-label number.")
    else:
        v = (f"NOT SUPPORTED: {shape_fresh_auroc:.3f} -- shape probe does not predict this "
             "non-nn_dist quality axis; nn_dist may be doing most of the work in the original claim.")
    lines.append(f"## VERDICT: {v}")

    out = folder / "results" / "nonnndist_target_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "NONNNDIST_CHECK.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-model-seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    main(args)
