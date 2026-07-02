"""Post-hoc energy/OOD head — lowest-risk test of Yilun's question.

Can a scalar E_psi(x_final), trained post-hoc on frozen cached latents with a
margin-ranking loss against metacog-probe-mined hard negatives, beat the dead
endpoint-energy baselines (~0.605-0.609 AUROC) and approach the SHAPE probe
(~0.81)? Base EqM model is never touched; this reads only the cached
runs/b2_vanilla shard. See documentation/energy-ood-head-design-2026-07-02.md.

Run: python energy_ood_head.py --folder runs/b2_vanilla --seed 0
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from learned_probe import auc, cv_oof_preds, shape_feats


# --------------------------------------------------------------------------- #
def load(folder):
    folder = Path(folder)
    shards = sorted(glob.glob(str(folder / "logs" / "traj_rank*.npz")))
    P = {k: [] for k in ["sample_id", "norm", "dot", "l2", "step_dot", "x_final"]}
    for s in shards:
        d = np.load(s)
        if d["sample_id"].shape[0] == 0:
            continue
        for k in P:
            P[k].append(d[k])
    out = {k: np.concatenate(v, 0) for k, v in P.items()}

    lab, nn_dist = {}, {}
    with open(folder / "labels.csv") as fh:
        for r in csv.DictReader(fh):
            lab[int(r["sample_id"])] = r["label"]
            nn_dist[int(r["sample_id"])] = float(r["nn_dist"])
    y = np.array([1.0 if lab.get(int(i)) == "garbage"
                  else (0.0 if lab.get(int(i)) == "good" else -1)
                  for i in out["sample_id"]])
    amb = np.array([1.0 if lab.get(int(i)) == "ambiguous" else 0.0 for i in out["sample_id"]])
    nnd = np.array([nn_dist.get(int(i), np.nan) for i in out["sample_id"]])
    out["y"] = y          # -1 ambiguous, 0 good, 1 garbage
    out["ambiguous"] = amb
    out["nn_dist"] = nnd
    return out


def existing_baselines(d):
    """Old scalar energy-like baselines, evaluated only on the clean good/garbage
    split. NOTE: nn_dist is excluded here -- labels.csv derives good/garbage by
    thresholding nn_dist itself (see thresholds.json tau_low/tau_high), so
    nn_dist-as-a-baseline is circular (AUROC 1.0 by construction), not a real
    comparison point."""
    keep = d["y"] >= 0
    y = d["y"][keep]
    norm, dot, sd = d["norm"][keep], d["dot"][keep], d["step_dot"][keep]
    rows = {
        "endpoint_dot": auc(y, dot[:, -1]),
        "path_integral_dot": auc(y, sd.sum(1)),
        "final_norm_magnitude": auc(y, norm[:, -1]),
    }
    return rows, keep


def shape_probe_reference(d):
    """Reproduce the existing SHAPE-only descent-dynamics probe AUROC (reference ceiling)."""
    keep = d["y"] >= 0
    y = d["y"][keep]
    norm, dot = d["norm"][keep], d["dot"][keep]
    shp = np.nan_to_num(shape_feats(norm, dot), nan=0.0, posinf=0.0, neginf=0.0)
    oof = cv_oof_preds(shp, y, k=5, seed=0, l2=1.0)
    return auc(y, oof)


# --------------------------------------------------------------------------- #
# E_psi: small MLP over the flattened frozen VAE latent x_final
# --------------------------------------------------------------------------- #
class EnergyHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mine_hard_negatives(d, top_frac=0.5):
    """p_bad from the existing SHAPE probe, restricted to garbage+ambiguous
    (near-manifold pool by construction: same generative chain population,
    not paired with a separate kNN/image OOD filter). Returns a boolean mask
    over the FULL array selecting the mined hard negatives."""
    norm, dot = d["norm"], d["dot"]
    shp = np.nan_to_num(shape_feats(norm, dot), nan=0.0, posinf=0.0, neginf=0.0)
    labeled = d["y"] >= 0
    y_bin = np.where(d["y"] == 1, 1.0, 0.0)  # placeholder target for CV fit on labeled subset
    # fit shape probe on the clean good/garbage subset, score everyone (garbage+ambiguous pool)
    idx_lab = np.where(labeled)[0]
    mu = shp[idx_lab].mean(0); sd = shp[idx_lab].std(0) + 1e-8
    from learned_probe import fit_logreg
    w, b = fit_logreg((shp[idx_lab] - mu) / sd, y_bin[idx_lab], l2=1.0)
    p_bad_all = 1.0 / (1.0 + np.exp(-np.clip(((shp - mu) / sd) @ w + b, -30, 30)))

    pool = (d["y"] == 1) | (d["ambiguous"] == 1)  # garbage + ambiguous = near-manifold OOD pool
    pool_idx = np.where(pool)[0]
    k = max(1, int(round(top_frac * len(pool_idx))))
    top = pool_idx[np.argsort(-p_bad_all[pool_idx])[:k]]
    mask = np.zeros(len(d["y"]), dtype=bool)
    mask[top] = True
    return mask, p_bad_all


def train_energy_head(x_pos, x_neg, seed=0, epochs=200, margin=1.0, lr=1e-3):
    torch.manual_seed(seed)
    in_dim = x_pos.shape[1]
    model = EnergyHead(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xp = torch.tensor(x_pos, dtype=torch.float32)
    xn = torch.tensor(x_neg, dtype=torch.float32)
    mu = torch.cat([xp, xn], 0).mean(0, keepdim=True)
    sd = torch.cat([xp, xn], 0).std(0, keepdim=True) + 1e-6
    xp, xn = (xp - mu) / sd, (xn - mu) / sd
    n = min(len(xp), len(xn))
    for _ep in range(epochs):
        perm_p = torch.randperm(len(xp))[:n]
        perm_n = torch.randperm(len(xn))[:n]
        e_pos = model(xp[perm_p])
        e_neg = model(xn[perm_n])
        loss = torch.clamp(margin + e_pos - e_neg, min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model, mu, sd


def eval_energy_head(model, mu, sd, x_final_held, y_held):
    model.eval()
    with torch.no_grad():
        x = (torch.tensor(x_final_held, dtype=torch.float32) - mu) / sd
        e = model(x).numpy()
    return auc(y_held, e)


# --------------------------------------------------------------------------- #
def main(args):
    rng = np.random.default_rng(args.seed)
    folder = Path(args.folder)
    d = load(folder)
    n_good = int((d["y"] == 0).sum()); n_garb = int((d["y"] == 1).sum()); n_amb = int((d["ambiguous"] == 1).sum())
    print(f"[energy_ood_head] good={n_good} garbage={n_garb} ambiguous={n_amb}", flush=True)

    baselines, _keep_clean = existing_baselines(d)
    shape_ref = shape_probe_reference(d)

    hard_neg_mask, _p_bad_all = mine_hard_negatives(d, top_frac=args.top_frac)
    good_idx = np.where(d["y"] == 0)[0]

    # 70/30 held-out split on the clean good/garbage pool (never touches mined negs from ambiguous)
    clean_idx = np.where(d["y"] >= 0)[0]
    perm = rng.permutation(len(clean_idx))
    n_te = int(0.3 * len(clean_idx))
    te_idx, tr_idx = clean_idx[perm[:n_te]], clean_idx[perm[n_te:]]

    x_final_flat = d["x_final"].reshape(len(d["x_final"]), -1)

    # training positives: 'good' samples in the train split only
    pos_tr = np.intersect1d(tr_idx, good_idx)
    # training negatives: mined hard negatives, excluding anything in the held-out split
    neg_tr = np.setdiff1d(np.where(hard_neg_mask)[0], te_idx)

    seeds = list(range(args.n_model_seeds))
    aurocs = []
    for s in seeds:
        model, mu, sd = train_energy_head(x_final_flat[pos_tr], x_final_flat[neg_tr],
                                           seed=s, epochs=args.epochs, margin=args.margin, lr=args.lr)
        a = eval_energy_head(model, mu, sd, x_final_flat[te_idx], d["y"][te_idx])
        aurocs.append(a)
    e_psi_mean, e_psi_std = float(np.mean(aurocs)), float(np.std(aurocs))

    # sanity: overlap between mined hard negatives and true garbage labels
    mined_garbage_frac = float(d["y"][np.where(hard_neg_mask)[0]] .__eq__(1).mean()) if hard_neg_mask.sum() else float("nan")

    rows = [
        ("endpoint_dot (dead energy)", baselines["endpoint_dot"]),
        ("path_integral_dot (dead energy)", baselines["path_integral_dot"]),
        ("final_norm_magnitude", baselines["final_norm_magnitude"]),
        (f"E_psi post-hoc energy head ({args.n_model_seeds} seeds)", e_psi_mean),
        ("SHAPE-only descent probe (reference ceiling)", shape_ref),
    ]

    out = folder / "results" / "energy_ood_head"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "auroc_table.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "auroc"])
        for name, v in rows:
            w.writerow([name, f"{v:.4f}"])

    lines = ["# Post-hoc energy/OOD head — results", "",
             f"good={n_good} garbage={n_garb} ambiguous={n_amb}",
             "NOTE: knn_dist/nn_dist excluded as a baseline -- labels are derived by "
             "thresholding nn_dist itself (thresholds.json), so it is circular (AUROC=1.0 "
             "by construction), not a real comparison point.",
             f"mined hard negatives: {int(hard_neg_mask.sum())} "
             f"(fraction that are true-labeled garbage: {mined_garbage_frac:.3f}, "
             f"rest ambiguous — near-manifold pool)",
             f"E_psi: {args.n_model_seeds}-seed mean {e_psi_mean:.4f} +/- {e_psi_std:.4f}", "",
             "| method | AUROC |", "|---|---|"]
    for name, v in rows:
        lines.append(f"| {name} | {v:.4f} |")

    old_best = max(baselines.values())
    if e_psi_mean >= shape_ref - 0.03:
        verdict = f"GREEN: E_psi ({e_psi_mean:.3f}) approaches the SHAPE probe ceiling ({shape_ref:.3f})."
    elif e_psi_mean > old_best + 0.05:
        verdict = (f"PARTIAL: E_psi ({e_psi_mean:.3f}) beats old energy baselines "
                   f"(best {old_best:.3f}) but well short of SHAPE probe ({shape_ref:.3f}). "
                   "Some endpoint-only signal recoverable via hard-negative mining, "
                   "but most of the metacog signal is still in trajectory shape.")
    else:
        verdict = (f"NEGATIVE: E_psi ({e_psi_mean:.3f}) does not clear old energy baselines "
                   f"(best {old_best:.3f}). Confirms the signal is not a static function of "
                   "the endpoint, however parameterized -- it lives in descent shape.")
    lines += ["", f"## VERDICT: {verdict}"]
    (out / "ENERGY_OOD_HEAD_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-model-seeds", type=int, default=5)
    ap.add_argument("--top-frac", type=float, default=0.5,
                     help="fraction of garbage+ambiguous pool to mine as hard negatives")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    main(args)
