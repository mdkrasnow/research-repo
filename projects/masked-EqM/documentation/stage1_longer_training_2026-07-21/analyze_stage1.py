import json
import glob
import os
import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
RAW = [f"{SP}/raw/holylabs", f"{SP}/raw/home03"]

SEEDS = [0, 1, 2]
EPOCHS = [1, 2, 3, 5]
FAMILIES = ["gaussian", "mask", "gm"]

# name patterns per family/seed (post-quota-incident naming drifted across v2/v3/v4 suffixes)
NAME_PATTERNS = {
    ("gaussian", 0): ["stage1_gaussian_seed0_v2_epoch{}", "stage1_gaussian_seed0_epoch{}"],
    ("gaussian", 1): ["stage1_gaussian_seed1_epoch{}"],
    ("gaussian", 2): ["stage1_gaussian_seed2_epoch{}"],
    ("mask", 0): ["stage1_mask_seed0_v3_epoch{}", "stage1_mask_seed0_epoch{}"],
    ("mask", 1): ["stage1_mask_seed1_v4_epoch{}", "stage1_mask_seed1_epoch{}"],
    ("mask", 2): ["stage1_mask_seed2_epoch{}"],
    ("gm", 0): ["stage1_gm_seed0_v3_epoch{}", "stage1_gm_seed0_epoch{}"],
    ("gm", 1): ["stage1_gm_seed1_v4_epoch{}", "stage1_gm_seed1_epoch{}"],
    ("gm", 2): ["stage1_gm_seed2_epoch{}"],
}


def find_file(base_patterns, epoch, suffix):
    for d in RAW:
        for pat in base_patterns:
            candidate = f"{d}/{pat.format(epoch)}{suffix}"
            if os.path.exists(candidate):
                return candidate
    return None


def load_fourier(base_patterns, epoch):
    f = find_file(base_patterns, epoch, "_fourier.json")
    assert f is not None, f"missing fourier for {base_patterns} epoch {epoch}"
    j = json.load(open(f))
    pc = j["per_cutoff"]["0.1"]
    per_image = {p["index"]: (p["lpips"], p["mse"]) for p in pc["per_image"]}
    return pc["mean_lpips"], pc["mean_mse"], per_image


def load_fid(base_patterns, epoch):
    f = find_file(base_patterns, epoch, "_fid.json")
    assert f is not None, f"missing fid for {base_patterns} epoch {epoch}"
    return json.load(open(f))["fid"]


def load_maskrecovery(base_patterns, epoch):
    f = find_file(base_patterns, epoch, "_maskrecovery.json")
    assert f is not None, f"missing maskrecovery for {base_patterns} epoch {epoch}"
    return json.load(open(f))["mean_lpips"]


# ---- load everything ----
data = {}  # data[epoch][family][seed] = {mean_lpips, mean_mse, per_image, fid, mr_lpips}
for epoch in EPOCHS:
    data[epoch] = {}
    for fam in FAMILIES:
        data[epoch][fam] = {}
        for seed in SEEDS:
            pats = NAME_PATTERNS[(fam, seed)]
            mean_lpips, mean_mse, per_image = load_fourier(pats, epoch)
            fid = load_fid(pats, epoch)
            mr = load_maskrecovery(pats, epoch)
            data[epoch][fam][seed] = {
                "mean_lpips": mean_lpips,
                "mean_mse": mean_mse,
                "per_image": per_image,
                "fid": fid,
                "mr_lpips": mr,
            }

print("Loaded all 9 arms x 4 epochs successfully.")

# sanity: same image-index set across all arms/seeds at a given epoch (frozen manifest)
ref_indices = None
for epoch in EPOCHS:
    for fam in FAMILIES:
        for seed in SEEDS:
            idxs = set(data[epoch][fam][seed]["per_image"].keys())
            if ref_indices is None:
                ref_indices = idxs
            else:
                assert idxs == ref_indices, f"index mismatch at epoch{epoch} {fam} seed{seed}"
print(f"Manifest consistency OK: {len(ref_indices)} images, identical index set across all 36 combos.")
IMAGE_INDICES = sorted(ref_indices)

rng = np.random.default_rng(12345)
N_BOOT = 10000


def hierarchical_bootstrap_delta(epoch, fam_a, fam_b):
    """delta(t) = LPIPS_a - LPIPS_b, bootstrap over seeds then images within seed."""
    per_seed_arrays_a = []
    per_seed_arrays_b = []
    for seed in SEEDS:
        a = data[epoch][fam_a][seed]["per_image"]
        b = data[epoch][fam_b][seed]["per_image"]
        arr_a = np.array([a[i][0] for i in IMAGE_INDICES])
        arr_b = np.array([b[i][0] for i in IMAGE_INDICES])
        per_seed_arrays_a.append(arr_a)
        per_seed_arrays_b.append(arr_b)

    n_seeds = len(SEEDS)
    n_images = len(IMAGE_INDICES)
    boot_deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        seed_sample = rng.integers(0, n_seeds, size=n_seeds)
        delta_sum = 0.0
        for s_idx in seed_sample:
            img_sample = rng.integers(0, n_images, size=n_images)
            delta_sum += per_seed_arrays_a[s_idx][img_sample].mean() - per_seed_arrays_b[s_idx][img_sample].mean()
        boot_deltas[b] = delta_sum / n_seeds

    point = np.mean([per_seed_arrays_a[i].mean() - per_seed_arrays_b[i].mean() for i in range(n_seeds)])
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    # two-sided bootstrap p-value (proportion of bootstrap distribution crossing 0, doubled)
    p_boot = 2 * min((boot_deltas <= 0).mean(), (boot_deltas >= 0).mean())
    p_boot = min(p_boot, 1.0)
    return point, ci_lo, ci_hi, p_boot


def holm_adjust(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in original order."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


results = {}
raw_pvals = []
pval_keys = []
for epoch in EPOCHS:
    dg_point, dg_lo, dg_hi, dg_p = hierarchical_bootstrap_delta(epoch, "gaussian", "gm")
    dm_point, dm_lo, dm_hi, dm_p = hierarchical_bootstrap_delta(epoch, "mask", "gm")
    results[epoch] = {
        "delta_g": dg_point, "ci_g": [dg_lo, dg_hi], "p_g_raw": dg_p,
        "delta_m": dm_point, "ci_m": [dm_lo, dm_hi], "p_m_raw": dm_p,
    }
    raw_pvals.extend([dg_p, dm_p])
    pval_keys.extend([(epoch, "g"), (epoch, "m")])

adjusted = holm_adjust(np.array(raw_pvals))
for (epoch, which), p_adj in zip(pval_keys, adjusted):
    results[epoch][f"p_{which}_holm"] = p_adj

# per-epoch aggregate means/fid/mr + win rates + seeds-beat-both
for epoch in EPOCHS:
    for fam in FAMILIES:
        vals = [data[epoch][fam][s]["mean_lpips"] for s in SEEDS]
        results[epoch][f"mean_lpips_{fam}"] = float(np.mean(vals))
        results[epoch][f"seed_lpips_{fam}"] = vals
        results[epoch][f"mean_mse_{fam}"] = float(np.mean([data[epoch][fam][s]["mean_mse"] for s in SEEDS]))
        results[epoch][f"mean_fid_{fam}"] = float(np.mean([data[epoch][fam][s]["fid"] for s in SEEDS]))
        results[epoch][f"mean_mr_lpips_{fam}"] = float(np.mean([data[epoch][fam][s]["mr_lpips"] for s in SEEDS]))

    # per-image win rate: fraction of images where GM beats gaussian / GM beats mask
    win_g_fracs = []
    win_m_fracs = []
    seeds_beat_both = 0
    for seed in SEEDS:
        gm_img = data[epoch]["gm"][seed]["per_image"]
        g_img = data[epoch]["gaussian"][seed]["per_image"]
        m_img = data[epoch]["mask"][seed]["per_image"]
        wins_g = np.mean([gm_img[i][0] < g_img[i][0] for i in IMAGE_INDICES])
        wins_m = np.mean([gm_img[i][0] < m_img[i][0] for i in IMAGE_INDICES])
        win_g_fracs.append(wins_g)
        win_m_fracs.append(wins_m)
        gm_mean = data[epoch]["gm"][seed]["mean_lpips"]
        g_mean = data[epoch]["gaussian"][seed]["mean_lpips"]
        m_mean = data[epoch]["mask"][seed]["mean_lpips"]
        if gm_mean < g_mean and gm_mean < m_mean:
            seeds_beat_both += 1
    results[epoch]["win_rate_g"] = float(np.mean(win_g_fracs))
    results[epoch]["win_rate_m"] = float(np.mean(win_m_fracs))
    results[epoch]["seed_win_rates_g"] = win_g_fracs
    results[epoch]["seed_win_rates_m"] = win_m_fracs
    results[epoch]["seeds_beat_both"] = seeds_beat_both

# best LPIPS + step of best (per family, across the 4 measured epochs)
best = {}
for fam in FAMILIES:
    vals = {ep: results[ep][f"mean_lpips_{fam}"] for ep in EPOCHS}
    best_ep = min(vals, key=vals.get)
    best[fam] = {"best_lpips": vals[best_ep], "best_epoch": best_ep}

output = {
    "epochs": EPOCHS,
    "seeds": SEEDS,
    "num_images": len(IMAGE_INDICES),
    "n_bootstrap": N_BOOT,
    "results": {str(k): v for k, v in results.items()},
    "best": best,
}
json.dump(output, open(f"{SP}/stage1_analysis.json", "w"), indent=2)

print("\n=== Stage 1 results table ===")
header = f"{'ep':>3} {'LPIPS_G':>8} {'LPIPS_M':>8} {'LPIPS_GM':>9} {'d_G':>8} {'CI_G':>20} {'p_G_holm':>9} {'d_M':>8} {'CI_M':>20} {'p_M_holm':>9} {'beat_both':>9} {'winG':>6} {'winM':>6} {'FID_G':>7} {'FID_M':>7} {'FID_GM':>7}"
print(header)
for ep in EPOCHS:
    r = results[ep]
    print(f"{ep:>3} {r['mean_lpips_gaussian']:>8.4f} {r['mean_lpips_mask']:>8.4f} {r['mean_lpips_gm']:>9.4f} "
          f"{r['delta_g']:>+8.4f} [{r['ci_g'][0]:+.4f},{r['ci_g'][1]:+.4f}] {r['p_g_holm']:>9.4f} "
          f"{r['delta_m']:>+8.4f} [{r['ci_m'][0]:+.4f},{r['ci_m'][1]:+.4f}] {r['p_m_holm']:>9.4f} "
          f"{r['seeds_beat_both']:>7}/3 {r['win_rate_g']:>6.3f} {r['win_rate_m']:>6.3f} "
          f"{r['mean_fid_gaussian']:>7.2f} {r['mean_fid_mask']:>7.2f} {r['mean_fid_gm']:>7.2f}")

print("\n=== Best LPIPS per family (across measured epochs 1,2,3,5) ===")
for fam, b in best.items():
    print(f"{fam}: best={b['best_lpips']:.4f} at epoch{b['best_epoch']}")

print("\n=== Mask-recovery LPIPS (trained task) ===")
for ep in EPOCHS:
    r = results[ep]
    print(f"epoch{ep}: G={r['mean_mr_lpips_gaussian']:.4f} M={r['mean_mr_lpips_mask']:.4f} GM={r['mean_mr_lpips_gm']:.4f}")
