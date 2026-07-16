"""
Fourier severity-curve analysis: 6 cutoffs (0.05,0.10,0.15,0.20,0.30,0.4181),
5 matched seeds x {gaussian,mask,gm}, plus fourier-only specialist (seed0).
Hierarchical paired bootstrap (resample seeds, then images) for delta_G/delta_M
at each cutoff, Holm-corrected across the 12 tests (2 deltas x 6 cutoffs).
"""
import json, math, os

SP = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
NEW_CUTOFFS = ["0.05", "0.1", "0.15"]

def rng_stream(seed):
    # deterministic LCG, no random module (avoid Date.now()-style nondeterminism concerns / keep simple)
    state = [seed]
    def nxt():
        state[0] = (1103515245 * state[0] + 12345) & 0x7fffffff
        return state[0]
    return nxt

def load_new(model, seed, cutoff_key):
    d = json.load(open(f"{SP}/severity_{model}_{seed}.json"))
    return d["per_cutoff"][cutoff_key]["per_image"]

def load_old(model, seed, which):
    # which in {"primary","secondary"} old replication files
    d = json.load(open(f"{SP}/fourier_repl_{model}_seed{seed}_{which}.json"))
    return d

def get_old_cutoff(model, seed, cutoff_key):
    # primary file only has cutoff 0.4181 (single, not per_cutoff dict); secondary has grid incl 0.20/0.30
    if cutoff_key == "0.4181":
        d = load_old(model, seed, "primary")
        return d["per_cutoff"][cutoff_key]["per_image"] if "per_cutoff" in d else d["per_image"]
    else:
        d = load_old(model, seed, "secondary")
        return d["per_cutoff"][cutoff_key]["per_image"]

def matched(model_a_imgs, model_b_imgs):
    a = {im["index"]: im for im in model_a_imgs}
    b = {im["index"]: im for im in model_b_imgs}
    common = sorted(set(a) & set(b))
    return [(a[i], b[i]) for i in common]

def hierarchical_bootstrap(seed_diffs, n_boot, seed_val):
    # seed_diffs: list over seeds, each a list of per-image (a-b) diffs
    nxt = rng_stream(seed_val)
    n_seeds = len(seed_diffs)
    reps = []
    for _ in range(n_boot):
        # resample seeds with replacement
        chosen_seed_idxs = [nxt() % n_seeds for _ in range(n_seeds)]
        total = 0.0
        count = 0
        for si in chosen_seed_idxs:
            diffs = seed_diffs[si]
            n = len(diffs)
            # resample images with replacement within this seed
            s = 0.0
            for _ in range(n):
                s += diffs[nxt() % n]
            total += s
            count += n
        reps.append(total / count)
    return reps

def ci95(reps):
    r = sorted(reps)
    n = len(r)
    lo = r[int(0.025 * n)]
    hi = r[int(0.975 * n)]
    return lo, hi

def two_sided_p(reps):
    n = len(reps)
    frac_le0 = sum(1 for x in reps if x <= 0) / n
    p = 2 * min(frac_le0, 1 - frac_le0)
    return min(p, 1.0)

def holm_correct(pvals_with_labels):
    order = sorted(range(len(pvals_with_labels)), key=lambda i: pvals_with_labels[i][1])
    m = len(pvals_with_labels)
    adj = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        label, p = pvals_with_labels[idx]
        val = min((m - rank) * p, 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return [(pvals_with_labels[i][0], adj[i]) for i in range(m)]

CUTOFFS = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.4181"]

results = {}
for cutoff in CUTOFFS:
    g_imgs, m_imgs, gm_imgs = {}, {}, {}
    for seed in SEEDS:
        if cutoff in NEW_CUTOFFS:
            g_imgs[seed] = load_new("gaussian", seed, cutoff)
            m_imgs[seed] = load_new("mask", seed, cutoff)
            gm_imgs[seed] = load_new("gm", seed, cutoff)
        else:
            g_imgs[seed] = get_old_cutoff("gaussian", seed, cutoff)
            m_imgs[seed] = get_old_cutoff("mask", seed, cutoff)
            gm_imgs[seed] = get_old_cutoff("gm", seed, cutoff)

    seed_g_minus_gm = []
    seed_m_minus_gm = []
    per_seed_means = {"gaussian": [], "mask": [], "gm": []}
    seeds_beat_both = 0
    total_wins_g = 0
    total_wins_m = 0
    total_n = 0
    per_seed_mse = {"gaussian": [], "mask": [], "gm": []}

    for seed in SEEDS:
        pg = matched(g_imgs[seed], gm_imgs[seed])
        pm = matched(m_imgs[seed], gm_imgs[seed])
        g_lpips = [a["lpips"] for a, b in pg]
        gm_lpips_from_pg = [b["lpips"] for a, b in pg]
        m_lpips = [a["lpips"] for a, b in pm]
        gm_lpips_from_pm = [b["lpips"] for a, b in pm]

        mean_g = sum(g_lpips) / len(g_lpips)
        mean_m = sum(m_lpips) / len(m_lpips)
        mean_gm_viag = sum(gm_lpips_from_pg) / len(gm_lpips_from_pg)
        mean_gm_viam = sum(gm_lpips_from_pm) / len(gm_lpips_from_pm)
        per_seed_means["gaussian"].append(mean_g)
        per_seed_means["mask"].append(mean_m)
        per_seed_means["gm"].append((mean_gm_viag + mean_gm_viam) / 2)

        diffs_g = [a["lpips"] - b["lpips"] for a, b in pg]  # gaussian - gm
        diffs_m = [a["lpips"] - b["lpips"] for a, b in pm]  # mask - gm
        seed_g_minus_gm.append(diffs_g)
        seed_m_minus_gm.append(diffs_m)

        beats_both = (mean_gm_viag < mean_g) and (mean_gm_viam < mean_m)
        if beats_both:
            seeds_beat_both += 1

        total_wins_g += sum(1 for d in diffs_g if d > 0)
        total_wins_m += sum(1 for d in diffs_m if d > 0)
        total_n += len(diffs_g)

        # MSE
        g_mse = [a["mse"] for a, b in pg]
        m_mse = [a["mse"] for a, b in pm]
        gm_mse = [b["mse"] for a, b in pg]
        per_seed_mse["gaussian"].append(sum(g_mse)/len(g_mse))
        per_seed_mse["mask"].append(sum(m_mse)/len(m_mse))
        per_seed_mse["gm"].append(sum(gm_mse)/len(gm_mse))

    reps_g = hierarchical_bootstrap(seed_g_minus_gm, 10000, seed_val=hash(("g", cutoff)) & 0xffff)
    reps_m = hierarchical_bootstrap(seed_m_minus_gm, 10000, seed_val=hash(("m", cutoff)) & 0xffff)
    ci_g = ci95(reps_g)
    ci_m = ci95(reps_m)
    p_g = two_sided_p(reps_g)
    p_m = two_sided_p(reps_m)

    results[cutoff] = {
        "mean_gaussian": sum(per_seed_means["gaussian"]) / 5,
        "mean_mask": sum(per_seed_means["mask"]) / 5,
        "mean_gm": sum(per_seed_means["gm"]) / 5,
        "seed_means_gaussian": per_seed_means["gaussian"],
        "seed_means_mask": per_seed_means["mask"],
        "seed_means_gm": per_seed_means["gm"],
        "mean_mse_gaussian": sum(per_seed_mse["gaussian"]) / 5,
        "mean_mse_mask": sum(per_seed_mse["mask"]) / 5,
        "mean_mse_gm": sum(per_seed_mse["gm"]) / 5,
        "delta_g": sum(per_seed_means["gaussian"])/5 - sum(per_seed_means["gm"])/5,
        "delta_m": sum(per_seed_means["mask"])/5 - sum(per_seed_means["gm"])/5,
        "ci_g": ci_g, "ci_m": ci_m,
        "p_g_raw": p_g, "p_m_raw": p_m,
        "seeds_beat_both": seeds_beat_both,
        "win_rate_g": total_wins_g / total_n,
        "win_rate_m": total_wins_m / total_n,
    }

# fourier specialist (seed0 only) for gap-closure calc
fspec = {}
for cutoff in CUTOFFS:
    if cutoff in NEW_CUTOFFS:
        imgs = load_new("fourier_only", 0, cutoff)
    elif cutoff == "0.4181":
        d = json.load(open(f"{SP}/fourier_repl_fourier_only_seed0_primary.json"))
        imgs = d["per_cutoff"][cutoff]["per_image"]
    else:
        d = json.load(open(f"{SP}/harder_fourier_only_seed0.json"))
        imgs = d["per_cutoff"][cutoff]["per_image"]
    fspec[cutoff] = sum(im["lpips"] for im in imgs) / len(imgs)

# Holm correction across all 12 tests
pvals = []
for cutoff in CUTOFFS:
    pvals.append((("delta_g", cutoff), results[cutoff]["p_g_raw"]))
    pvals.append((("delta_m", cutoff), results[cutoff]["p_m_raw"]))
adjusted = dict(holm_correct(pvals))

for cutoff in CUTOFFS:
    results[cutoff]["p_g_holm"] = adjusted[("delta_g", cutoff)]
    results[cutoff]["p_m_holm"] = adjusted[("delta_m", cutoff)]
    lpips_g = results[cutoff]["mean_gaussian"]
    lpips_gm = results[cutoff]["mean_gm"]
    lpips_spec = fspec[cutoff]
    denom = lpips_g - lpips_spec
    results[cutoff]["frac_gap_closed"] = (lpips_g - lpips_gm) / denom if abs(denom) > 1e-9 else float("nan")
    results[cutoff]["fourier_specialist_lpips"] = lpips_spec

out = {"cutoffs": CUTOFFS, "results": results}
json.dump(out, open(f"{SP}/severity_curve_analysis.json", "w"), indent=2)

print(f"{'cutoff':>8} {'G':>7} {'M':>7} {'GM':>7} {'Spec':>7} {'dG':>8} {'dM':>8} {'CI_G':>20} {'CI_M':>20} {'pG_holm':>9} {'pM_holm':>9} {'seeds4/5':>9} {'winG':>6} {'winM':>6} {'gap%':>7}")
for cutoff in CUTOFFS:
    r = results[cutoff]
    print(f"{cutoff:>8} {r['mean_gaussian']:.4f} {r['mean_mask']:.4f} {r['mean_gm']:.4f} {r['fourier_specialist_lpips']:.4f} "
          f"{r['delta_g']:+.4f} {r['delta_m']:+.4f} "
          f"[{r['ci_g'][0]:+.4f},{r['ci_g'][1]:+.4f}] [{r['ci_m'][0]:+.4f},{r['ci_m'][1]:+.4f}] "
          f"{r['p_g_holm']:.4f} {r['p_m_holm']:.4f} {r['seeds_beat_both']}/5 "
          f"{r['win_rate_g']:.3f} {r['win_rate_m']:.3f} {r['frac_gap_closed']*100:.1f}")
