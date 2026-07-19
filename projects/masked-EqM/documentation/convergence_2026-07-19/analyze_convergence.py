"""
Recovery convergence-curve analysis: single trajectory per (model, seed) recorded
at steps 0/25/50/100/250/500/1000, fourier cutoff 0.10 fixed. 5 matched seeds x
{gaussian, mask, gm}. Hierarchical paired bootstrap (resample seeds, then images)
for delta_G(t)/delta_M(t) at each step, Holm-corrected across the 14 tests
(2 deltas x 7 steps). Machinery ported verbatim from analyze_severity.py
(severity-sweep 2026-07-16 experiment) with cutoff-dimension swapped for step.
"""
import json, os

SP = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
STEPS = ["0", "25", "50", "100", "250", "500", "1000"]
CUTOFF = "0.1"


def rng_stream(seed):
    state = [seed]
    def nxt():
        state[0] = (1103515245 * state[0] + 12345) & 0x7fffffff
        return state[0]
    return nxt


def load(model, seed, step):
    d = json.load(open(f"{SP}/convergence_{model}_{seed}.json"))
    return d["per_cutoff"][CUTOFF]["per_step"][step]["per_image"]


def matched(model_a_imgs, model_b_imgs):
    a = {im["index"]: im for im in model_a_imgs}
    b = {im["index"]: im for im in model_b_imgs}
    common = sorted(set(a) & set(b))
    return [(a[i], b[i]) for i in common]


def hierarchical_bootstrap(seed_diffs, n_boot, seed_val):
    nxt = rng_stream(seed_val)
    n_seeds = len(seed_diffs)
    reps = []
    for _ in range(n_boot):
        chosen_seed_idxs = [nxt() % n_seeds for _ in range(n_seeds)]
        total = 0.0
        count = 0
        for si in chosen_seed_idxs:
            diffs = seed_diffs[si]
            n = len(diffs)
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


results = {}
for step in STEPS:
    g_imgs, m_imgs, gm_imgs = {}, {}, {}
    for seed in SEEDS:
        g_imgs[seed] = load("gaussian", seed, step)
        m_imgs[seed] = load("mask", seed, step)
        gm_imgs[seed] = load("gm", seed, step)

    seed_g_minus_gm = []
    seed_m_minus_gm = []
    per_seed_means = {"gaussian": [], "mask": [], "gm": []}
    per_seed_mse = {"gaussian": [], "mask": [], "gm": []}
    seeds_beat_both = 0
    total_wins_g = 0
    total_wins_m = 0
    total_n = 0

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

        g_mse = [a["mse"] for a, b in pg]
        m_mse = [a["mse"] for a, b in pm]
        gm_mse = [b["mse"] for a, b in pg]
        per_seed_mse["gaussian"].append(sum(g_mse) / len(g_mse))
        per_seed_mse["mask"].append(sum(m_mse) / len(m_mse))
        per_seed_mse["gm"].append(sum(gm_mse) / len(gm_mse))

    reps_g = hierarchical_bootstrap(seed_g_minus_gm, 10000, seed_val=hash(("g", step)) & 0xffff)
    reps_m = hierarchical_bootstrap(seed_m_minus_gm, 10000, seed_val=hash(("m", step)) & 0xffff)
    ci_g = ci95(reps_g)
    ci_m = ci95(reps_m)
    p_g = two_sided_p(reps_g)
    p_m = two_sided_p(reps_m)

    results[step] = {
        "mean_gaussian": sum(per_seed_means["gaussian"]) / 5,
        "mean_mask": sum(per_seed_means["mask"]) / 5,
        "mean_gm": sum(per_seed_means["gm"]) / 5,
        "seed_means_gaussian": per_seed_means["gaussian"],
        "seed_means_mask": per_seed_means["mask"],
        "seed_means_gm": per_seed_means["gm"],
        "mean_mse_gaussian": sum(per_seed_mse["gaussian"]) / 5,
        "mean_mse_mask": sum(per_seed_mse["mask"]) / 5,
        "mean_mse_gm": sum(per_seed_mse["gm"]) / 5,
        "delta_g": sum(per_seed_means["gaussian"]) / 5 - sum(per_seed_means["gm"]) / 5,
        "delta_m": sum(per_seed_means["mask"]) / 5 - sum(per_seed_means["gm"]) / 5,
        "ci_g": ci_g, "ci_m": ci_m,
        "p_g_raw": p_g, "p_m_raw": p_m,
        "seeds_beat_both": seeds_beat_both,
        "win_rate_g": total_wins_g / total_n,
        "win_rate_m": total_wins_m / total_n,
    }

pvals = []
for step in STEPS:
    pvals.append((("delta_g", step), results[step]["p_g_raw"]))
    pvals.append((("delta_m", step), results[step]["p_m_raw"]))
adjusted = dict(holm_correct(pvals))
for step in STEPS:
    results[step]["p_g_holm"] = adjusted[("delta_g", step)]
    results[step]["p_m_holm"] = adjusted[("delta_m", step)]

# best LPIPS + step-of-best per model (from the mean-per-step curve)
best = {}
for model in ["gaussian", "mask", "gm"]:
    key = f"mean_{model}"
    curve = [(int(s), results[s][key]) for s in STEPS]
    best_step, best_val = min(curve, key=lambda x: x[1])
    best[model] = {"best_lpips": best_val, "best_step": best_step,
                   "curve": curve}

# step-to-threshold: does gaussian/mask ever reach GM's step-250 (the original
# horizon) mean LPIPS, and if so at what step? Answers "convergence-speed" framing.
gm_250 = results["250"]["mean_gm"]
gm_best_val = best["gm"]["best_lpips"]
step_to_threshold = {}
for model in ["gaussian", "mask"]:
    key = f"mean_{model}"
    curve_sorted = sorted([(int(s), results[s][key]) for s in STEPS], key=lambda x: x[0])
    step_to_threshold[model] = {}
    for label, thresh in [("gm_step250", gm_250), ("gm_best", gm_best_val)]:
        hit = next((s for s, v in curve_sorted if v <= thresh), None)
        step_to_threshold[model][label] = hit

# monotonicity: fraction of images whose LPIPS is non-increasing across all
# 7 recorded steps (per model, averaged across the 5 seeds)
monotonic_frac = {}
for model in ["gaussian", "mask", "gm"]:
    seed_fracs = []
    for seed in SEEDS:
        per_step_imgs = {step: {im["index"]: im["lpips"] for im in load(model, seed, step)} for step in STEPS}
        idxs = sorted(per_step_imgs["0"].keys())
        n_mono = 0
        for idx in idxs:
            seq = [per_step_imgs[step][idx] for step in STEPS]
            if all(seq[i + 1] <= seq[i] + 1e-9 for i in range(len(seq) - 1)):
                n_mono += 1
        seed_fracs.append(n_mono / len(idxs))
    monotonic_frac[model] = sum(seed_fracs) / len(seed_fracs)

out = {
    "steps": STEPS,
    "cutoff": CUTOFF,
    "results": results,
    "best": {m: {"best_lpips": best[m]["best_lpips"], "best_step": best[m]["best_step"]} for m in best},
    "step_to_threshold": step_to_threshold,
    "monotonic_frac": monotonic_frac,
}
json.dump(out, open(f"{SP}/convergence_analysis.json", "w"), indent=2)

print(f"{'step':>6} {'G':>7} {'M':>7} {'GM':>7} {'dG':>8} {'dM':>8} {'CI_G':>20} {'CI_M':>20} "
      f"{'pG_holm':>9} {'pM_holm':>9} {'seeds4/5':>9} {'winG':>6} {'winM':>6} {'MSE_G':>7} {'MSE_M':>7} {'MSE_GM':>7}")
for step in STEPS:
    r = results[step]
    print(f"{step:>6} {r['mean_gaussian']:.4f} {r['mean_mask']:.4f} {r['mean_gm']:.4f} "
          f"{r['delta_g']:+.4f} {r['delta_m']:+.4f} "
          f"[{r['ci_g'][0]:+.4f},{r['ci_g'][1]:+.4f}] [{r['ci_m'][0]:+.4f},{r['ci_m'][1]:+.4f}] "
          f"{r['p_g_holm']:.4f} {r['p_m_holm']:.4f} {r['seeds_beat_both']}/5 "
          f"{r['win_rate_g']:.3f} {r['win_rate_m']:.3f} "
          f"{r['mean_mse_gaussian']:.4f} {r['mean_mse_mask']:.4f} {r['mean_mse_gm']:.4f}")

print()
print("best LPIPS (step of best):")
for m in ["gaussian", "mask", "gm"]:
    print(f"  {m}: {best[m]['best_lpips']:.4f} @ step {best[m]['best_step']}")

print()
print("step-to-threshold (first step reaching threshold, None = never within 1000):")
print(json.dumps(step_to_threshold, indent=2))

print()
print("fraction of images with monotonically non-increasing LPIPS across all 7 steps:")
print(json.dumps(monotonic_frac, indent=2))
