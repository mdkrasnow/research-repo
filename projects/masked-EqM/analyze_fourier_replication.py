# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Preregistered analysis for the G+M 1:1 -> unseen-Fourier replication
(2026-07-15 runbook). Consumes per-image eval_fourier_recovery.py JSON
outputs for matched Gaussian-only / mask-only / G+M-1:1 seed triplets and
computes:
  - per-seed mean LPIPS/MSE per cutoff
  - hierarchical paired bootstrap (resample seeds, then images within the
    resampled seed) for delta_G = LPIPS(gaussian) - LPIPS(GM) and
    delta_M = LPIPS(mask) - LPIPS(GM), Holm-corrected across the two tests
  - per-seed win check (does G+M beat both parents' mean LPIPS)
  - win-rate by image (fraction of images where G+M beats the best parent)
  - cutoff-response table across the secondary grid

No Date.now()/random() usage other than numpy's seeded Generator, so this
is fully reproducible given the same input JSONs.
"""
import argparse
import json

import numpy as np


def load_per_image(path, cutoff_key):
    with open(path) as f:
        d = json.load(f)
    pc = d["per_cutoff"][cutoff_key]
    by_index = {r["index"]: r for r in pc["per_image"]}
    return by_index


def matched_arrays(gauss_by_idx, mask_by_idx, gm_by_idx):
    """Intersect indices present in all three (should be the full manifest
    if everything ran cleanly) and return aligned LPIPS arrays."""
    common = sorted(set(gauss_by_idx) & set(mask_by_idx) & set(gm_by_idx))
    g = np.array([gauss_by_idx[i]["lpips"] for i in common])
    m = np.array([mask_by_idx[i]["lpips"] for i in common])
    gm = np.array([gm_by_idx[i]["lpips"] for i in common])
    return common, g, m, gm


def hierarchical_bootstrap(seed_arrays, n_boot, rng):
    """seed_arrays: list of 1D delta arrays, one per matched training seed.
    Resample seeds with replacement, then resample images within the
    chosen seed with replacement, average -> one bootstrap replicate of
    the grand mean delta. Returns array of n_boot replicate means."""
    n_seeds = len(seed_arrays)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        seed_choice = rng.integers(0, n_seeds, size=n_seeds)
        seed_means = []
        for s in seed_choice:
            arr = seed_arrays[s]
            img_choice = rng.integers(0, len(arr), size=len(arr))
            seed_means.append(arr[img_choice].mean())
        reps[b] = np.mean(seed_means)
    return reps


def holm_correct(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def two_sided_p_from_bootstrap(reps):
    """p-value for H0: mean delta <= 0, via bootstrap: fraction of
    replicates <= 0, doubled (two-sided), capped at 1."""
    p_one_sided = (reps <= 0).mean()
    return min(1.0, 2 * p_one_sided)


def main(args):
    rng = np.random.default_rng(args.rng_seed)
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"=== Primary cutoff {args.cutoff} ===")
    delta_g_by_seed, delta_m_by_seed, synergy_by_seed = [], [], []
    seed_table = []
    for seed in seeds:
        gauss = load_per_image(args.gaussian_json.format(seed=seed), args.cutoff)
        mask = load_per_image(args.mask_json.format(seed=seed), args.cutoff)
        gm = load_per_image(args.gm_json.format(seed=seed), args.cutoff)
        common, g, m, gmv = matched_arrays(gauss, mask, gm)
        dg = g - gmv
        dm = m - gmv
        syn = np.minimum(g, m) - gmv
        delta_g_by_seed.append(dg)
        delta_m_by_seed.append(dm)
        synergy_by_seed.append(syn)
        beats_both = (gmv.mean() < g.mean()) and (gmv.mean() < m.mean())
        seed_table.append({
            "seed": seed, "n_images": len(common),
            "mean_gaussian_lpips": float(g.mean()), "mean_mask_lpips": float(m.mean()),
            "mean_gm_lpips": float(gmv.mean()), "mean_delta_g": float(dg.mean()),
            "mean_delta_m": float(dm.mean()), "beats_both_parents": bool(beats_both),
        })
        print(f"seed{seed}: n={len(common)} gaussian={g.mean():.4f} mask={m.mean():.4f} "
              f"gm={gmv.mean():.4f} beats_both={beats_both}")

    reps_g = hierarchical_bootstrap(delta_g_by_seed, args.n_boot, rng)
    reps_m = hierarchical_bootstrap(delta_m_by_seed, args.n_boot, rng)
    p_g = two_sided_p_from_bootstrap(reps_g)
    p_m = two_sided_p_from_bootstrap(reps_m)
    p_adj = holm_correct(np.array([p_g, p_m]))

    ci_g = np.percentile(reps_g, [2.5, 97.5])
    ci_m = np.percentile(reps_m, [2.5, 97.5])

    n_beats_both = sum(1 for r in seed_table if r["beats_both_parents"])

    result = {
        "cutoff": args.cutoff,
        "n_boot": args.n_boot,
        "seed_table": seed_table,
        "delta_g": {"mean": float(np.mean([d.mean() for d in delta_g_by_seed])),
                    "ci95": ci_g.tolist(), "p_raw": float(p_g), "p_holm": float(p_adj[0]),
                    "excludes_zero": bool(ci_g[0] > 0)},
        "delta_m": {"mean": float(np.mean([d.mean() for d in delta_m_by_seed])),
                    "ci95": ci_m.tolist(), "p_raw": float(p_m), "p_holm": float(p_adj[1]),
                    "excludes_zero": bool(ci_m[0] > 0)},
        "seeds_beating_both_parents": f"{n_beats_both}/{len(seeds)}",
        "gate_2of3_seeds": n_beats_both >= 2 if len(seeds) == 3 else None,
        "gate_4of5_seeds": n_beats_both >= 4 if len(seeds) == 5 else None,
    }

    print(f"\ndelta_G (gaussian - gm): mean={result['delta_g']['mean']:.5f} "
          f"95% CI={ci_g} holm_p={p_adj[0]:.4f} excludes_zero={result['delta_g']['excludes_zero']}")
    print(f"delta_M (mask - gm): mean={result['delta_m']['mean']:.5f} "
          f"95% CI={ci_m} holm_p={p_adj[1]:.4f} excludes_zero={result['delta_m']['excludes_zero']}")
    print(f"seeds beating both parents: {n_beats_both}/{len(seeds)}")

    # secondary cutoff grid: just the direction (mean delta sign), no bootstrap
    if args.cutoff_grid:
        grid_table = []
        for c in args.cutoff_grid.split(","):
            dgs, dms = [], []
            for seed in seeds:
                gauss = load_per_image(args.gaussian_json.format(seed=seed), c)
                mask = load_per_image(args.mask_json.format(seed=seed), c)
                gm = load_per_image(args.gm_json.format(seed=seed), c)
                _, g, m, gmv = matched_arrays(gauss, mask, gm)
                dgs.append((g - gmv).mean())
                dms.append((m - gmv).mean())
            positive = np.mean(dgs) > 0 and np.mean(dms) > 0
            grid_table.append({"cutoff": c, "mean_delta_g": float(np.mean(dgs)),
                                "mean_delta_m": float(np.mean(dms)), "positive_direction": bool(positive)})
            print(f"cutoff={c}: delta_g={np.mean(dgs):.5f} delta_m={np.mean(dms):.5f} positive={positive}")
        result["cutoff_grid"] = grid_table
        n_positive = sum(1 for r in grid_table if r["positive_direction"])
        result["cutoffs_positive"] = f"{n_positive}/{len(grid_table)}"
        print(f"cutoffs with positive direction: {n_positive}/{len(grid_table)}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-json", type=str, required=True,
                         help="path template with {seed} placeholder, e.g. .../fourier_gaussian_seed{seed}.json")
    parser.add_argument("--mask-json", type=str, required=True)
    parser.add_argument("--gm-json", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--cutoff", type=str, default="0.4181")
    parser.add_argument("--cutoff-grid", type=str, default="0.2,0.3,0.4181,0.55,0.7")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="eval_results/fourier_replication_analysis.json")
    args = parser.parse_args()
    main(args)
