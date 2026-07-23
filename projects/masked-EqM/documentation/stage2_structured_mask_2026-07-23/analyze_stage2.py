"""Preregistered Stage 2 structured-mask gate analysis.

Consumes the 27 frozen outputs produced by
eval_stage2_structured_mask_full.sbatch:
  3 training seeds x 3 arms x {Fourier recovery, structured-mask
  recovery, FID}.

The five gate conditions are those registered in
stage2_structured_mask_proposal_2026-07-22.md. The proposal describes
condition 5 qualitatively ("structured-mask recovery is strong"). Before
looking at results, this script operationalizes it as: both trained arms
(structured-mask specialist and Gaussian+structured-mask treatment) have
lower mean recovery LPIPS than Gaussian-only, and each does so in all
3 matched seeds. No post-result magnitude threshold is introduced.
"""

import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
SEEDS = [0, 1, 2]
FAMILIES = ["gaussian", "structmask", "gm"]
N_BOOT = 10_000
RNG_SEED = 20260723


def load_json(name):
    path = os.path.join(RAW, name)
    with open(path) as f:
        return json.load(f)


def load_fourier(family, seed):
    data = load_json(f"stage2_{family}_seed{seed}_fourier.json")
    cutoff = data["per_cutoff"]["0.1"]
    per_image = {
        int(row["index"]): {"lpips": float(row["lpips"]), "mse": float(row["mse"])}
        for row in cutoff["per_image"]
    }
    return {
        "mean_lpips": float(cutoff["mean_lpips"]),
        "mean_mse": float(cutoff["mean_mse"]),
        "per_image": per_image,
    }


def load_recovery(family, seed):
    data = load_json(f"stage2_{family}_seed{seed}_maskrecovery.json")
    return {
        "mean_lpips": float(data["mean_lpips"]),
        "mean_masked_mse": float(data["mean_masked_mse"]),
    }


def load_fid(family, seed):
    return float(load_json(f"stage2_{family}_seed{seed}_fid.json")["fid"])


def hierarchical_bootstrap(seed_deltas, rng):
    n_seeds = len(seed_deltas)
    reps = np.empty(N_BOOT)
    for b in range(N_BOOT):
        chosen_seeds = rng.integers(0, n_seeds, size=n_seeds)
        means = []
        for seed_idx in chosen_seeds:
            values = seed_deltas[seed_idx]
            chosen_images = rng.integers(0, len(values), size=len(values))
            means.append(values[chosen_images].mean())
        reps[b] = np.mean(means)
    return reps


def two_sided_bootstrap_p(reps):
    return float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())))


def holm_adjust(pvals):
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = (len(pvals) - rank) * pvals[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def main():
    data = {
        family: {
            seed: {
                "fourier": load_fourier(family, seed),
                "recovery": load_recovery(family, seed),
                "fid": load_fid(family, seed),
            }
            for seed in SEEDS
        }
        for family in FAMILIES
    }

    reference_indices = None
    for family in FAMILIES:
        for seed in SEEDS:
            indices = set(data[family][seed]["fourier"]["per_image"])
            if reference_indices is None:
                reference_indices = indices
            else:
                assert indices == reference_indices, (
                    f"Frozen-manifest mismatch: {family} seed{seed}"
                )
    image_indices = sorted(reference_indices)
    assert len(image_indices) == 1024

    delta_g_by_seed = []
    delta_s_by_seed = []
    seed_rows = []
    for seed in SEEDS:
        g = data["gaussian"][seed]["fourier"]["per_image"]
        s = data["structmask"][seed]["fourier"]["per_image"]
        gm = data["gm"][seed]["fourier"]["per_image"]
        g_arr = np.array([g[i]["lpips"] for i in image_indices])
        s_arr = np.array([s[i]["lpips"] for i in image_indices])
        gm_arr = np.array([gm[i]["lpips"] for i in image_indices])
        delta_g = g_arr - gm_arr
        delta_s = s_arr - gm_arr
        delta_g_by_seed.append(delta_g)
        delta_s_by_seed.append(delta_s)
        seed_rows.append(
            {
                "seed": seed,
                "gaussian_lpips": float(g_arr.mean()),
                "structmask_lpips": float(s_arr.mean()),
                "gm_lpips": float(gm_arr.mean()),
                "delta_g": float(delta_g.mean()),
                "delta_structmask": float(delta_s.mean()),
                "gm_beats_both": bool(
                    gm_arr.mean() < g_arr.mean() and gm_arr.mean() < s_arr.mean()
                ),
                "win_rate_vs_gaussian": float((gm_arr < g_arr).mean()),
                "win_rate_vs_structmask": float((gm_arr < s_arr).mean()),
            }
        )

    rng = np.random.default_rng(RNG_SEED)
    reps_g = hierarchical_bootstrap(delta_g_by_seed, rng)
    reps_s = hierarchical_bootstrap(delta_s_by_seed, rng)
    raw_p = [two_sided_bootstrap_p(reps_g), two_sided_bootstrap_p(reps_s)]
    holm_p = holm_adjust(raw_p)

    mean_delta_g = float(np.mean([values.mean() for values in delta_g_by_seed]))
    mean_delta_s = float(np.mean([values.mean() for values in delta_s_by_seed]))
    ci_g = np.percentile(reps_g, [2.5, 97.5]).tolist()
    ci_s = np.percentile(reps_s, [2.5, 97.5]).tolist()
    seeds_beating_both = sum(row["gm_beats_both"] for row in seed_rows)
    win_rate_g = float(np.mean([row["win_rate_vs_gaussian"] for row in seed_rows]))
    win_rate_s = float(np.mean([row["win_rate_vs_structmask"] for row in seed_rows]))

    family_summary = {}
    for family in FAMILIES:
        family_summary[family] = {
            "fourier_lpips_mean": float(
                np.mean([data[family][s]["fourier"]["mean_lpips"] for s in SEEDS])
            ),
            "fourier_lpips_by_seed": [
                data[family][s]["fourier"]["mean_lpips"] for s in SEEDS
            ],
            "fourier_mse_mean": float(
                np.mean([data[family][s]["fourier"]["mean_mse"] for s in SEEDS])
            ),
            "structured_recovery_lpips_mean": float(
                np.mean([data[family][s]["recovery"]["mean_lpips"] for s in SEEDS])
            ),
            "structured_recovery_lpips_by_seed": [
                data[family][s]["recovery"]["mean_lpips"] for s in SEEDS
            ],
            "structured_recovery_mse_mean": float(
                np.mean(
                    [data[family][s]["recovery"]["mean_masked_mse"] for s in SEEDS]
                )
            ),
            "fid_mean": float(np.mean([data[family][s]["fid"] for s in SEEDS])),
            "fid_by_seed": [data[family][s]["fid"] for s in SEEDS],
        }

    recovery_specialist_wins = [
        data["structmask"][s]["recovery"]["mean_lpips"]
        < data["gaussian"][s]["recovery"]["mean_lpips"]
        for s in SEEDS
    ]
    recovery_treatment_wins = [
        data["gm"][s]["recovery"]["mean_lpips"]
        < data["gaussian"][s]["recovery"]["mean_lpips"]
        for s in SEEDS
    ]
    recovery_strong = (
        family_summary["structmask"]["structured_recovery_lpips_mean"]
        < family_summary["gaussian"]["structured_recovery_lpips_mean"]
        and family_summary["gm"]["structured_recovery_lpips_mean"]
        < family_summary["gaussian"]["structured_recovery_lpips_mean"]
        and all(recovery_specialist_wins)
        and all(recovery_treatment_wins)
    )

    fid_delta = (
        family_summary["gm"]["fid_mean"] - family_summary["gaussian"]["fid_mean"]
    )
    gates = {
        "1_mean_delta_g_at_least_0.010": bool(mean_delta_g >= 0.010),
        "2_gm_beats_both_parents_3of3": bool(seeds_beating_both == 3),
        "3_win_rate_vs_gaussian_over_0.75": bool(win_rate_g > 0.75),
        "4_fid_within_plus_15": bool(fid_delta <= 15.0),
        "5_structured_recovery_strong": bool(recovery_strong),
    }

    output = {
        "num_images": len(image_indices),
        "n_bootstrap": N_BOOT,
        "rng_seed": RNG_SEED,
        "family_summary": family_summary,
        "seed_rows": seed_rows,
        "delta_g": {
            "mean": mean_delta_g,
            "ci95": ci_g,
            "p_raw": raw_p[0],
            "p_holm": float(holm_p[0]),
        },
        "delta_structmask": {
            "mean": mean_delta_s,
            "ci95": ci_s,
            "p_raw": raw_p[1],
            "p_holm": float(holm_p[1]),
        },
        "seeds_beating_both": f"{seeds_beating_both}/3",
        "win_rate_vs_gaussian": win_rate_g,
        "win_rate_vs_structmask": win_rate_s,
        "fid_delta_gm_minus_gaussian": fid_delta,
        "recovery_specialist_beats_gaussian_by_seed": recovery_specialist_wins,
        "recovery_treatment_beats_gaussian_by_seed": recovery_treatment_wins,
        "gates": gates,
        "overall_gate_pass": bool(all(gates.values())),
    }

    out_path = os.path.join(HERE, "stage2_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("=== Stage 2 structured-mask gate analysis ===")
    for row in seed_rows:
        print(
            f"seed{row['seed']}: G={row['gaussian_lpips']:.4f} "
            f"S={row['structmask_lpips']:.4f} GM={row['gm_lpips']:.4f} "
            f"dG={row['delta_g']:+.4f} beats_both={row['gm_beats_both']} "
            f"winG={row['win_rate_vs_gaussian']:.3f}"
        )
    print(
        f"delta_G={mean_delta_g:+.5f}, CI={ci_g}, Holm p={holm_p[0]:.4f}"
    )
    print(
        f"delta_S={mean_delta_s:+.5f}, CI={ci_s}, Holm p={holm_p[1]:.4f}"
    )
    print(
        f"FID G={family_summary['gaussian']['fid_mean']:.2f}, "
        f"S={family_summary['structmask']['fid_mean']:.2f}, "
        f"GM={family_summary['gm']['fid_mean']:.2f}, "
        f"GM-G={fid_delta:+.2f}"
    )
    print("Gates:")
    for name, passed in gates.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"OVERALL: {'PASS' if output['overall_gate_pass'] else 'FAIL'}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
