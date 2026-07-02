"""D2 step 5 — confirmatory readout against pre-registered branches
(preregistration-d2.md: C1-C4, B1/B2/B3; everything per-rung descriptive).

PRIMARY: gap_{rs} = acc_RC - acc_no-mine (paired by seed) regressed on
[1, H, S, rho_tail, rho_tail^2]; cluster bootstrap (rungs, then seeds) for
the C1 CI on q2.

Writes results/d2_verdict.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main():
    t0 = time.time()
    pf = json.load(open(os.path.join(RESULTS, "d2_preflight.json")))
    rows = json.load(open(os.path.join(RESULTS, "d2_ladder_results.json")))
    rung_meta = {r["tag"]: r for r in pf["rungs"]}
    desig = pf["designated"]

    def accs(tag, arm, alpha=0.10, mode=None):
        return {r["seed"]: r["probe_acc"] for r in rows
                if r["rung"] == tag and r["arm"] == arm
                and r["alpha"] == alpha and r.get("noise_mode") == mode}

    # ---- assemble paired gaps on full rungs ----
    full = [r for r in pf["rungs"] if r["full_arms"]]
    points = []          # (tag, H, S, rho, seed, gap)
    for r in full:
        tag = r["tag"]
        rc = accs(tag, "rc_hpm")
        nm = {s: v for s, v in enumerate(r["no_mine_accs"])}
        for s in rc:
            if s in nm:
                points.append((tag, r["H"], r["S"], r["rho_tail"], s,
                               rc[s] - nm[s]))
    tags = sorted({p[0] for p in points})

    verdict = {"designated": desig, "n_full_rungs": len(full),
               "descriptive": {}, "primary": {}}

    # descriptive per-rung table (all arms)
    for r in pf["rungs"]:
        tag = r["tag"]
        d = dict(rho_tail=r["rho_tail"], S=r["S"], H=r["H"],
                 H_bin=r["H_bin"], gamma=r.get("gamma"),
                 full_arms=r["full_arms"],
                 no_mine=float(np.mean(r["no_mine_accs"])),
                 supcon=float(np.mean(r["supcon_accs"])))
        for arm in ("rc_hpm", "rc_neg_only", "cert_random_k", "rince",
                    "naive_neg", "naive_pos"):
            a = accs(tag, arm)
            if a:
                d[arm] = float(np.mean(list(a.values())))
                d[arm + "_sd"] = float(np.std(list(a.values()), ddof=1))
        verdict["descriptive"][tag] = d

    # ---- PRIMARY regression ----
    if len(tags) >= 3 and points:
        def design(pts):
            X = np.array([[1.0, p[1], p[2], p[3], p[3] ** 2] for p in pts])
            y = np.array([p[5] for p in pts])
            return X, y
        X, y = design(points)
        beta = fit_ols(X, y)
        b0, b1, b2, q1, q2 = beta
        rng = np.random.default_rng(0)
        by_tag = {t: [p for p in points if p[0] == t] for t in tags}
        q2_bs, peak_bs = [], []
        for _ in range(2000):
            sample = []
            for t in rng.choice(tags, len(tags), replace=True):
                pts = by_tag[t]
                idx = rng.integers(0, len(pts), len(pts))
                sample += [pts[i] for i in idx]
            Xb, yb = design(sample)
            try:
                bb = fit_ols(Xb, yb)
            except Exception:                              # noqa: BLE001
                continue
            q2_bs.append(bb[4])
            if abs(bb[4]) > 1e-12:
                peak_bs.append(-bb[3] / (2 * bb[4]))
        q2_lo, q2_hi = np.percentile(q2_bs, [2.5, 97.5])
        peak = -q1 / (2 * q2) if abs(q2) > 1e-12 else None
        # C3: best cell gap
        cell_stats = {}
        for t in tags:
            g = [p[5] for p in by_tag[t]]
            cell_stats[t] = (float(np.mean(g)),
                             float(np.std(g, ddof=1) / np.sqrt(len(g))))
        best_tag = max(cell_stats, key=lambda t: cell_stats[t][0])
        best_mean, best_se = cell_stats[best_tag]
        C1 = bool(q2 < 0 and q2_hi < 0)
        C2 = bool(peak is not None and 0.10 < peak < 0.50)
        C3 = bool(best_mean > 2 * best_se)
        C4 = bool(b1 >= 0 and b2 >= 0)
        B1 = bool(C1 and C2 and C3 and C4)
        B2 = bool(all(cell_stats[t][0] <= 2 * cell_stats[t][1] for t in tags))
        verdict["primary"] = dict(
            beta=dict(b0=float(b0), b1_H=float(b1), b2_S=float(b2),
                      q1=float(q1), q2=float(q2)),
            q2_ci95=[float(q2_lo), float(q2_hi)], fitted_peak=peak,
            cell_gaps={t: cell_stats[t] for t in tags},
            best_cell=dict(tag=best_tag, mean=best_mean, se=best_se),
            C1=C1, C2=C2, C3=C3, C4=C4, B1_band_found=B1,
            B2_no_band=B2)
    else:
        verdict["primary"] = dict(error="insufficient full rungs for the "
                                  "regression", B1_band_found=False,
                                  B2_no_band=None)

    # ---- H-B' retention (B3) at designated rung ----
    if desig:
        r = rung_meta[desig]
        nm = float(np.mean(r["no_mine_accs"]))
        nv = accs(desig, "naive_neg")
        rc = accs(desig, "rc_hpm")
        rn = accs(desig, "rc_neg_only")
        if nv and rc and rn:
            nv_m = float(np.mean(list(nv.values())))
            damage = nm - nv_m
            pooled_sd = float(np.std(r["no_mine_accs"], ddof=1))
            valid = damage > 2 * pooled_sd
            rc_m = float(np.mean(list(rc.values())))
            rn_m = float(np.mean(list(rn.values())))
            denom = rc_m - nv_m
            retention = (rn_m - nv_m) / denom if abs(denom) > 1e-9 else None
            verdict["hb_prime"] = dict(
                valid=bool(valid), damage=damage, retention=retention,
                B3_full_gate_loadbearing=bool(valid and retention is not None
                                              and retention < 0.80))
        # FP-pull probe
        np_ = accs(desig, "naive_pos")
        if np_:
            verdict["fp_pull_probe"] = dict(
                naive_pos=float(np.mean(list(np_.values()))),
                no_mine=nm,
                damage=float(nm - np.mean(list(np_.values()))))
        # alpha frontier
        verdict["alpha_frontier"] = {
            str(a): (float(np.mean(list(accs(desig, "rc_hpm", alpha=a)
                                        .values())))
                     if accs(desig, "rc_hpm", alpha=a) else None)
            for a in (0.05, 0.10, 0.20, 0.40)}
        # RINCE foil
        foil = {}
        for mode in ("concentrated", "diffuse"):
            sub = {arm: accs(desig, arm, mode=mode)
                   for arm in ("no_mine", "rc_hpm", "rince")}
            if all(sub.values()):
                rcv = np.array(list(sub["rc_hpm"].values()))
                riv = np.array(list(sub["rince"].values()))
                pooled = float(np.sqrt((rcv.var(ddof=1) + riv.var(ddof=1)) / 2))
                abst = [r_.get("abstention") for r_ in rows
                        if r_["rung"] == desig and r_["arm"] == "rc_hpm"
                        and r_.get("noise_mode") == mode
                        and r_.get("abstention") is not None]
                foil[mode] = dict(
                    no_mine=float(np.mean(list(sub["no_mine"].values()))),
                    rc=float(rcv.mean()), rince=float(riv.mean()),
                    rc_minus_rince=float(rcv.mean() - riv.mean()),
                    pooled_sd=pooled,
                    rc_abstention=float(np.mean(abst)) if abst else None)
        if foil:
            clean_abst = [r_.get("abstention") for r_ in rows
                          if r_["rung"] == desig and r_["arm"] == "rc_hpm"
                          and r_.get("noise_mode") is None
                          and r_.get("abstention") is not None]
            ca = float(np.mean(clean_abst)) if clean_abst else None
            conc, diff = foil.get("concentrated"), foil.get("diffuse")
            sep = bool(conc and conc["rc_minus_rince"] > 2 * conc["pooled_sd"])
            visible = bool(conc and ca is not None and
                           conc["rc_abstention"] is not None and
                           conc["rc_abstention"] - ca > 0.05)
            matched = bool(diff and abs(diff["rc_minus_rince"]) <=
                           2 * diff["pooled_sd"])
            foil["clean_abstention"] = ca
            foil["foil_pass"] = bool(sep and visible and matched)
            verdict["rince_foil"] = foil

    # ---- branch summary ----
    pr = verdict["primary"]
    verdict["branches"] = dict(
        B1=pr.get("B1_band_found", False),
        B2=pr.get("B2_no_band"),
        B3=verdict.get("hb_prime", {}).get("B3_full_gate_loadbearing"))
    verdict["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "d2_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict["branches"], indent=2))
    print(json.dumps(verdict["primary"], indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
