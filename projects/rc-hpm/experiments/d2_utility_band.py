"""D2 utility band (deviation D5) — the REAL B1 test, after the linchpin showed
supply is threshold-limited not structural.

Pre-flight measured S at alpha_0=0.10 only, which throttled certification to ~0
and faked an H-S anticorrelation. The linchpin (d2_supply_alpha.json) shows
the high-headroom rungs have abundant certified-hard supply at the
pre-registered looser endpoints (alpha=0.20, 0.40). So the band IS testable.

This runs the full UTILITY arm set at alpha=0.40 (P2 tertiary endpoint 4*alpha_0)
on every high-H rung — uniform alpha so the band regression over rho_tail is
unconfounded by alpha. no_mine + supcon reused from pre-flight (mining-free,
alpha-independent). Then the pre-registered C1-C4 regression -> B1/B2/B3.

Writes results/d2_utility_band.json + appends to d2_ladder_results for readout.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm.ladder import Rung, calibrate_rung, supply_S, train_arm  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
BAND_ALPHA = 0.40                       # P2 tertiary endpoint (4 * alpha_0)
N_SEEDS = 10
MINE_ARMS = ["rc_hpm", "rc_neg_only", "cert_random_k", "rince", "naive_neg"]
_CAL = {}


def _calib(spec, alpha):
    key = (spec["K"], spec["sigma"], spec["a"], alpha)
    if key not in _CAL:
        _CAL[key] = calibrate_rung(Rung(K=spec["K"], sigma=spec["sigma"],
                                        a=spec["a"]), seed=0, alpha=alpha)
    return _CAL[key]


def job(args):
    spec, arm, seed = args
    rung = Rung(K=spec["K"], sigma=spec["sigma"], a=spec["a"])
    q_fn = calib = None
    if arm in ("rc_hpm", "rc_neg_only", "cert_random_k"):
        q_fn, calib = _calib(spec, BAND_ALPHA)
    r = train_arm(rung, arm, seed, alpha=BAND_ALPHA, q_fn=q_fn, calib=calib)
    r.update(rung=rung.tag(), spec=spec)
    return r


def main():
    t0 = time.time()
    pf = json.load(open(os.path.join(RESULTS, "d2_preflight.json")))
    lp = json.load(open(os.path.join(RESULTS, "d2_supply_alpha.json")))
    S_band = {r["tag"]: r["S_by_alpha"].get(str(BAND_ALPHA), {}).get("S", 0.0)
              for r in lp["rungs"]}

    # band rungs: high-H (in linchpin set) with S>0.10 at BAND_ALPHA
    band = [r for r in pf["rungs"] if r["tag"] in S_band
            and S_band[r["tag"]] > 0.10 and r["H"] > 2 * r["noise_floor"]]
    print("band rungs:", [(r["tag"], round(r["rho_tail"], 2), round(r["H"], 3),
                           round(S_band[r["tag"]], 2)) for r in band], flush=True)

    jobs = []
    for r in band:
        for arm in MINE_ARMS:
            jobs += [(r["spec"], arm, s) for s in range(N_SEEDS)]
    print(f"{len(jobs)} jobs at alpha={BAND_ALPHA}", flush=True)

    rows = []
    with ProcessPoolExecutor(8) as ex:
        for i, rr in enumerate(ex.map(job, jobs)):
            rows.append(rr)
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(jobs)} ({(time.time()-t0)/60:.0f}m)",
                      flush=True)

    # assemble per-rung gaps + regression points (reuse no_mine from pre-flight)
    nm = {r["tag"]: r["no_mine_accs"] for r in pf["rungs"]}
    sc = {r["tag"]: np.mean(r["supcon_accs"]) for r in pf["rungs"]}
    points = []
    table = {}
    for r in band:
        tag = r["tag"]
        rc = {x["seed"]: x["probe_acc"] for x in rows
              if x["rung"] == tag and x["arm"] == "rc_hpm"}
        nmv = nm[tag]
        gaps = [rc[s] - nmv[s] for s in rc if s < len(nmv)]
        for s in rc:
            if s < len(nmv):
                points.append((tag, r["H"], S_band[tag], r["rho_tail"], s,
                               rc[s] - nmv[s]))
        arm_means = {}
        for arm in MINE_ARMS:
            v = [x["probe_acc"] for x in rows
                 if x["rung"] == tag and x["arm"] == arm]
            arm_means[arm] = float(np.mean(v)) if v else None
        table[tag] = dict(rho_tail=r["rho_tail"], H=r["H"], S=S_band[tag],
                          no_mine=float(np.mean(nmv)), supcon=float(sc[tag]),
                          gap_mean=float(np.mean(gaps)),
                          gap_se=float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))),
                          **arm_means)

    # pre-registered regression: gap ~ 1 + H + S + rho + rho^2
    def design(pts):
        X = np.array([[1, p[1], p[2], p[3], p[3] ** 2] for p in pts])
        y = np.array([p[5] for p in pts])
        return X, y
    out = dict(band_alpha=BAND_ALPHA, table=table, n_band_rungs=len(band))
    if len(band) >= 3 and points:
        X, y = design(points)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        b0, b1, b2, q1, q2 = [float(v) for v in beta]
        tags = sorted({p[0] for p in points})
        by = {t: [p for p in points if p[0] == t] for t in tags}
        rng = np.random.default_rng(0)
        q2bs, peaks = [], []
        for _ in range(2000):
            samp = []
            for t in rng.choice(tags, len(tags), replace=True):
                pts = by[t]
                idx = rng.integers(0, len(pts), len(pts))
                samp += [pts[i] for i in idx]
            try:
                bb, *_ = np.linalg.lstsq(*design(samp), rcond=None)
            except Exception:                              # noqa: BLE001
                continue
            q2bs.append(bb[4])
            if abs(bb[4]) > 1e-9:
                peaks.append(-bb[3] / (2 * bb[4]))
        q2lo, q2hi = [float(v) for v in np.percentile(q2bs, [2.5, 97.5])]
        peak = -q1 / (2 * q2) if abs(q2) > 1e-9 else None
        cell = {t: (float(np.mean([p[5] for p in by[t]])),
                    float(np.std([p[5] for p in by[t]], ddof=1) /
                          np.sqrt(len(by[t])))) for t in tags}
        best = max(cell, key=lambda t: cell[t][0])
        C1 = bool(q2 < 0 and q2hi < 0)
        C2 = bool(peak is not None and 0.10 < peak < 0.50)
        C3 = bool(cell[best][0] > 2 * cell[best][1])
        C4 = bool(b1 >= 0 and b2 >= 0)
        out["regression"] = dict(
            beta=dict(b0=b0, b1_H=b1, b2_S=b2, q1=q1, q2=q2),
            q2_ci95=[q2lo, q2hi], peak=peak, cells=cell,
            best_cell=dict(tag=best, mean=cell[best][0], se=cell[best][1]),
            C1=C1, C2=C2, C3=C3, C4=C4,
            B1_band_found=bool(C1 and C2 and C3 and C4),
            B2_no_band=bool(all(cell[t][0] <= 2 * cell[t][1] for t in tags)))
    out["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "d2_utility_band.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out.get("regression", {"error": "too few rungs"}),
                     indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
