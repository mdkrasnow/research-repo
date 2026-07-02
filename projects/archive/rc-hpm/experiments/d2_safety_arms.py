"""D2 deviation D3 — safety arms on supply-bearing rungs.

The pre-flight 'full_arms' gate bundles (a) H>2xfloor [a UTILITY precondition]
with (b) S>0.10 [a SAFETY precondition]. B3 (H-B' retention) and the RINCE
foil are SAFETY questions: they need certified-hard SUPPLY (b), not headroom
(a). No rung passed (a)+(b) (the H-S anticorrelation finding), so under the
strict reading B3/foil are untestable. This deviation runs the safety arm set
on rungs passing (b) ALONE, where naive mining demonstrably damages — exactly
the regime the pre-registered B3/foil were written for.

Designated SAFETY rung = supply-bearing rung with the largest naive damage
(K40_s1.2_a0.8: S=0.573, naive damage 0.395). Logged in deviations.md (D3).

Arms (10 seeds): no_mine / naive_neg / rc_hpm / rc_neg_only / cert_random_k /
rince / naive_pos (FP-pull) + foil {concentrated, diffuse} x {no_mine, rc_hpm,
rince} + alpha frontier {0.05,0.10,0.20,0.40} on rc_hpm.

Writes results/d2_safety_results.json.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm.ladder import (Rung, calibrate_rung, make_view_noise,
                           train_arm)                              # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
N_SEEDS = 10
SAFETY_ARMS = ["no_mine", "naive_neg", "rc_hpm", "rc_neg_only",
               "cert_random_k", "rince", "naive_pos"]
ALPHA_SWEEP = (0.05, 0.10, 0.20, 0.40)
_CAL = {}


def _calib(spec, alpha):
    key = (spec["K"], spec["sigma"], spec["a"], alpha)
    if key not in _CAL:
        _CAL[key] = calibrate_rung(Rung(K=spec["K"], sigma=spec["sigma"],
                                        a=spec["a"]), seed=0, alpha=alpha)
    return _CAL[key]


def job(args):
    spec, arm, seed, alpha, mode = args
    rung = Rung(K=spec["K"], sigma=spec["sigma"], a=spec["a"])
    q_fn = calib = None
    if arm in ("rc_hpm", "rc_neg_only", "cert_random_k"):
        q_fn, calib = _calib(spec, alpha)
    vn = make_view_noise(rung, 0.15, mode) if mode else None
    r = train_arm(rung, arm, seed, alpha=alpha, q_fn=q_fn, calib=calib,
                  view_noise=vn)
    r.update(rung=rung.tag(), noise_mode=mode, spec=spec)
    return r


def pick_safety_rung(pf, rows):
    nm = {r["tag"]: np.mean(r["no_mine_accs"]) for r in pf["rungs"]}
    best = None
    for r in pf["rungs"]:
        if r["kind"] != "toy" or r["S"] <= 0.10:
            continue
        nv = [x["probe_acc"] for x in rows
              if x["rung"] == r["tag"] and x["arm"] == "naive_neg"]
        if not nv:
            continue
        dmg = nm[r["tag"]] - np.mean(nv)
        if best is None or dmg > best[1]:
            best = (r, dmg)
    return best


def main():
    t0 = time.time()
    pf = json.load(open(os.path.join(RESULTS, "d2_preflight.json")))
    band = json.load(open(os.path.join(RESULTS, "d2_ladder_results.json")))
    picked = pick_safety_rung(pf, band)
    if picked is None:
        print("no supply-bearing rung with damage — safety arms N/A")
        json.dump(dict(skipped=True, reason="no S>0.10 rung"),
                  open(os.path.join(RESULTS, "d2_safety_results.json"), "w"))
        return
    rung_meta, damage = picked
    spec = rung_meta["spec"]
    print(f"safety rung {rung_meta['tag']} S={rung_meta['S']:.3f} "
          f"naive_damage={damage:.3f}", flush=True)

    jobs = []
    for arm in SAFETY_ARMS:
        jobs += [(spec, arm, s, 0.10, None) for s in range(N_SEEDS)]
    for a in ALPHA_SWEEP:
        jobs += [(spec, "rc_hpm", s, a, None) for s in range(N_SEEDS)]
    for mode in ("concentrated", "diffuse"):
        for arm in ("no_mine", "rc_hpm", "rince"):
            jobs += [(spec, arm, s, 0.10, mode) for s in range(N_SEEDS)]

    print(f"{len(jobs)} jobs", flush=True)
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(job, jobs)):
            rows.append(r)
            if (i + 1) % 30 == 0:
                print(f"{i + 1}/{len(jobs)} ({(time.time()-t0)/60:.0f}m)",
                      flush=True)

    out = dict(safety_rung=rung_meta["tag"], spec=spec,
               naive_damage=damage, S=rung_meta["S"],
               no_mine_accs=rung_meta["no_mine_accs"], rows=rows,
               wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "d2_safety_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # quick H-B' + foil readout
    def acc(arm, alpha=0.10, mode=None):
        v = [r["probe_acc"] for r in rows if r["arm"] == arm
             and r["alpha"] == alpha and r.get("noise_mode") == mode]
        return float(np.mean(v)) if v else None
    nm = float(np.mean(rung_meta["no_mine_accs"]))
    nv, rc, rn = acc("naive_neg"), acc("rc_hpm"), acc("rc_neg_only")
    denom = rc - nv if rc and nv else None
    ret = (rn - nv) / denom if denom and abs(denom) > 1e-9 else None
    print(f"\nH-B': no_mine={nm:.3f} naive={nv:.3f} rc={rc:.3f} "
          f"rc_neg_only={rn:.3f} retention={ret}")
    print(f"FP-pull (naive_pos)={acc('naive_pos'):.3f} vs no_mine={nm:.3f}")
    for m in ("concentrated", "diffuse"):
        print(f"foil/{m}: no_mine={acc('no_mine',mode=m)} "
              f"rc={acc('rc_hpm',mode=m)} rince={acc('rince',mode=m)}")
    print("alpha frontier:", {a: acc("rc_hpm", alpha=a) for a in ALPHA_SWEEP})


if __name__ == "__main__":
    main()
