"""D2 linchpin check — is certified-hard supply S=0 a threshold artifact or
mechanism? Re-measure S at alpha in {0.10, 0.20, 0.40} on every candidate
rung. If high-headroom rungs stay S<0.10 even at loose alpha, the
headroom/supply anticorrelation is structural (band genuinely empty), not a
gate-strictness artifact.

Generator-only + one calibration per (rung, alpha). Writes
results/d2_supply_alpha.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm.ladder import Rung, calibrate_rung, supply_S, rho_tail   # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    t0 = time.time()
    pf = json.load(open(os.path.join(RESULTS, "d2_preflight.json")))
    out = []
    for r in pf["rungs"]:
        if r["kind"] != "toy":
            continue
        spec = r["spec"]
        rung = Rung(K=spec["K"], sigma=spec["sigma"], a=spec["a"])
        row = dict(tag=r["tag"], rho_tail=r["rho_tail"], H=r["H"],
                   H_bin=r["H_bin"], S_by_alpha={})
        for alpha in (0.10, 0.20, 0.40):
            try:
                q_fn, calib = calibrate_rung(rung, alpha=alpha)
                S = supply_S(rung, q_fn, calib)
                row["S_by_alpha"][str(alpha)] = dict(
                    S=S, aborted=bool(calib.aborted),
                    lam=list(calib.lam) if not calib.aborted else None)
            except Exception as e:                          # noqa: BLE001
                row["S_by_alpha"][str(alpha)] = dict(error=str(e))
        out.append(row)
        print(f"{r['tag']} H={r['H']:.3f}: " + " ".join(
            f"a{a}->S={row['S_by_alpha'][str(a)].get('S', 'err'):.3f}"
            if isinstance(row['S_by_alpha'][str(a)].get('S'), float)
            else f"a{a}->{row['S_by_alpha'][str(a)]}"
            for a in (0.10, 0.20, 0.40)), flush=True)

    # mechanism test: among high/med-H rungs (H > 0.04), does ANY reach S>0.10
    # at ANY alpha?
    hi_h = [r for r in out if r["H"] > 0.04]
    any_supply = any(
        max((r["S_by_alpha"][a].get("S", 0) for a in r["S_by_alpha"]),
            default=0) > 0.10 for r in hi_h)
    verdict = dict(rungs=out, high_H_rungs=len(hi_h),
                   any_high_H_reaches_supply=bool(any_supply),
                   conclusion=("supply is THRESHOLD-limited (loosen alpha "
                               "to open band)" if any_supply else
                               "headroom/supply ANTICORRELATION is STRUCTURAL "
                               "— band empty at all alpha (B2 destination)"),
                   wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "d2_supply_alpha.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print("\n", verdict["conclusion"])


if __name__ == "__main__":
    main()
