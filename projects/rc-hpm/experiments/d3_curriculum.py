"""D3 — certification curriculum without hardness. Runs the new certified
selectors on the D2 band rungs at alpha=0.40, evaluates the four pre-registered
gates (safety/utility/mechanism/anti-hardness) -> branch C1/C2/C3.

Reuses no_mine/naive/supcon/rc_hard/cert_random_k from d2_utility_band.json +
d2_preflight.json; runs cert_conf_easy/mid_band/diverse/curriculum fresh.

Writes results/d3_results.json + results/d3_verdict.json.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm.ladder import Rung, calibrate_rung                    # noqa: E402
from rc_hpm.curriculum import train_d3_arm                        # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.40
N_SEEDS = 10
NEW_ARMS = ["cert_conf_easy", "cert_mid_band", "cert_diverse",
            "cert_curriculum"]
_CAL = {}


def _calib(spec):
    key = (spec["K"], spec["sigma"], spec["a"])
    if key not in _CAL:
        _CAL[key] = calibrate_rung(Rung(K=spec["K"], sigma=spec["sigma"],
                                        a=spec["a"]), seed=0, alpha=ALPHA)
    return _CAL[key]


def job(args):
    spec, arm, seed = args
    rung = Rung(K=spec["K"], sigma=spec["sigma"], a=spec["a"])
    q_fn, calib = _calib(spec)
    r = train_d3_arm(rung, arm, seed, q_fn, calib, alpha=ALPHA)
    r["spec"] = spec
    return r


def main():
    t0 = time.time()
    band = json.load(open(os.path.join(RESULTS, "d2_utility_band.json")))
    pf = json.load(open(os.path.join(RESULTS, "d2_preflight.json")))
    pf_by = {r["tag"]: r for r in pf["rungs"]}
    band_tags = list(band["table"].keys())
    specs = {t: pf_by[t]["spec"] for t in band_tags}

    jobs = [(specs[t], arm, s) for t in band_tags for arm in NEW_ARMS
            for s in range(N_SEEDS)]
    print(f"{len(jobs)} D3 jobs", flush=True)
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(job, jobs)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(jobs)} ({(time.time()-t0)/60:.0f}m)",
                      flush=True)
    with open(os.path.join(RESULTS, "d3_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # ---- assemble table (reuse D2 arms) ----
    def d3_mean(tag, arm):
        v = [r["probe_acc"] for r in rows if r["rung"] == tag
             and r["arm"] == arm and r["probe_acc"] is not None]
        return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) \
            if v else (None, None)

    table = {}
    for t in band_tags:
        bt = band["table"][t]
        row = dict(rho_tail=bt["rho_tail"], H=bt["H"], S=bt["S"],
                   no_mine=bt["no_mine"], supcon=bt["supcon"],
                   naive=bt.get("naive_neg"), rc_hard=bt.get("rc_hpm"),
                   cert_random_k=bt.get("cert_random_k"))
        for arm in NEW_ARMS:
            m, se = d3_mean(t, arm)
            row[arm] = m
            row[arm + "_se"] = se
        # no_mine seed-SE from pre-flight accs
        nm = pf_by[t]["no_mine_accs"]
        row["no_mine_se"] = float(np.std(nm, ddof=1) / np.sqrt(len(nm)))
        # realized risk for curriculum
        cr = [r for r in rows if r["rung"] == t and r["arm"] == "cert_curriculum"]
        if cr:
            row["curriculum_risk_max"] = float(np.max(
                [r["realized_risk_max"] for r in cr]))
        table[t] = row

    # ---- pre-registered gates ----
    def beats(a_mean, a_se, b_mean, b_se, factor=2.0):
        if None in (a_mean, b_mean):
            return False
        pooled = np.sqrt((a_se or 0) ** 2 + (b_se or 0) ** 2)
        return bool(a_mean - b_mean > factor * pooled)

    gate_util, gate_mech, gate_anti, gate_safe = {}, {}, {}, {}
    for t, row in table.items():
        cc, cc_se = row["cert_curriculum"], row["cert_curriculum_se"]
        cr, cr_se = row["cert_random_k"], None
        # cert_random_k SE from D2 (approx via band table not stored) -> use
        # curriculum SE as conservative proxy
        cr_se = cc_se
        gate_util[t] = beats(cc, cc_se, row["no_mine"], row["no_mine_se"])
        gate_mech[t] = beats(cc, cc_se, cr, cr_se)
        gate_anti[t] = bool(row["rc_hard"] is not None and cc is not None
                            and row["rc_hard"] < cc)
        gate_safe[t] = bool(row.get("curriculum_risk_max", 1.0) <= ALPHA)

    # also: best SIMPLE certified selector vs no_mine (for C2)
    simple_util = {}
    for arm in ["cert_random_k", "cert_conf_easy", "cert_mid_band",
                "cert_diverse"]:
        wins = []
        for t, row in table.items():
            m = row.get(arm)
            se = row.get(arm + "_se") or row["no_mine_se"]
            wins.append(beats(m, se, row["no_mine"], row["no_mine_se"]))
        simple_util[arm] = int(sum(wins))

    g_safety = all(gate_safe.values())
    g_utility = any(gate_util.values())
    g_mechanism = any(gate_mech.values())
    g_anti = all(gate_anti.values())

    if g_utility and g_mechanism and g_safety:
        branch = "C1 (BEST): certification curriculum produces UTILITY"
    elif max(simple_util.values()) > 0:
        branch = ("C2 (MIDDLE): certified pairs a cheap safe regularizer; "
                  f"best simple selector wins on "
                  f"{max(simple_util.values())} rungs "
                  f"({max(simple_util, key=simple_util.get)})")
    else:
        branch = ("C3 (WORST): certification is ONLY a safety guardrail -> "
                  "bounded-harm / risk-dial paper, stop chasing utility")

    verdict = dict(alpha=ALPHA, table=table,
                   gates=dict(safety=g_safety, utility=g_utility,
                              mechanism=g_mechanism, anti_hardness=g_anti,
                              utility_per_rung=gate_util,
                              mechanism_per_rung=gate_mech),
                   simple_selector_wins=simple_util, branch=branch,
                   wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "d3_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print("\nGATES:", verdict["gates"]["safety"], verdict["gates"]["utility"],
          verdict["gates"]["mechanism"], verdict["gates"]["anti_hardness"])
    print("simple selector wins:", simple_util)
    print("BRANCH:", branch)
    for t, row in table.items():
        print(f"{t} rho={row['rho_tail']:.2f}: nomine={row['no_mine']:.3f} "
              f"rand={row['cert_random_k']:.3f} easy={row['cert_conf_easy']} "
              f"mid={row['cert_mid_band']} div={row['cert_diverse']} "
              f"curr={row['cert_curriculum']} rc_hard={row['rc_hard']:.3f}")


if __name__ == "__main__":
    main()
