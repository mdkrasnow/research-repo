"""E0.1 — synthetic contrastive toy, calibration validity. Gate G0.

20 seeds x alpha grid {0.05 stretch, 0.10 primary, 0.20 loose} x
{learned, oracle (+), random (-)} gates. Realized risk on 200 fresh
population batches per seed.

G0 (preregistration.md + Amendment A1), evaluated at alpha_0 = 0.10:
 (a)  exceedances (either risk: mean - 2SE > alpha) among certifying seeds
      <= 4  [binomial Bin(20, delta=0.1) upper at 0.05]
 (a') pooled mean + 2 SE_pooled <= alpha + delta(1-alpha), each risk
 (b)  random gate: every seed ABORT or abstention >= 0.95; risk controlled
      whenever mining occurred
 informativeness: >= 15/20 learned-gate seeds certify at alpha_0

Writes results/e0_1_results.json + results/e0_1_verdict.json.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm.toy import ToyConfig, run_seed   # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

N_SEEDS = 20
ALPHAS = [0.05, 0.10, 0.20]
ALPHA_PRIMARY = 0.10
DELTA = 0.10
EXCEED_CUTOFF = 4          # smallest c with P(Bin(20, 0.1) >= c) < 0.05 is 5


def one(args):
    seed, alpha, gate = args
    r = run_seed(seed, ToyConfig(), alpha, gate_kind=gate)
    d = asdict(r)
    d.update(seed=seed, alpha=alpha, gate=gate)
    return d


def main():
    t0 = time.time()
    jobs = [(s, a, g) for g in ("learned", "oracle", "random")
            for a in ALPHAS for s in range(N_SEEDS)]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs))
    with open(os.path.join(RESULTS, "e0_1_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    def sel(gate, alpha):
        return [r for r in rows if r["gate"] == gate and r["alpha"] == alpha]

    def exceed(r, alpha):
        return (not r["aborted"]) and (
            r["realized_minus"] - 2 * r["se_minus"] > alpha or
            r["realized_plus"] - 2 * r["se_plus"] > alpha)

    verdict = {"gate": "G0", "alpha_primary": ALPHA_PRIMARY, "criteria": {}}
    crit = verdict["criteria"]

    for alpha in ALPHAS:
        L = sel("learned", alpha)
        certs = [r for r in L if not r["aborted"]]
        exc = sum(1 for r in L if exceed(r, alpha))
        pooled = {}
        for k, sk in (("realized_minus", "se_minus"), ("realized_plus", "se_plus")):
            if certs:
                means = np.array([r[k] for r in certs])
                pm = means.mean()
                se_p = means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else 0
                pooled[k] = dict(pooled_mean=float(pm), upper=float(pm + 2 * se_p),
                                 bound=alpha + DELTA * (1 - alpha),
                                 ok=bool(pm + 2 * se_p <= alpha + DELTA * (1 - alpha)))
            else:
                pooled[k] = dict(pooled_mean=None, ok=True)  # vacuous (all abort)
        crit[f"alpha={alpha}"] = dict(
            certified=len(certs), exceedances=exc,
            a_ok=bool(exc <= EXCEED_CUTOFF),
            a_prime=pooled,
            informative=bool(len(certs) >= 15) if alpha == ALPHA_PRIMARY else None,
            mean_abstention=float(np.mean([r["abstention"] for r in certs]))
            if certs else None,
            lams=sorted({str(r["lam"]) for r in certs}))

    # (b) random-gate negative control, primary alpha
    Rb = sel("random", ALPHA_PRIMARY)
    b_ok = all(r["aborted"] or r["abstention"] >= 0.95 or not r["mined_any"]
               or not exceed(r, ALPHA_PRIMARY) for r in Rb)
    b_strict = all(r["aborted"] or r["abstention"] >= 0.95 for r in Rb)
    crit["random_gate"] = dict(
        aborted=sum(1 for r in Rb if r["aborted"]),
        abst95=sum(1 for r in Rb if r["abstention"] >= 0.95),
        risk_ok_when_mined=b_ok, fully_vacuous=b_strict)

    # oracle positive control sanity
    Ob = sel("oracle", ALPHA_PRIMARY)
    crit["oracle_gate"] = dict(
        certified=sum(1 for r in Ob if not r["aborted"]),
        exceedances=sum(1 for r in Ob if exceed(r, ALPHA_PRIMARY)))

    p = crit[f"alpha={ALPHA_PRIMARY}"]
    passed = (p["a_ok"] and p["informative"]
              and p["a_prime"]["realized_minus"]["ok"]
              and p["a_prime"]["realized_plus"]["ok"]
              and b_ok
              and crit["oracle_gate"]["exceedances"] == 0)
    verdict["passed"] = bool(passed)
    verdict["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "e0_1_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))
    print("\nG0 PASS" if passed else "\nG0 FAIL")


if __name__ == "__main__":
    main()
