"""D2 steps 2+3: generator knob search (no arm training) + gamma-probe
validation against known-K synthetic rungs.

Step 2 procedure (preregistration-d2.md):
 1. rho_tail for every (K, sigma, a) combo — generator-only.
 2. keep combos within +/-0.05 of a rho_tail target; select <=3 candidates
    per target spanning predicted-H range (ordering proxy: K then a).
Step 3: gamma probe must recover true K on >=3 of 4 known-K rungs, else
gamma is "uninstrumented" for all of D2.

Writes results/d2_knob_search.json.
"""
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm.ladder import Rung, rho_tail, gamma_probe, teacher_embed, draw  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
KS = (4, 10, 20, 40)
SIGMAS = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
AS = (0.0, 0.8, 1.6)
RHO_TARGETS = (0.05, 0.20, 0.40, 0.60)


def one(combo):
    K, s, a = combo
    r = Rung(K=K, sigma=s, a=a)
    return dict(K=K, sigma=s, a=a, rho_tail=rho_tail(r))


def gamma_validation():
    out = {}
    hits = 0
    for K in KS:
        r = Rung(K=K, sigma=0.8, a=0.0)
        rng = np.random.default_rng(3)
        x, _ = draw(r, 1500, rng)
        K_hat, gam, head = gamma_probe(teacher_embed(r, x), true_K=K)
        ok = (K_hat == K)
        hits += int(ok)
        out[f"K={K}"] = dict(K_hat=K_hat, gamma_at_true_K=gam, hit=ok,
                             eigs_head=head)
    out["validated"] = bool(hits >= 3)
    out["hits"] = hits
    return out


def main():
    t0 = time.time()
    combos = list(itertools.product(KS, SIGMAS, AS))
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(one, combos))

    candidates = {}
    for tgt in RHO_TARGETS:
        near = [r for r in rows if abs(r["rho_tail"] - tgt) <= 0.05]
        # span predicted-H range: sort by (K, a) — more classes / heavier tail
        # predicted to open headroom; take low/mid/high of that ordering
        near.sort(key=lambda r: (r["K"], r["a"]))
        if not near:
            candidates[str(tgt)] = []
            continue
        picks = sorted({0, len(near) // 2, len(near) - 1})
        candidates[str(tgt)] = [near[i] for i in picks]

    gv = gamma_validation()
    out = dict(all_combos=rows, candidates=candidates,
               gamma_validation=gv,
               gamma_status="instrumented" if gv["validated"]
               else "UNINSTRUMENTED (excluded from analysis per prereg)",
               wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "d2_knob_search.json"), "w") as f:
        json.dump(out, f, indent=2)
    for tgt, cs in candidates.items():
        print(f"rho_tail target {tgt}: " + "; ".join(
            f"K{c['K']}/s{c['sigma']}/a{c['a']} (rho={c['rho_tail']:.3f})"
            for c in cs) if cs else f"rho_tail target {tgt}: UNREACHABLE",
            flush=True)
    print("gamma:", out["gamma_status"], gv)


if __name__ == "__main__":
    main()
