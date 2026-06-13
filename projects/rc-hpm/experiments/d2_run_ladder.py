"""D2 step 4b — train the arm grid on pre-flighted rungs.

Per d2_preflight.json:
  full rungs:  rc_hpm / rc_neg_only / cert_random_k / rince / naive_neg
               (10 seeds; no_mine + supcon accs reused from pre-flight)
  failed rungs: naive_neg only (cheap band-ends)
  designated rung extras: naive_pos (FP-pull probe) + alpha sweep
               {0.05, 0.20, 0.40} on rc_hpm
  foil (at designated rung): {concentrated, diffuse} x {no_mine, rc_hpm,
               rince}, marginal crossing rate 0.15

Calibration is memoized per (rung, alpha) inside each worker process
(closures are unpicklable; recalibration is deterministic at seed 0).

Writes results/d2_ladder_results.json.
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
from experiments.d2_preflight import (cifar_patch, make_cifar_rung)  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FULL_ARMS = ["rc_hpm", "rc_neg_only", "cert_random_k", "rince", "naive_neg"]
N_SEEDS = 10
ALPHA_SWEEP = (0.05, 0.20, 0.40)

_CAL_CACHE = {}


def _get_rung(spec):
    if spec.get("kind") == "cifar":
        cifar_patch()
        return make_cifar_rung()
    return Rung(K=spec["K"], sigma=spec["sigma"], a=spec["a"])


def _get_calib(spec, alpha):
    key = (json.dumps(spec, sort_keys=True), alpha)
    if key not in _CAL_CACHE:
        _CAL_CACHE[key] = calibrate_rung(_get_rung(spec), seed=0, alpha=alpha)
    return _CAL_CACHE[key]


def job(args):
    spec, arm, seed, alpha, noise_mode = args
    rung = _get_rung(spec)
    q_fn = calib = None
    if arm in ("rc_hpm", "rc_neg_only", "cert_random_k"):
        q_fn, calib = _get_calib(spec, alpha)
    vn = make_view_noise(rung, 0.15, noise_mode) if noise_mode else None
    r = train_arm(rung, arm, seed, alpha=alpha, q_fn=q_fn, calib=calib,
                  view_noise=vn)
    r.update(rung=rung.tag(), noise_mode=noise_mode,
             spec=spec)
    return r


def main():
    t0 = time.time()
    pf = json.load(open(os.path.join(RESULTS, "d2_preflight.json")))
    jobs = []
    for r in pf["rungs"]:
        spec = r["spec"]
        if r["full_arms"]:
            for arm in FULL_ARMS:
                jobs += [(spec, arm, s, 0.10, None) for s in range(N_SEEDS)]
        else:
            jobs += [(spec, "naive_neg", s, 0.10, None)
                     for s in range(N_SEEDS)]
        if r["tag"] == pf["designated"]:
            jobs += [(spec, "naive_pos", s, 0.10, None)
                     for s in range(N_SEEDS)]
            for a in ALPHA_SWEEP:
                jobs += [(spec, "rc_hpm", s, a, None) for s in range(N_SEEDS)]
            for mode in ("concentrated", "diffuse"):
                for arm in ("no_mine", "rc_hpm", "rince"):
                    jobs += [(spec, arm, s, 0.10, mode)
                             for s in range(N_SEEDS)]

    print(f"{len(jobs)} jobs", flush=True)
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(job, jobs)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(jobs)} done "
                      f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    with open(os.path.join(RESULTS, "d2_ladder_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("wall_minutes", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
