"""E0.0 bug-injection suite — gate G-1 (entry condition for everything).

Each deliberately re-introduced bug must be DETECTED: realized/true risk
inflated beyond the clean envelope, or caught loudly by a validity checker.
Clean run must pass with no checker noise. Writes results/e0_0_verdict.json.

Run: python3 experiments/e0_0_bug_injection.py
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm import core                      # noqa: E402
from rc_hpm.toy import (ToyConfig, Injections, run_seed, make_population,
                        draw, teacher_embed, aug_view, train_gate,
                        lam_grid)            # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

DELTA_R = 0.05
ALPHA_PRIMARY = 0.10

SMALL = ToyConfig(n_gate=400, n_fold=1000, n_batch=32, m=15, m_fit=15,
                  n_pairs=400, eval_batches=150)
DEGEN = ToyConfig(K=2, n_gate=200, n_fold=400, n_batch=3, m=33, m_fit=20,
                  n_pairs=400, eval_batches=200)


def _exceed(r, alpha):
    return (not r.aborted) and (
        r.realized_minus - 2 * r.se_minus > alpha or
        r.realized_plus - 2 * r.se_plus > alpha)


def _detected(r, alpha):
    """Generic detection: checker fired, monitor alarmed, or risk exceeded."""
    return (r.checker_error is not None) or (r.monitor_alarms > 0) or \
        _exceed(r, alpha)


# ---------------------------------------------------------------------------
# T1 fold reuse — statistical false-certification test on a knife-edge alpha
# ---------------------------------------------------------------------------

def _t1_pilot_alpha():
    """alpha = 0.95 x best achievable max(L-, L+) over the grid on the FIXED
    population -> no lambda truly satisfies both risks; any certification is
    a false certification."""
    cfg = SMALL
    rng = np.random.default_rng(99)
    pop = make_population(0, cfg)
    xg, lg = draw(pop, cfg.n_gate, rng)
    eg = teacher_embed(pop, xg)
    q_fn = train_gate(eg, lg, 0)
    xf, lf = draw(pop, 4000, rng)
    ef = teacher_embed(pop, xf)
    rho_hat, rho_plus, rho_amb = core.fit_rhos(ef, lf, q_fn, rng, n_pairs=4000)
    best = np.inf
    for lam in lam_grid(cfg):
        Lm, Lp = [], []
        for _ in range(120):
            xb, lb = draw(pop, cfg.n_batch, rng)
            eb = teacher_embed(pop, xb)
            ab = teacher_embed(pop, aug_view(xb, rng, cfg.sigma_aug))
            qb = core.q_matrix(q_fn, eb, lb)
            mined = core.mine_batch(eb, qb, lam[0], lam[1], rho_hat, rho_plus,
                                    cfg.k_plus, cfg.k_minus)
            ys = (lb[:, None] == lb[None, :]).astype(float)
            lm, lp = core.batch_risks(mined, ys, rho_hat, rho_plus, rho_amb,
                                      (eb * ab).sum(1))
            Lm.append(lm); Lp.append(lp)
        best = min(best, max(np.mean(Lm), np.mean(Lp)))
    return 0.95 * best


def _t1_one(args):
    seed, alpha, injected = args
    inj = Injections(fold_reuse=True) if injected else Injections()
    r = run_seed(seed, SMALL, alpha, gate_kind="learned", inj=inj, pop_seed=0)
    # false certification = certified anything at an unachievable alpha
    return (not r.aborted)


def t1_fold_reuse():
    """Data-flow detection (preregistration E0.0 amendment A2): 1-D isotonic
    fold reuse has near-zero realized-risk signature at toy scale; the
    structural violation is caught by the structural rho/path-vs-test-split
    overlap guard inside calibrate_ltt."""
    clean = run_seed(0, SMALL, ALPHA_PRIMARY, pop_seed=0)
    inj = run_seed(0, SMALL, ALPHA_PRIMARY, pop_seed=0,
                   inj=Injections(fold_reuse=True))
    ok_clean = clean.checker_error is None
    ok_inj = inj.checker_error is not None and "fold reuse" in inj.checker_error
    return dict(name="fold_reuse", passed=bool(ok_clean and ok_inj),
                inj_err=inj.checker_error)


# ---------------------------------------------------------------------------
# T2/T7/T8 — pipeline-level statistical/checker detections
# ---------------------------------------------------------------------------

def _detect_one(args):
    seed, inj_kwargs, gate_kind, cfg, alpha, ev = args
    r = run_seed(seed, cfg, alpha, gate_kind=gate_kind,
                 inj=Injections(**inj_kwargs), eval_batches=ev)
    if r.aborted and r.checker_error is None:
        return None        # LTT abort: vacuous, uninformative for detection
    return _detected(r, alpha)


def _detect_many(inj_kwargs, n_seeds, need, cfg=None, name=""):
    cfg = cfg or ToyConfig(eval_batches=80)
    args = [(s, inj_kwargs, "learned", cfg, ALPHA_PRIMARY, 80)
            for s in range(n_seeds)]
    with ProcessPoolExecutor(8) as ex:
        out = [o for o in ex.map(_detect_one, args) if o is not None]
    det = int(np.sum(out))
    return dict(name=name, passed=bool(det >= need and len(out) >= need),
                detected=det, informative_seeds=len(out), n_seeds=n_seeds)


# ---------------------------------------------------------------------------
# T3/T4/T5/T9 — loud checker detections (deterministic)
# ---------------------------------------------------------------------------

def t3_partition_bentkus():
    r = run_seed(0, SMALL, ALPHA_PRIMARY, inj=Injections(partition_bentkus=True),
                 pop_seed=0)
    ok = r.checker_error is not None and "Bentkus" in r.checker_error
    return dict(name="partition_bentkus", passed=bool(ok), err=r.checker_error)


def t4_wprime_mismatch():
    r = run_seed(0, ToyConfig(eval_batches=10), ALPHA_PRIMARY,
                 inj=Injections(wprime_mismatch=True))
    ok = r.checker_error is not None and "pinned" in r.checker_error
    return dict(name="wprime_mismatch", passed=bool(ok), err=r.checker_error)


def t5_skip_degenerate():
    clean = run_seed(0, DEGEN, 0.20, gate_kind="oracle", eval_batches=200)
    inj = run_seed(0, DEGEN, 0.20, gate_kind="oracle", eval_batches=200,
                   inj=Injections(skip_degenerate=True))
    ok_clean = (not clean.aborted) and clean.checker_error is None
    ok_inj = inj.aborted and inj.checker_error is not None and \
        "NaN" in inj.checker_error
    return dict(name="skip_degenerate", passed=bool(ok_clean and ok_inj),
                clean_aborted=clean.aborted, inj_err=inj.checker_error)


def t9_overlapping_batches():
    rng = np.random.default_rng(0)
    raised_overlap = False
    try:
        batches = [np.array([0, 1, 2]), np.array([2, 3, 4])]
        core.check_disjoint_batches(batches, pool_mode=False)
    except core.ValidityError:
        raised_overlap = True
    raised_budget = False
    try:
        core.make_batches(100, 32, 15, rng, "disjoint")   # 480 > 100
    except core.ValidityError:
        raised_budget = True
    return dict(name="overlapping_batches", passed=bool(raised_overlap and
                raised_budget))


# ---------------------------------------------------------------------------
# T6 monitor Bonferroni — unit-level false-alarm counting
# ---------------------------------------------------------------------------

def t6_no_bonferroni(n_checks=400):
    rng = np.random.default_rng(0)
    F_ref = rng.normal(0, 1, 300)
    a_ref = 0.1
    mon_c = core.DriftMonitor(F_ref, a_ref, n_checks_planned=n_checks,
                              bonferroni=True)
    mon_i = core.DriftMonitor(F_ref, a_ref, n_checks_planned=n_checks,
                              bonferroni=False)
    for _ in range(n_checks):
        w = rng.normal(0, 1, 25)
        mon_c.check(w, a_ref)
        mon_i.check(w, a_ref)
    return dict(name="no_bonferroni", passed=bool(mon_c.alarms <= 2 and
                mon_i.alarms >= 6), clean_alarms=mon_c.alarms,
                injected_alarms=mon_i.alarms)


# ---------------------------------------------------------------------------
# T0 clean run
# ---------------------------------------------------------------------------

def _clean_one(seed):
    return run_seed(seed, ToyConfig(eval_batches=80), ALPHA_PRIMARY,
                    gate_kind="learned", eval_batches=80)


def t0_clean(n_seeds=6):
    with ProcessPoolExecutor(6) as ex:
        rs = list(ex.map(_clean_one, range(n_seeds)))
    certs = sum(1 for r in rs if not r.aborted)
    checker_noise = sum(1 for r in rs if r.checker_error is not None)
    alarms = sum(r.monitor_alarms for r in rs)
    exceed = sum(1 for r in rs if _exceed(r, ALPHA_PRIMARY))
    return dict(name="clean", passed=bool(certs >= n_seeds - 1 and
                checker_noise == 0 and exceed == 0 and alarms <= 1),
                certified=certs, checker_noise=checker_noise,
                monitor_alarms=alarms, exceedances=exceed, n_seeds=n_seeds)


def main():
    t0 = time.time()
    results = []
    results.append(t0_clean());                          print(results[-1], flush=True)
    results.append(t3_partition_bentkus());              print(results[-1], flush=True)
    results.append(t4_wprime_mismatch());                print(results[-1], flush=True)
    results.append(t5_skip_degenerate());                print(results[-1], flush=True)
    results.append(t6_no_bonferroni());                  print(results[-1], flush=True)
    results.append(t9_overlapping_batches());            print(results[-1], flush=True)
    results.append(_detect_many(dict(a1prime_noise=0.5), 8, 6,
                                name="a1prime_asymmetry")); print(results[-1], flush=True)
    results.append(_detect_many(dict(live_student_drift=1.0), 8, 7,
                                name="live_student"));    print(results[-1], flush=True)
    results.append(_detect_many(dict(gamma_pooled=True), 3, 3,
                                name="gamma_pooled"));    print(results[-1], flush=True)
    results.append(t1_fold_reuse());                     print(results[-1], flush=True)

    g_minus_1 = all(r["passed"] for r in results)
    verdict = dict(gate="G-1", passed=bool(g_minus_1), tests=results,
                   wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "e0_0_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print("\nG-1 PASS" if g_minus_1 else "\nG-1 FAIL")


if __name__ == "__main__":
    main()
