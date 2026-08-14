"""Aggregate the five-atom campaign into the report tables and the gate verdict.

Emits Table A (toy transport), Table B (FD numerical accuracy), the gradient
noise table, and an explicit PASS/FAIL against the pre-registered gate:

    MAE_D <= max(0.015, 2 * MAE_V)

plus the four ancillary gate conditions.  Prints markdown; also writes a JSON
summary next to the input.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

ARM_LABEL = {
    "btm_vector": "V  vector BTM",
    "btm_scalar_exact": "G  scalar exact",
    "btm_scalar_action_exact": "A  action exact",
    "btm_scalar_fd_directional": "D  FD directional",
    "btm_scalar_fd_action": "F  FD action",
    "eqm_legacy_vector": "-  EqM legacy (vec)",
    "eqm_legacy_scalar": "-  EqM legacy (scalar)",
}


def load(paths):
    """Return (rows, bad_lines).

    A single partially-flushed line from an in-flight run used to abort the
    entire gate evaluation via an uncaught JSONDecodeError.  Bad lines are now
    counted and REPORTED (silently skipping them would hide truncation).
    """
    rows, bad = [], []
    for p in paths:
        for pat in ("*.jsonl",) if os.path.isdir(p) else ("",):
            for f in (glob.glob(os.path.join(p, pat)) if pat else [p]):
                with open(f) as fh:
                    for i, line in enumerate(fh, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            bad.append(f"{f}:{i}: {exc.msg}")
    return rows, bad


def _agg(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return st.mean(vals), st.stdev(vals), len(vals)


def _seed_of(r):
    return r.get("config", {}).get("seed", r.get("seed"))


def table_a(rows, stage="main"):
    groups = defaultdict(list)
    for r in rows:
        if r.get("stage") != stage or "config" not in r:
            continue
        c = r["config"]
        key = (c["arm"], c["K"], c["eps_fd"], c["tc"], c.get("geometry", "ring"))
        groups[key].append(r)

    lines = ["| arm | K | eps_fd | tc | geometry | seeds | crashed | "
             "mass MAE (mean±std) | median | unresolved | "
             "R_weak (median rel) | stable |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for key in sorted(groups, key=lambda k: (list(ARM_LABEL).index(k[0]), k[1],
                                             k[2], k[3])):
        arm, K, eps, tc, geom = key
        rs = groups[key]
        # Deduplicate by seed: `n` used to count RECORDS, so passing the same
        # path twice doubled it and could satisfy an "n >= 10 seeds" gate with
        # five real seeds.  Last record per seed wins.
        by_seed = {}
        for i, r in enumerate(rs):
            by_seed.setdefault(_seed_of(r) if _seed_of(r) is not None
                               else f"_anon{i}", r)
        seeds = list(by_seed.values())
        n_seeds = len(seeds)
        maes = [r.get("mass_mae") for r in seeds]
        # A crashed seed is written as {mass_mae: NaN, stable: False}; dropping
        # NaN made it vanish from BOTH the count and the stability denominator,
        # so 12 seeds with 2 crashes reported "10/10 stable".
        crashed = [r for r in seeds
                   if r.get("mass_mae") is None
                   or (isinstance(r.get("mass_mae"), float)
                       and math.isnan(r["mass_mae"]))
                   or r.get("error")]
        n_crashed = len(crashed)
        m, s, n_finite = _agg(maes)
        finite = [v for v in maes if v is not None and not math.isnan(v)]
        med = st.median(finite) if finite else float("nan")
        unres, _, _ = _agg([r.get("unresolved_frac") for r in seeds])
        wc, _, _ = _agg([r.get("R_overall_median_rel") for r in seeds])
        nstable = sum(1 for r in seeds if r.get("stable"))
        scalar_fd = arm in ("btm_scalar_fd_directional", "btm_scalar_fd_action")
        lines.append(
            f"| {ARM_LABEL.get(arm, arm)} | {K if scalar_fd else '-'} | "
            f"{eps:g} | {tc:g} | {geom} | {n_seeds} | {n_crashed} | "
            f"{m:.4f} ± {s:.4f} | {med:.4f} | {unres:.4f} | "
            f"{wc:.3f} | {nstable}/{n_seeds} |")
        # tc varies BY DESIGN on the tc_sweep stage; omitting it from the key
        # collapsed every tc but one, rendering a 40x MAE difference as two
        # apparently-duplicate rows.
        out[f"{arm}|K{K}|eps{eps:g}|tc{tc:g}|{geom}"] = {
            "mean": m, "std": s, "median": med,
            "n": n_seeds, "n_finite": n_finite, "n_crashed": n_crashed,
            "unresolved": unres, "weak_residual": wc,
            "n_stable": nstable, "tc": tc, "geometry": geom}
    return "\n".join(lines), out


def table_b(rows):
    cal = [r for r in rows if r.get("stage") == "fd_calibration"]
    if not cal:
        return "(no calibration records)", {}
    lines = ["| train step | eps_fd | h (mean) | K | rel RMSE | corr | cosine | "
             "bias | var | cancel ratio | nonfinite |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(cal, key=lambda r: (r["train_step"], r["eps_fd"])):
        lines.append(
            f"| {r['train_step']} | {r['eps_fd']:g} | {r['h_mean']:.3g} | "
            f"{r['K']} | {r['rel_rmse']:.3g} | {r.get('corr', float('nan')):.5f} | "
            f"{r.get('cosine', float('nan')):.5f} | {r['bias']:.3g} | "
            f"{r['var']:.3g} | {r['cancel_ratio']:.3g} | {r['nonfinite_frac']:.2g} |")
    return "\n".join(lines), {"n_rows": len(cal)}


def table_noise(rows):
    gn = [r for r in rows if r.get("stage") == "gradnoise"]
    if not gn:
        return "(no gradient-noise records)", {}
    lines = ["| arm | K | train step | E|g|^2 | |E g|^2 | noise scale | "
             "mean pairwise cos | SNR |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(gn, key=lambda r: (r["arm"], r["K"], r["train_step"])):
        lines.append(
            f"| {ARM_LABEL.get(r['arm'], r['arm'])} | {r['K']} | "
            f"{r['train_step']} | {r['E_norm_sq']:.4g} | {r['mean_norm_sq']:.4g} | "
            f"{r['noise_scale']:.4g} | {r['mean_pairwise_cosine']:.4f} | "
            f"{r['snr']:.4g} |")
    return "\n".join(lines), {"n_rows": len(gn)}


def gate(summary, rows):
    """The six pre-registered Arm-D conditions."""
    def pick(arm, prefer_K=None):
        cands = {k: v for k, v in summary.items() if k.startswith(arm + "|")}
        if not cands:
            return None, None
        if prefer_K is not None:
            for k, v in cands.items():
                if f"|K{prefer_K}|" in k:
                    return k, v
        best = min(cands.items(), key=lambda kv: kv[1]["median"])
        return best

    kV, V = pick("btm_vector")
    kD, D = pick("btm_scalar_fd_directional")
    kG, G = pick("btm_scalar_exact")
    kE, E = pick("eqm_legacy_vector")
    kEs, Es = pick("eqm_legacy_scalar")

    checks = {}
    if V is None or D is None:
        return {"verdict": "INCONCLUSIVE", "checks": {},
                "missing": ["btm_vector" if V is None else None,
                            "btm_scalar_fd_directional" if D is None else None],
                "reason": "missing V or D records"}

    thr = max(0.015, 2.0 * V["median"])
    checks["3_mae_close_to_vector"] = {
        "pass": D["median"] <= thr, "measured": True,
        "detail": f"median MAE_D={D['median']:.4f} vs threshold "
                  f"max(0.015, 2*MAE_V={V['median']:.4f}) = {thr:.4f}"}
    # Crashed seeds are seeds.  They belong in the denominator; excluding them
    # let a run with crashes report full stability.
    checks["2_stable_across_seeds"] = {
        "pass": D["n_stable"] == D["n"] and D["n"] >= 10
                and D.get("n_crashed", 0) == 0,
        "measured": True,
        "detail": f"{D['n_stable']}/{D['n']} stable distinct seeds "
                  f"({D.get('n_crashed', 0)} crashed) (>=10 seeds, 0 crashes "
                  f"required)"}
    legacy = None
    for cand in (E, Es):
        if cand is not None:
            legacy = cand if legacy is None else (
                legacy if legacy["median"] < cand["median"] else cand)
    if legacy is not None:
        checks["4_beats_legacy_eqm"] = {
            "pass": D["median"] < 0.5 * legacy["median"], "measured": True,
            "detail": f"MAE_D={D['median']:.4f} vs best legacy-EqM control "
                      f"{legacy['median']:.4f} (needs < half)"}
    else:
        # AGENTS.md mandates a negative control.  Its ABSENCE must not silently
        # delete the condition and leave a PASS on 5 of 6 -- it is unmeasured,
        # which makes the whole verdict INCONCLUSIVE.
        checks["4_beats_legacy_eqm"] = {
            "pass": False, "measured": False,
            "detail": "NOT MEASURED: the legacy-EqM negative control "
                      "(eqm_legacy_vector / eqm_legacy_scalar) produced no "
                      "records, so this pre-registered condition could not be "
                      "evaluated"}
    cal_ok = any(r.get("stage") == "fd_calibration" and r["rel_rmse"] < 0.05
                 and r["nonfinite_frac"] == 0 for r in rows)
    checks["1_fd_estimator_validated"] = {
        "pass": cal_ok, "measured": True,
        "detail": "some calibrated eps achieves rel RMSE < 5% with no "
                  "non-finite estimates" if cal_ok else "no usable eps found"}
    # Not computed from these records at all -- it is a static code-level
    # guarantee asserted elsewhere.  Labelling it "PASS" here overstated what
    # this gate evaluation measured.
    checks["5_no_mixed_derivative"] = {
        "pass": True, "measured": False, "external": True,
        "detail": "NOT MEASURED BY THIS ANALYZER (verified externally): "
                  "enforced mechanically by "
                  "fd.assert_no_double_backward() around both loss and "
                  "backward; tests/test_btm_math.py includes a positive "
                  "control that Arm G trips the guard"}
    checks["6_exact_field_transports"] = {
        "pass": D["unresolved"] < 0.05, "measured": True,
        "detail": f"unresolved fraction {D['unresolved']:.4f} under the exact "
                  f"grad phi evaluation drift (needs < 0.05)"}

    # "external" = the condition is a static code-level guarantee asserted by
    # the test suite, not by these records; it is reported as not-measured here
    # but does not by itself make the data inconclusive.  "unmeasured" = a
    # pre-registered condition whose DATA is missing -> INCONCLUSIVE.
    unmeasured = sorted(k for k, c in checks.items()
                        if not c.get("measured", True)
                        and not c.get("external", False))
    measured = [c for c in checks.values() if c.get("measured", True)]
    if not all(c["pass"] for c in measured):
        verdict = "FAIL"
    elif unmeasured:
        # Every measured condition passed but at least one pre-registered
        # condition was never evaluated -> not a PASS.
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "checks": checks, "unmeasured": unmeasured,
            "not_measured_here": sorted(k for k, c in checks.items()
                                        if not c.get("measured", True)),
            "arm_keys": {"V": kV, "D": kD, "G": kG,
                         "legacy_vec": kE, "legacy_scalar": kEs},
            "medians": {k: (v["median"] if v else None) for k, v in
                        (("V", V), ("D", D), ("G", G), ("A", pick(
                            "btm_scalar_action_exact")[1]),
                         ("F", pick("btm_scalar_fd_action")[1]),
                         ("legacy_vec", E), ("legacy_scalar", Es))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, bad = load(args.paths)
    ta, summary = table_a(rows)
    tb, _ = table_b(rows)
    tn, _ = table_noise(rows)
    tgrid, grid_summary = table_a(rows, stage="fd_grid")
    ttc, tc_summary = table_a(rows, stage="tc_sweep")
    g = gate(summary, rows)

    print(f"\n## Table A — toy transport (main comparison, {len(rows)} records)\n")
    print(ta)
    print("\n## Table B — FD numerical accuracy\n")
    print(tb)
    print("\n## Gradient-estimator noise\n")
    print(tn)
    print("\n## FD (K, eps) grid\n")
    print(tgrid)
    print("\n## tc sweep\n")
    print(ttc)
    if bad:
        print(f"\n## WARNING — {len(bad)} unparseable JSONL line(s)\n")
        for b in bad[:20]:
            print(f"- {b}")
        if len(bad) > 20:
            print(f"- ... and {len(bad) - 20} more")

    print(f"\n## Pre-registered toy gate: **{g['verdict']}**\n")
    for name in sorted(g.get("checks", {})):
        c = g["checks"][name]
        if not c.get("measured", True):
            state = "NOT MEASURED"
        else:
            state = "PASS" if c["pass"] else "FAIL"
        print(f"- [{state}] {name}: {c['detail']}")
    if g.get("unmeasured"):
        print(f"\nVerdict is INCONCLUSIVE because these pre-registered "
              f"conditions have no data: {', '.join(g['unmeasured'])}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "grid": grid_summary,
                       "tc_sweep": tc_summary, "gate": g,
                       "n_records": len(rows), "bad_lines": bad}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
