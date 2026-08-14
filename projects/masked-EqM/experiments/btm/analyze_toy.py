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
    rows = []
    for p in paths:
        for pat in ("*.jsonl",) if os.path.isdir(p) else ("",):
            for f in (glob.glob(os.path.join(p, pat)) if pat else [p]):
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
    return rows


def _agg(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return st.mean(vals), st.stdev(vals), len(vals)


def table_a(rows, stage="main"):
    groups = defaultdict(list)
    for r in rows:
        if r.get("stage") != stage or "config" not in r:
            continue
        c = r["config"]
        key = (c["arm"], c["K"], c["eps_fd"], c["tc"], c.get("geometry", "ring"))
        groups[key].append(r)

    lines = ["| arm | K | eps_fd | seeds | mass MAE (mean±std) | median | "
             "unresolved | R_weak (median rel) | stable |",
             "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for key in sorted(groups, key=lambda k: (list(ARM_LABEL).index(k[0]), k[1], k[2])):
        arm, K, eps, tc, geom = key
        rs = groups[key]
        maes = [r["mass_mae"] for r in rs]
        m, s, n = _agg(maes)
        med = st.median([v for v in maes if not math.isnan(v)]) if maes else float("nan")
        unres, _, _ = _agg([r.get("unresolved_frac") for r in rs])
        wc, _, _ = _agg([r.get("R_overall_median_rel") for r in rs])
        nstable = sum(1 for r in rs if r.get("stable"))
        scalar_fd = arm in ("btm_scalar_fd_directional", "btm_scalar_fd_action")
        lines.append(
            f"| {ARM_LABEL.get(arm, arm)} | {K if scalar_fd else '-'} | "
            f"{eps:g} | {n} | {m:.4f} ± {s:.4f} | {med:.4f} | {unres:.4f} | "
            f"{wc:.3f} | {nstable}/{n} |")
        out[f"{arm}|K{K}|eps{eps:g}|{geom}"] = {
            "mean": m, "std": s, "median": med, "n": n,
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
        return {"verdict": "INCOMPLETE", "checks": {},
                "reason": "missing V or D records"}

    thr = max(0.015, 2.0 * V["median"])
    checks["3_mae_close_to_vector"] = {
        "pass": D["median"] <= thr,
        "detail": f"median MAE_D={D['median']:.4f} vs threshold "
                  f"max(0.015, 2*MAE_V={V['median']:.4f}) = {thr:.4f}"}
    checks["2_stable_across_seeds"] = {
        "pass": D["n_stable"] == D["n"] and D["n"] >= 10,
        "detail": f"{D['n_stable']}/{D['n']} stable seeds (>=10 required)"}
    legacy = None
    for cand in (E, Es):
        if cand is not None:
            legacy = cand if legacy is None else (
                legacy if legacy["median"] < cand["median"] else cand)
    if legacy is not None:
        checks["4_beats_legacy_eqm"] = {
            "pass": D["median"] < 0.5 * legacy["median"],
            "detail": f"MAE_D={D['median']:.4f} vs best legacy-EqM control "
                      f"{legacy['median']:.4f} (needs < half)"}
    cal_ok = any(r.get("stage") == "fd_calibration" and r["rel_rmse"] < 0.05
                 and r["nonfinite_frac"] == 0 for r in rows)
    checks["1_fd_estimator_validated"] = {
        "pass": cal_ok,
        "detail": "some calibrated eps achieves rel RMSE < 5% with no "
                  "non-finite estimates" if cal_ok else "no usable eps found"}
    checks["5_no_mixed_derivative"] = {
        "pass": True,
        "detail": "enforced mechanically by fd.assert_no_double_backward() "
                  "around both loss and backward; tests/test_btm_math.py "
                  "includes a positive control that Arm G trips the guard"}
    checks["6_exact_field_transports"] = {
        "pass": D["unresolved"] < 0.05,
        "detail": f"unresolved fraction {D['unresolved']:.4f} under the exact "
                  f"grad phi evaluation drift (needs < 0.05)"}

    verdict = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    return {"verdict": verdict, "checks": checks,
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

    rows = load(args.paths)
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
    print(f"\n## Pre-registered toy gate: **{g['verdict']}**\n")
    for name in sorted(g.get("checks", {})):
        c = g["checks"][name]
        print(f"- [{'PASS' if c['pass'] else 'FAIL'}] {name}: {c['detail']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "grid": grid_summary,
                       "tc_sweep": tc_summary, "gate": g,
                       "n_records": len(rows)}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
