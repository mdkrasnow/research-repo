"""v17 collector: read all v17_*.json results, aggregate across seeds, evaluate the pre-registered gates,
and emit a markdown report + JSON verdict. Fan-in gates determine whether the next phase is authorized.

Usage: python v17_collect_results.py [--phase 0|1|2|all]
"""
import argparse
import glob
import json
import os
from statistics import mean

import v17_common as C

RD = C.RESULTS_DIR
VALID, DECOY = C.VALID, C.DECOY


def _load(pat):
    out = []
    for p in sorted(glob.glob(os.path.join(RD, pat))):
        with open(p) as fh:
            out.append((os.path.basename(p), json.load(fh)))
    return out


def _agg(dicts, key_path):
    vals = []
    for d in dicts:
        x = d
        ok = True
        for k in key_path:
            if isinstance(x, dict) and k in x:
                x = x[k]
            else:
                ok = False; break
        if ok and isinstance(x, (int, float)):
            vals.append(x)
    return (mean(vals) if vals else float("nan")), vals


# ---------------------------------------------------------------------------
def collect_phase0():
    lines = ["## Phase 0 — Calibration\n"]
    verdict = {}
    for part in ["0A", "0B", "0C", "0D"]:
        files = _load("v17_calib_%s_*.json" % part)
        if not files:
            lines.append("- %s: (no results)\n" % part); verdict[part] = None; continue
        if part == "0B":
            # recompute EqM-primary gate from saved arm numbers (robust to gate-definition changes)
            passes = []
            for _, d in files:
                a = d["arms"]
                ok = (a["KNOWN_ORACLE_MULTI"]["eqm_gap"] < a["BASE"]["eqm_gap"]
                      and a["KNOWN_ORACLE_MULTI"]["eqm_gap"] < a["RANDOM_VALID"]["eqm_gap"]
                      and a["RANDOM_VALID"]["eqm_gap"] < a["RANDOM_WITH_DECOYS"]["eqm_gap"])
                passes.append(ok)
        else:
            passes = [d.get("pass") for _, d in files]
        p = all(passes)
        verdict[part] = p
        extra = ""
        if part == "0C":
            sel = [d.get("selected_anchor") for _, d in files]
            extra = " selected_anchor=%s" % sel
        if part == "0B":
            g = files[0][1].get("gate", {})
            extra = " gate=%s" % {k: v for k, v in g.items()}
        lines.append("- **%s** pass=%s%s\n" % (part, p, extra))
    fanin = all(v for v in verdict.values() if v is not None) and verdict.get("0B") is True
    lines.append("\n**Phase 0 fan-in: %s**\n" % ("PASS" if fanin else "FAIL"))
    return "".join(lines), {"verdict": verdict, "fanin": fanin}


def collect_phase1():
    lines = ["## Phase 1 — Discovery\n"]
    files = _load("v17_discovery_*.json")
    if not files:
        return "## Phase 1 — Discovery\n(no results)\n", {"fanin": False, "tasks": {}}
    # group by task
    bytask = {}
    for _, d in files:
        bytask.setdefault(d["task"], []).append(d)
    tasks_verdict = {}
    for task, ds in sorted(bytask.items()):
        true_f = ds[0]["true_families"]
        def armagg(arm, *kp):
            return _agg([d["arms"][arm] for d in ds if arm in d["arms"]], list(kp))[0]
        learned = "LEARNED_MULTI_POLICY"
        single = "LEARNED_SINGLE_POLICY"
        cov_l = armagg(learned, "coverage", "heldout_coverage")
        cov_s = armagg(single, "coverage", "heldout_coverage")
        cov_rv = armagg("RANDOM_VALID_POLICY", "coverage", "heldout_coverage")
        cov_or = armagg("KNOWN_ORACLE", "coverage", "heldout_coverage")
        cov_base = armagg("BASE_IDENTITY", "coverage", "heldout_coverage")
        decoy_l = armagg(learned, "quality", "decoy_usage")
        truse_l = armagg(learned, "quality", "true_family_usage")
        recall_l = armagg(learned, "quality", "true_family_recall")
        cov_na = armagg("LEARNED_MULTI_NO_ANCHOR", "coverage", "heldout_coverage")
        decoy_na = armagg("LEARNED_MULTI_NO_ANCHOR", "quality", "decoy_usage")
        cov_nd = armagg("LEARNED_MULTI_NO_DIVERSITY", "coverage", "heldout_coverage")
        val_l = armagg(learned, "collapse_validity", "validity_rate")
        val_rd = armagg("RANDOM_WITH_DECOYS", "collapse_validity", "validity_rate")
        val_na = armagg("LEARNED_MULTI_NO_ANCHOR", "collapse_validity", "validity_rate")
        is_multi = "multi" in task
        impossible = "impossible" in task
        is_decoy = "decoy" in task
        # Honest, universal discovery criteria: (1) recovers TRUE families, (2) AVOIDS decoys,
        # (3) stays VALID without being told the valid set, (4) the ANCHOR matters -> NO_ANCHOR ablation
        # loses validity (anchor's job is on-manifold/decoy-avoidance, NOT coverage: per-factor coverage
        # can stay high under brute spreading, but invalidly). Multi adds: multi>single coverage.
        anchor_matters = val_na < val_l - 0.05
        valid_beats_decoyarm = val_l > val_rd + 0.05
        if impossible:
            passed = cov_l < 0.15
            note = "no-hallucination (cov<0.15)"
        elif is_decoy:
            passed = (recall_l >= 0.5 and decoy_l < 0.10 and valid_beats_decoyarm and anchor_matters)
            note = "decoy_pressure: recall>=.5, STRONG decoy-avoid(<.10), valid>decoy-arm, NO_ANCHOR validity worse"
        elif is_multi:
            passed = (recall_l >= 0.66 and decoy_l < 0.34 and cov_l > cov_s + 0.05
                      and valid_beats_decoyarm and anchor_matters)
            note = "multi: recall>=.66, low-decoy, cov>single, valid>decoy-arm, NO_ANCHOR validity worse"
        else:
            passed = (truse_l > decoy_l and decoy_l < 0.34 and valid_beats_decoyarm and anchor_matters)
            note = "single: true>decoy usage, low-decoy, valid>decoy-arm, NO_ANCHOR validity worse"
        tasks_verdict[task] = bool(passed)
        lines.append(
            "\n### %s (true=%s) — %s\n" % (task, true_f, "PASS" if passed else "FAIL"))
        lines.append("| arm | cov | | arm | cov |\n|---|---|---|---|---|\n")
        lines.append("| BASE | %.3f | | ORACLE | %.3f |\n" % (cov_base, cov_or))
        lines.append("| RANDOM_VALID | %.3f | | LEARNED_SINGLE | %.3f |\n" % (cov_rv, cov_s))
        lines.append("| **LEARNED_MULTI** | **%.3f** | | NO_ANCHOR | %.3f |\n" % (cov_l, cov_na))
        lines.append("| NO_DIVERSITY | %.3f | | | |\n" % cov_nd)
        lines.append("- learned: true_use=%.2f decoy_use=%.2f recall=%.2f | NO_ANCHOR decoy_use=%.2f (ablation should rise) | gate: %s\n"
                     % (truse_l, decoy_l, recall_l, decoy_na, note))
    fanin = all(tasks_verdict.get(t, False) for t in tasks_verdict
                if t not in ("impossible_control",)) and len(tasks_verdict) > 0
    # require impossible_control to pass too (no hallucination)
    if "impossible_control" in tasks_verdict:
        fanin = fanin and tasks_verdict["impossible_control"]
    lines.append("\n**Phase 1 fan-in: %s** %s\n" % ("PASS" if fanin else "FAIL", tasks_verdict))
    return "".join(lines), {"fanin": fanin, "tasks": tasks_verdict}


def collect_phase2():
    lines = ["## Phase 2 — Payoff\n"]
    verdict = {}
    for proxy, key, better in [("classifier", "heldout_acc", "higher"),
                               ("eqm_lite", "robustness_gap", "lower")]:
        files = _load("v17_payoff_*_%s_*.json" % proxy)
        if not files:
            lines.append("\n### %s: (no results)\n" % proxy); verdict[proxy] = None; continue
        bytask = {}
        for _, d in files:
            bytask.setdefault(d["task"], []).append(d)
        proxy_pass = []
        for task, ds in sorted(bytask.items()):
            def a(arm):
                return _agg([d["arms"][arm] for d in ds if arm in d["arms"]], [key])[0]
            base, om, rv = a("BASE"), a("KNOWN_ORACLE_MULTI"), a("RANDOM_VALID_POLICY")
            dm, dsng = a("DISCOVERED_MULTI"), a("DISCOVERED_SINGLE")
            if better == "higher":
                ok = dm > base and dm >= rv - 1e-3 and dm >= dsng - 1e-3
            else:
                ok = dm < base and dm <= rv + 1e-3 and dm <= dsng + 1e-3
            proxy_pass.append(ok)
            lines.append("\n### %s / %s — %s (%s better)\n" % (proxy, task, "PASS" if ok else "FAIL", better))
            lines.append("| BASE | ORACLE | RANDOM_VALID | DISC_SINGLE | **DISC_MULTI** |\n|---|---|---|---|---|\n")
            lines.append("| %.4f | %.4f | %.4f | %.4f | **%.4f** |\n" % (base, om, rv, dsng, dm))
        verdict[proxy] = all(proxy_pass) if proxy_pass else None
    cls_ok = verdict.get("classifier"); eqm_ok = verdict.get("eqm_lite")
    authorize = bool(cls_ok and eqm_ok)
    lines.append("\n**Phase 2: classifier=%s eqm_lite=%s -> EqM/FID authorized: %s**\n"
                 % (cls_ok, eqm_ok, "YES" if authorize else "NO (still gated)"))
    return "".join(lines), {"classifier": cls_ok, "eqm_lite": eqm_ok, "authorize_fid": authorize}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["0", "1", "2", "all"])
    a = ap.parse_args()
    secs = []
    verdicts = {}
    if a.phase in ("0", "all"):
        t, v = collect_phase0(); secs.append(t); verdicts["phase0"] = v
    if a.phase in ("1", "all"):
        t, v = collect_phase1(); secs.append(t); verdicts["phase1"] = v
    if a.phase in ("2", "all"):
        t, v = collect_phase2(); secs.append(t); verdicts["phase2"] = v
    report = "# v17 MorphismGym — Results Report\n\n" + "\n".join(secs)
    with open(os.path.join(RD, "v17_report.md"), "w") as fh:
        fh.write(report)
    C.save_json(verdicts, "v17_verdicts.json")
    print(report)
    print("\n[verdicts]", json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
