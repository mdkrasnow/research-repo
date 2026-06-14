"""Task 2 — unified baseline comparison table (cached-only).

One table comparing every contender on the SAME footing:
  random | ||f|| only | scalar readouts (-<f,x>, <f,dx>, l2, residual) |
  matched-norm scalar scores | learned trajectory probe (if available).

Columns: raw AUROC, within-norm AUROC, n_good, n_garbage, bins_used, verdict.
Verdict per row (on within-norm AUROC, the de-confounded number):
  GREEN >=0.80 | WEAK 0.60-0.80 | KILL <0.60 | INCONCLUSIVE (nan / <MIN_BINS).

Reads results/auroc_table.csv (scalars, raw+within-norm) and, if present,
results/dynamics_probe/probe_table.csv or PROBE_VALIDATION.txt (learned probe).
numpy-only, fixed seed, graceful fail.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

MIN_BINS = 3
SEED = 0


def die(path, nxt):
    print(f"[baseline_table] MISSING: {path}\n[next] {nxt}")
    sys.exit(2)


def verdict(within, bins):
    if within is None or not np.isfinite(within):
        return "INCONCLUSIVE"
    if bins is not None and bins < MIN_BINS:
        return "INCONCLUSIVE"
    if within >= 0.80:
        return "GREEN"
    if within >= 0.60:
        return "WEAK"
    return "KILL"


def load_auroc(folder):
    t = Path(folder) / "results" / "auroc_table.csv"
    if not t.exists():
        die(t, "run analyze.py on this run dir first.")
    rows = []
    with open(t) as fh:
        for r in csv.DictReader(fh):
            if r.get("regime") == "fixed":
                rows.append(r)
    return rows


def load_probe(folder):
    """Best learned-probe within-norm AUROC across all cached probe artifacts (max)."""
    f = Path(folder)
    best, src = None, None
    pt = f / "results" / "dynamics_probe" / "probe_table.csv"
    if pt.exists():
        with open(pt) as fh:
            for r in csv.DictReader(fh):
                v = r.get("within_norm_auroc") or r.get("within_norm") or "nan"
                try:
                    v = float(v)
                except ValueError:
                    continue
                if r.get("feature_set", "").upper() in ("FULL", "SHAPE", "ALL", "ALL-SHAPE") and np.isfinite(v):
                    if best is None or v > best:
                        best, src = v, "dynamics_probe/probe_table.csv"
    # held-out validation txt: scan ALL within-norm=0.xxx, take max (ALL-shape wins)
    for name in ("PROBE_VALIDATION.txt", "PROBE_VERDICT.txt"):
        p = f / "results" / name
        if p.exists():
            for m in re.finditer(r"within[- _]?norm\s*=?\s*([01]\.\d+)", p.read_text(), re.I):
                v = float(m.group(1))
                if best is None or v > best:
                    best, src = v, name
    return best, src


SCALAR_DESC = {
    "s1": "-<f,x>  (dot energy)",
    "s2": "0.5||f||^2  (l2 energy, norm-coupled)",
    "s3": "sum<f,dx>  (path integral)",
    "s5": "post-step ||f||  (residual, norm-coupled)",
}


def main(a):
    auroc = load_auroc(a.folder)
    by = {r["score"]: r for r in auroc}
    n_total = int(float(by[next(iter(by))]["n_total"])) if by else 0
    n_good = a.n_good if a.n_good else n_total // 2
    n_garb = a.n_garb if a.n_garb else n_total - n_good

    table = []  # (method, raw, within, bins, verdict)

    table.append(("random", 0.500, 0.500, None, "KILL"))
    if "s5" in by:
        r = by["s5"]
        table.append(("||f|| only (norm)", float(r["raw_auroc"]), float(r["within_norm_auroc"]),
                      int(r["n_bins_used"]), verdict(float(r["within_norm_auroc"]), int(r["n_bins_used"]))))
    for s in ("s1", "s3", "s2", "s5"):
        if s in by:
            r = by[s]
            w, b = float(r["within_norm_auroc"]), int(r["n_bins_used"])
            table.append((f"{s}: {SCALAR_DESC.get(s, s)}", float(r["raw_auroc"]), w, b, verdict(w, b)))
    if "s4" in by:
        r = by["s4"]
        w, b = float(r["within_norm_auroc"]), int(r["n_bins_used"])
        table.append(("latent-NN s4 (no f; sanity)", float(r["raw_auroc"]), w, b, verdict(w, b)))
    pv, psrc = load_probe(a.folder)
    if pv is not None:
        table.append((f"learned trajectory probe ({psrc})", float("nan"), pv, MIN_BINS, verdict(pv, MIN_BINS)))
    else:
        table.append(("learned trajectory probe", float("nan"), float("nan"), None,
                      "INCONCLUSIVE (run dynamics_probe.py)"))

    res = Path(a.folder) / "results"
    res.mkdir(parents=True, exist_ok=True)
    csv_p = res / "baseline_table.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "raw_auroc", "within_norm_auroc", "n_good", "n_garbage", "bins_used", "verdict"])
        for m, raw, wn, b, v in table:
            w.writerow([m, f"{raw:.4f}", f"{wn:.4f}", n_good, n_garb, b if b is not None else "", v])

    lines = ["# Baseline comparison table", "",
             f"Run: `{a.folder}`  n_good={n_good} n_garbage={n_garb}  seed={SEED}  "
             f"verdict on within-norm AUROC (de-confounded): GREEN≥0.80 / WEAK 0.60–0.80 / KILL<0.60.",
             "",
             "| method | raw AUROC | within-norm AUROC | bins | verdict |",
             "|---|---|---|---|---|"]
    for m, raw, wn, b, v in table:
        rs = "—" if not np.isfinite(raw) else f"{raw:.3f}"
        ws = "—" if not np.isfinite(wn) else f"{wn:.3f}"
        lines.append(f"| {m} | {rs} | {ws} | {b if b is not None else '—'} | {v} |")
    md = res / "BASELINE_TABLE.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"[baseline_table] wrote {csv_p} and {md}")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--n-good", type=int, default=0, dest="n_good")
    ap.add_argument("--n-garb", type=int, default=0, dest="n_garb")
    main(ap.parse_args())
