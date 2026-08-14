"""Aggregate Phase II training logs into Table C (optimizer) and Table D (image).

Reads the `gradient_metrics.jsonl` each run writes (the repo's existing
per-step diagnostic stream, which the BTM branch extends with btm_* fields) and
produces, per arm and per training window:

  Table C  grad norm before/after clip, clip rate, update norm, nonfinite
           count, and (only when the run actually logged the required metric)
           the unclipped norm and the update/param ratio
  Table D  exact-target MSE / cosine (evaluation-only grad_x phi vs the BTM
           target), energy statistics, FD estimator accuracy in situ

The comparison that matters is G vs D over TRAINING TIME, so every quantity is
reported in early / mid / late windows rather than as a single average -- the
original direct-scalar failure was invisible in the loss and only showed up as a
growing clip rate late in training.

Three conventions this analyzer enforces, each fixing a defect that silently
corrupted the cross-run comparison:

1. WINDOWS ARE ABSOLUTE.  early/mid/late are thirds of the PLANNED step budget
   (`--planned-steps`, else `max_steps` from results/btm/manifest.jsonl), not of
   each run's own observed range.  Run-relative windows scored a run preempted
   at 3k on its steps 2050-3000 as "late" next to a complete run's 13350-20000 --
   which, for a campaign whose entire hypothesis is "scalar arms degrade LATE",
   makes a truncated arm look healthy.  A run whose maximum observed step is
   materially below the target (`--complete-frac`, default 0.9) is EXCLUDED from
   the cross-run aggregate and listed in an explicit "incomplete runs" section.

2. ARM AND K COME FROM THE MANIFEST.  `results/btm/manifest.jsonl` records
   `run_tag` -> (`arm`, `fd_k`) per submitted job.  Substring matching over the
   full filesystem path collapsed K=1 and K=4 into one bucket (the K=4 arm name
   contains the K=1 arm name) and could relabel every run under a root whose
   path happens to contain another arm's name.  An arm that cannot be resolved
   is a hard error, not a "?" sentinel that still gets averaged.

3. THE TWO RECORD TYPES ARE SEPARATED.  train.py writes two structurally
   different records that share a `step`: the GRAD record (emitted before
   `opt.step()`; keys `grad_norm` / `clipped` / `learning_rate`) and the PROBE /
   checklist record (emitted after; keys `delta_theta_norm` / `probe_delta_L` /
   `P_t` / `eta_func`).  Pooling them doubled the reported sample count and made
   `update_over_param` a ratio of medians taken over two different populations.
   Records are discriminated by which of those key sets is present.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

ARM_OF = {
    "btm_vector": "V  vector",
    "btm_scalar_exact": "G  scalar exact",
    "btm_scalar_action_exact": "A  action exact",
    "btm_scalar_fd_directional": "D  FD directional",
    "btm_scalar_fd_directional4": "D  FD directional",
    "btm_scalar_fd_action": "F  FD action",
}

# FD arms are the only ones for which the K suffix is meaningful.
FD_ARMS = {"btm_scalar_fd_directional", "btm_scalar_fd_directional4",
           "btm_scalar_fd_action"}

# Short RUN_TAG aliases some launches use instead of the long-form arm name.
TAG_ALIASES = (
    ("_D4_", "btm_scalar_fd_directional", 4),
    ("_D1_", "btm_scalar_fd_directional", 1),
    ("_V_", "btm_vector", None),
    ("_G_", "btm_scalar_exact", None),
    ("_A_", "btm_scalar_action_exact", None),
    ("_F_", "btm_scalar_fd_action", None),
)

# Keys that identify each of the two record populations train.py emits.
GRAD_KEYS = ("grad_norm", "clipped", "learning_rate")
PROBE_KEYS = ("delta_theta_norm", "probe_delta_L", "P_t", "eta_func")

NAN = float("nan")


def load_run(path):
    """Return rows for one gradient_metrics.jsonl (bad lines counted)."""
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    rows = [r for r in rows if "step" in r]
    rows.sort(key=lambda r: r["step"])
    return rows, bad


def is_grad_record(r):
    return any(k in r for k in GRAD_KEYS)


def is_probe_record(r):
    return any(k in r for k in PROBE_KEYS)


def load_manifest(path):
    """run_tag -> {arm, fd_k, max_steps}; last record for a tag wins."""
    index = {}
    if not path or not os.path.exists(path):
        return index
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = rec.get("run_tag")
            if tag:
                index[tag] = {"arm": rec.get("arm"),
                              "fd_k": rec.get("fd_k"),
                              "max_steps": rec.get("max_steps")}
    return index


def resolve_arm(tag, manifest):
    """(arm, fd_k) for a run tag.  Manifest first, then TAG-ONLY matching.

    Never looks at the full path: a root directory named after some other arm
    used to relabel every run underneath it (last match wins, no break).
    """
    entry = manifest.get(tag)
    if entry and entry.get("arm"):
        return entry["arm"], entry.get("fd_k")
    # Longest arm name that is a substring of the TAG wins, so
    # btm_scalar_fd_directional4 is not swallowed by btm_scalar_fd_directional.
    hits = [k for k in ARM_OF if k in tag]
    if hits:
        arm = max(hits, key=len)
        fd_k = 4 if arm.endswith("4") else (1 if arm in FD_ARMS else None)
        return arm, fd_k
    for alias, arm, fd_k in TAG_ALIASES:
        if alias in tag:
            return arm, fd_k
    return None, None


def windows(rows, target_steps, n=3):
    """Split by ABSOLUTE step into n equal windows of the planned budget.

    Intervals are half-open [lo, hi) so a record landing exactly on an interior
    edge is counted once, not twice.  The final window is closed on the right so
    the last planned step is included.
    """
    if not rows or not target_steps or target_steps <= 0:
        return []
    edges = [target_steps * i / n for i in range(n + 1)]
    names = ["early", "mid", "late"] if n == 3 else [f"w{i}" for i in range(n)]
    out = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        last = i == n - 1
        sel = [r for r in rows
               if lo <= r["step"] and (r["step"] <= hi if last
                                       else r["step"] < hi)]
        if sel:
            out.append((names[i], (lo, hi, last), sel))
    return out


def _f(rows, key):
    return [r[key] for r in rows if key in r and r[key] is not None
            and isinstance(r[key], (int, float))
            and not isinstance(r[key], bool)
            and not math.isnan(r[key])]


def _nonfinite_count(rows, keys=("grad_norm", "loss")):
    n = 0
    for r in rows:
        for k in keys:
            v = r.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and not math.isfinite(v):
                n += 1
                break
    return n


def summarize(rows, tag, arm, label, seed, target_steps):
    recs = []
    for wname, bounds, w in windows(rows, target_steps):
        lo, hi, last = bounds
        grad_w = [r for r in w if is_grad_record(r)]
        probe_w = [r for r in w if is_probe_record(r)]
        gn = _f(grad_w, "grad_norm")
        ugn = _f(grad_w, "unclipped_grad_norm")
        clipped = [r["clipped"] for r in grad_w if "clipped" in r]
        dtn = _f(probe_w, "delta_theta_norm")
        pn = _f(probe_w, "param_norm") or _f(grad_w, "param_norm")
        rec = {
            "run": tag, "arm": arm, "label": label, "seed": seed,
            "window": wname,
            "window_bounds": f"{lo:.0f}-{hi:.0f}{']' if last else ')'}",
            "steps": f"{w[0]['step']}-{w[-1]['step']}",
            "n_grad": len(grad_w), "n_probe": len(probe_w),
            "grad_norm_median": st.median(gn) if gn else NAN,
            "grad_norm_p95": (sorted(gn)[int(0.95 * (len(gn) - 1))]
                              if gn else NAN),
            # train.py does NOT log unclipped_grad_norm; substituting grad_norm
            # here made the "unclipped" column a silent duplicate of the
            # grad-norm column.  Absent stays absent.
            "unclipped_median": st.median(ugn) if ugn else NAN,
            "unclipped_max": max(ugn) if ugn else NAN,
            "unclipped_available": bool(ugn),
            "clip_rate_pct": (100.0 * sum(bool(c) for c in clipped)
                              / len(clipped) if clipped else NAN),
            "delta_theta_median": st.median(dtn) if dtn else NAN,
            "param_norm_median": st.median(pn) if pn else NAN,
            "param_norm_available": bool(pn),
            "nonfinite_count": _nonfinite_count(grad_w),
        }
        # Delta/||theta|| needs param_norm, which train.py does not log; and
        # both medians must come from the SAME record population.
        if dtn and pn:
            rec["update_over_param"] = rec["delta_theta_median"] / (
                rec["param_norm_median"] + 1e-30)
        for k in ("target_cosine", "target_mse_per_dim", "target_norm_ratio",
                  "E_mean", "E_std", "field_norm", "fd_h_mean", "fd_gap_abs",
                  "target_cosine_near_data", "target_cosine_far",
                  "probe_delta_L"):
            v = _f(w, k)
            if v:
                rec[k] = st.median(v)
        recs.append(rec)
    return recs


def _fmt(v, spec=".4g"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+",
                    help="dirs to search for gradient_metrics.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=None,
                    help="results/btm/manifest.jsonl (arm/fd_k/max_steps per "
                         "run_tag); defaults to the in-repo path")
    ap.add_argument("--planned-steps", type=int, default=None,
                    help="absolute planned step budget defining early/mid/late; "
                         "overrides the manifest's max_steps")
    ap.add_argument("--complete-frac", type=float, default=0.9,
                    help="a run whose max observed step is below this fraction "
                         "of the planned budget is excluded from the cross-run "
                         "aggregate and reported as incomplete")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(os.path.dirname(here))
    manifest_path = args.manifest or os.path.join(
        project, "results", "btm", "manifest.jsonl")
    manifest = load_manifest(manifest_path)

    allrecs, incomplete, unresolved, bad_lines = [], [], [], 0
    runinfo = []
    for root in args.roots:
        for p in sorted(glob.glob(
                os.path.join(root, "**", "gradient_metrics.jsonl"),
                recursive=True)):
            tag = os.path.basename(os.path.dirname(os.path.dirname(p)))
            arm, fd_k = resolve_arm(tag, manifest)
            if arm is None:
                unresolved.append((tag, p))
                continue
            rows, bad = load_run(p)
            bad_lines += bad
            if not rows:
                continue
            entry = manifest.get(tag) or {}
            target = (args.planned_steps or entry.get("max_steps")
                      or max(m.get("max_steps") or 0
                             for m in manifest.values()) or None)
            if not target:
                raise RuntimeError(
                    f"no planned step budget for run {tag!r}: pass "
                    f"--planned-steps or record max_steps in {manifest_path}")
            seed = "?"
            if "_s" in tag:
                seed = tag.rsplit("_s", 1)[-1][:1]
            ksuffix = (f" K={fd_k}" if arm in FD_ARMS and fd_k else "")
            label = ARM_OF[arm] + ksuffix
            maxstep = rows[-1]["step"]
            complete = maxstep >= args.complete_frac * target
            runinfo.append({"run": tag, "label": label, "seed": seed,
                            "max_step": maxstep, "planned": target,
                            "complete": complete, "path": p})
            recs = summarize(rows, tag, arm, label, seed, target)
            for r in recs:
                r["planned_steps"] = target
                r["complete"] = complete
            if complete:
                allrecs += recs
            else:
                incomplete += recs

    if unresolved:
        raise RuntimeError(
            "could not resolve an arm for these runs (add them to "
            f"{manifest_path} or rename the run tag): "
            + ", ".join(t for t, _ in unresolved))

    if not allrecs and not incomplete:
        print("no gradient_metrics.jsonl found under: " + ", ".join(args.roots))
        return

    if bad_lines:
        print(f"\n**{bad_lines} unparseable JSONL line(s) skipped** "
              "(expected for a run still in flight; not expected otherwise)")

    def table_c(recs, title):
        print(f"\n## {title}\n")
        print("| arm | seed | window | planned window | observed steps | "
              "n grad | n probe | grad norm (med) | grad p95 | unclipped max | "
              "clip rate % | nonfinite | Δθ (med) | Δθ/‖θ‖ |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(recs, key=lambda r: (r["label"], r["seed"],
                                             r["window_bounds"])):
            print(f"| {r['label']} | {r['seed']} | {r['window']} "
                  f"| {r['window_bounds']} | {r['steps']} "
                  f"| {r['n_grad']} | {r['n_probe']} "
                  f"| {_fmt(r['grad_norm_median'])} "
                  f"| {_fmt(r['grad_norm_p95'])} "
                  f"| {'not logged' if not r['unclipped_available'] else _fmt(r['unclipped_max'])} "
                  f"| {_fmt(r['clip_rate_pct'], '.2f')} "
                  f"| {r['nonfinite_count']} "
                  f"| {_fmt(r.get('delta_theta_median'), '.3g')} "
                  f"| {'not logged (param_norm absent)' if not r['param_norm_available'] else _fmt(r.get('update_over_param'), '.3g')} |")

    table_c(allrecs, "Table C — optimizer behaviour by training window "
                     "(complete runs)")

    print("\n## Table D — learned field vs the corrected BTM target "
          "(evaluation-only exact grad_x phi)\n")
    print("| arm | seed | window | target cosine | cos near-data | cos far | "
          "target MSE/dim | norm ratio | E mean | E std | FD h | FD gap |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(allrecs, key=lambda r: (r["label"], r["seed"],
                                            r["window_bounds"])):
        if "target_cosine" not in r:
            continue
        g = lambda k: r.get(k, NAN)
        print(f"| {r['label']} | {r['seed']} | {r['window']} "
              f"| {_fmt(g('target_cosine'), '.4f')} "
              f"| {_fmt(g('target_cosine_near_data'), '.4f')} "
              f"| {_fmt(g('target_cosine_far'), '.4f')} "
              f"| {_fmt(g('target_mse_per_dim'))} "
              f"| {_fmt(g('target_norm_ratio'), '.3f')} | {_fmt(g('E_mean'))} "
              f"| {_fmt(g('E_std'))} | {_fmt(g('fd_h_mean'), '.3g')} "
              f"| {_fmt(g('fd_gap_abs'), '.3g')} |")

    # arm-level aggregation of the two quantities the hypothesis turns on.
    # COMPLETE RUNS ONLY -- a truncated run's "late" window is early training.
    print("\n## Key mechanistic comparison — clip rate and target cosine, "
          "late window (absolute), mean ± std over seeds, COMPLETE RUNS ONLY\n")
    by = defaultdict(list)
    for r in allrecs:
        if r["window"] == "late":
            by[r["label"]].append(r)
    print("| arm | seeds | late clip rate % | late target cosine |")
    print("|---|---|---|---|")
    for arm, rs in sorted(by.items()):
        cr = [r["clip_rate_pct"] for r in rs
              if not math.isnan(r["clip_rate_pct"])]
        tc = [r["target_cosine"] for r in rs if "target_cosine" in r]

        def ms(v):
            if not v:
                return "n/a"
            return (f"{st.mean(v):.4g} ± {st.stdev(v):.3g}" if len(v) > 1
                    else f"{v[0]:.4g}")
        print(f"| {arm} | {len(rs)} | {ms(cr)} | {ms(tc)} |")

    incomplete_runs = [r for r in runinfo if not r["complete"]]
    print("\n## Incomplete runs — EXCLUDED from the aggregate above\n")
    if not incomplete_runs:
        print("(none; every run reached its planned step budget)")
    else:
        print("| run | arm | seed | max observed step | planned steps | "
              "fraction |")
        print("|---|---|---|---|---|---|")
        for r in sorted(incomplete_runs, key=lambda r: r["run"]):
            print(f"| {r['run']} | {r['label']} | {r['seed']} "
                  f"| {r['max_step']} | {r['planned']} "
                  f"| {r['max_step'] / r['planned']:.2%} |")
        table_c(incomplete, "Table C (incomplete runs — window labels refer to "
                            "the ABSOLUTE planned budget, so these runs have "
                            "no late window)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"complete": allrecs, "incomplete": incomplete,
                       "runs": runinfo, "bad_lines": bad_lines}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
