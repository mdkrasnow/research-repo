"""
WFB-EqM Stage 3: post-hoc analysis of the 3-arm 300-step comparison
(direct/WFB/CG-FBGN, no-Adam Armijo trainer). Run once all three
wfb_stage3_lm_trainer.py jobs' metrics.jsonl files are fetched locally.

Per documentation/wfb-eqm-stage3-proposal-2026-08-13.md's promotion/kill
rule: judged on the PROBE-LOSS TRAJECTORY over the full run, not any single
step (established batch-to-batch noise floor makes single-step reads
unreliable -- see Stage 2.6b's rho analysis and the 20-step smokes).

Usage:
    python analyze_stage3_results.py \
        --direct /path/to/wfb_stage3_full_alpha0p0_..._metrics.jsonl \
        --wfb /path/to/wfb_stage3_full_alpha0p5_..._metrics.jsonl \
        --fbgn-cg /path/to/wfb_stage3_full_alpha1p0_cg_..._metrics.jsonl
"""
import argparse
import json
import statistics


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def windowed_probe_loss(rows):
    """rows with a 'probe_loss' key are logged every `probe_every` steps (see
    wfb_stage3_lm_trainer.py) -- returns list of (step, probe_loss)."""
    return [(r["step"] + 1, r["probe_loss"]) for r in rows if "probe_loss" in r]


def accept_rate(rows):
    accepted = sum(1 for r in rows if r.get("accepted"))
    return accepted / len(rows) if rows else 0.0


def skip_breakdown(rows):
    reasons = {}
    for r in rows:
        if not r.get("accepted"):
            reasons[r.get("skip_reason")] = reasons.get(r.get("skip_reason"), 0) + 1
    return reasons


def linear_trend_slope(points):
    """OLS slope of probe_loss vs step -- simple, robust-enough summary of
    the trajectory's net direction over noisy per-checkpoint values."""
    if len(points) < 2:
        return None
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den > 0 else None


def analyze_arm(name, path):
    rows = load_jsonl(path)
    pl = windowed_probe_loss(rows)
    ar = accept_rate(rows)
    skips = skip_breakdown(rows)
    slope = linear_trend_slope(pl)
    dLs = [r["actual_delta_L"] for r in rows if r.get("actual_delta_L") is not None]

    print(f"\n=== {name} ({path}) ===")
    print(f"  n_steps_logged={len(rows)}  accept_rate={ar:.3f}  skip_breakdown={skips}")
    if dLs:
        print(f"  same-batch dL: median={statistics.median(dLs):.4f}  mean={statistics.mean(dLs):.4f}")
    if pl:
        print(f"  probe_loss checkpoints (step, value): {[(s, round(v, 3)) for s, v in pl]}")
        print(f"  probe_loss: first={pl[0][1]:.4f}  last={pl[-1][1]:.4f}  "
              f"min={min(v for _, v in pl):.4f}  max={max(v for _, v in pl):.4f}  "
              f"median={statistics.median(v for _, v in pl):.4f}")
        if slope is not None:
            print(f"  OLS trend slope (probe_loss per step): {slope:.6f} "
                  f"({'IMPROVING' if slope < 0 else 'WORSENING'} net trend over the run)")
    return {"name": name, "rows": rows, "probe_loss_points": pl, "accept_rate": ar,
            "skip_breakdown": skips, "trend_slope": slope}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--direct", required=True)
    p.add_argument("--wfb", required=True)
    p.add_argument("--fbgn-cg", required=True)
    args = p.parse_args()

    results = {
        "direct": analyze_arm("direct (alpha=0, negative control)", args.direct),
        "wfb": analyze_arm("WFB (alpha=0.5)", args.wfb),
        "fbgn_cg": analyze_arm("FBGN via CG (alpha=1.0)", args.fbgn_cg),
    }

    print("\n=== COMPARISON SUMMARY ===")
    print(f"{'arm':<12} {'accept_rate':<12} {'probe_last':<12} {'probe_median':<14} {'trend_slope':<14}")
    for key in ("direct", "wfb", "fbgn_cg"):
        r = results[key]
        pl = r["probe_loss_points"]
        last = pl[-1][1] if pl else float("nan")
        med = statistics.median(v for _, v in pl) if pl else float("nan")
        slope = r["trend_slope"] if r["trend_slope"] is not None else float("nan")
        print(f"{key:<12} {r['accept_rate']:<12.3f} {last:<12.4f} {med:<14.4f} {slope:<14.6f}")

    print("\nPer proposal doc promotion rule: WFB or FBGN promotes if it shows a clearly higher "
          "accept rate AND net-negative probe-loss trajectory (trend_slope < 0, ideally "
          "meaningfully more negative than direct's) than the alpha=0 negative control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
