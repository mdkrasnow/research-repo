"""Phase II aggregation over the telemetry event log — gated, not best-effort.

This replaces the identity-by-path-parsing, window-by-whatever-you-reached
aggregation in ``analyze_image.py``.  The old script is retained unchanged so
that runs in flight under the legacy format can still be read; new analysis
should use this one.

What changed, and why each change is load-bearing
-------------------------------------------------
**Identity is read, not parsed.**  ``analyze_image.py`` recovered the arm by
substring-matching a hand-maintained alias table against a directory name.  That
decoding was non-injective -- ``btm_scalar_fd_directional`` is a prefix of
``btm_scalar_fd_directional4``, so K=1 and K=4 runs decoded to the same arm and
were silently averaged together -- and non-total, with ``arm = "?"`` a reachable
state that was still aggregated.  Here the arm comes from the run's own recorded
spec.  The alias table is gone; the bug class is gone with it.

**Windows are absolute.**  The old windows were computed from each run's own
first and last observed step, so "late" meant "the last third of however far this
run got".  A run killed at step 3,000 contributed its steps 2k-3k to the same
column as a complete run's 13k-20k.  Here windows come from ``planned_steps``,
and :func:`telemetry.read.shared_windows` refuses to build a common axis for runs
that were not planned to the same length.

**Incomplete runs are quarantined, loudly.**  Nothing was checking whether a run
finished, because nothing recorded it.  Now every table states which runs it is
built from, and every excluded run is printed with its reason.  A report that
silently omits the three arms that died is not a shorter report -- it is a wrong
one.

**Measurement populations cannot be pooled by accident.**  The per-step gradient
record and the post-step probe record are separate ``kind``s.  ``update_over_param``
used to divide a median over probe records by a median over gradient records; that
is now impossible to express without naming both kinds explicitly.

**Comparisons are checked for confounds.**  Before printing the headline G-vs-D
table, the specs of the arms being compared are diffed.  If they differ in more
than the manipulated variable -- a different git sha, a different batch size, a
different planned horizon -- the comparison is reported as confounded rather than
presented as a result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from telemetry import EventType  # noqa: E402
from telemetry.invariants import check_group, check_run  # noqa: E402
from telemetry.read import (  # noqa: E402
    CompletenessPolicy,
    LogicalRun,
    RunLog,
    controlled_comparison,
    load_campaign,
    rate,
    shared_windows,
    windowed,
)

#: Display names only.  Never a join key -- the join key is the ``arm`` field the
#: run itself recorded.  Every historical identity bug in this codebase came from
#: a human-readable label being pressed into service as an identifier.
ARM_LABEL = {
    "btm_vector": "V  vector",
    "btm_scalar_exact": "G  scalar exact",
    "btm_scalar_action_exact": "A  action exact",
    "btm_scalar_fd_directional": "D  FD directional",
    "btm_scalar_fd_action": "F  FD action",
}

#: The two measurement populations on the PROGRESS stream.  Naming them here,
#: once, is what prevents a caller from silently pooling them.
KIND_GRAD = "grad"
KIND_PROBE = "probe"


def label_for(run: LogicalRun) -> str:
    """Display label, including the parameter that distinguishes sibling arms.

    ``fd_k`` is read from the recorded spec rather than sniffed out of a name
    suffix.  This is exactly the collapse the old ``_D1_``/``_D4_`` alias logic
    got wrong whenever a run used a long-form tag.
    """
    arm = str(run.arm or "unknown")
    base = str(ARM_LABEL.get(arm, arm))
    params = dict(run.spec.params) if run.spec else {}
    fd_k = params.get("fd_k")
    if fd_k is not None and "fd" in arm:
        base += f" K={fd_k}"
    return base


def _fmt(value: Optional[float], spec: str = ".4g") -> str:
    return "n/a" if value is None else format(value, spec)


def _summary_cell(summary) -> str:
    """Render a statistic together with the sample size behind it.

    A median over three points and a median over three hundred are different
    claims, and a table that renders them identically invites the reader to treat
    them the same.  ``n`` travels with every number here.
    """
    if summary is None:
        return "n/a"
    text = f"{summary.median:.4g} (n={summary.n})"
    if summary.nonfinite:
        text += f" !{summary.nonfinite}nf"
    return text


def build_report(root: str, *, campaign: Optional[str], phase: Optional[str],
                 policy: CompletenessPolicy, n_windows: int = 3,
                 allow_incomplete: bool = False) -> Dict[str, Any]:
    campaign_data = load_campaign(root)
    selected = [r for r in campaign_data.runs
                if (campaign is None or (r.spec and r.spec.campaign == campaign))
                and (phase is None or (r.spec and r.spec.phase == phase))]

    admitted: List[Tuple[LogicalRun, RunLog]] = []
    quarantined: List[Tuple[LogicalRun, Any]] = []
    for run in selected:
        execution, rejection = run.analyzable(policy)
        if execution is not None:
            admitted.append((run, execution))
        else:
            quarantined.append((run, rejection))

    report: Dict[str, Any] = {
        "root": root,
        "campaign": campaign,
        "phase": phase,
        "n_selected": len(selected),
        "n_admitted": len(admitted),
        "quarantined": [
            {"run_uid": r.run_uid, "arm": r.arm, "seed": r.seed,
             "label": label_for(r), "reason": rej.reason, "detail": rej.detail}
            for r, rej in quarantined
        ],
        "tables": {},
        "confounds": [],
        "invariants": [],
    }

    if not admitted:
        report["verdict"] = (
            "NO ANALYZABLE RUNS. Every selected run failed the completeness gate; "
            "see 'quarantined' for why. No table is produced, because any table "
            "built from these runs would compare non-comparable step ranges.")
        return report

    executions = [e for _, e in admitted]
    try:
        windows = shared_windows(executions, n=n_windows)
    except ValueError as exc:
        # Different planned horizons across arms is a design-level confound, not
        # something to paper over by rescaling one arm onto the other's axis.
        report["verdict"] = f"NOT COMPARABLE: {exc}"
        report["confounds"].append(str(exc))
        return report
    report["windows"] = [{"name": w.name, "lo": w.lo, "hi": w.hi} for w in windows]

    # -- Table C: optimizer behaviour, per absolute window --------------------
    table_c: List[Dict[str, Any]] = []
    for run, execution in admitted:
        grad_norm = windowed(execution, "grad_norm", windows, kind=KIND_GRAD)
        unclipped = windowed(execution, "unclipped_grad_norm", windows, kind=KIND_GRAD)
        clip_rate = rate(execution, "clipped", windows, kind=KIND_GRAD)
        delta_theta = windowed(execution, "delta_theta_norm", windows, kind=KIND_PROBE)
        param_norm = windowed(execution, "param_norm", windows, kind=KIND_PROBE)
        for window in windows:
            summary = grad_norm[window.name]
            unclipped_summary = unclipped[window.name] or summary
            delta_summary = delta_theta[window.name]
            param_summary = param_norm[window.name]
            # Both operands come from the SAME kind, so this ratio is now a
            # within-population quantity. Under the old code the numerator came
            # from probe records and the denominator from gradient records.
            ratio = (delta_summary.median / param_summary.median
                     if delta_summary and param_summary and param_summary.median
                     else None)
            table_c.append({
                "run_uid": run.run_uid, "label": label_for(run), "seed": run.seed,
                "window": window.name, "steps": f"{window.lo}-{window.hi}",
                "n": summary.n if summary else 0,
                "grad_norm_median": summary.median if summary else None,
                "grad_norm_p95": summary.p95 if summary else None,
                "unclipped_max": unclipped_summary.maximum if unclipped_summary else None,
                "clip_rate_pct": clip_rate[window.name],
                "delta_theta_median": delta_summary.median if delta_summary else None,
                "update_over_param": ratio,
                "nonfinite": summary.nonfinite if summary else 0,
            })
    report["tables"]["C"] = table_c

    # -- Table D: learned field vs the corrected BTM target -------------------
    eval_keys = ("target_cosine", "target_cosine_near_data", "target_cosine_far",
                 "target_mse_per_dim", "target_norm_ratio", "E_mean", "E_std",
                 "fd_h_mean", "fd_gap_abs")
    table_d: List[Dict[str, Any]] = []
    for run, execution in admitted:
        per_key = {
            key: windowed(execution, key, windows, event=EventType.EVAL)
            for key in eval_keys
        }
        for window in windows:
            row: Dict[str, Any] = {
                "run_uid": run.run_uid, "label": label_for(run), "seed": run.seed,
                "window": window.name, "steps": f"{window.lo}-{window.hi}",
            }
            for key in eval_keys:
                summary = per_key[key][window.name]
                row[key] = summary.median if summary else None
                row[f"{key}_n"] = summary.n if summary else 0
            table_d.append(row)
    report["tables"]["D"] = table_d

    # -- Headline comparison, gated on being controlled -----------------------
    by_arm: Dict[str, List[Tuple[LogicalRun, RunLog]]] = defaultdict(list)
    for run, execution in admitted:
        by_arm[label_for(run)].append((run, execution))

    for i, left in enumerate(sorted(by_arm)):
        for right in sorted(by_arm)[i + 1:]:
            a_run = by_arm[left][0][0]
            b_run = by_arm[right][0][0]
            check = controlled_comparison(a_run, b_run)
            if not check["controlled"]:
                report["confounds"].append({
                    "left": left, "right": right,
                    "reason": check.get("reason", ""),
                    "differing_fields": {
                        k: list(v) for k, v in
                        (check.get("differing_fields") or {}).items()},
                })

    late = windows[-1]
    headline: List[Dict[str, Any]] = []
    for arm_label, members in sorted(by_arm.items()):
        clip_values, cosine_values = [], []
        for _, execution in members:
            clip = rate(execution, "clipped", windows, kind=KIND_GRAD)[late.name]
            if clip is not None:
                clip_values.append(clip)
            cosine = windowed(execution, "target_cosine", windows,
                              event=EventType.EVAL)[late.name]
            if cosine is not None:
                cosine_values.append(cosine.median)
        headline.append({
            "label": arm_label,
            "n_runs": len(members),
            "late_clip_rate_pct": clip_values,
            "late_target_cosine": cosine_values,
        })
    report["tables"]["headline"] = headline

    # -- Declared invariants --------------------------------------------------
    violations = []
    for _, execution in admitted:
        violations += check_run(execution)
    violations += check_group([e for _, e in admitted],
                              [label_for(r) for r, _ in admitted])
    report["invariants"] = [
        {"invariant": v.invariant, "kind": v.kind.value, "severity": v.severity,
         "detail": v.detail, "rationale": v.rationale} for v in violations
    ]

    blocking = [v for v in violations if v.severity == "error"]
    if blocking:
        report["verdict"] = (
            f"INVALID: {len(blocking)} declared invariant(s) were violated. The "
            "arms were not measured under the conditions the comparison assumes, "
            "so the tables below are not a valid basis for a decision.")
    elif report["confounds"]:
        report["verdict"] = (
            f"CONFOUNDED: {len(report['confounds'])} arm pair(s) differ in more "
            "than the manipulated variable. Read the tables descriptively only.")
    elif quarantined and not allow_incomplete:
        report["verdict"] = (
            f"PARTIAL: {len(admitted)} of {len(selected)} runs admitted; "
            f"{len(quarantined)} quarantined. The tables are valid for the "
            "admitted runs; the campaign is not yet decidable.")
    else:
        report["verdict"] = f"OK: all {len(admitted)} selected runs admitted."
    return report


def render(report: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append(f"# Phase II telemetry report\n")
    out.append(f"**Verdict — {report['verdict']}**\n")

    if report["quarantined"]:
        out.append(f"\n## Quarantined runs ({len(report['quarantined'])})\n")
        out.append("These are excluded from every table below. This list is part "
                   "of the result, not an appendix.\n")
        out.append("| run | label | seed | reason | detail |")
        out.append("|---|---|---|---|---|")
        for row in report["quarantined"]:
            out.append(f"| `{row['run_uid']}` | {row['label']} | {row['seed']} "
                       f"| **{row['reason']}** | {row['detail']} |")

    if report.get("invariants"):
        out.append("\n## Declared-invariant violations\n")
        out.append("| invariant | kind | severity | detail |")
        out.append("|---|---|---|---|")
        for row in report["invariants"]:
            out.append(f"| {row['invariant']} | {row['kind']} | "
                       f"**{row['severity']}** | {row['detail']} |")

    if report.get("confounds"):
        out.append("\n## Confounded comparisons\n")
        for row in report["confounds"]:
            if isinstance(row, str):
                out.append(f"- {row}")
            else:
                fields = ", ".join(f"`{k}`: {v[0]!r} vs {v[1]!r}"
                                   for k, v in row["differing_fields"].items())
                out.append(f"- **{row['left']}** vs **{row['right']}** — {fields}")

    if "windows" not in report:
        return "\n".join(out) + "\n"

    edges = ", ".join(f"{w['name']} {w['lo']}-{w['hi']}" for w in report["windows"])
    out.append(f"\n## Windows (absolute, on the planned step axis)\n\n{edges}\n")
    out.append("Identical for every arm by construction — a 'late' number for one "
               "arm covers the same training interval as every other arm's.\n")

    out.append("\n## Table C — optimizer behaviour by window\n")
    out.append("| arm | seed | window | steps | grad norm (med) | grad p95 | "
               "unclipped max | clip rate % | Δθ (med) | Δθ/‖θ‖ | nonfinite |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in report["tables"]["C"]:
        out.append(
            f"| {row['label']} | {row['seed']} | {row['window']} | {row['steps']} "
            f"| {_fmt(row['grad_norm_median'])} (n={row['n']}) "
            f"| {_fmt(row['grad_norm_p95'])} | {_fmt(row['unclipped_max'])} "
            f"| {_fmt(row['clip_rate_pct'], '.2f')} "
            f"| {_fmt(row['delta_theta_median'], '.3g')} "
            f"| {_fmt(row['update_over_param'], '.3g')} | {row['nonfinite']} |")

    out.append("\n## Table D — learned field vs the corrected BTM target\n")
    out.append("| arm | seed | window | target cos | cos near | cos far | "
               "MSE/dim | norm ratio | E mean | E std | FD h | FD gap |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in report["tables"]["D"]:
        if row.get("target_cosine") is None:
            continue
        out.append(
            f"| {row['label']} | {row['seed']} | {row['window']} "
            f"| {_fmt(row['target_cosine'], '.4f')} "
            f"| {_fmt(row['target_cosine_near_data'], '.4f')} "
            f"| {_fmt(row['target_cosine_far'], '.4f')} "
            f"| {_fmt(row['target_mse_per_dim'])} "
            f"| {_fmt(row['target_norm_ratio'], '.3f')} "
            f"| {_fmt(row['E_mean'])} | {_fmt(row['E_std'])} "
            f"| {_fmt(row['fd_h_mean'], '.3g')} | {_fmt(row['fd_gap_abs'], '.3g')} |")

    out.append("\n## Headline — late window, the two quantities the hypothesis turns on\n")
    out.append("| arm | runs | late clip rate % | late target cosine |")
    out.append("|---|---|---|---|")
    for row in report["tables"]["headline"]:
        def stat(values: Sequence[float]) -> str:
            if not values:
                return "n/a"
            if len(values) == 1:
                return f"{values[0]:.4g}"
            mean = sum(values) / len(values)
            spread = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
            return f"{mean:.4g} ± {spread:.3g}"
        out.append(f"| {row['label']} | {row['n_runs']} "
                   f"| {stat(row['late_clip_rate_pct'])} "
                   f"| {stat(row['late_target_cosine'])} |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/telemetry",
                        help="telemetry root (one directory per logical run)")
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="admit truncated and unsealed runs. Explicitly unsafe for the "
             "late-window comparison; every table is stamped accordingly.")
    args = parser.parse_args()

    policy = CompletenessPolicy()
    if args.allow_incomplete:
        policy = CompletenessPolicy(require_end=False, require_status=(),
                                    forbid_truncated=False)

    report = build_report(args.root, campaign=args.campaign, phase=args.phase,
                          policy=policy, n_windows=args.windows,
                          allow_incomplete=args.allow_incomplete)
    print(render(report))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True, default=str)
        print(f"wrote {args.out}")

    # Non-zero exit when the report is not a valid basis for a decision, so a
    # pipeline cannot consume an invalid table and proceed.
    return 1 if report["verdict"].split(":")[0] in ("INVALID", "CONFOUNDED",
                                                    "NO ANALYZABLE RUNS") else 0


if __name__ == "__main__":
    raise SystemExit(main())
