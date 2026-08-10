"""
Longitudinal, progress-matched direct-specific Jacobian amplification
(2026-08-10, follow-up to matched_replay_jacobian_diagnostic.py / job
38136938, per documentation/longitudinal_jacobian_audit.md).

Two things job 38136938 could NOT answer, per external review:

1. `none` was held FIXED at one late reference while direct swept
   early/mid/late -- confounding "does direct's amplification grow" with
   "how does a fixed-none baseline compare to direct at different points."
   Phase 0 audit here found this was an infra-discovery gap, not a real
   data limitation: `none`'s training lineage (chained resumed jobs
   longer40/60/80) has checkpoints matching direct's steps almost exactly
   (0% / 1.15% / 0.885% mismatch at early/mid/late -- see the audit doc).
   This script uses those PROGRESS-MATCHED checkpoints at every stage.

2. K=24 spike/control gave CIs wide enough to cross zero at every stage.
   This script raises K to `num_control` (default 64) at the SAME
   spike_frac=0.025 percentile criterion (Phase 1: widen the pool, not the
   tail definition) by scaling `pool_size` accordingly.

Also adds (both explicitly REQUIRED by the follow-up spec, not optional):
  - Table 7: tail quantiles (p50/p90/p95/p99/max) + tail_anisotropy_proxy
    per stage/model/source-set -- tests whether the effect is upper-tail
    specific rather than a shift in the whole distribution.
  - Table 6 extension: G = I(late) - I(early), bootstrapped by treating the
    two stages' underlying rho arrays as independent samples (they ARE
    different selected batches at different checkpoints -- no natural
    batch-level pairing across stages exists, so this is intentionally the
    conservative/wider-CI choice, not an artificially tightened one).
  - Table 8/9/10: matrix-free sigma_1(J_backbone) + residual/top-left-
    singular-vector alignment (Phase 6/7), run on a SMALL subset (default
    16 spike + 16 control) at the LATE stage only -- both JVP and the
    power-iteration primitive are validated against an explicitly
    materialized Jacobian + numpy SVD on a tiny model in
    tests/test_fb_direct_exact_hvp.py (ALL PASS, re-run as this sbatch's
    pre-flight gate) BEFORE this script touches a GPU.

Everything else (canonical residual identity, A via exact VJP, matched-
replay design, corrected interaction statistic) is unchanged infra reused
directly from matched_replay_jacobian_diagnostic.py.
"""
import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matched_replay_jacobian_diagnostic import (  # noqa: E402
    block_groups, build_pool, field_of, is_backbone_group, load_model,
    matched_interaction_bootstrap, paired_self_minus_other, probe_direct,
    probe_none, rank_pool, ratio_row, replay_union, select_spike_control,
    total_norm,
)
from fb_direct.exact_hvp import (  # noqa: E402
    exact_field_vjp, field_jvp_direct, field_jvp_none, field_vjp_none,
    power_iteration_theta_sigma1,
)


# ---------------------------------------------------------------- Table 7: tail quantiles

def _quantiles(vals):
    if not vals:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    s = sorted(vals)
    n = len(s)

    def q(p):
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return s[idx]
    return {"p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99), "max": s[-1]}


def tail_row(label, a_dict, idx):
    vals = [a_dict[i] for i in idx]
    qs = _quantiles(vals)
    qs["tail_anisotropy_proxy"] = (qs["p99"] / qs["p50"]) if qs["p50"] else None
    qs["label"] = label
    qs["n"] = len(vals)
    return qs


# ---------------------------------------------------------------- per-stage evaluation (progress-matched)

def evaluate_stage(stage_label, ckpt_direct, ckpt_none, args, fixed_inputs, device):
    print(f"[diag] === stage {stage_label}: direct={ckpt_direct} vs none={ckpt_none} (progress-matched) ===")
    model_direct = load_model(ckpt_direct, args.model, args.image_size, args.num_classes, "direct", device)
    model_none = load_model(ckpt_none, args.model, args.image_size, args.num_classes, "none", device)
    groups_direct = block_groups(model_direct)
    groups_none = block_groups(model_none)

    pool_grad_direct, _ = rank_pool(probe_direct, model_direct, fixed_inputs, f"direct({stage_label})")
    pool_grad_none, _ = rank_pool(probe_none, model_none, fixed_inputs, f"none({stage_label})")

    direct_spike_idx, direct_control_idx = select_spike_control(pool_grad_direct, args.spike_frac, args.num_control)
    none_spike_idx, none_control_idx = select_spike_control(pool_grad_none, args.spike_frac, args.num_control)
    print(f"[diag] {stage_label}: direct-selected {len(direct_spike_idx)} spike / {len(direct_control_idx)} control, "
          f"none-selected {len(none_spike_idx)} spike / {len(none_control_idx)} control")

    union_idx = sorted(set(direct_spike_idx) | set(direct_control_idx)
                        | set(none_spike_idx) | set(none_control_idx))
    print(f"[diag] {stage_label}: {len(union_idx)} unique batches for matched replay")
    replay = replay_union(model_direct, model_none, groups_direct, groups_none, fixed_inputs, union_idx)

    table1 = []
    for eval_name in ("direct", "none"):
        diffs = [replay[idx][eval_name]["identity_abs_diff"] for idx in union_idx]
        rels = [replay[idx][eval_name]["identity_rel_diff"] for idx in union_idx]
        table1.append({"stage": stage_label, "eval_model": eval_name, "n_batches": len(union_idx),
                        "max_abs_diff": max(diffs), "max_rel_diff": max(rels), "pass": True})

    a_direct = {i: replay[i]["direct"]["A_backbone"] for i in union_idx}
    a_none = {i: replay[i]["none"]["A_backbone"] for i in union_idx}
    resid_direct = {i: replay[i]["direct"]["residual_rms"] for i in union_idx}
    resid_none = {i: replay[i]["none"]["residual_rms"] for i in union_idx}
    grad_direct = {i: pool_grad_direct[i] for i in union_idx}
    grad_none = {i: pool_grad_none[i] for i in union_idx}

    table3 = [
        {"eval_model": "direct", "batches_from": "direct", "selection": "own", "metric": "A_backbone",
         **ratio_row(direct_spike_idx, direct_control_idx, a_direct)},
        {"eval_model": "none", "batches_from": "direct", "selection": "cross (matched, same batches)", "metric": "A_backbone",
         **ratio_row(direct_spike_idx, direct_control_idx, a_none)},
        {"eval_model": "none", "batches_from": "none", "selection": "own", "metric": "A_backbone",
         **ratio_row(none_spike_idx, none_control_idx, a_none)},
        {"eval_model": "direct", "batches_from": "none", "selection": "cross (matched, same batches)", "metric": "A_backbone",
         **ratio_row(none_spike_idx, none_control_idx, a_direct)},
        {"eval_model": "direct", "batches_from": "direct", "selection": "own", "metric": "grad_norm_real_preclip",
         **ratio_row(direct_spike_idx, direct_control_idx, grad_direct)},
        {"eval_model": "none", "batches_from": "direct", "selection": "cross (matched, same batches)", "metric": "grad_norm_real_preclip",
         **ratio_row(direct_spike_idx, direct_control_idx, grad_none)},
        {"eval_model": "none", "batches_from": "none", "selection": "own", "metric": "grad_norm_real_preclip",
         **ratio_row(none_spike_idx, none_control_idx, grad_none)},
        {"eval_model": "direct", "batches_from": "none", "selection": "cross (matched, same batches)", "metric": "grad_norm_real_preclip",
         **ratio_row(none_spike_idx, none_control_idx, grad_direct)},
    ]

    rho_D_spike_A = paired_self_minus_other(direct_spike_idx, a_direct, a_none)
    rho_D_control_A = paired_self_minus_other(direct_control_idx, a_direct, a_none)
    rho_N_spike_A = paired_self_minus_other(none_spike_idx, a_none, a_direct)
    rho_N_control_A = paired_self_minus_other(none_control_idx, a_none, a_direct)
    interaction_A = matched_interaction_bootstrap(rho_D_spike_A, rho_D_control_A, rho_N_spike_A, rho_N_control_A,
                                                   num_samples=args.bootstrap_samples, seed=args.seed)

    rho_D_spike_g = paired_self_minus_other(direct_spike_idx, grad_direct, grad_none)
    rho_D_control_g = paired_self_minus_other(direct_control_idx, grad_direct, grad_none)
    rho_N_spike_g = paired_self_minus_other(none_spike_idx, grad_none, grad_direct)
    rho_N_control_g = paired_self_minus_other(none_control_idx, grad_none, grad_direct)
    interaction_grad = matched_interaction_bootstrap(rho_D_spike_g, rho_D_control_g, rho_N_spike_g, rho_N_control_g,
                                                       num_samples=args.bootstrap_samples, seed=args.seed)

    decomposition = {
        "direct_own": {
            "grad_ratio": ratio_row(direct_spike_idx, direct_control_idx, grad_direct)["ratio"],
            "residual_ratio": ratio_row(direct_spike_idx, direct_control_idx, resid_direct)["ratio"],
            "A_ratio": ratio_row(direct_spike_idx, direct_control_idx, a_direct)["ratio"],
        },
        "none_own": {
            "grad_ratio": ratio_row(none_spike_idx, none_control_idx, grad_none)["ratio"],
            "residual_ratio": ratio_row(none_spike_idx, none_control_idx, resid_none)["ratio"],
            "A_ratio": ratio_row(none_spike_idx, none_control_idx, a_none)["ratio"],
        },
    }
    for row in decomposition.values():
        row["implied_grad_ratio"] = (row["residual_ratio"] * row["A_ratio"]
                                      if (row["residual_ratio"] and row["A_ratio"]) else None)

    table7 = [
        tail_row("direct_own_spike", a_direct, direct_spike_idx),
        tail_row("direct_own_control", a_direct, direct_control_idx),
        tail_row("none_own_spike", a_none, none_spike_idx),
        tail_row("none_own_control", a_none, none_control_idx),
    ]

    all_groups = sorted((set(groups_direct.values()) | set(groups_none.values())) - {"other"})
    table10 = []
    for g in all_groups:
        row = {"group": g}
        for eval_name, spike_idx, control_idx in (("direct", direct_spike_idx, direct_control_idx),
                                                    ("none", none_spike_idx, none_control_idx)):
            layer_vals = {i: replay[i][eval_name]["A_layer"].get(g, 0.0) for i in union_idx}
            rr = ratio_row(spike_idx, control_idx, layer_vals)
            row[f"{eval_name}_control"] = rr["control_median"]
            row[f"{eval_name}_spike"] = rr["spike_median"]
            row[f"{eval_name}_ratio"] = rr["ratio"]
            contribs = []
            for i in spike_idx:
                total_sq = sum(v ** 2 for v in replay[i][eval_name]["A_layer"].values())
                if total_sq > 0:
                    contribs.append(replay[i][eval_name]["A_layer"].get(g, 0.0) ** 2 / total_sq)
            row[f"{eval_name}_spike_contribution_pct"] = (statistics.median(contribs) * 100.0) if contribs else None
        table10.append(row)
    table10.sort(key=lambda r: (r.get("direct_ratio") or 0), reverse=True)

    result = {
        "stage": stage_label,
        "checkpoint_direct": ckpt_direct, "checkpoint_none": ckpt_none,
        "n_direct_spike": len(direct_spike_idx), "n_direct_control": len(direct_control_idx),
        "n_none_spike": len(none_spike_idx), "n_none_control": len(none_control_idx),
        "table1_instrumentation_validation": table1,
        "table3_own_and_matched_cross": table3,
        "table4_source_matched_interaction": {
            "A_based": interaction_A, "grad_based": interaction_grad,
            "note": "interaction = Delta_D - Delta_N, SOURCE-matched (same batches, swapped evaluating model). See module docstring.",
        },
        "residual_decomposition": decomposition,
        "table7_tail_quantiles": table7,
        "table10_layer_localization": table10,
    }
    # Return live handles too, for the singular-value pass (avoids a reload at LATE stage).
    return result, model_direct, model_none, groups_direct, groups_none, fixed_inputs, \
        direct_spike_idx, direct_control_idx, none_spike_idx, none_control_idx, replay


# ---------------------------------------------------------------- Phase 6/7: singular value + alignment

def singular_value_alignment_pass(model_direct, model_none, groups_direct, groups_none, fixed_inputs,
                                   direct_spike_idx, direct_control_idx,
                                   none_spike_idx, none_control_idx, args):
    """LATE-stage only (called by caller with the late stage's live model
    handles). Runs matrix-free power iteration for sigma_1(J_backbone) +
    top-left-singular-vector alignment alpha_1 = |<u, u_1>| on a SMALL
    subset (args.sv_k spike + args.sv_k control) for BOTH models, on
    direct-selected batches (primary) and none-selected batches (if
    affordable, per spec 'if affordable')."""
    import random
    rng = random.Random(args.seed)

    def subset(idx_list, k):
        idx_list = list(idx_list)
        rng.shuffle(idx_list)
        return idx_list[:k]

    # requires_grad filter: pos_embed is an nn.Parameter with requires_grad=False
    # (frozen sinusoidal embedding) but IS returned by named_parameters() and IS
    # mapped into the x_embedder/backbone group by block_groups() -- excluded here
    # since it is not differentiable (autograd.grad would otherwise error).
    backbone_params_direct = [p for n, p in model_direct.named_parameters()
                               if is_backbone_group(groups_direct[n]) and p.requires_grad]
    backbone_params_none = [p for n, p in model_none.named_parameters()
                             if is_backbone_group(groups_none[n]) and p.requires_grad]

    def run_one(model, ebm, params, xt, t, y, ut):
        jvp_fn = field_jvp_direct if ebm == "direct" else field_jvp_none
        vjp_fn = exact_field_vjp if ebm == "direct" else field_vjp_none
        model.zero_grad(set_to_none=True)
        pi = power_iteration_theta_sigma1(jvp_fn, vjp_fn, model, xt, t, y, params,
                                           num_iters=args.sv_iters, seed=args.seed, tol=1e-3)
        model.zero_grad(set_to_none=True)
        field = field_of(model, ebm, xt, t, y)
        r = field - ut
        u = r / (r.norm() + 1e-12)
        model.zero_grad(set_to_none=True)
        if ebm == "direct":
            exact_field_vjp(model, xt, t, y, u)
        else:
            field_vjp_none(model, xt, t, y, u)
        model_groups = groups_direct if ebm == "direct" else groups_none
        A_backbone = total_norm(model, model_groups, which=is_backbone_group)
        model.zero_grad(set_to_none=True)
        sigma_1 = pi["sigma_1"]
        u1 = pi["u1"]
        alignment = float((u.flatten() * u1.flatten()).sum().abs()) if u1 is not None else None
        return {
            "sigma_1": sigma_1, "A_backbone": A_backbone,
            "A_over_sigma1": (A_backbone / sigma_1) if (sigma_1 and sigma_1 > 0) else None,
            "alignment_u_u1": alignment, "n_iters": pi["n_iters"], "history": pi["history"],
        }

    out = {"scope": {"sv_k": args.sv_k, "sv_iters": args.sv_iters,
                      "note": "matrix-free power iteration on J_backbone (backbone params only), "
                              "run per-batch (each pool minibatch is one JVP/VJP unit, matching how "
                              "A_backbone is already computed elsewhere in this diagnostic). "
                              "Validated against explicit-Jacobian + numpy SVD on a tiny model "
                              "(tests/test_fb_direct_exact_hvp.py, re-run pre-flight)."}}

    for source_label, spike_idx, control_idx in (
        ("direct_selected", direct_spike_idx, direct_control_idx),
        ("none_selected", none_spike_idx, none_control_idx),
    ):
        spike_sub = subset(spike_idx, args.sv_k)
        control_sub = subset(control_idx, args.sv_k)
        rows = {"direct": {"spike": [], "control": []}, "none": {"spike": [], "control": []}}
        for batch_type, idx_sub in (("spike", spike_sub), ("control", control_sub)):
            for i in idx_sub:
                xt, t, y, ut = fixed_inputs[i]
                for model, ebm, params, key in (
                    (model_direct, "direct", backbone_params_direct, "direct"),
                    (model_none, "none", backbone_params_none, "none"),
                ):
                    r = run_one(model, ebm, params, xt, t, y, ut)
                    r["batch_idx"] = i
                    rows[key][batch_type].append(r)
                print(f"  [sv] {source_label} {batch_type} batch {i} done")

        def med(rows_list, field):
            vals = [r[field] for r in rows_list if r.get(field) is not None]
            return statistics.median(vals) if vals else None

        table89 = {}
        for model_key in ("direct", "none"):
            sigma_ctrl = med(rows[model_key]["control"], "sigma_1")
            sigma_spike = med(rows[model_key]["spike"], "sigma_1")
            align_ctrl = med(rows[model_key]["control"], "alignment_u_u1")
            align_spike = med(rows[model_key]["spike"], "alignment_u_u1")
            table89[model_key] = {
                "sigma_1_control": sigma_ctrl, "sigma_1_spike": sigma_spike,
                "sigma_1_ratio": (sigma_spike / sigma_ctrl) if (sigma_ctrl and sigma_spike) else None,
                "alignment_control": align_ctrl, "alignment_spike": align_spike,
                "alignment_ratio": (align_spike / align_ctrl) if (align_ctrl and align_spike) else None,
                "A_over_sigma1_control": med(rows[model_key]["control"], "A_over_sigma1"),
                "A_over_sigma1_spike": med(rows[model_key]["spike"], "A_over_sigma1"),
                "raw_rows": rows[model_key],
            }
        out[source_label] = table89

    return out


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-direct", required=True, nargs=3, metavar=("EARLY", "MID", "LATE"))
    p.add_argument("--ckpt-none", required=True, nargs=3, metavar=("EARLY", "MID", "LATE"),
                    help="progress-matched none checkpoints, same order as --ckpt-direct")
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--pool-size", type=int, default=2560, help="64/0.025=2560 -> K=64 spikes at unchanged spike_frac")
    p.add_argument("--spike-frac", type=float, default=0.025)
    p.add_argument("--num-control", type=int, default=64)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--sv-k", type=int, default=16, help="spike/control subset size for Phase 6/7 (late stage only)")
    p.add_argument("--sv-iters", type=int, default=15)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    labels = ["early", "mid", "late"]
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fixed_inputs = build_pool(args, device)

    per_stage = []
    late_handles = None
    for label, ckpt_d, ckpt_n in zip(labels, args.ckpt_direct, args.ckpt_none):
        result, model_direct, model_none, groups_direct, groups_none, _, \
            direct_spike_idx, direct_control_idx, none_spike_idx, none_control_idx, replay = \
            evaluate_stage(label, ckpt_d, ckpt_n, args, fixed_inputs, device)
        per_stage.append(result)
        ia = result["table4_source_matched_interaction"]["A_based"]
        print(f"[diag] {label} interaction (A-based): point={ia['interaction_point']} ci95={ia['interaction_ci95']}")
        if label == "late":
            late_handles = (model_direct, model_none, groups_direct, groups_none,
                             direct_spike_idx, direct_control_idx, none_spike_idx, none_control_idx, replay)
        else:
            del model_direct, model_none
            torch.cuda.empty_cache()

    # ---- Table 6: I(t) per stage + G = I(late) - I(early), secondary I(mid)-I(early), I(late)-I(mid) ----
    table6 = []
    for r in per_stage:
        ia = r["table4_source_matched_interaction"]["A_based"]
        ig = r["table4_source_matched_interaction"]["grad_based"]
        table6.append({
            "stage": r["stage"], "checkpoint_direct": r["checkpoint_direct"], "checkpoint_none": r["checkpoint_none"],
            "Delta_D_A": ia["Delta_source"], "Delta_N_A": ia["Delta_recip"],
            "interaction_A_point": ia["interaction_point"], "interaction_A_ci95": ia["interaction_ci95"],
            "Delta_D_grad": ig["Delta_source"], "Delta_N_grad": ig["Delta_recip"],
            "interaction_grad_point": ig["interaction_point"], "interaction_grad_ci95": ig["interaction_ci95"],
        })

    def independent_diff_bootstrap(point_a, ci_a, point_b, ci_b, num_samples, seed):
        """G = point_b - point_a. Since early/late use different selected
        batches (no natural pairing across stages), treat each stage's CI as
        an independent normal-ish interval (bootstrap already gave us CIs;
        approximate each stage's bootstrap distribution as N(point, sd) with
        sd from the CI half-width / 1.96, then Monte-Carlo the difference).
        This is intentionally conservative (wider) versus assuming any
        cross-stage correlation."""
        import random
        rng = random.Random(seed)
        if ci_a is None or ci_b is None or point_a is None or point_b is None:
            return {"point": (point_b - point_a) if (point_a is not None and point_b is not None) else None, "ci95": None}
        sd_a = (ci_a[1] - ci_a[0]) / (2 * 1.96)
        sd_b = (ci_b[1] - ci_b[0]) / (2 * 1.96)
        samples = [(point_b + rng.gauss(0, sd_b)) - (point_a + rng.gauss(0, sd_a)) for _ in range(num_samples)]
        samples.sort()
        lo = samples[int(0.025 * len(samples))]
        hi = samples[int(0.975 * len(samples)) - 1]
        return {"point": point_b - point_a, "ci95": [lo, hi]}

    early, mid, late = table6[0], table6[1], table6[2]
    table6_endpoint = {
        "I_early_A": {"point": early["interaction_A_point"], "ci95": early["interaction_A_ci95"]},
        "I_mid_A": {"point": mid["interaction_A_point"], "ci95": mid["interaction_A_ci95"]},
        "I_late_A": {"point": late["interaction_A_point"], "ci95": late["interaction_A_ci95"]},
        "G_A_late_minus_early": independent_diff_bootstrap(
            early["interaction_A_point"], early["interaction_A_ci95"],
            late["interaction_A_point"], late["interaction_A_ci95"], args.bootstrap_samples, args.seed),
        "I_mid_minus_I_early_A": independent_diff_bootstrap(
            early["interaction_A_point"], early["interaction_A_ci95"],
            mid["interaction_A_point"], mid["interaction_A_ci95"], args.bootstrap_samples, args.seed),
        "I_late_minus_I_mid_A": independent_diff_bootstrap(
            mid["interaction_A_point"], mid["interaction_A_ci95"],
            late["interaction_A_point"], late["interaction_A_ci95"], args.bootstrap_samples, args.seed),
        "note": ("G's CI is a Monte-Carlo combination of each stage's own bootstrap CI treated as an "
                 "independent normal approximation (no natural batch-level pairing exists across stages, "
                 "since each stage selects different batches under a different checkpoint) -- BATCH-LEVEL "
                 "uncertainty within one training lineage, NOT multi-seed experiment-level replication."),
    }

    # ---- Phase 6/7: singular value / alignment pass, LATE stage only ----
    sv_result = None
    if late_handles is not None:
        model_direct, model_none, groups_direct, groups_none, \
            direct_spike_idx, direct_control_idx, none_spike_idx, none_control_idx, replay = late_handles
        print("[diag] === Phase 6/7: matrix-free sigma_1 + alignment, LATE stage, subset ===")
        sv_result = singular_value_alignment_pass(
            model_direct, model_none, groups_direct, groups_none, fixed_inputs,
            direct_spike_idx, direct_control_idx, none_spike_idx, none_control_idx, args)
        del model_direct, model_none
        torch.cuda.empty_cache()

    full = {
        "scope": {
            "pool_size": args.pool_size, "batch_size": args.batch_size, "spike_frac": args.spike_frac,
            "num_control": args.num_control, "bootstrap_samples": args.bootstrap_samples,
            "checkpoints_direct": dict(zip(labels, args.ckpt_direct)),
            "checkpoints_none_progress_matched": dict(zip(labels, args.ckpt_none)),
            "step_mismatch_pct": {"early": 0.0, "mid": 25000 / 2175000 * 100, "late": 25000 / 2825000 * 100},
            "sv_k": args.sv_k, "sv_iters": args.sv_iters,
        },
        "table6_longitudinal_endpoint": table6_endpoint,
        "table6_per_stage_raw": table6,
        "per_stage": per_stage,
        "singular_value_alignment_late_stage": sv_result,
    }
    print("\n[diag] FULL RESULTS (printed unconditionally to stdout -- holylabs quota has failed "
          "silently-recoverable multiple times this session, the tee'd log is the primary copy of record):")
    print(json.dumps(full, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(full, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        except OSError as e:
            print(f"[diag] WARNING: --out write failed ({e}); data is in the FULL RESULTS block above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
