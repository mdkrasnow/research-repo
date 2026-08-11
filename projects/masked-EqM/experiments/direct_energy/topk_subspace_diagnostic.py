"""
Stage A: top-k high-gain SUBSPACE confirmation (2026-08-10).

Primary scientific question (per documentation/topk_subspace_audit.md):
does the normalized residual increasingly concentrate in a small high-gain
singular SUBSPACE of the ENERGY-HEAD parameter->field Jacobian J_h on
genuine direct gradient-tail events, more so than in none, and does this
strengthen as the tail gets more extreme?

Do NOT read this as "does the residual align with u_1" (a single, possibly
unstable vector under near-degenerate spectra) -- P_k/Q_k below are
SUBSPACE quantities by construction (sum over the top-k orthonormal U_k
recovered by block_subspace_iteration_theta), which is exactly the object
A7 requires.

Reuses, unchanged: load_model/block_groups/is_backbone_group/build_pool/
probe_direct/probe_none/select_spike_control/replay_union/ratio_row/
paired_self_minus_other/matched_interaction_bootstrap (matched_replay_
jacobian_diagnostic.py), field_jvp_direct/field_jvp_none/exact_field_vjp/
field_vjp_none/block_subspace_iteration_theta (fb_direct/exact_hvp.py).
"""
import argparse
import json
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matched_replay_jacobian_diagnostic import (  # noqa: E402
    block_groups, build_pool, is_backbone_group, load_model,
    probe_direct, probe_none, rank_pool, select_spike_control,
)
from fb_direct.exact_hvp import (  # noqa: E402
    block_subspace_iteration_theta, exact_field_vjp, field_jvp_direct,
    field_jvp_none, field_vjp_none,
)

HEAD_GROUP = {"direct": "energy_head", "none": "final_layer"}


def head_params(model, groups, ebm):
    g = HEAD_GROUP[ebm]
    return [(n, p) for n, p in model.named_parameters() if groups[n] == g]


def residual_and_A(model, ebm, groups, xt, t, y, ut, eps=1e-12):
    """Canonical residual r, unit direction u_r = r/||r||, and A_head =
    ||J_head^T u_r|| via the EXISTING exact_field_vjp/field_vjp_none
    (restricting the readback to head params only -- these VJP functions
    already populate .grad for every parameter; we only read the head
    subset back, exactly the technique power_iteration_theta_sigma1 already
    uses for a params subset)."""
    if ebm == "direct":
        z = xt.detach().clone().requires_grad_(True)
        E = model(z, t, y, energy_only=True)
        g = torch.autograd.grad(E.sum(), z, create_graph=False)[0].detach()
        field = -g
    else:
        with torch.no_grad():
            field = model(xt, t, y, train=False)
    r = field - ut
    u_r = r / (r.norm() + eps)

    model.zero_grad(set_to_none=True)
    if ebm == "direct":
        exact_field_vjp(model, xt, t, y, u_r)
    else:
        field_vjp_none(model, xt, t, y, u_r)
    hp = head_params(model, groups, ebm)
    A_head_sq = sum(float(p.grad.detach().float().square().sum()) for _, p in hp if p.grad is not None)
    A_backbone = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if is_backbone_group(groups[n]):
            A_backbone += float(p.grad.detach().float().square().sum())
    A_total_sq = A_head_sq + A_backbone
    model.zero_grad(set_to_none=True)
    return {
        "u_r": u_r.detach(), "residual_rms": float((r.float() ** 2).mean().sqrt()),
        "A_head": A_head_sq ** 0.5, "A_backbone": A_backbone ** 0.5,
        "A_total": A_total_sq ** 0.5,
        "C_head": (A_head_sq / A_total_sq) if A_total_sq > 0 else None,
        "C_backbone": (A_backbone / A_total_sq) if A_total_sq > 0 else None,
    }


def subspace_metrics(model, ebm, groups, xt, t, y, ut, k, num_iters, seed):
    """One batch's P_k/Q_k + spectral diagnostics against the HEAD Jacobian."""
    hp = head_params(model, groups, ebm)
    params = [p for _, p in hp]
    jvp_fn = field_jvp_direct if ebm == "direct" else field_jvp_none
    vjp_fn = exact_field_vjp if ebm == "direct" else field_vjp_none

    model.zero_grad(set_to_none=True)
    sub = block_subspace_iteration_theta(jvp_fn, vjp_fn, model, xt, t, y, params,
                                          k=k, num_iters=num_iters, seed=seed, tol=1e-3)
    model.zero_grad(set_to_none=True)

    rA = residual_and_A(model, ebm, groups, xt, t, y, ut)
    u_r = rA["u_r"].reshape(-1)
    sigma = sub["sigma"]  # (k,)
    U = sub["U"]  # (n_out, k)
    proj = (U.T @ u_r)  # (k,)
    proj_sq = (proj ** 2)

    A_head_sq = rA["A_head"] ** 2
    P_cum, Q_cum = {}, {}
    running_p, running_qs = 0.0, 0.0
    for i in range(sigma.shape[0]):
        running_p += float(proj_sq[i])
        running_qs += float(sigma[i] ** 2 * proj_sq[i])
        if (i + 1) in (1, 4, 8, 16):
            P_cum[i + 1] = running_p
            Q_cum[i + 1] = (running_qs / A_head_sq) if A_head_sq > 0 else None

    return {
        "sigma": sigma.tolist(), "sigma_1_over_2": (float(sigma[0] / sigma[1]) if sigma.shape[0] > 1 and sigma[1] > 0 else None),
        "sigma_1_over_median": (float(sigma[0] / statistics.median(sigma.tolist())) if sigma.shape[0] > 0 else None),
        "P": P_cum, "Q": Q_cum,
        "ortho_error_V": sub["ortho_error_V"], "ortho_error_U": sub["ortho_error_U"],
        "n_iters": sub["n_iters"],
        "residual_rms": rA["residual_rms"], "A_head": rA["A_head"], "A_backbone": rA["A_backbone"],
        "A_total": rA["A_total"], "C_head": rA["C_head"], "C_backbone": rA["C_backbone"],
    }


def median_field(rows, path):
    vals = []
    for r in rows:
        v = r
        for key in path:
            v = v[key] if v is not None else None
        if v is not None:
            vals.append(v)
    return statistics.median(vals) if vals else None


def tail_group(pool_grad, spike_frac, num_control, K_target, label):
    n = len(pool_grad)
    spike_idx, control_idx = select_spike_control(pool_grad, spike_frac, num_control)
    return {"label": label, "n_pool": n, "spike_frac": spike_frac,
            "spike_idx": spike_idx, "control_idx": control_idx,
            "n_spike": len(spike_idx), "n_control": len(control_idx), "K_target": K_target}


def run_group(model, ebm, groups, fixed_inputs, idx_list, k, num_iters, seed, label):
    rows = []
    for n_done, i in enumerate(idx_list):
        xt, t, y, ut = fixed_inputs[i]
        m = subspace_metrics(model, ebm, groups, xt, t, y, ut, k, num_iters, seed)
        m["batch_idx"] = i
        rows.append(m)
        if (n_done + 1) % 8 == 0:
            print(f"  [subspace] {label} {ebm} {n_done + 1}/{len(idx_list)} done")
    return rows


def ratio(spike_rows, control_rows, field):
    sv = [r[field] for r in spike_rows if r.get(field) is not None]
    cv = [r[field] for r in control_rows if r.get(field) is not None]
    ms, mc = (statistics.median(sv) if sv else None), (statistics.median(cv) if cv else None)
    return {"spike_median": ms, "control_median": mc,
            "ratio": (ms / mc) if (ms is not None and mc and mc > 0) else None,
            "n_spike": len(sv), "n_control": len(cv)}


def q_ratio(spike_rows, control_rows, k):
    sv = [r["Q"].get(k) for r in spike_rows if r["Q"].get(k) is not None]
    cv = [r["Q"].get(k) for r in control_rows if r["Q"].get(k) is not None]
    ms, mc = (statistics.median(sv) if sv else None), (statistics.median(cv) if cv else None)
    return {"spike_median": ms, "control_median": mc,
            "ratio": (ms / mc) if (ms is not None and mc and mc > 0) else None}


def source_interaction_Q(direct_rows, k, num_samples, seed):
    """I_Q analog of the already-validated corrected interaction statistic:
    Delta_D^Q = log(median Q_k(direct spike)/Q_k(direct control)) evaluated
    for the DIRECT model on direct-selected batches; the "recip" term for
    none is computed by the caller with none-selected batches the same way,
    and I_Q = Delta_D^Q - Delta_N^Q."""
    import random
    rng = random.Random(seed)

    def qvals(rows):
        return [math.log(max(r["Q"].get(k, 1e-30) or 1e-30, 1e-30)) for r in rows]

    def point(vals):
        return statistics.median(vals) if vals else None

    def resample(vals):
        return statistics.median([vals[rng.randrange(len(vals))] for _ in range(len(vals))]) if vals else None

    sv, cv = qvals(direct_rows["spike"]), qvals(direct_rows["control"])
    delta_point = (point(sv) - point(cv)) if (sv and cv) else None
    samples = []
    for _ in range(num_samples):
        s, c = resample(sv), resample(cv)
        if s is None or c is None:
            continue
        samples.append(s - c)
    samples.sort()
    ci = [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]] if samples else None
    return {"Delta_point": delta_point, "Delta_ci95": ci, "n_bootstrap": len(samples)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-direct", required=True)
    p.add_argument("--ckpt-none", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--k2", type=int, default=16, help="opportunistic second k, 0 to skip")
    p.add_argument("--subspace-iters", type=int, default=30)
    p.add_argument("--clip-threshold", type=float, default=6.87141,
                    help="locked training max_grad_norm (job 36806020 calibration)")
    p.add_argument("--vae", default="ema")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- tail-severity pools: reuse the K=64@2.5% pool for the widest pool; build the
    # narrower-percentile pools by drawing a FRESH wider pool (Phase 1: widen the pool
    # for a rarer percentile, don't relax the criterion). All pools use the SAME seed
    # for image sampling per index -- NOT literally the same tensors across pool sizes
    # (build_probe_bank's sampler is deterministic given seed+size, so smaller pools are
    # prefixes of larger ones), documented in topk_subspace_audit.md Sec 4.
    tail_specs = [
        ("top2.5pct", 0.025, 2560, 64),
        ("top1.0pct", 0.010, 4800, 48),
        ("top0.5pct", 0.005, 6400, 32),
    ]
    max_pool = max(t[2] for t in tail_specs)

    import types
    pool_args = types.SimpleNamespace(data_path=args.data_path, pool_size=max_pool, batch_size=args.batch_size,
                                       image_size=args.image_size, seed=args.seed, vae=args.vae)
    fixed_inputs = build_pool(pool_args, device)

    print("[stageA] loading matched late-stage checkpoints...")
    model_direct = load_model(args.ckpt_direct, args.model, args.image_size, args.num_classes, "direct", device)
    model_none = load_model(args.ckpt_none, args.model, args.image_size, args.num_classes, "none", device)
    groups_direct = block_groups(model_direct)
    groups_none = block_groups(model_none)

    head_d = head_params(model_direct, groups_direct, "direct")
    head_n = head_params(model_none, groups_none, "none")
    print(f"[stageA] energy_head: {len(head_d)} tensors, {sum(p.numel() for _, p in head_d)} params: "
          f"{[(n, tuple(p.shape)) for n, p in head_d]}")
    print(f"[stageA] final_layer: {len(head_n)} tensors, {sum(p.numel() for _, p in head_n)} params: "
          f"{[(n, tuple(p.shape)) for n, p in head_n]}")

    print("[stageA] ranking full pool under both models (real pre-clip grad_norm)...")
    pool_grad_direct, _ = rank_pool(probe_direct, model_direct, fixed_inputs, "direct(late)")
    pool_grad_none, _ = rank_pool(probe_none, model_none, fixed_inputs, "none(late)")

    table_a1 = []
    tail_groups_direct, tail_groups_none = {}, {}
    for label, frac, pool_n, K in tail_specs:
        sub_pool_d = pool_grad_direct[:pool_n]
        sub_pool_n = pool_grad_none[:pool_n]
        gd = tail_group(sub_pool_d, frac, K, K, label)
        gn = tail_group(sub_pool_n, frac, K, K, label)
        tail_groups_direct[label] = gd
        tail_groups_none[label] = gn
        table_a1.append({"tail": label, "pool_size": pool_n,
                          "direct_spike_n": gd["n_spike"], "direct_control_n": gd["n_control"],
                          "none_spike_n": gn["n_spike"], "none_control_n": gn["n_control"]})

    # actual clip events: reuse the top2.5pct pool (2560), threshold-based, not percentile-based
    clip_pool = pool_grad_direct[:2560]
    clip_idx = [i for i, v in enumerate(clip_pool) if v > args.clip_threshold]
    control_2p5 = tail_groups_direct["top2.5pct"]["control_idx"]
    table_a1.append({"tail": "actual_clip_events", "pool_size": 2560,
                      "direct_spike_n": len(clip_idx), "direct_control_n": len(control_2p5),
                      "none_spike_n": None, "none_control_n": None,
                      "note": f"threshold={args.clip_threshold} (locked training max_grad_norm, job 36806020); "
                              "direct-only (this is a direct-side diagnostic pool; no none-side clip threshold "
                              "reconstruction attempted -- out of scope, see audit doc)."})
    tail_groups_direct["actual_clip_events"] = {"label": "actual_clip_events", "spike_idx": clip_idx,
                                                 "control_idx": control_2p5, "n_spike": len(clip_idx),
                                                 "n_control": len(control_2p5)}

    print(json.dumps({"table_a1_tail_populations": table_a1}, indent=2))

    ks = [args.k] + ([args.k2] if args.k2 and args.k2 != args.k else [])

    results = {"table_a1_tail_populations": table_a1, "by_tail": {}}
    for label, frac, pool_n, K in tail_specs + [("actual_clip_events", None, 2560, None)]:
        gd = tail_groups_direct[label]
        idx_all = sorted(set(gd["spike_idx"]) | set(gd["control_idx"]))
        print(f"[stageA] === tail={label}: {len(idx_all)} unique batches (direct-selected) ===")
        rows_direct = {bt: run_group(model_direct, "direct", groups_direct, fixed_inputs,
                                      gd[f"{bt}_idx"], max(ks), args.subspace_iters, args.seed, f"{label}/direct")
                       for bt in ("spike", "control")}
        rows_none_on_direct = {bt: run_group(model_none, "none", groups_none, fixed_inputs,
                                              gd[f"{bt}_idx"], max(ks), args.subspace_iters, args.seed, f"{label}/none-on-direct-batches")
                                for bt in ("spike", "control")}

        tail_result = {"n_spike": gd["n_spike"], "n_control": gd["n_control"], "by_k": {}}
        for k in ks:
            table_a2 = {"direct": ratio(rows_direct["spike"], rows_direct["control"], "residual_rms"),
                        "none": ratio(rows_none_on_direct["spike"], rows_none_on_direct["control"], "residual_rms")}
            table_a3 = {"direct_spike_sigma_median": [statistics.median([r["sigma"][i] for r in rows_direct["spike"]])
                                                       for i in range(k)] if rows_direct["spike"] else None,
                        "direct_control_sigma_median": [statistics.median([r["sigma"][i] for r in rows_direct["control"]])
                                                         for i in range(k)] if rows_direct["control"] else None,
                        "sigma_1_over_2_spike": median_field(rows_direct["spike"], ["sigma_1_over_2"]),
                        "sigma_1_over_2_control": median_field(rows_direct["control"], ["sigma_1_over_2"])}
            table_a4 = {}
            for src_label, rows in (("direct", rows_direct), ("none_on_direct_batches", rows_none_on_direct)):
                table_a4[src_label] = {}
                for bt in ("spike", "control"):
                    table_a4[src_label][bt] = {
                        "P1": median_field(rows[bt], ["P", 1]), "P4": median_field(rows[bt], ["P", 4]),
                        "P8": median_field(rows[bt], ["P", 8]) if k >= 8 else None,
                        "P16": median_field(rows[bt], ["P", 16]) if k >= 16 else None,
                        "Q1": median_field(rows[bt], ["Q", 1]), "Q4": median_field(rows[bt], ["Q", 4]),
                        "Q8": median_field(rows[bt], ["Q", 8]) if k >= 8 else None,
                        "Q16": median_field(rows[bt], ["Q", 16]) if k >= 16 else None,
                    }
            table_a5 = {"direct": {"P8": q_ratio(rows_direct["spike"], rows_direct["control"], 8) if k >= 8 else None,
                                    "Q8": q_ratio(rows_direct["spike"], rows_direct["control"], 8) if k >= 8 else None},
                        "none_on_direct_batches": {
                            "P8": q_ratio(rows_none_on_direct["spike"], rows_none_on_direct["control"], 8) if k >= 8 else None,
                            "Q8": q_ratio(rows_none_on_direct["spike"], rows_none_on_direct["control"], 8) if k >= 8 else None}}
            table_a6 = None
            if k >= 8:
                Delta_D_Q = source_interaction_Q(rows_direct, 8, args.bootstrap_samples, args.seed)
                Delta_N_Q = source_interaction_Q(rows_none_on_direct, 8, args.bootstrap_samples, args.seed + 1)
                I_Q = (Delta_D_Q["Delta_point"] - Delta_N_Q["Delta_point"]) \
                    if (Delta_D_Q["Delta_point"] is not None and Delta_N_Q["Delta_point"] is not None) else None
                table_a6 = {"Delta_D_Q": Delta_D_Q, "Delta_N_Q": Delta_N_Q, "I_Q_point": I_Q,
                            "note": "I_Q on DIRECT-selected batches only this pass (none-selected reciprocal "
                                    "pool would double the job's compute; direct-selected is the primary "
                                    "per-spec test of whether direct's own hard batches show head-subspace "
                                    "concentration more than none does on those SAME batches)."}
            table_a7 = {"direct": {"A_head": ratio(rows_direct["spike"], rows_direct["control"], "A_head"),
                                    "A_backbone": ratio(rows_direct["spike"], rows_direct["control"], "A_backbone"),
                                    "A_total": ratio(rows_direct["spike"], rows_direct["control"], "A_total"),
                                    "C_head_spike_median": median_field(rows_direct["spike"], ["C_head"]),
                                    "C_head_control_median": median_field(rows_direct["control"], ["C_head"])},
                        "none_on_direct_batches": {
                            "A_head": ratio(rows_none_on_direct["spike"], rows_none_on_direct["control"], "A_head"),
                            "A_backbone": ratio(rows_none_on_direct["spike"], rows_none_on_direct["control"], "A_backbone"),
                            "A_total": ratio(rows_none_on_direct["spike"], rows_none_on_direct["control"], "A_total"),
                            "C_head_spike_median": median_field(rows_none_on_direct["spike"], ["C_head"]),
                            "C_head_control_median": median_field(rows_none_on_direct["control"], ["C_head"])}}
            ortho_err = {"V_max": max([r["ortho_error_V"] for r in rows_direct["spike"] + rows_direct["control"]], default=None),
                         "U_max": max([r["ortho_error_U"] for r in rows_direct["spike"] + rows_direct["control"]], default=None)}
            tail_result["by_k"][k] = {
                "table_a2_residual_sanity": table_a2, "table_a3_spectral_structure": table_a3,
                "table_a4_subspace_projection": table_a4, "table_a5_spike_control_ratios": table_a5,
                "table_a6_source_interaction": table_a6, "table_a7_head_dominance": table_a7,
                "ortho_error": ortho_err,
            }
        results["by_tail"][label] = tail_result
        print(f"[stageA] {label} k={ks[0]}: Q8 direct spike/control ratio = "
              f"{tail_result['by_k'][ks[0]]['table_a5_spike_control_ratios']['direct']['Q8']}")

    print("\n[stageA] FULL RESULTS:")
    print(json.dumps(results, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        except OSError as e:
            print(f"[stageA] WARNING: --out write failed ({e}); data is in the FULL RESULTS block above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
