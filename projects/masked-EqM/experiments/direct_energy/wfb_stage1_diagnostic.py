"""
WFB-EqM Stage 1: zero-update real-checkpoint causal diagnostic (2026-08-11).

PRIMARY STAGE-1 QUESTION (per spec Section 8): on batches where raw direct
training exhibits a large gradient spike, does WFB (compute_wfb_gradient,
fb_direct/exact_hvp.py) remove the parameter-gradient amplification while
receiving the EXACT SAME residual and EXACT SAME frozen model -- no weight
updates, no retraining?

Methodology mirrors topk_subspace_diagnostic.py's already-validated design
(reused verbatim: load_model/block_groups/is_backbone_group/build_pool/
probe_direct/rank_pool/select_spike_control from
matched_replay_jacobian_diagnostic.py), swapping the head-only top-k
subspace metrics for WFB's g_raw/g_wfb pair (fb_direct/exact_hvp.py's
compute_wfb_gradient). Batch selection uses the STANDARD ranking metric
(real pre-clip grad_norm via probe_direct, i.e. exact_fwrev_backward's
double-backward gradient norm) -- independent of WFB itself, so there is no
selection-bias circularity in "does WFB suppress batches selected by a
DIFFERENT, ordinary criterion."

Checkpoint: CKPT_DIRECT_LATE (fwrev_ep80_lambda0_job37780076, step 2825000)
-- confirmed-working path from this session's prior jobs (netscratch root,
NOT holylabs; see documentation/longitudinal_jacobian_audit.md). This
checkpoint is well past the ~1.5M-step instability onset, in the elevated
(~24x grown) clip-rate regime -- i.e. "during the region where clip
frequency rises substantially" per spec Section 8, without an extra
cluster round-trip to enumerate earlier checkpoints.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matched_replay_jacobian_diagnostic import (  # noqa: E402
    block_groups, build_pool, is_backbone_group, load_model,
    probe_direct, rank_pool, select_spike_control,
)
from fb_direct.exact_hvp import compute_wfb_gradient  # noqa: E402


def group_split_norm(name_by_id, groups, params_used, tensors):
    """Split a list of per-parameter tensors into head/backbone/total L2
    norms via the same block_groups taxonomy topk_subspace/matched_replay
    already use. Aligns by id(param) via `name_by_id` (not position/zip
    against model.named_parameters()) because compute_wfb_gradient's
    returned `params` may have fewer entries than the caller's original
    list -- e.g. this codebase's `pos_embed` is registered as an
    nn.Parameter with requires_grad=False (a fixed sinusoidal embedding),
    which compute_wfb_gradient filters out (see its docstring) since
    autograd.grad raises unconditionally on a requires_grad=False input
    regardless of allow_unused."""
    head_sq, backbone_sq, total_sq = 0.0, 0.0, 0.0
    for p, t in zip(params_used, tensors):
        sq = float(t.detach().float().square().sum())
        total_sq += sq
        g = groups[name_by_id[id(p)]]
        if is_backbone_group(g):
            backbone_sq += sq
        elif g != "other":
            head_sq += sq
    return {"total": total_sq ** 0.5, "head": head_sq ** 0.5, "backbone": backbone_sq ** 0.5}


def cosine(a_list, b_list):
    dot = sum(float((a * b).sum()) for a, b in zip(a_list, b_list))
    na = sum(float((a ** 2).sum()) for a in a_list) ** 0.5
    nb = sum(float((b ** 2).sum()) for b in b_list) ** 0.5
    return dot / (na * nb + 1e-30)


def run_batch(model, groups, name_by_id, xt, t, y, ut, rho, k, lambda_max_num_iters, seed, track_memory):
    if track_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = compute_wfb_gradient(model, xt, t, y, ut, params=list(model.parameters()),
                                   rho=rho, k=k, lambda_max_num_iters=lambda_max_num_iters, seed=seed)
    wall_s = time.time() - t0
    peak_mem_mb = float(torch.cuda.max_memory_allocated() / 1e6) if (track_memory and torch.cuda.is_available()) else None

    raw_split = group_split_norm(name_by_id, groups, result["params"], result["g_raw"])
    wfb_split = group_split_norm(name_by_id, groups, result["params"], result["g_wfb"])

    row = {
        "r_norm": result["r_norm"],
        "g_raw_norm": result["g_raw_norm"], "g_wfb_norm": result["g_wfb_norm"],
        "raw_head": raw_split["head"], "raw_backbone": raw_split["backbone"],
        "wfb_head": wfb_split["head"], "wfb_backbone": wfb_split["backbone"],
        "lambda_max": result["lambda_max"], "lam": result["lam"], "T_eigmax": result["T_eigmax"],
        "m_lanczos": result["m"], "breakdown": result["breakdown"], "breakdown_reason": result["breakdown_reason"],
        "ortho_error": result["ortho_error"],
        "cosine_raw_wfb": cosine(result["g_raw"], result["g_wfb"]),
        "wall_s": wall_s, "peak_mem_mb": peak_mem_mb,
    }
    model.zero_grad(set_to_none=True)
    return row


def run_group(model, groups, name_by_id, fixed_inputs, idx_list, rho, k, lambda_max_num_iters, seed, label, track_memory):
    rows = []
    for n_done, i in enumerate(idx_list):
        xt, t, y, ut = fixed_inputs[i]
        row = run_batch(model, groups, name_by_id, xt, t, y, ut, rho, k, lambda_max_num_iters, seed, track_memory)
        row["batch_idx"] = i
        rows.append(row)
        if (n_done + 1) % 4 == 0 or (n_done + 1) == len(idx_list):
            print(f"  [stage1] {label} {n_done + 1}/{len(idx_list)}: "
                  f"g_raw={row['g_raw_norm']:.4f} g_wfb={row['g_wfb_norm']:.4f} "
                  f"m={row['m_lanczos']} breakdown={row['breakdown_reason']} wall={row['wall_s']:.2f}s")
    return rows


def ratio_field(spike_rows, control_rows, field):
    sv = [r[field] for r in spike_rows if r.get(field) is not None]
    cv = [r[field] for r in control_rows if r.get(field) is not None]
    ms, mc = (statistics.median(sv) if sv else None), (statistics.median(cv) if cv else None)
    return {"spike_median": ms, "control_median": mc,
            "ratio": (ms / mc) if (ms is not None and mc and mc > 0) else None,
            "n_spike": len(sv), "n_control": len(cv)}


def k_convergence_check(model, groups, fixed_inputs, idx_subset, rho, ks, lambda_max_num_iters, seed):
    """Spec Section 7.A / 8: 'k=2,4,8,12 where affordable' -- run on a SMALL
    subsample (not the full spike/control population, to bound compute) to
    confirm k=8 (the production default) has converged relative to k=12."""
    rows = []
    for i in idx_subset:
        xt, t, y, ut = fixed_inputs[i]
        by_k = {}
        for k in ks:
            model.zero_grad(set_to_none=True)
            t0 = time.time()
            result = compute_wfb_gradient(model, xt, t, y, ut, params=list(model.parameters()),
                                           rho=rho, k=k, lambda_max_num_iters=lambda_max_num_iters, seed=seed)
            by_k[k] = {"g_wfb_norm": result["g_wfb_norm"], "T_eigmax": result["T_eigmax"],
                       "m_lanczos": result["m"], "wall_s": time.time() - t0}
            model.zero_grad(set_to_none=True)
        rows.append({"batch_idx": i, "by_k": by_k})
        print(f"  [stage1] k-convergence batch {i}: " +
              ", ".join(f"k={k}:g_wfb={by_k[k]['g_wfb_norm']:.4f}" for k in ks))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-direct", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-size", type=int, default=1280)
    p.add_argument("--spike-frac", type=float, default=0.025)
    p.add_argument("--num-control", type=int, default=32)
    p.add_argument("--rho", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lambda-max-num-iters", type=int, default=20)
    p.add_argument("--clip-threshold", type=float, default=6.87141,
                    help="locked training max_grad_norm (job 36806020 calibration)")
    p.add_argument("--k-convergence-n", type=int, default=4,
                    help="number of top spike batches to run the k=2,4,8,12 convergence check on")
    p.add_argument("--vae", default="ema")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import types
    pool_args = types.SimpleNamespace(data_path=args.data_path, pool_size=args.pool_size, batch_size=args.batch_size,
                                       image_size=args.image_size, seed=args.seed, vae=args.vae)
    fixed_inputs = build_pool(pool_args, device)

    print("[stage1] loading direct checkpoint (instability regime)...")
    model = load_model(args.ckpt_direct, args.model, args.image_size, args.num_classes, "direct", device)
    groups = block_groups(model)
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    if n_frozen:
        print(f"[stage1] note: {n_frozen} parameter tensor(s) have requires_grad=False "
              f"(e.g. fixed pos_embed) -- compute_wfb_gradient filters these out automatically.")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage1] model params: {n_params}")

    print("[stage1] ranking pool via probe_direct (real pre-clip grad_norm, ordinary criterion "
          "-- independent of WFB, no selection-bias circularity)...")
    pool_grad, pool_loss = rank_pool(probe_direct, model, fixed_inputs, "direct(late)")

    spike_idx, control_idx = select_spike_control(pool_grad, args.spike_frac, args.num_control)
    clip_idx = [i for i, v in enumerate(pool_grad) if v > args.clip_threshold]
    print(f"[stage1] pool={len(pool_grad)}, spike_n={len(spike_idx)} (top {args.spike_frac*100:.1f}%), "
          f"control_n={len(control_idx)} (40th-60th pctile band), actual_clip_events={len(clip_idx)} "
          f"(threshold={args.clip_threshold})")

    track_memory = torch.cuda.is_available()
    rows = {}
    for label, idx_list in (("spike", spike_idx), ("control", control_idx), ("actual_clip_events", clip_idx)):
        if not idx_list:
            rows[label] = []
            continue
        print(f"[stage1] === {label}: {len(idx_list)} batches ===")
        rows[label] = run_group(model, groups, name_by_id, fixed_inputs, idx_list, args.rho, args.k,
                                 args.lambda_max_num_iters, args.seed, label, track_memory)

    table_1_residual_sanity = ratio_field(rows["spike"], rows["control"], "r_norm")
    table_2_raw_vs_wfb_ratio = {
        "g_raw": ratio_field(rows["spike"], rows["control"], "g_raw_norm"),
        "g_wfb": ratio_field(rows["spike"], rows["control"], "g_wfb_norm"),
    }
    table_3_head_backbone = {
        "raw_head": ratio_field(rows["spike"], rows["control"], "raw_head"),
        "raw_backbone": ratio_field(rows["spike"], rows["control"], "raw_backbone"),
        "wfb_head": ratio_field(rows["spike"], rows["control"], "wfb_head"),
        "wfb_backbone": ratio_field(rows["spike"], rows["control"], "wfb_backbone"),
    }
    table_4_lambda_diagnostics = {
        "lambda_max_spike_median": statistics.median([r["lambda_max"] for r in rows["spike"]]) if rows["spike"] else None,
        "lambda_max_control_median": statistics.median([r["lambda_max"] for r in rows["control"]]) if rows["control"] else None,
        "T_eigmax_vs_lambda_max_max_rel_err": max(
            [abs(r["T_eigmax"] - r["lambda_max"]) / (abs(r["lambda_max"]) + 1e-30)
             for r in rows["spike"] + rows["control"]], default=None),
        "ortho_error_max": max([r["ortho_error"] for r in rows["spike"] + rows["control"]], default=None),
        "breakdown_reasons": sorted(set(r["breakdown_reason"] for r in rows["spike"] + rows["control"] if r["breakdown"])),
    }
    table_5_cosine = {
        "spike_median": statistics.median([r["cosine_raw_wfb"] for r in rows["spike"]]) if rows["spike"] else None,
        "control_median": statistics.median([r["cosine_raw_wfb"] for r in rows["control"]]) if rows["control"] else None,
    }
    table_6_cost = {
        "wall_s_median": statistics.median([r["wall_s"] for r in rows["spike"] + rows["control"]]) if (rows["spike"] or rows["control"]) else None,
        "peak_mem_mb_median": statistics.median([r["peak_mem_mb"] for r in rows["spike"] + rows["control"] if r["peak_mem_mb"] is not None])
            if any(r["peak_mem_mb"] is not None for r in rows["spike"] + rows["control"]) else None,
    }
    table_7_actual_clip_events = ratio_field(rows["actual_clip_events"], rows["control"], "g_wfb_norm") if rows["actual_clip_events"] else None

    top_spike_by_raw = sorted(rows["spike"], key=lambda r: -r["g_raw_norm"])[:args.k_convergence_n]
    kconv_idx = [r["batch_idx"] for r in top_spike_by_raw]
    print(f"[stage1] k-convergence check on {len(kconv_idx)} most-severe spike batches...")
    table_8_k_convergence = k_convergence_check(model, groups, fixed_inputs, kconv_idx, args.rho,
                                                 (2, 4, 8, 12), args.lambda_max_num_iters, args.seed)

    results = {
        "checkpoint": args.ckpt_direct, "n_params": n_params,
        "pool_size": len(pool_grad), "n_spike": len(spike_idx), "n_control": len(control_idx),
        "n_actual_clip_events": len(clip_idx), "rho": args.rho, "k": args.k,
        "table_1_residual_sanity": table_1_residual_sanity,
        "table_2_raw_vs_wfb_spike_control_ratio": table_2_raw_vs_wfb_ratio,
        "table_3_head_backbone_ratios": table_3_head_backbone,
        "table_4_lambda_diagnostics": table_4_lambda_diagnostics,
        "table_5_cosine_raw_wfb": table_5_cosine,
        "table_6_cost": table_6_cost,
        "table_7_actual_clip_events_vs_control": table_7_actual_clip_events,
        "table_8_k_convergence": table_8_k_convergence,
        "rows": rows,
    }

    print("\n[stage1] SUMMARY:")
    print(json.dumps({k: v for k, v in results.items() if k != "rows"}, indent=2,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"\n[stage1] PRIMARY RESULT: raw spike/control ratio={table_2_raw_vs_wfb_ratio['g_raw']['ratio']}, "
          f"WFB spike/control ratio={table_2_raw_vs_wfb_ratio['g_wfb']['ratio']}")

    print("\n[stage1] FULL RESULTS:")
    print(json.dumps(results, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        except OSError as e:
            print(f"[stage1] WARNING: --out write failed ({e}); data is in the FULL RESULTS block above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
