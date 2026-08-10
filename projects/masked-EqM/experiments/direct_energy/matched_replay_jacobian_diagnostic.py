"""
Matched-batch residual-conditioned Jacobian amplification (2026-08-10,
correcting spike_batch_amplification_diagnostic.py after Phase 0 audit).

============================================================
PHASE 0 AUDIT -- what was wrong with the previous instrumentation
============================================================
spike_batch_amplification_diagnostic.py computed, for `direct`:

    gain_direct = ||dL/dtheta|| / w_norm,   w_norm = ||exact_fwrev_backward's w||

and for `none`:

    gain_none = ||dL/dtheta|| / ||field - target||

exact_hvp.py's derivation (module docstring) gives, for gp_lambda=0:

    w = (2/(B*D)) * (g + u) = -(2/(B*D)) * (field - u) = -(2/(B*D)) * r

i.e. w_norm = (2/(B*D)) * ||r|| exactly -- w is the SAME residual r used by
`none`'s gain, rescaled by the constant 2/(B*D) (~1e-4 to 1e-5 for this
model's B, D). This is NOT a directional or model-specific quantity; it is
the per-element-mean reduction convention `direct`'s training loss happens
to use, baked into w by construction. So the two gains differ by
approximately B*D/2 for a structural, not scientific, reason -- confirmed
algebraically AND at FP64 machine precision by
tests/test_fb_direct_exact_hvp.py::test_field_vjp_direct_sign_relation_to_fwrev_w
(exact_field_vjp(model, x, t, y, v=-w) reproduces exact_fwrev_backward's
.grad bit-for-bit). Dividing the old direct/none gain ratios (~284/0.011 and
~84/0.0038, both ~2.2-2.6e4) by B*D/2 (=16384 at B=8, D=4*32*32=4096) brings
them to the SAME order of magnitude (~0.017 vs 0.011, ~0.005 vs 0.004) --
consistent with the normalization-mismatch explanation dominating the
previously reported gap, though a smaller genuine effect cannot be ruled
out from that back-of-envelope check alone. This script eliminates the
ambiguity by construction: both arms use the IDENTICAL canonical residual
r = field - target (raw, unreduced, no B*D rescaling) and the IDENTICAL VJP
convention A(x,r) = ||J^T (r/||r||)||, computed via exact_field_vjp /
field_vjp_none (fb_direct/exact_hvp.py), independently unit-tested against
double-backward / torch.autograd.grad references and a finite-difference
ground truth (tests/test_fb_direct_exact_hvp.py, all FP64 CPU, ALL PASS
before this script touched a GPU).

grad_norm reported in TABLE 2 is the REAL theta-space training gradient
norm (probe_direct / probe_none below, PRE-clip -- clipping is applied by
train.py's optimizer step, never inside these probes), used ONLY for spike
selection, exactly as in the original diagnostic; loss_main is
transport-plan MSE with mean_flat(...).mean() reduction (train.py's
convention). Neither is used to build the primary amplification metric.

============================================================
PHASE 1 -- canonical residual + validation identity
============================================================
r = field - target, BEFORE any reduction. Both arms satisfy
    loss_main == mean(r**2)   (mean_flat(...).mean() == global mean over
                                all elements, since every sample has the
                                same feature count D)
Runtime-asserted every batch; the script aborts loudly on violation.

============================================================
PHASE 2 -- matched batch bank (prospective live probe, per project
discipline: exact historical spike-step reconstruction would require
deterministic DataLoader sampler replay from epoch 0 -- impractical: same
scope note as the superseded diagnostic)
============================================================
ONE shared pool of `pool_size` real batches (fixed images/t/y/noise, same
seed -> IDENTICAL tensors for every model/checkpoint). Ranked TWICE, once
under each model's REAL pre-clip training grad_norm, giving the reciprocal
2x2 design:

    direct-selected spike / control   (top spike_frac / 40-60th pct by direct's grad_norm)
    none-selected   spike / control   (top spike_frac / 40-60th pct by none's grad_norm)

Every selected batch (deduplicated by pool index) is replayed through BOTH
checkpoints with the canonical VJP metric -- matched replay, not
independently-selected-then-compared groups (Hypothesis D's selection
artifact is exactly what this design is built to catch).

============================================================
PHASE 3 -- primary metric
============================================================
A_global   = ||J_theta^T (r/||r||)||           (all parameters)
A_backbone = ||J_theta^T (r/||r||)||            (blocks.*, x/t/y_embedder, pos_embed --
                                                  SHARED architecture across direct/none)
A_head     = ||J_theta^T (r/||r||)||            (energy_head for direct, final_layer for none --
                                                  model-specific, not cross-model-comparable)
A_layer[g] = ||J_theta_g^T (r/||r||)||          (per block_groups() group)

Scope note (compute-budget decision, documented per instructions rather
than silently applied): single LATE checkpoint pair (matches the most
unstable region already characterized by prior diagnostics), K spike
batches bounded by pool_size * spike_frac rather than the requested K=32 if
that would exceed the gpu_test time budget -- reported explicitly in the
JSON `scope` block, not hidden.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curvature_clip_diagnostic import build_probe_bank  # noqa: E402
from fb_direct.exact_hvp import exact_field_vjp, exact_fwrev_backward, field_vjp_none  # noqa: E402
from transport.utils import mean_flat  # noqa: E402


# ---------------------------------------------------------------- loading

def load_model(ckpt_path, model_name, image_size, num_classes, ebm, device):
    from models import EqM_models
    latent_size = image_size // 8
    model = EqM_models[model_name](input_size=latent_size, num_classes=num_classes, ebm=ebm).to(device)
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [warn] load_state_dict({ckpt_path}): missing={missing} unexpected={unexpected}")
    model.eval()  # deterministic: no CFG label dropout, same batch -> same output every call
    return model


def block_groups(model):
    groups = {}
    for name, _ in model.named_parameters():
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            groups[name] = f"block{idx}"
        elif name.startswith("energy_head."):
            groups[name] = "energy_head"
        elif name.startswith("final_layer."):
            groups[name] = "final_layer"
        elif name.startswith("x_embedder.") or name == "pos_embed":
            groups[name] = "x_embedder"
        elif name.startswith("t_embedder."):
            groups[name] = "t_embedder"
        elif name.startswith("y_embedder."):
            groups[name] = "y_embedder"
        else:
            groups[name] = "other"
    return groups


BACKBONE_GROUPS_PREFIX = ("block", "x_embedder", "t_embedder", "y_embedder")


def is_backbone_group(g):
    return g.startswith(BACKBONE_GROUPS_PREFIX)


def group_norms(model, groups, which=None):
    """which: None (all groups), or a predicate on group name -> bool."""
    sq = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = groups[name]
        if which is not None and not which(g):
            continue
        sq[g] = sq.get(g, 0.0) + float(p.grad.detach().float().square().sum())
    return {g: v ** 0.5 for g, v in sq.items()}


def total_norm(model, groups, which=None):
    sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if which is not None and not which(groups[name]):
            continue
        sq += float(p.grad.detach().float().square().sum())
    return sq ** 0.5


# ---------------------------------------------------------------- ranking probes (REAL training grad_norm)

def probe_direct(model, xt, t, y, ut):
    model.zero_grad(set_to_none=True)
    stats = exact_fwrev_backward(model, xt, t, y, ut, gp_lambda=0.0)
    grad_norm = total_norm(model, block_groups(model))
    model.zero_grad(set_to_none=True)
    return grad_norm, stats["loss_main"]


def probe_none(model, xt, t, y, ut):
    model.zero_grad(set_to_none=True)
    field = model(xt, t, y, train=True)
    loss = mean_flat((field - ut) ** 2).mean()
    loss.backward()
    grad_norm = total_norm(model, block_groups(model))
    model.zero_grad(set_to_none=True)
    return grad_norm, float(loss.detach())


# ---------------------------------------------------------------- canonical residual + primary VJP metric

def field_of(model, ebm, xt, t, y):
    """Field only, no grad tracking needed for the tensor itself."""
    if ebm == "direct":
        z = xt.detach().clone().requires_grad_(True)
        E = model(z, t, y, energy_only=True)
        g = torch.autograd.grad(E.sum(), z, create_graph=False)[0].detach()
        return -g
    with torch.no_grad():
        return model(xt, t, y, train=False)


def canonical_residual_and_validate(model, ebm, xt, t, y, ut, tol=1e-5):
    field = field_of(model, ebm, xt, t, y)
    r = field - ut
    residual_l2 = float(r.norm())
    residual_rms = float((r.float() ** 2).mean().sqrt())
    residual_abs_max = float(r.abs().max())
    pred_rms = float((field.float() ** 2).mean().sqrt())
    target_rms = float((ut.float() ** 2).mean().sqrt())

    # Recompute loss_main via the SAME reduction convention train.py/exact_hvp.py use
    # (mean_flat -> per-sample mean over features -> .mean() over batch), and validate
    # the identity loss_main == mean(r**2) == residual_rms**2.
    loss_main = float(mean_flat((field - ut) ** 2).mean())
    identity_lhs = loss_main
    identity_rhs = residual_rms ** 2
    abs_diff = abs(identity_lhs - identity_rhs)
    rel_diff = abs_diff / (abs(identity_rhs) + 1e-30)
    passed = rel_diff < tol
    if not passed:
        raise RuntimeError(
            f"[FATAL] canonical residual identity violated for ebm={ebm}: "
            f"loss_main={identity_lhs} vs residual_rms**2={identity_rhs} "
            f"(abs_diff={abs_diff}, rel_diff={rel_diff}, tol={tol}). "
            f"Aborting per Phase-1 requirement -- do not silently continue."
        )
    return {
        "field": field, "r": r,
        "residual_numel": int(r.numel()), "residual_l2": residual_l2,
        "residual_rms": residual_rms, "residual_abs_max": residual_abs_max,
        "prediction_rms": pred_rms, "target_rms": target_rms,
        "loss_main": loss_main, "identity_abs_diff": abs_diff,
        "identity_rel_diff": rel_diff, "identity_pass": passed,
    }


def amplification(eval_model, ebm, groups, xt, t, y, ut, eps=1e-12):
    """Primary metric: A(x,r) = ||J_theta^T (r/||r||)|| via exact VJP."""
    diag = canonical_residual_and_validate(eval_model, ebm, xt, t, y, ut)
    r = diag.pop("r")
    diag.pop("field")
    v = r / (r.norm() + eps)

    eval_model.zero_grad(set_to_none=True)
    if ebm == "direct":
        exact_field_vjp(eval_model, xt, t, y, v)
    else:
        field_vjp_none(eval_model, xt, t, y, v)

    a_global = total_norm(eval_model, groups)
    a_backbone = total_norm(eval_model, groups, which=is_backbone_group)
    a_head = total_norm(eval_model, groups, which=lambda g: not is_backbone_group(g) and g != "other")
    a_layer = group_norms(eval_model, groups)
    eval_model.zero_grad(set_to_none=True)

    diag.update({"A_global": a_global, "A_backbone": a_backbone, "A_head": a_head, "A_layer": a_layer})
    return diag


# ---------------------------------------------------------------- selection + bootstrap

def select_spike_control(pool_grad_norms, spike_frac, num_control):
    n = len(pool_grad_norms)
    order = sorted(range(n), key=lambda i: pool_grad_norms[i], reverse=True)
    n_spike = max(1, int(round(spike_frac * n)))
    spike_idx = order[:n_spike]
    mid_lo, mid_hi = int(0.4 * n), int(0.6 * n)
    mid_band = sorted(order[mid_lo:mid_hi])
    control_idx = mid_band[:num_control]
    return spike_idx, control_idx


def bootstrap_delta_ci(log_a_direct_spike, log_a_direct_control, log_a_none_spike, log_a_none_control,
                        num_samples=2000, seed=0):
    """Delta = median(logA_direct[spike]-logA_direct[control]) -
               median(logA_none[spike]-logA_none[control])
    Bootstrap over BATCHES (paired within model where possible), reported
    explicitly as batch-level uncertainty, not a multi-seed CI."""
    import random
    rng = random.Random(seed)

    def resample_median_diff(spike_vals, control_vals):
        if not spike_vals or not control_vals:
            return None
        rs = [spike_vals[rng.randrange(len(spike_vals))] for _ in range(len(spike_vals))]
        rc = [control_vals[rng.randrange(len(control_vals))] for _ in range(len(control_vals))]
        rs_sorted, rc_sorted = sorted(rs), sorted(rc)
        med_s = rs_sorted[len(rs_sorted) // 2]
        med_c = rc_sorted[len(rc_sorted) // 2]
        return med_s - med_c

    deltas = []
    for _ in range(num_samples):
        d_direct = resample_median_diff(log_a_direct_spike, log_a_direct_control)
        d_none = resample_median_diff(log_a_none_spike, log_a_none_control)
        if d_direct is None or d_none is None:
            continue
        deltas.append(d_direct - d_none)
    if not deltas:
        return {"point": None, "ci95": None, "n_bootstrap": 0}
    deltas.sort()
    import statistics
    med_direct = statistics.median(log_a_direct_spike) - statistics.median(log_a_direct_control) \
        if log_a_direct_spike and log_a_direct_control else None
    med_none = statistics.median(log_a_none_spike) - statistics.median(log_a_none_control) \
        if log_a_none_spike and log_a_none_control else None
    point_est = (med_direct - med_none) if (med_direct is not None and med_none is not None) else None
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]
    return {"point": point_est, "ci95": [lo, hi], "n_bootstrap": len(deltas)}


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-direct", required=True)
    p.add_argument("--ckpt-none", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8, help="gpu_test MIG 20G slice needs <=8 for exact_fwrev")
    p.add_argument("--pool-size", type=int, default=960)
    p.add_argument("--spike-frac", type=float, default=0.025)
    p.add_argument("--num-control", type=int, default=24)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--out-md", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers.models import AutoencoderKL
    from transport import create_transport
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    transport = create_transport("Linear", "velocity", None, None, None)

    print(f"[diag] pool: {args.pool_size} batches x {args.batch_size} (real ImageNet), seed={args.seed}")
    probe_bank = build_probe_bank(args.data_path, args.pool_size, args.batch_size, args.image_size, args.seed)
    fixed_inputs = []
    for x, y in probe_bank:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
        t, x0, x1 = transport.sample(x1)
        t = t.to(x1)
        t, xt, ut = transport.path_sampler.plan(t, x0, x1)
        ut = ut * transport.get_ct(t)[:, None, None, None]
        fixed_inputs.append((xt.detach(), t.detach(), y, ut.detach()))
    del vae
    torch.cuda.empty_cache()

    print("[diag] loading direct + none checkpoints (both resident)...")
    model_direct = load_model(args.ckpt_direct, args.model, args.image_size, args.num_classes, "direct", device)
    model_none = load_model(args.ckpt_none, args.model, args.image_size, args.num_classes, "none", device)
    groups_direct = block_groups(model_direct)
    groups_none = block_groups(model_none)

    # ---------------- Phase 2a: rank the shared pool under BOTH models' real training grad_norm ----------------
    print("[diag] ranking pool under direct model (real pre-clip grad_norm, exact_fwrev_backward)...")
    pool_grad_direct, pool_loss_direct = [], []
    for i, (xt, t, y, ut) in enumerate(fixed_inputs):
        gn, lm = probe_direct(model_direct, xt, t, y, ut)
        pool_grad_direct.append(gn)
        pool_loss_direct.append(lm)
        if (i + 1) % 100 == 0:
            print(f"  [direct ranking] {i + 1}/{len(fixed_inputs)}")

    print("[diag] ranking pool under none model (real pre-clip grad_norm)...")
    pool_grad_none, pool_loss_none = [], []
    for i, (xt, t, y, ut) in enumerate(fixed_inputs):
        gn, lm = probe_none(model_none, xt, t, y, ut)
        pool_grad_none.append(gn)
        pool_loss_none.append(lm)
        if (i + 1) % 100 == 0:
            print(f"  [none ranking] {i + 1}/{len(fixed_inputs)}")

    direct_spike_idx, direct_control_idx = select_spike_control(pool_grad_direct, args.spike_frac, args.num_control)
    none_spike_idx, none_control_idx = select_spike_control(pool_grad_none, args.spike_frac, args.num_control)

    print(f"[diag] direct-selected: {len(direct_spike_idx)} spike, {len(direct_control_idx)} control "
          f"(grad_norm spike range {min(pool_grad_direct[i] for i in direct_spike_idx):.3f}-"
          f"{max(pool_grad_direct[i] for i in direct_spike_idx):.3f})")
    print(f"[diag] none-selected:   {len(none_spike_idx)} spike, {len(none_control_idx)} control "
          f"(grad_norm spike range {min(pool_grad_none[i] for i in none_spike_idx):.3f}-"
          f"{max(pool_grad_none[i] for i in none_spike_idx):.3f})")

    groups_selection = {
        "direct_spike": direct_spike_idx, "direct_control": direct_control_idx,
        "none_spike": none_spike_idx, "none_control": none_control_idx,
    }
    all_selected_idx = sorted(set(direct_spike_idx) | set(direct_control_idx)
                               | set(none_spike_idx) | set(none_control_idx))
    print(f"[diag] {len(all_selected_idx)} unique batches selected for matched 2x2 replay")

    # ---------------- Phase 3: matched replay -- every selected batch through BOTH models ----------------
    replay = {}  # idx -> {"direct": {...}, "none": {...}}
    identity_failures = []
    for n_done, idx in enumerate(all_selected_idx):
        xt, t, y, ut = fixed_inputs[idx]
        try:
            r_direct = amplification(model_direct, "direct", groups_direct, xt, t, y, ut)
        except RuntimeError as e:
            identity_failures.append(str(e))
            r_direct = None
        try:
            r_none = amplification(model_none, "none", groups_none, xt, t, y, ut)
        except RuntimeError as e:
            identity_failures.append(str(e))
            r_none = None
        replay[idx] = {"direct": r_direct, "none": r_none}
        if (n_done + 1) % 20 == 0:
            print(f"  [replay] {n_done + 1}/{len(all_selected_idx)}")

    if identity_failures:
        print(f"[diag] FATAL: {len(identity_failures)} canonical-residual identity violations:")
        for msg in identity_failures[:5]:
            print("   ", msg)
        raise SystemExit(1)

    # ---------------- TABLE 1: instrumentation validation ----------------
    table1 = []
    for eval_name, ckpt in (("direct", args.ckpt_direct), ("none", args.ckpt_none)):
        diffs = [replay[idx][eval_name]["identity_abs_diff"] for idx in all_selected_idx]
        rels = [replay[idx][eval_name]["identity_rel_diff"] for idx in all_selected_idx]
        table1.append({
            "model": eval_name, "checkpoint": ckpt, "n_batches": len(all_selected_idx),
            "max_abs_diff": max(diffs), "max_rel_diff": max(rels), "pass": True,
        })

    # ---------------- TABLE 2: matched replay summary (per batch, per eval model) ----------------
    def batch_source_labels(idx):
        labels = []
        for gname, idxs in groups_selection.items():
            if idx in idxs:
                labels.append(gname)
        return labels

    table2 = []
    for idx in all_selected_idx:
        for eval_name in ("direct", "none"):
            d = replay[idx][eval_name]
            table2.append({
                "batch_idx": idx, "source_groups": batch_source_labels(idx), "eval_model": eval_name,
                "residual_rms": d["residual_rms"], "residual_l2": d["residual_l2"],
                "residual_abs_max": d["residual_abs_max"],
                "grad_norm_direct_ranking": pool_grad_direct[idx], "grad_norm_none_ranking": pool_grad_none[idx],
                "A_backbone": d["A_backbone"], "A_head": d["A_head"], "A_global": d["A_global"],
            })

    # ---------------- TABLE 3 + 4: primary aggregation + cross-model interaction ----------------
    import math
    import statistics

    def vals(group_idx, eval_name, key="A_backbone"):
        return [replay[idx][eval_name][key] for idx in group_idx]

    table3 = []
    for eval_name in ("direct", "none"):
        # own-selection ordinary/spike, then (below) cross-selection against the OTHER model's batches
        own_spike = direct_spike_idx if eval_name == "direct" else none_spike_idx
        own_control = direct_control_idx if eval_name == "direct" else none_control_idx
        sv = vals(own_spike, eval_name)
        cv = vals(own_control, eval_name)
        ratio = (statistics.median(sv) / statistics.median(cv)) if sv and cv and statistics.median(cv) > 0 else None
        table3.append({
            "eval_model": eval_name, "selection": "own",
            "A_backbone_control_median": statistics.median(cv) if cv else None,
            "A_backbone_spike_median": statistics.median(sv) if sv else None,
            "spike_over_control_ratio": ratio,
            "n_control": len(cv), "n_spike": len(sv),
        })
        # cross-selection: this eval model's response to the OTHER model's spike/control batches
        other_spike = none_spike_idx if eval_name == "direct" else direct_spike_idx
        other_control = none_control_idx if eval_name == "direct" else direct_control_idx
        sv2 = vals(other_spike, eval_name)
        cv2 = vals(other_control, eval_name)
        ratio2 = (statistics.median(sv2) / statistics.median(cv2)) if sv2 and cv2 and statistics.median(cv2) > 0 else None
        table3.append({
            "eval_model": eval_name, "selection": "cross (other model's batches)",
            "A_backbone_control_median": statistics.median(cv2) if cv2 else None,
            "A_backbone_spike_median": statistics.median(sv2) if sv2 else None,
            "spike_over_control_ratio": ratio2,
            "n_control": len(cv2), "n_spike": len(sv2),
        })

    def log_vals(group_idx, eval_name):
        return [math.log(max(v, 1e-30)) for v in vals(group_idx, eval_name)]

    log_a_direct_spike = log_vals(direct_spike_idx, "direct")
    log_a_direct_control = log_vals(direct_control_idx, "direct")
    log_a_none_spike = log_vals(none_spike_idx, "none")
    log_a_none_control = log_vals(none_control_idx, "none")

    ci = bootstrap_delta_ci(log_a_direct_spike, log_a_direct_control, log_a_none_spike, log_a_none_control,
                             num_samples=args.bootstrap_samples, seed=args.seed)
    r_direct = statistics.median(vals(direct_spike_idx, "direct")) / statistics.median(vals(direct_control_idx, "direct")) \
        if direct_spike_idx and direct_control_idx else None
    r_none = statistics.median(vals(none_spike_idx, "none")) / statistics.median(vals(none_control_idx, "none")) \
        if none_spike_idx and none_control_idx else None
    table4 = [{
        "checkpoint_pair": [args.ckpt_direct, args.ckpt_none],
        "R_direct_own_selection": r_direct, "R_none_own_selection": r_none,
        "Delta_point": ci["point"], "Delta_ci95": ci["ci95"], "n_bootstrap": ci["n_bootstrap"],
        "note": "batch-level bootstrap uncertainty only, NOT multi-seed replication",
    }]

    # ---------------- TABLE 5: layer localization ----------------
    all_groups = sorted((set(groups_direct.values()) | set(groups_none.values())) - {"other"})
    table5 = []
    for g in all_groups:
        row = {"group": g}
        for eval_name, spike_idx, control_idx in (("direct", direct_spike_idx, direct_control_idx),
                                                    ("none", none_spike_idx, none_control_idx)):
            sv = [replay[idx][eval_name]["A_layer"].get(g, 0.0) for idx in spike_idx]
            cv = [replay[idx][eval_name]["A_layer"].get(g, 0.0) for idx in control_idx]
            med_s = statistics.median(sv) if sv else None
            med_c = statistics.median(cv) if cv else None
            row[f"{eval_name}_ordinary"] = med_c
            row[f"{eval_name}_spike"] = med_s
            row[f"{eval_name}_ratio"] = (med_s / med_c) if (med_s is not None and med_c and med_c > 0) else None
        table5.append(row)
    table5.sort(key=lambda r: (r.get("direct_ratio") or 0), reverse=True)

    summary = {
        "scope": {
            "pool_size": args.pool_size, "batch_size": args.batch_size, "spike_frac": args.spike_frac,
            "num_control": args.num_control, "requested_K_spike": 32,
            "actual_K_direct_spike": len(direct_spike_idx), "actual_K_none_spike": len(none_spike_idx),
            "checkpoints": {"direct": args.ckpt_direct, "none": args.ckpt_none},
            "single_checkpoint_pair_only": True,
        },
        "table1_instrumentation_validation": table1,
        "table3_primary_aggregation": table3,
        "table4_cross_model_interaction": table4,
        "table5_layer_localization": table5,
        "n_table2_rows": len(table2),
    }
    print(f"\n[diag] SUMMARY:\n{json.dumps(summary, indent=2)}")

    full = dict(summary)
    full["table2_matched_replay"] = table2
    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(full, f, indent=2)
        except OSError as e:
            print(f"[diag] WARNING: --out write failed ({e}); data is in the SUMMARY block above.")
    if args.out_md:
        try:
            with open(args.out_md, "w") as f:
                f.write(render_markdown(summary, table5))
        except OSError as e:
            print(f"[diag] WARNING: --out-md write failed ({e}).")
    return 0


def render_markdown(summary, table5):
    lines = ["# matched_replay_jacobian_diagnostic\n"]
    lines.append("## Table 1 -- instrumentation validation\n")
    lines.append("| model | checkpoint | n | max_abs_diff | max_rel_diff | pass |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary["table1_instrumentation_validation"]:
        lines.append(f"| {r['model']} | {r['checkpoint']} | {r['n_batches']} | "
                      f"{r['max_abs_diff']:.3e} | {r['max_rel_diff']:.3e} | {r['pass']} |")
    lines.append("\n## Table 3 -- primary aggregation (A_backbone)\n")
    lines.append("| eval_model | selection | control_median | spike_median | ratio | n_c | n_s |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in summary["table3_primary_aggregation"]:
        lines.append(f"| {r['eval_model']} | {r['selection']} | {r['A_backbone_control_median']} | "
                      f"{r['A_backbone_spike_median']} | {r['spike_over_control_ratio']} | "
                      f"{r['n_control']} | {r['n_spike']} |")
    lines.append("\n## Table 4 -- cross-model interaction\n")
    for r in summary["table4_cross_model_interaction"]:
        lines.append(f"R_direct={r['R_direct_own_selection']}, R_none={r['R_none_own_selection']}, "
                      f"Delta={r['Delta_point']}, 95% CI={r['Delta_ci95']} (n_bootstrap={r['n_bootstrap']})")
    lines.append("\n## Table 5 -- layer localization (top by direct ratio)\n")
    lines.append("| group | direct_ord | direct_spike | direct_ratio | none_ord | none_spike | none_ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in table5[:15]:
        lines.append(f"| {r['group']} | {r.get('direct_ordinary')} | {r.get('direct_spike')} | "
                      f"{r.get('direct_ratio')} | {r.get('none_ordinary')} | {r.get('none_spike')} | "
                      f"{r.get('none_ratio')} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
