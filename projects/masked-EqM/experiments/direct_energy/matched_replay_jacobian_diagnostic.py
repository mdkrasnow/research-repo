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


# ---------------------------------------------------------------- selection + matched interaction
#
# CORRECTNESS NOTE (2026-08-10, post-hoc fix after external review): the
# first version of this script computed
#     Delta_WRONG = log(median A_direct[direct_spike]/A_direct[direct_control])
#                 - log(median A_none[none_spike]/A_none[none_control])
# i.e. it compared EACH model on its OWN independently-selected batches --
# exactly the unmatched comparison Hypothesis D warns against, and exactly
# what the matched-replay design exists to avoid. The reviewer's correct
# statistic holds the SOURCE of the batches fixed and swaps only the
# EVALUATING model:
#
#     Delta_D = log(A_direct[direct_spike]/A_direct[direct_control])
#             - log(A_none[direct_spike]/A_none[direct_control])          (source = direct's own spikes)
#     Delta_N = log(A_none[none_spike]/A_none[none_control])
#             - log(A_direct[none_spike]/A_direct[none_control])          (source = none's own spikes)
#     interaction = Delta_D - Delta_N
#
# Delta_D asks: for the batches direct itself flags as hard, does direct's
# OWN Jacobian amplify them more than none's Jacobian does (evaluated on
# the identical tensors)? That is the actual test of "direct-specific
# geometry," not "is direct's own worst case worse than none's own worst
# case" (which conflates the amplification effect with which batches each
# model's selection criterion happens to prefer).


def select_spike_control(pool_grad_norms, spike_frac, num_control):
    n = len(pool_grad_norms)
    order = sorted(range(n), key=lambda i: pool_grad_norms[i], reverse=True)
    n_spike = max(1, int(round(spike_frac * n)))
    spike_idx = order[:n_spike]
    mid_lo, mid_hi = int(0.4 * n), int(0.6 * n)
    mid_band = sorted(order[mid_lo:mid_hi])
    control_idx = mid_band[:num_control]
    return spike_idx, control_idx


def paired_self_minus_other(idxs, a_self, a_other):
    """Per-batch log(a_self[i]) - log(a_other[i]) for i in idxs. a_self/a_other:
    dict idx -> value (same batch, two models -- this is what makes it a
    matched/paired comparison rather than an independent-samples one)."""
    import math
    return [math.log(max(a_self[i], 1e-30)) - math.log(max(a_other[i], 1e-30)) for i in idxs]


def matched_interaction_bootstrap(rho_source_spike, rho_source_control,
                                   rho_recip_spike, rho_recip_control,
                                   num_samples=2000, seed=0):
    """rho_source_*: per-batch log(A_source_model) - log(A_other_model) on the
    SOURCE model's own-selected spike/control batches -> Delta_source.
    rho_recip_*: the reciprocal, per-batch log(A_recip_model) -
    log(A_other_model) on the RECIPROCAL model's own-selected batches ->
    Delta_recip. Returns point estimates for both and a bootstrap CI for
    Delta_source - Delta_recip (the interaction Table 4 should report)."""
    import random
    import statistics
    rng = random.Random(seed)

    def point(vals):
        return statistics.median(vals) if vals else None

    def resample_median(vals):
        if not vals:
            return None
        r = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        return statistics.median(r)

    delta_source_point = (point(rho_source_spike) - point(rho_source_control)) \
        if (rho_source_spike and rho_source_control) else None
    delta_recip_point = (point(rho_recip_spike) - point(rho_recip_control)) \
        if (rho_recip_spike and rho_recip_control) else None
    interaction_point = (delta_source_point - delta_recip_point) \
        if (delta_source_point is not None and delta_recip_point is not None) else None

    samples = []
    for _ in range(num_samples):
        d_s = resample_median(rho_source_spike)
        d_c = resample_median(rho_source_control)
        n_s = resample_median(rho_recip_spike)
        n_c = resample_median(rho_recip_control)
        if None in (d_s, d_c, n_s, n_c):
            continue
        samples.append((d_s - d_c) - (n_s - n_c))
    if not samples:
        return {"Delta_source": delta_source_point, "Delta_recip": delta_recip_point,
                "interaction_point": interaction_point, "interaction_ci95": None, "n_bootstrap": 0}
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples)) - 1]
    return {"Delta_source": delta_source_point, "Delta_recip": delta_recip_point,
            "interaction_point": interaction_point, "interaction_ci95": [lo, hi], "n_bootstrap": len(samples)}


# ---------------------------------------------------------------- pool + ranking + replay helpers

def build_pool(args, device):
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
    return fixed_inputs


def rank_pool(probe_fn, model, fixed_inputs, label):
    grads, losses = [], []
    for i, (xt, t, y, ut) in enumerate(fixed_inputs):
        gn, lm = probe_fn(model, xt, t, y, ut)
        grads.append(gn)
        losses.append(lm)
        if (i + 1) % 100 == 0:
            print(f"  [{label} ranking] {i + 1}/{len(fixed_inputs)}")
    return grads, losses


def replay_union(model_direct, model_none, groups_direct, groups_none, fixed_inputs, union_idx):
    replay = {}
    identity_failures = []
    for n_done, idx in enumerate(union_idx):
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
            print(f"  [replay] {n_done + 1}/{len(union_idx)}")
    if identity_failures:
        print(f"[diag] FATAL: {len(identity_failures)} canonical-residual identity violations:")
        for msg in identity_failures[:5]:
            print("   ", msg)
        raise SystemExit(1)
    return replay


def ratio_row(spike_idx, control_idx, d, extra=None):
    import statistics
    sv = [d[i] for i in spike_idx]
    cv = [d[i] for i in control_idx]
    med_s = statistics.median(sv) if sv else None
    med_c = statistics.median(cv) if cv else None
    ratio = (med_s / med_c) if (med_s is not None and med_c and med_c > 0) else None
    row = {"control_median": med_c, "spike_median": med_s, "ratio": ratio,
           "n_control": len(cv), "n_spike": len(sv)}
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------- per-checkpoint-pairing evaluation

def evaluate_checkpoint_pairing(ckpt_label, model_direct, model_none, groups_direct, groups_none,
                                 fixed_inputs, pool_grad_none, none_spike_idx, none_control_idx, args):
    print(f"[diag] === checkpoint pairing: direct={ckpt_label} vs none=FIXED reference ===")
    pool_grad_direct, _ = rank_pool(probe_direct, model_direct, fixed_inputs, f"direct({ckpt_label})")

    direct_spike_idx, direct_control_idx = select_spike_control(pool_grad_direct, args.spike_frac, args.num_control)
    print(f"[diag] direct({ckpt_label})-selected: {len(direct_spike_idx)} spike, {len(direct_control_idx)} control "
          f"(grad_norm spike range {min(pool_grad_direct[i] for i in direct_spike_idx):.3f}-"
          f"{max(pool_grad_direct[i] for i in direct_spike_idx):.3f})")

    union_idx = sorted(set(direct_spike_idx) | set(direct_control_idx)
                        | set(none_spike_idx) | set(none_control_idx))
    print(f"[diag] {len(union_idx)} unique batches for matched replay at {ckpt_label}")
    replay = replay_union(model_direct, model_none, groups_direct, groups_none, fixed_inputs, union_idx)

    table1 = []
    for eval_name in ("direct", "none"):
        diffs = [replay[idx][eval_name]["identity_abs_diff"] for idx in union_idx]
        rels = [replay[idx][eval_name]["identity_rel_diff"] for idx in union_idx]
        table1.append({
            "model": eval_name, "checkpoint_label": ckpt_label if eval_name == "direct" else "none_FIXED",
            "n_batches": len(union_idx), "max_abs_diff": max(diffs), "max_rel_diff": max(rels), "pass": True,
        })

    def source_labels(idx):
        out = []
        if idx in direct_spike_idx:
            out.append("direct_spike")
        if idx in direct_control_idx:
            out.append("direct_control")
        if idx in none_spike_idx:
            out.append("none_spike")
        if idx in none_control_idx:
            out.append("none_control")
        return out

    table2 = []
    for idx in union_idx:
        for eval_name in ("direct", "none"):
            d = replay[idx][eval_name]
            table2.append({
                "batch_idx": idx, "source_groups": source_labels(idx), "eval_model": eval_name,
                "residual_rms": d["residual_rms"], "residual_l2": d["residual_l2"],
                "residual_abs_max": d["residual_abs_max"],
                "grad_norm_direct_ranking": pool_grad_direct[idx], "grad_norm_none_ranking": pool_grad_none[idx],
                "A_backbone": d["A_backbone"], "A_head": d["A_head"], "A_global": d["A_global"],
            })

    a_direct = {idx: replay[idx]["direct"]["A_backbone"] for idx in union_idx}
    a_none = {idx: replay[idx]["none"]["A_backbone"] for idx in union_idx}
    resid_direct = {idx: replay[idx]["direct"]["residual_rms"] for idx in union_idx}
    resid_none = {idx: replay[idx]["none"]["residual_rms"] for idx in union_idx}
    grad_direct = {idx: pool_grad_direct[idx] for idx in union_idx}
    grad_none = {idx: pool_grad_none[idx] for idx in union_idx}

    # TABLE 3: own + CORRECTLY-matched cross (same batches, other model), both A-based and real-grad-based.
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

    # TABLE 4: the CORRECTED source-matched interaction (Delta_D - Delta_N), A-based and grad-based.
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

    # Decomposition |g| ~ (2/(B*D)) * |r| * A -- the (2/(B*D)) constant is IDENTICAL for direct and none
    # at fixed batch_size/latent shape, so it cancels exactly in a same-model spike/control ratio; grad_ratio
    # should equal residual_ratio * A_ratio for each model's OWN selection, up to that shared factor cancelling.
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

    all_groups = sorted((set(groups_direct.values()) | set(groups_none.values())) - {"other"})
    table5 = []
    for g in all_groups:
        row = {"group": g}
        for eval_name, spike_idx, control_idx in (("direct", direct_spike_idx, direct_control_idx),
                                                    ("none", none_spike_idx, none_control_idx)):
            layer_vals = {idx: replay[idx][eval_name]["A_layer"].get(g, 0.0) for idx in union_idx}
            rr = ratio_row(spike_idx, control_idx, layer_vals)
            row[f"{eval_name}_ordinary"] = rr["control_median"]
            row[f"{eval_name}_spike"] = rr["spike_median"]
            row[f"{eval_name}_ratio"] = rr["ratio"]
        table5.append(row)
    table5.sort(key=lambda r: (r.get("direct_ratio") or 0), reverse=True)

    return {
        "checkpoint_label": ckpt_label,
        "n_direct_spike": len(direct_spike_idx), "n_direct_control": len(direct_control_idx),
        "table1_instrumentation_validation": table1,
        "table2_matched_replay": table2,
        "table3_own_and_matched_cross": table3,
        "table4_source_matched_interaction": {
            "A_based": interaction_A, "grad_based": interaction_grad,
            "note": ("interaction_point = Delta_source - Delta_recip, where Delta_source = "
                     "log(R_direct-on-direct-selected) - log(R_none-on-direct-selected) [SAME "
                     "batches, both models] and Delta_recip is the reciprocal with none-selected "
                     "batches. This is the CORRECTED statistic -- see module-level correctness note."),
        },
        "residual_decomposition": decomposition,
        "table5_layer_localization": table5,
    }


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-direct", required=True, nargs="+",
                    help="one or more direct checkpoints, temporal order (early..late); "
                         "the temporal-trend table needs >=2")
    p.add_argument("--ckpt-direct-labels", nargs="+", default=None,
                    help="labels matching --ckpt-direct (default ckpt0, ckpt1, ...)")
    p.add_argument("--ckpt-none", required=True, help="single FIXED none reference checkpoint")
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
    args = p.parse_args()

    labels = args.ckpt_direct_labels or [f"ckpt{i}" for i in range(len(args.ckpt_direct))]
    assert len(labels) == len(args.ckpt_direct), "--ckpt-direct-labels must match --ckpt-direct in count"

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fixed_inputs = build_pool(args, device)

    print("[diag] loading FIXED none reference checkpoint...")
    model_none = load_model(args.ckpt_none, args.model, args.image_size, args.num_classes, "none", device)
    groups_none = block_groups(model_none)

    print("[diag] ranking pool under FIXED none model (once; reused for every direct checkpoint)...")
    pool_grad_none, _ = rank_pool(probe_none, model_none, fixed_inputs, "none")
    none_spike_idx, none_control_idx = select_spike_control(pool_grad_none, args.spike_frac, args.num_control)
    print(f"[diag] none-selected: {len(none_spike_idx)} spike, {len(none_control_idx)} control "
          f"(grad_norm spike range {min(pool_grad_none[i] for i in none_spike_idx):.3f}-"
          f"{max(pool_grad_none[i] for i in none_spike_idx):.3f})")

    per_checkpoint = []
    for label, ckpt_path in zip(labels, args.ckpt_direct):
        model_direct = load_model(ckpt_path, args.model, args.image_size, args.num_classes, "direct", device)
        groups_direct = block_groups(model_direct)
        result = evaluate_checkpoint_pairing(label, model_direct, model_none, groups_direct, groups_none,
                                              fixed_inputs, pool_grad_none, none_spike_idx, none_control_idx, args)
        result["checkpoint_path"] = ckpt_path
        per_checkpoint.append(result)
        print(f"[diag] {label} interaction (A-based): point={result['table4_source_matched_interaction']['A_based']['interaction_point']} "
              f"ci95={result['table4_source_matched_interaction']['A_based']['interaction_ci95']}")
        del model_direct
        torch.cuda.empty_cache()

    table6 = []
    for r in per_checkpoint:
        ia = r["table4_source_matched_interaction"]["A_based"]
        ig = r["table4_source_matched_interaction"]["grad_based"]
        table6.append({
            "checkpoint_label": r["checkpoint_label"], "checkpoint_path": r["checkpoint_path"],
            "Delta_D_A": ia["Delta_source"], "Delta_N_A": ia["Delta_recip"],
            "interaction_A_point": ia["interaction_point"], "interaction_A_ci95": ia["interaction_ci95"],
            "Delta_D_grad": ig["Delta_source"], "Delta_N_grad": ig["Delta_recip"],
            "interaction_grad_point": ig["interaction_point"], "interaction_grad_ci95": ig["interaction_ci95"],
        })

    full = {
        "scope": {
            "pool_size": args.pool_size, "batch_size": args.batch_size, "spike_frac": args.spike_frac,
            "num_control": args.num_control, "bootstrap_samples": args.bootstrap_samples,
            "checkpoints_direct": dict(zip(labels, args.ckpt_direct)),
            "checkpoint_none_FIXED": args.ckpt_none,
            "none_held_fixed_reason": (
                "none arm's absolute step numbering does not map cleanly onto direct's "
                "epoch40-continuation range (direct: 1.6M~=epoch40 start .. ~3.2M=epoch80 target, "
                "TIMEOUT at 2.825M; none's earliest surviving checkpoint here is 2.45M with unknown "
                "epoch correspondence to direct's numbering). Rather than guess a wrong early/mid "
                "none checkpoint, none is held at its one well-characterized late reference "
                "(epoch80.pt) across the whole direct-checkpoint sweep. This directly answers "
                "'does direct's own-batch amplification grow over direct's training relative to a "
                "stable reference' -- the load-bearing half of the temporal question -- but does "
                "NOT characterize any temporal structure in none's reciprocal Delta_N (held fixed "
                "at whatever none's own late-checkpoint value is)."
            ),
        },
        "table6_temporal_trend": table6,
        "per_checkpoint": per_checkpoint,
    }
    print("\n[diag] FULL RESULTS (printed unconditionally to stdout regardless of --out outcome -- "
          "holylabs disk quota has failed silently-recoverable multiple times this session, "
          "the tee'd log is now the primary copy of record):")
    print(json.dumps(full, indent=2))

    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(full, f, indent=2)
        except OSError as e:
            print(f"[diag] WARNING: --out write failed ({e}); data is in the FULL RESULTS block above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
