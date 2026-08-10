"""
Spike-batch residual-conditioned amplification + layerwise top-sensitivity
(2026-08-10, targeted follow-up after two mean/isotropic diagnostics --
z-Hessian curvature and random-direction field-sensitivity U_t -- came back
null or wrong-signed for the growing-clip-rate mechanism. Both share a blind
spot: MEAN-based or ISOTROPIC-direction measurements can't see a TAIL-driven
phenomenon. This instruments the actual tail instead.

Scope note: reproducing the EXACT historical spike steps (e.g. lambda0 step
2811423, grad_norm=80839) would require deterministically replaying the
DataLoader's sampler order from epoch 0 -- impractical. Instead: live-
classify spike vs ordinary batches at the existing checkpoints by drawing a
large pool of REAL batches and ranking by the training-identical grad_norm.
Top percentile = spike, matched middle-band sample = ordinary. Same
phenomenon at accessible scale (same logic this project already applies to
CIFAR-as-proxy-for-IN-1K).

Two measurements, both DIRECTION-INFORMED (not random, not mean-based):

1. Residual-conditioned amplification (global + per-block, ~free): for
   `direct`, gain = ||dL/dtheta|| / ||w||, where w is the REAL JVP/residual
   direction exact_fwrev_backward already computes internally (now returned
   as w_norm) -- not a random direction. For `none`, the analogous residual
   is r = field - target (what the loss literally differentiates), gain =
   ||dL/dtheta|| / ||r||. Per-block: ||dL/dtheta_block|| / ||w or r||, a
   layerwise gain PROFILE -- does one block's relative contribution spike
   specifically on spike batches?

2. Layerwise top-sensitivity (bounded batch/block subset): per block,
   finite-difference perturbation ALONG THAT BLOCK'S OWN REAL GRADIENT
   DIRECTION (a power-iteration warm start -- much better estimate of a
   dominant singular direction than a random Gaussian one), measuring
   ||df||/||dtheta_block|| -- does perturbing block b toward where it's
   already being pushed amplify the field disproportionately more on spike
   batches, and is that pattern direct-specific vs none?

Run (single GPU):
  python experiments/direct_energy/spike_batch_amplification_diagnostic.py \
      --ckpt-direct <path> --ckpt-none <path> --data-path <dir>
"""
import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curvature_clip_diagnostic import build_probe_bank  # noqa: E402
from fb_direct.exact_hvp import exact_fwrev_backward  # noqa: E402
from transport.utils import mean_flat  # noqa: E402


def load_model(ckpt_path, model_name, image_size, num_classes, ebm, device):
    from models import EqM_models
    latent_size = image_size // 8
    model = EqM_models[model_name](input_size=latent_size, num_classes=num_classes, ebm=ebm).to(device)
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [warn] load_state_dict: missing={missing} unexpected={unexpected}")
    # eval() -> self.training=False -> CFG label dropout disabled, so the
    # SAME batch always produces the SAME gradient across repeated probe_*
    # calls (needed: the directional-sensitivity step re-calls probe_fn on
    # an already-classified batch to repopulate .grad, and must see the
    # identical computation, not a fresh random dropout draw).
    model.eval()
    return model


def block_groups(model):
    """name-prefix -> group label, coarse (per-SiTBlock) granularity."""
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


def per_group_grad_norms(model, groups):
    sq = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = groups[name]
        sq[g] = sq.get(g, 0.0) + float(p.grad.detach().float().square().sum())
    return {g: v ** 0.5 for g, v in sq.items()}


def probe_direct(model, xt, t, y, ut):
    model.zero_grad(set_to_none=True)
    stats = exact_fwrev_backward(model, xt, t, y, ut, gp_lambda=0.0)
    grad_norm = float(torch.stack(
        [p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None]
    ).sum() ** 0.5)
    return grad_norm, stats["w_norm"], stats["loss_main"]


def probe_none(model, xt, t, y, ut):
    model.zero_grad(set_to_none=True)
    field = model(xt, t, y, train=True)
    residual = (field - ut).detach()
    loss = mean_flat((field - ut) ** 2).mean()
    loss.backward()
    grad_norm = float(torch.stack(
        [p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None]
    ).sum() ** 0.5)
    return grad_norm, float(residual.norm()), float(loss.detach())


def directional_layerwise_sensitivity(model, ebm, xt, t, y, groups, group_list, eps, field_fn):
    """For each group in group_list: perturb ONLY that group's params along
    ITS OWN real gradient direction (already populated on model.parameters()
    by the caller's probe_* call), measure ||df||/||dtheta_group||."""
    base_state = copy.deepcopy(model.state_dict())
    with torch.no_grad() if ebm == "none" else torch.enable_grad():
        base_field = field_fn(model, xt, t, y).detach()
    grads_by_name = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}

    out = {}
    for g in group_list:
        names = [n for n, gg in groups.items() if gg == g and n in grads_by_name]
        if not names:
            continue
        perturbed = dict(base_state)
        total_delta_sq = 0.0
        for name in names:
            grad = grads_by_name[name]
            gnorm = float(grad.norm())
            if gnorm < 1e-12:
                continue
            direction = grad / gnorm
            v = base_state[name]
            delta = eps * float(v.detach().float().norm()) * direction.to(v.dtype)
            total_delta_sq += float((delta.float() ** 2).sum())
            perturbed[name] = v + delta
        delta_norm = total_delta_sq ** 0.5
        if delta_norm < 1e-20:
            continue
        model.load_state_dict(perturbed)
        with torch.no_grad() if ebm == "none" else torch.enable_grad():
            pert_field = field_fn(model, xt, t, y).detach()
        model.load_state_dict(base_state)
        out[g] = float((pert_field - base_field).float().norm()) / delta_norm
    return out


def field_of(model, xt, t, y):
    return model(xt, t, y, train=False)


def run_arm(arm_name, ckpt_path, ebm, probe_fn, fixed_inputs, args, device):
    model = load_model(ckpt_path, args.model, args.image_size, args.num_classes, ebm, device)
    groups = block_groups(model)
    depth = len(model.blocks)

    print(f"[diag] {arm_name}: classifying {len(fixed_inputs)} pool batches by grad_norm...")
    pool_results = []
    for i, (xt, t, y, ut) in enumerate(fixed_inputs):
        grad_norm, resid_norm, loss_main = probe_fn(model, xt, t, y, ut)
        group_norms = per_group_grad_norms(model, groups)
        pool_results.append({
            "idx": i, "grad_norm": grad_norm, "resid_norm": resid_norm,
            "loss_main": loss_main, "gain": grad_norm / (resid_norm + 1e-30),
            "group_norms": group_norms,
        })

    pool_results.sort(key=lambda r: r["grad_norm"], reverse=True)
    n = len(pool_results)
    n_spike = max(1, int(round(args.spike_frac * n)))
    spikes = pool_results[:n_spike]
    mid_lo, mid_hi = int(0.4 * n), int(0.6 * n)
    ordinary = sorted(pool_results[mid_lo:mid_hi], key=lambda r: r["idx"])[:args.num_ordinary]

    def summarize(rows, label):
        gains = [r["gain"] for r in rows]
        grad_norms = [r["grad_norm"] for r in rows]
        print(f"[diag] {arm_name} {label}: n={len(rows)} "
              f"grad_norm(mean/max)={sum(grad_norms)/len(grad_norms):.3f}/{max(grad_norms):.3f} "
              f"gain=grad_norm/resid(mean/max)={sum(gains)/len(gains):.4f}/{max(gains):.4f}")
        # per-group mean SHARE of total grad_norm^2 (which block dominates)
        shares = {}
        for r in rows:
            total_sq = sum(v ** 2 for v in r["group_norms"].values()) + 1e-30
            for g, v in r["group_norms"].items():
                shares.setdefault(g, []).append(v ** 2 / total_sq)
        mean_shares = {g: sum(v) / len(v) for g, v in shares.items()}
        top3 = sorted(mean_shares.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print(f"    top-3 grad^2-share groups: {top3}")
        return {"gains": gains, "mean_shares": mean_shares}

    spike_summary = summarize(spikes, "SPIKE")
    ordinary_summary = summarize(ordinary, "ORDINARY")

    # Directional layerwise top-sensitivity on a bounded subset.
    directional = {"spike": [], "ordinary": []}
    dir_groups = [f"block{i}" for i in range(depth)] + ["energy_head", "final_layer", "x_embedder"]
    dir_groups = [g for g in dir_groups if g in set(groups.values())]
    for label, rows in (("spike", spikes[:args.num_directional_batches]),
                         ("ordinary", ordinary[:args.num_directional_batches])):
        for r in rows:
            xt, t, y, ut = fixed_inputs[r["idx"]]
            probe_fn(model, xt, t, y, ut)  # repopulate .grad for this exact batch
            sens = directional_layerwise_sensitivity(
                model, ebm, xt, t, y, groups, dir_groups, args.eps, field_of,
            )
            directional[label].append({"idx": r["idx"], "sensitivity": sens})
            print(f"[diag] {arm_name} {label} idx={r['idx']}: "
                  f"top sensitivity groups = "
                  f"{sorted(sens.items(), key=lambda kv: kv[1], reverse=True)[:3]}")

    del model
    torch.cuda.empty_cache()
    return {
        "spike_summary": spike_summary, "ordinary_summary": ordinary_summary,
        "directional": directional,
        "n_spike": n_spike, "n_ordinary": len(ordinary),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-direct", required=True)
    p.add_argument("--ckpt-none", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32, help="match training's global batch")
    p.add_argument("--pool-size", type=int, default=500)
    p.add_argument("--spike-frac", type=float, default=0.02)
    p.add_argument("--num-ordinary", type=int, default=10)
    p.add_argument("--num-directional-batches", type=int, default=3)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers.models import AutoencoderKL
    from transport import create_transport
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    transport = create_transport("Linear", "velocity", None, None, None)

    print(f"[diag] pool: {args.pool_size} batches x {args.batch_size} (real ImageNet)")
    probe_bank = build_probe_bank(args.data_path, args.pool_size, args.batch_size,
                                   args.image_size, args.seed)
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

    summary = {
        "direct": run_arm("direct", args.ckpt_direct, "direct", probe_direct, fixed_inputs, args, device),
        "none": run_arm("none", args.ckpt_none, "none", probe_none, fixed_inputs, args, device),
    }
    print(f"\n[diag] SUMMARY:\n{json.dumps(summary, indent=2)}")
    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=2)
        except OSError as e:
            print(f"[diag] WARNING: --out write failed ({e}); data is in the SUMMARY block above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
