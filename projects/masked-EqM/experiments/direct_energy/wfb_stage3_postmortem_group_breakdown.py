"""
WFB-EqM Stage 3 postmortem: is the FBGN run's lambda_max growth (observed in
wfb_stage3_full_alpha1p0_cg's metrics.jsonl -- ~36x average growth over 300
steps, spiking to ~19,000x the initial value, coinciding exactly with every
same-batch loss spike) localized to BACKBONE parameters or the energy HEAD?

Loads the step100 and step300 checkpoints saved during that run (before/after
most of the growth) and compares top-singular-value estimates of the
field-parameter Jacobian, computed SEPARATELY for backbone vs head groups,
on the SAME held-out probe batches for a fair before/after comparison.
Reuses block_subspace_iteration_theta (already validated against explicit
SVD in tests/test_fb_direct_exact_hvp.py) and the backbone/head group
partition already used in the Stage A top-k-subspace diagnostic (which
found, on the ORIGINAL Adam-trained instability, that backbone -- not the
energy head -- dominates; this checks whether the same holds for the FBGN
run's curvature growth).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matched_replay_jacobian_diagnostic import (  # noqa: E402
    build_pool, block_groups, is_backbone_group,
)
from fb_direct.exact_hvp import block_subspace_iteration_theta, exact_field_vjp, field_jvp_direct  # noqa: E402


def load_model_from_stage3_ckpt(ckpt_path, model_name, image_size, num_classes, device):
    from models import EqM_models
    latent_size = image_size // 8
    model = EqM_models[model_name](input_size=latent_size, num_classes=num_classes, ebm="direct").to(device)
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [warn] load_state_dict({ckpt_path}): missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def sigma1_for_group(model, xt, t, y, params, k=4, num_iters=40, seed=0):
    result = block_subspace_iteration_theta(field_jvp_direct, exact_field_vjp, model, xt, t, y,
                                             params, k=k, num_iters=num_iters, seed=seed)
    return result["sigma"].tolist(), result["n_iters"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-before", required=True, help="e.g. wfb_stage3_full_alpha1p0_cg_step100.pt")
    p.add_argument("--ckpt-after", required=True, help="e.g. wfb_stage3_full_alpha1p0_cg_step300.pt")
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-probe-batches", type=int, default=4)
    p.add_argument("--k", type=int, default=4, help="top-k subspace dimension")
    p.add_argument("--num-iters", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae", default="ema")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import types
    pool_args = types.SimpleNamespace(data_path=args.data_path, pool_size=args.n_probe_batches, batch_size=args.batch_size,
                                       image_size=args.image_size, seed=args.seed, vae=args.vae)
    fixed_inputs = build_pool(pool_args, device)

    results = {"before": {"backbone": [], "head": []}, "after": {"backbone": [], "head": []}}
    for label, ckpt in (("before", args.ckpt_before), ("after", args.ckpt_after)):
        print(f"[postmortem] loading {label} checkpoint: {ckpt}")
        model = load_model_from_stage3_ckpt(ckpt, args.model, args.image_size, args.num_classes, device)
        groups = block_groups(model)
        backbone_params = [p for n, p in model.named_parameters() if is_backbone_group(groups[n]) and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if not is_backbone_group(groups[n]) and groups[n] != "other" and p.requires_grad]
        print(f"  backbone_params={len(backbone_params)} head_params={len(head_params)}")

        for i, (xt, t, y, ut) in enumerate(fixed_inputs):
            sigma_bb, n_iter_bb = sigma1_for_group(model, xt, t, y, backbone_params, k=args.k, num_iters=args.num_iters, seed=args.seed)
            sigma_hd, n_iter_hd = sigma1_for_group(model, xt, t, y, head_params, k=args.k, num_iters=args.num_iters, seed=args.seed)
            results[label]["backbone"].append(sigma_bb)
            results[label]["head"].append(sigma_hd)
            print(f"  [{label} batch {i}] backbone top-{args.k} sigma={[round(s,2) for s in sigma_bb]} "
                  f"(n_iters={n_iter_bb}) | head top-{args.k} sigma={[round(s,2) for s in sigma_hd]} (n_iters={n_iter_hd})")
        del model
        torch.cuda.empty_cache()

    print("\n[postmortem] SUMMARY (median top-1 sigma per group, before vs after):")
    for group in ("backbone", "head"):
        before_top1 = sorted(s[0] for s in results["before"][group])
        after_top1 = sorted(s[0] for s in results["after"][group])
        b_med = before_top1[len(before_top1) // 2]
        a_med = after_top1[len(after_top1) // 2]
        ratio = a_med / (b_med + 1e-30)
        print(f"  {group}: before_median_sigma1={b_med:.3f} after_median_sigma1={a_med:.3f} ratio={ratio:.3f}x")

    print("\n[postmortem] FULL RESULTS:")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
