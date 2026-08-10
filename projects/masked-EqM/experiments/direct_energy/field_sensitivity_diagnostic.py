"""
Field-sensitivity + none-control diagnostic (2026-08-10, follow-up to
activation_scale_diagnostic.py per external review).

That review made two corrections to the 2026-08-10 activation-scale result:
  1. No `none` control was run -- weight-specnorm growth might be generic to
     this Transformer backbone, not `direct`-specific. This script adds it.
  2. Raw |E| growth is gauge-ambiguous (E -> E + C(theta) doesn't change the
     field f = -grad_z E; ScalarEnergyHead.linear has a bias term that can
     drift for free). The mechanistically relevant, gauge-safe quantity is
     the FUNCTION-SPACE sensitivity of the field itself to a parameter
     perturbation:

        U = ||f_{theta+dtheta}(x) - f_theta(x)|| / ||dtheta||

     estimated here by finite difference along random unit directions in
     parameter space (each tensor perturbed proportional to its own norm,
     so no layer dominates by scale alone). This directly tests whether
     `direct`'s second-order (grad_z E) construction amplifies a given
     amount of parameter drift more than `none`'s first-order field output
     does -- the actual candidate mechanism, not a proxy for it.

Cheap: forward-only for `none`, one ordinary (non-double) backward per
forward for `direct` (train=False -> create_graph=False). No training.

Run (single GPU):
  python experiments/direct_energy/field_sensitivity_diagnostic.py \
      --ckpts-direct <path1> ... --ckpts-none <path1> ... --data-path <dir>
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
from activation_scale_diagnostic import attn_logit_stats, weight_spectral_norms  # noqa: E402


def load_model(ckpt_path, model_name, image_size, num_classes, ebm, device):
    from models import EqM_models
    latent_size = image_size // 8
    model = EqM_models[model_name](input_size=latent_size, num_classes=num_classes, ebm=ebm).to(device)
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [warn] load_state_dict: missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def field_of(model, xt, t, y):
    """f_theta(x): for `direct`, one ordinary (create_graph=False) backward
    inside EqM.forward; for `none`, a plain forward. No grad tracked on our
    side -- only theta needs grad for the caller's perturbed-vs-base diff."""
    return model(xt, t, y, train=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts-direct", nargs="+", required=True)
    p.add_argument("--ckpts-none", nargs="+", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=3)
    p.add_argument("--num-directions", type=int, default=5)
    p.add_argument("--eps", type=float, default=1e-3, help="relative per-tensor perturbation scale")
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

    print(f"[diag] probe bank: {args.num_batches} x {args.batch_size}")
    probe_bank = build_probe_bank(args.data_path, args.num_batches, args.batch_size,
                                   args.image_size, args.seed)
    # Fix (xt, t, y) once so every checkpoint/arm/direction sees IDENTICAL
    # inputs -- required for a clean paired U_t comparison.
    fixed_inputs = []
    for x, y in probe_bank:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
        t, x0, x1 = transport.sample(x1)
        t = t.to(x1)
        t, xt, _ut = transport.path_sampler.plan(t, x0, x1)
        fixed_inputs.append((xt.detach(), t.detach(), y))

    def run_arm(arm_name, ckpts, ebm):
        rows = []
        for ckpt_path in ckpts:
            model = load_model(ckpt_path, args.model, args.image_size, args.num_classes, ebm, device)
            wnorms = weight_spectral_norms(model)
            logit_rows = []
            for xt, t, y in fixed_inputs:
                logit_rows.append(attn_logit_stats(model, xt, t, y, device, energy_only=(ebm != "none")))

            # Base fields (no grad needed on our side).
            with torch.no_grad() if ebm == "none" else torch.enable_grad():
                base_fields = [field_of(model, xt, t, y).detach() for xt, t, y in fixed_inputs]

            base_state = copy.deepcopy(model.state_dict())
            base_theta_norm = float(torch.stack(
                [v.detach().float().norm() ** 2 for v in base_state.values() if v.is_floating_point()]
            ).sum().sqrt())

            gen = torch.Generator(device="cpu").manual_seed(args.seed)
            u_vals = []
            for d in range(args.num_directions):
                perturbed = {}
                total_delta_sq = 0.0
                for name, v in base_state.items():
                    if not v.is_floating_point():
                        perturbed[name] = v
                        continue
                    direction = torch.randn(v.shape, generator=gen).to(device=v.device, dtype=v.dtype)
                    direction = direction / (direction.norm() + 1e-12)
                    delta = args.eps * v.detach().float().norm().item() * direction
                    total_delta_sq += float((delta.float() ** 2).sum())
                    perturbed[name] = v + delta.to(v.dtype)
                delta_norm = total_delta_sq ** 0.5
                model.load_state_dict(perturbed)
                with torch.no_grad() if ebm == "none" else torch.enable_grad():
                    pert_fields = [field_of(model, xt, t, y).detach() for xt, t, y in fixed_inputs]
                model.load_state_dict(base_state)  # restore before next direction

                for base_f, pert_f in zip(base_fields, pert_fields):
                    df_norm = float((pert_f - base_f).float().norm())
                    u_vals.append(df_norm / (delta_norm + 1e-30))

            row = {
                "ckpt": ckpt_path,
                "U_mean": sum(u_vals) / len(u_vals),
                "U_max": max(u_vals),
                "base_theta_norm": base_theta_norm,
                "base_field_norm_mean": sum(float(f.norm()) for f in base_fields) / len(base_fields),
                "logit_abs_mean": sum(r["logit_abs_mean_pooled"] for r in logit_rows) / len(logit_rows),
                "logit_abs_max": max(r["logit_abs_max_pooled"] for r in logit_rows),
                **wnorms,
            }
            rows.append(row)
            print(f"[diag] {arm_name} {os.path.basename(ckpt_path)}: "
                  f"U(mean/max)={row['U_mean']:.4f}/{row['U_max']:.4f} "
                  f"theta_norm={row['base_theta_norm']:.2f} "
                  f"wspecnorm(median/max)={row['weight_specnorm_median']:.3f}/{row['weight_specnorm_max']:.3f} "
                  f"logit_abs(mean/max)={row['logit_abs_mean']:.3f}/{row['logit_abs_max']:.3f}")
            del model
            torch.cuda.empty_cache()
        return rows

    summary = {
        "direct": run_arm("direct", args.ckpts_direct, "direct"),
        "none": run_arm("none", args.ckpts_none, "none"),
    }
    print(f"\n[diag] SUMMARY:\n{json.dumps(summary, indent=2)}")
    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=2)
        except OSError as e:
            # Don't let a storage hiccup (e.g. quota) mask a successful run --
            # the SUMMARY above (captured by the caller's `tee`) is the data
            # of record either way. See 2026-08-10 activation_scale_diagnostic
            # incident (job 38066731): identical failure mode.
            print(f"[diag] WARNING: --out write failed ({e}); data is in the "
                  f"SUMMARY block above, printed via tee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
