"""
Offline activation/logit-scale diagnostic (2026-08-10, GPT-lineage scaling
follow-up to the (negative) z-Hessian curvature diagnostic).

Hypothesis (Wortsman et al. 2023 "Small-scale proxies for large-scale
Transformer training instabilities", DeepNet/Admin): the growing clip-rate
signature is generic Transformer-at-scale activation/logit growth over
training, not an EBM-specific z-space curvature effect. `direct`/fb_direct
amplify it because the theta-gradient is a mixed partial (product of
Jacobians along the same composed graph, differentiated twice), so the same
per-layer drift compounds harder than in `none`'s single-backward gradient.

Forward-pass-only, no backward needed -- cheap. Measures, per checkpoint:
  (a) pre-softmax attention logit scale (QK^T * scale) per block, pooled
  (b) raw scalar energy E(z) magnitude (energy_only=True forward)
  (c) per-block QKV/proj weight operator norm (top singular value)
Reports growth trend across checkpoints, both arms.

Run (single GPU):
  python experiments/direct_energy/activation_scale_diagnostic.py \
      --ckpts-lambda0 <path1> ... --ckpts-gp <path1> ... --data-path <dir>
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curvature_clip_diagnostic import build_probe_bank, load_model  # noqa: E402


def attn_logit_stats(model, x0, t, y, device):
    """Hook every block's qkv Linear, compute pre-softmax QK^T logits
    exactly as timm's Attention does, pool abs-value stats across blocks."""
    captured = {}

    def make_hook(name):
        def hook(_mod, _inp, out):
            captured[name] = out.detach()
        return hook

    handles = [b.attn.qkv.register_forward_hook(make_hook(i))
               for i, b in enumerate(model.blocks)]
    with torch.no_grad():
        model(x0, t, y, energy_only=True)
    for h in handles:
        h.remove()

    num_heads = model.num_heads
    all_abs_max, all_abs_mean = [], []
    for i, qkv in captured.items():
        B, N, threeD = qkv.shape
        head_dim = threeD // 3 // num_heads
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, _v = qkv.unbind(0)
        scale = head_dim ** -0.5
        logits = (q @ k.transpose(-2, -1)) * scale
        all_abs_max.append(float(logits.abs().max()))
        all_abs_mean.append(float(logits.abs().mean()))
    return {
        "logit_abs_mean_pooled": sum(all_abs_mean) / len(all_abs_mean),
        "logit_abs_max_pooled": max(all_abs_max),
        "logit_abs_max_per_block": all_abs_max,
    }


def weight_spectral_norms(model):
    norms = []
    for b in model.blocks:
        for w in (b.attn.qkv.weight, b.attn.proj.weight):
            norms.append(float(torch.linalg.svdvals(w.detach().float())[0]))
    return {"weight_specnorm_median": sorted(norms)[len(norms) // 2],
            "weight_specnorm_max": max(norms)}


def energy_stats(model, x0, t, y):
    with torch.no_grad():
        E = model(x0, t, y, energy_only=True)
    return {"E_mean": float(E.mean()), "E_abs_max": float(E.abs().max()), "E_std": float(E.std())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts-lambda0", nargs="+", required=True)
    p.add_argument("--ckpts-gp", nargs="+", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=5)
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

    summary = {}
    for arm_name, ckpts in [("lambda0", args.ckpts_lambda0), ("gp005", args.ckpts_gp)]:
        summary[arm_name] = []
        for ckpt_path in ckpts:
            model = load_model(ckpt_path, args.model, args.image_size, args.num_classes, device)
            wnorms = weight_spectral_norms(model)
            logit_rows, energy_rows = [], []
            for x, y in probe_bank:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
                t, x0, x1 = transport.sample(x1)
                t = t.to(x1)
                t, xt, _ut = transport.path_sampler.plan(t, x0, x1)
                logit_rows.append(attn_logit_stats(model, xt, t, y, device))
                energy_rows.append(energy_stats(model, xt, t, y))
            row = {
                "ckpt": ckpt_path,
                "logit_abs_mean": sum(r["logit_abs_mean_pooled"] for r in logit_rows) / len(logit_rows),
                "logit_abs_max": max(r["logit_abs_max_pooled"] for r in logit_rows),
                "E_abs_max": max(r["E_abs_max"] for r in energy_rows),
                "E_std_mean": sum(r["E_std"] for r in energy_rows) / len(energy_rows),
                **wnorms,
            }
            summary[arm_name].append(row)
            print(f"[diag] {arm_name} {os.path.basename(ckpt_path)}: "
                  f"logit_abs(mean/max)={row['logit_abs_mean']:.3f}/{row['logit_abs_max']:.3f} "
                  f"E(std_mean/abs_max)={row['E_std_mean']:.3f}/{row['E_abs_max']:.3f} "
                  f"wspecnorm(median/max)={row['weight_specnorm_median']:.3f}/{row['weight_specnorm_max']:.3f}")
            del model
            torch.cuda.empty_cache()

    print(f"\n[diag] SUMMARY:\n{json.dumps(summary, indent=2)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
