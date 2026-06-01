# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Deterministic fork of eqm-upstream/sample_gd.py for Experiment 1 (NFE/sampler
robustness). EVALUATION ONLY -- no training code is touched.

Minimal diff vs upstream sample_gd.py:
  1. Adds --init-latents / --labels: instead of drawing fresh randn/randint per
     batch, the fixed precomputed latents/labels are indexed by the SAME global
     sample index that names the output PNG (i*world+rank+total). This makes the
     vanilla and ANM arms see byte-identical noise + labels at every condition.
  2. Adds --step-mult: effective stepsize = stepsize * step_mult.
  3. Writes gen_stats.json (nan/divergence counts, latent-norm + field-norm
     stats, clip fraction) next to the samples for the driver to pick up.

Everything else (EMA load, VAE decode, CFG path, DDP, save format, the GD/NAG
loop, the `t += stepsize` time schedule, and the range(num_sampling_steps-1)
loop length) is kept identical to upstream so the trusted FID baselines
(vanilla 31.41, v10 29.01) remain reproducible. Because the loop runs
num_sampling_steps-1 field evaluations, the driver reports nfe_field = nfe-1.
"""
import math
import json
import os
import sys
import argparse
import numpy as np
from copy import deepcopy
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

# Make the upstream EqM modules importable (models.py, download.py, transport/,
# train_utils.py all live in projects/diff-EqM/eqm-upstream/).
EQM_UPSTREAM = Path(__file__).resolve().parents[2] / "eqm-upstream"
sys.path.insert(0, str(EQM_UPSTREAM))

from models import EqM_models                      # noqa: E402
from download import find_model                    # noqa: E402
from diffusers.models import AutoencoderKL          # noqa: E402
from train_utils import parse_transport_args        # noqa: E402


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def main(args):
    assert torch.cuda.is_available(), "Sampling requires at least one GPU."
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, \
        "Batch size must be divisible by world size."
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = int(os.environ["LOCAL_RANK"])
    seed = args.global_seed * world + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if rank == 0:
        print(f"[sample_gd_fixed] world={world} seed={args.global_seed} "
              f"sampler={args.sampler} steps={args.num_sampling_steps} "
              f"stepsize={args.stepsize}*{args.step_mult} cfg={args.cfg_scale}")

    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8

    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm,
    ).to(device)
    ema = deepcopy(model).to(device)

    assert args.ckpt is not None, "--ckpt is required"
    state_dict = find_model(args.ckpt)
    if "ema" in state_dict:
        # Sampling always uses the EMA weights (identical policy for both arms).
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
    else:
        model.load_state_dict(state_dict)
        ema.load_state_dict(state_dict)
    ema = ema.to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    model.train()   # keeps embedding dropout available for CFG (upstream behavior)
    ema.eval()

    # ---- effective stepsize ----
    eff_step = args.stepsize * args.step_mult

    # ---- fixed inputs ----
    fixed_latents = fixed_labels = None
    if args.init_latents:
        fixed_latents = torch.load(args.init_latents, map_location="cpu")
        assert fixed_latents.shape[1:] == (4, latent_size, latent_size), \
            f"init-latents shape {tuple(fixed_latents.shape)} != (*,4,{latent_size},{latent_size})"
    if args.labels:
        fixed_labels = torch.from_numpy(np.load(args.labels)).long()

    use_cfg = args.cfg_scale > 1.0
    model_fn = ema.forward_with_cfg if use_cfg else ema.forward

    if rank == 0:
        os.makedirs(args.folder, exist_ok=True)
    dist.barrier()

    total_samples = int(math.ceil(args.num_fid_samples / args.global_batch_size) * args.global_batch_size)
    assert total_samples % world == 0
    iterations = int(total_samples // args.global_batch_size)
    n = int(args.global_batch_size // world)

    # ---- stats accumulators ----
    s = dict(nan=0, div=0, n_samp=0, fln_sum=0.0, fln_sqsum=0.0,
             iln_sum=0.0, fgn_sum=0.0, fgn_n=0, first_gn_sum=0.0, first_gn_n=0,
             clip_num=0.0, clip_den=0.0)
    div_thresh = args.divergence_norm_thresh

    total = 0
    for it in range(iterations):
        with torch.no_grad():
            global_idx = torch.arange(n) * world + rank + total  # matches save indexing
            if fixed_latents is not None:
                z = fixed_latents[global_idx].to(device)
            else:
                z = torch.randn(n, 4, latent_size, latent_size, device=device)
            if fixed_labels is not None:
                y = fixed_labels[global_idx].to(device)
            else:
                y = torch.randint(0, args.num_classes, (n,), device=device)

            t = torch.ones((n,), device=device)
            if use_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([args.num_classes] * n, device=device)
                y = torch.cat([y, y_null], 0)
                t = torch.cat([t, t], 0)

            xt = z
            m = torch.zeros_like(xt)
            s["iln_sum"] += xt[:n].flatten(1).norm(dim=1).double().sum().item()

            gn = torch.zeros(n, device=device)
            for step_idx in range(args.num_sampling_steps - 1):
                if args.sampler == "gd":
                    out = model_fn(xt, t, y, args.cfg_scale)
                else:  # ngd
                    x_ = xt + eff_step * m * args.mu
                    out = model_fn(x_, t, y, args.cfg_scale)
                if not torch.is_tensor(out):
                    out = out[0]
                if args.sampler == "ngd":
                    m = out

                gn = out[:n].flatten(1).norm(dim=1)
                if step_idx == 0:
                    s["first_gn_sum"] += gn.double().sum().item(); s["first_gn_n"] += n
                xt = xt + out * eff_step
                t = t + eff_step

            s["fgn_sum"] += gn.double().sum().item(); s["fgn_n"] += n

            if use_cfg:
                xt, _ = xt.chunk(2, dim=0)

            # stability bookkeeping on the latent (per-sample)
            finite = torch.isfinite(xt).flatten(1).all(dim=1)
            norms = xt.flatten(1).norm(dim=1)
            s["nan"] += int((~finite).sum().item())
            s["div"] += int(((~finite) | (norms > div_thresh)).sum().item())
            fin_norms = norms[finite]
            s["fln_sum"] += fin_norms.double().sum().item()
            s["fln_sqsum"] += (fin_norms.double() ** 2).sum().item()
            s["n_samp"] += n

            xt = torch.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)
            raw = vae.decode(xt / 0.18215).sample
            raw = 127.5 * raw + 128.0
            s["clip_num"] += float(((raw < 0) | (raw > 255)).sum().item())
            s["clip_den"] += float(raw.numel())
            samples = torch.clamp(raw, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for i, sample in enumerate(samples):
                index = i * world + rank + total
                Image.fromarray(sample).save(f"{args.folder}/{index:06d}.png")
        total += args.global_batch_size
        dist.barrier()

    # ---- reduce stats across ranks ----
    keys = list(s.keys())
    tens = torch.tensor([s[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tens, op=dist.ReduceOp.SUM)
    s = {k: float(v) for k, v in zip(keys, tens.tolist())}

    if rank == 0:
        nman = max(s["n_samp"], 1)
        fln_mean = s["fln_sum"] / nman
        fln_var = max(s["fln_sqsum"] / nman - fln_mean ** 2, 0.0)
        stats = {
            "nan_count": int(s["nan"]),
            "divergence_count": int(s["div"]),
            "num_samples_counted": int(s["n_samp"]),
            "mean_init_latent_norm": s["iln_sum"] / nman,
            "mean_final_latent_norm": fln_mean,
            "std_final_latent_norm": fln_var ** 0.5,
            "mean_first_grad_norm": s["first_gn_sum"] / max(s["first_gn_n"], 1),
            "mean_final_grad_norm": s["fgn_sum"] / max(s["fgn_n"], 1),
            "clip_fraction": s["clip_num"] / max(s["clip_den"], 1.0),
            "effective_stepsize": eff_step,
            "nfe_field": args.num_sampling_steps - 1,
        }
        with open(os.path.join(args.folder, "gen_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[sample_gd_fixed] done: {stats}")
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--stepsize", type=float, default=0.003, help="base step size eta")
    parser.add_argument("--step-mult", type=float, default=1.0, help="multiplier applied to stepsize")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--folder", type=str, default="samples")
    parser.add_argument("--sampler", type=str, default="gd", choices=["gd", "ngd"])
    parser.add_argument("--mu", type=float, default=0.3, help="NAG-GD momentum")
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--uncond", type=bool, default=True)
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none")
    parser.add_argument("--init-latents", type=str, default=None,
                        help="path to fixed init latents .pt [N,4,H/8,W/8]")
    parser.add_argument("--labels", type=str, default=None,
                        help="path to fixed class labels .npy [N]")
    parser.add_argument("--divergence-norm-thresh", type=float, default=1e3,
                        help="per-sample latent L2 above this counts as divergence")
    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
