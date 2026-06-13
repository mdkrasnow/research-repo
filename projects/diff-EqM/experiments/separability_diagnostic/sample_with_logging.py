"""Stage 1 of the EqM Separability Diagnostic.

Fork of experiments/exp3_fidelity_diversity/sample_scheduled.py. The GD loop,
model load, VAE and decode are BYTE-FOR-BYTE the FID-trusted procedure
(sample_gd.py). The ONLY addition is per-step logging of the four signals we
need to test whether any cheap scalar separates good from garbage outputs:

  per step k, per sample:
    norm[k]     = ||f(x_k)||                 (gradient-field magnitude)
    dot[k]      = <f(x_k), x_k>              (the 'dot' energy proxy, raw field)
    l2[k]       = 0.5 * ||f(x_k)||^2         (the 'l2' energy proxy)
    step_dot[k] = <f(x_k), x_{k+1}-x_k>      (path-integral increment along path)

These are computed MANUALLY from the raw field f and x_k on the VANILLA
checkpoint (ebm='none' -> forward returns f directly, E=0). We do NOT use the
energy-trained weights; the diagnostic is whether the vanilla field already
carries a usable energy signal.

CRITICAL sign convention (verified against eqm-upstream/sample_gd.py:208):
    xt = xt + out * stepsize        # PLUS. The field already points noise->data.
    t  = t  + stepsize              # vestigial; uncond forward zeroes t internally.
Do NOT "fix" this to a minus sign.

Output (per rank): a shard .npz holding, for every sample this rank rendered,
the full per-step scalar trajectories + initial noise + final latent + label.
Stage 3 globs all shards. Final images are decoded to {id:06d}.png for Stage 2
(independent quality labels) and eyeball spot-checks.

Run via torch.distributed.run (see slurm/jobs/sep_diag.sbatch). Embarrassingly
parallel: each rank renders a disjoint stride of global indices, no NCCL.
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# eqm-upstream lives at projects/diff-EqM/eqm-upstream; this script sits at
# experiments/separability_diagnostic/, so go up TWO parents to diff-EqM.
_HERE = str(Path(__file__).resolve().parent)
_UPSTREAM = str(Path(__file__).resolve().parents[2] / "eqm-upstream")
for _p in (_HERE, _UPSTREAM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import EqM_models            # noqa: E402  (eqm-upstream on path)
from download import find_model          # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402

VAE_SCALE = 0.18215


def eqm_field(model, x, t, y):
    """One EqM forward -> raw field tensor (mirrors sample_scheduled.eqm_field).

    On the vanilla checkpoint (ebm='none') forward returns the field directly.
    forward() internally calls x0.requires_grad_(True); under torch.no_grad()
    no graph is built and E=0, so the autograd branches are skipped. We detach
    to be safe."""
    out = model(x, t, y)
    if not torch.is_tensor(out):
        out = out[0]
    return out.detach()


def gd_sample_logged(model, z, y, stepsize, num_steps, sampler, mu):
    """EqM GD/NAG-GD from noise, logging per-step signals.

    Returns:
        xt        final latent (N,4,H,W)
        logs dict of np.float32 arrays shaped (steps, N):
            'norm', 'dot', 'l2', 'step_dot'   (steps = num_steps - 1)
    Loop body is identical to sample_gd.py:196-209 plus pure-readout logging.
    """
    xt = z
    t = torch.zeros((xt.shape[0],), device=xt.device)
    m = torch.zeros_like(xt)
    norms, dots, l2s, step_dots = [], [], [], []
    for _ in range(num_steps - 1):
        if sampler == "ngd":
            x_look = xt + stepsize * m * mu
            out = eqm_field(model, x_look, t, y)
            m = out
        else:
            out = eqm_field(model, xt, t, y)
        # --- readouts at x_k (per-sample, reduce over C,H,W) ---
        f_flat = out.flatten(1)
        norm = f_flat.norm(dim=1)                          # ||f||
        dot = (out * xt).flatten(1).sum(dim=1)             # <f, x>
        l2 = 0.5 * (f_flat ** 2).sum(dim=1)                # 0.5||f||^2
        x_next = xt + out * stepsize                       # PLUS sign (verified)
        step_dot = (out * (x_next - xt)).flatten(1).sum(dim=1)  # <f, dx>
        norms.append(norm.to("cpu", torch.float32).numpy())
        dots.append(dot.to("cpu", torch.float32).numpy())
        l2s.append(l2.to("cpu", torch.float32).numpy())
        step_dots.append(step_dot.to("cpu", torch.float32).numpy())
        xt = x_next
        t = t + stepsize
    logs = {
        "norm": np.stack(norms, axis=0),       # (steps, N)
        "dot": np.stack(dots, axis=0),
        "l2": np.stack(l2s, axis=0),
        "step_dot": np.stack(step_dots, axis=0),
    }
    return xt, logs


def build_schedule(num_samples, base_seed, num_classes):
    """Deterministic per-sample (seed, label). seed[i] = base_seed + i so a
    sample id maps to a reproducible noise draw; labels drawn from one generator
    seeded by base_seed (fixed regardless of GPU count)."""
    g = torch.Generator().manual_seed(base_seed)
    labels = torch.randint(0, num_classes, (num_samples,), generator=g).numpy().astype(np.int64)
    seeds = (base_seed + np.arange(num_samples)).astype(np.int64)
    return labels, seeds


def main(args):
    assert torch.cuda.is_available()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)

    labels, seeds = build_schedule(args.num_samples, args.base_seed, args.num_classes)
    N = len(labels)
    if rank == 0:
        print(f"[sep-diag/sample] N={N} model={args.model} stepsize={args.stepsize} "
              f"steps={args.num_sampling_steps} cfg={args.cfg_scale} sampler={args.sampler}",
              flush=True)

    latent_size = args.image_size // 8
    model = EqM_models[args.model](input_size=latent_size,
                                   num_classes=args.num_classes,
                                   uncond=True, ebm="none").to(device)
    state = find_model(args.ckpt)
    if "ema" in state:
        model.load_state_dict(state["ema"])
    elif "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()

    os.makedirs(args.folder, exist_ok=True)
    log_dir = Path(args.folder) / "logs"
    os.makedirs(log_dir, exist_ok=True)

    my_indices = list(range(rank, N, world))
    bs = args.batch_size

    # per-rank accumulators
    all_ids, all_labels, all_x0, all_xfinal = [], [], [], []
    all_norm, all_dot, all_l2, all_step_dot = [], [], [], []

    done = 0
    for s in range(0, len(my_indices), bs):
        chunk = my_indices[s:s + bs]
        zs = []
        for i in chunk:
            g = torch.Generator(device=f"cuda:{device}").manual_seed(int(seeds[i]))
            zs.append(torch.randn(4, latent_size, latent_size, generator=g, device=device))
        z = torch.stack(zs)
        y = torch.tensor([int(labels[i]) for i in chunk], device=device)
        with torch.no_grad():
            xt, logs = gd_sample_logged(model, z, y, args.stepsize,
                                        args.num_sampling_steps, args.sampler, args.mu)
            imgs = vae.decode(xt / VAE_SCALE).sample
            imgs = torch.clamp(127.5 * imgs + 128.0, 0, 255)
            imgs = imgs.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        for j, i in enumerate(chunk):
            Image.fromarray(imgs[j]).save(f"{args.folder}/{i:06d}.png")
        # accumulate trajectories (transpose logs to (chunk, steps))
        all_ids.extend(int(i) for i in chunk)
        all_labels.extend(int(labels[i]) for i in chunk)
        all_x0.append(z.to("cpu", torch.float32).numpy())
        all_xfinal.append(xt.to("cpu", torch.float32).numpy())
        all_norm.append(logs["norm"].T)          # (chunk, steps)
        all_dot.append(logs["dot"].T)
        all_l2.append(logs["l2"].T)
        all_step_dot.append(logs["step_dot"].T)
        done += len(chunk)
        if rank == 0 and done % (bs * 5) < bs:
            print(f"[sep-diag/sample] rank0 {done}/{len(my_indices)}", flush=True)

    shard = log_dir / f"traj_rank{rank}.npz"
    np.savez_compressed(
        shard,
        sample_id=np.asarray(all_ids, dtype=np.int64),
        label=np.asarray(all_labels, dtype=np.int64),
        x0=np.concatenate(all_x0, axis=0) if all_x0 else np.zeros((0,)),
        x_final=np.concatenate(all_xfinal, axis=0) if all_xfinal else np.zeros((0,)),
        norm=np.concatenate(all_norm, axis=0) if all_norm else np.zeros((0,)),
        dot=np.concatenate(all_dot, axis=0) if all_dot else np.zeros((0,)),
        l2=np.concatenate(all_l2, axis=0) if all_l2 else np.zeros((0,)),
        step_dot=np.concatenate(all_step_dot, axis=0) if all_step_dot else np.zeros((0,)),
        stepsize=np.float32(args.stepsize),
    )
    n_png = len(list(Path(args.folder).glob("*.png")))
    print(f"[sep-diag/sample] rank{rank} DONE; shard={shard.name} "
          f"samples={len(all_ids)} folder_pngs={n_png}/{N}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EqM-B/2", choices=list(EqM_models.keys()))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--folder", required=True, help="output dir for PNGs + logs/")
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--vae", default="ema", choices=["ema", "mse"])
    ap.add_argument("--sampler", default="gd", choices=["gd", "ngd"])
    # B/2 GD uses eta=0.003, cfg=1.0 per the repo eval table (NOT the XL/2 argparse defaults).
    ap.add_argument("--stepsize", type=float, default=0.003)
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    assert abs(args.cfg_scale - 1.0) < 1e-9, \
        "Diagnostic holds cfg=1.0 (matches FID-trusted run); cfg>1 path not wired."
    try:
        main(args)
    except Exception:
        rank = os.environ.get("RANK", "0")
        tb = traceback.format_exc()
        try:
            os.makedirs(args.folder, exist_ok=True)
            with open(os.path.join(args.folder, f"ERROR_rank{rank}.txt"), "w") as f:
                f.write(tb)
        except Exception:
            pass
        print(f"[sep-diag/sample rank{rank}] FATAL:\n{tb}", flush=True)
        raise
