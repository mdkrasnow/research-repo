"""DDP sampler for Experiment 3 -- fork of eqm-upstream/sample_gd.py.

Differences from sample_gd.py (the only differences -- the GD loop, model load,
VAE and decode are byte-for-byte the procedure that produced FID 27.09/29.01):
  1. Labels come from a FIXED shared schedule (schedule.json), not torch.randint.
  2. Each sample's initial noise is seeded per global index (base_seed + i) via a
     dedicated torch.Generator, so vanilla and ANM get identical z and y at index
     i regardless of GPU count.
  3. Global indices are sharded contiguously-by-stride across ranks
     (range(rank, N, world)), and each PNG is named {global_index:06d}.png.
  4. Resume-aware: indices whose PNG already exists are skipped (unless --overwrite).
  5. NO filtering / NO rejection -- every generated image is written.

Run via torch.distributed.run (see slurm/jobs/exp3_generate.sbatch).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# eqm-upstream on path (sibling dir), same style as eval_capabilities.py
_UPSTREAM = str(Path(__file__).resolve().parent.parent / "eqm-upstream")
if _UPSTREAM not in sys.path:
    sys.path.insert(0, _UPSTREAM)

from models import EqM_models            # noqa: E402  (eqm-upstream on path)
from download import find_model          # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402

# schedule.py lives in this script's own dir (auto on sys.path as script dir)
from schedule import schedule_hash       # noqa: E402

VAE_SCALE = 0.18215


def provenance_fingerprint(args, sched):
    """Identifies the EXACT generation config of a sample folder, so a rerun
    with a different checkpoint/sampler can never silently reuse old PNGs."""
    import hashlib
    fp = {
        "ckpt": args.ckpt_id or os.path.basename(args.ckpt),
        "model": args.model,
        "sampler": args.sampler,
        "stepsize": args.stepsize,
        "num_sampling_steps": args.num_sampling_steps,
        "cfg_scale": args.cfg_scale,
        "image_size": args.image_size,
        "vae": args.vae,
        "schedule_hash": schedule_hash(sched),
    }
    fp["hash"] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    return fp


def eqm_field(model, x, t, y):
    """One EqM forward -> field tensor (mirrors eval_capabilities.eqm_field)."""
    out = model(x, t, y)
    if not torch.is_tensor(out):
        out = out[0]
    return out.detach()


def gd_sample(model, z, y, stepsize, num_steps, sampler, mu):
    """EqM gradient descent / NAG-GD from noise. Identical to sample_gd.py loop."""
    xt = z
    t = torch.zeros((xt.shape[0],), device=xt.device)
    m = torch.zeros_like(xt)
    for _ in range(num_steps - 1):
        if sampler == "ngd":
            x_look = xt + stepsize * m * mu
            out = eqm_field(model, x_look, t, y)
            m = out
        else:
            out = eqm_field(model, xt, t, y)
        xt = xt + out * stepsize
        t = t + stepsize
    return xt


def main(args):
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)

    # --- schedule (shared, identical across arms) ---
    sched = json.loads(Path(args.schedule).read_text())
    labels = np.asarray(sched["labels"], dtype=np.int64)
    seeds = np.asarray(sched["seeds"], dtype=np.int64)
    N = len(labels)
    if rank == 0:
        print(f"[exp3-sample] N={N} classes={sched['num_classes']} "
              f"per_class={sched['samples_per_class']} arm={args.tag}", flush=True)

    # --- model (EMA weights, the FID-trusted choice) ---
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

    # --- provenance guard: refuse to mix samples from a different config ---
    fp = provenance_fingerprint(args, sched)
    prov_path = Path(args.folder) / "gen_provenance.json"
    if rank == 0:
        if prov_path.exists():
            old = json.loads(prov_path.read_text())
            if old.get("hash") != fp["hash"]:
                if not args.overwrite:
                    raise RuntimeError(
                        f"[exp3-sample] folder {args.folder} already holds samples with "
                        f"provenance {old.get('hash')} but this run is {fp['hash']}. "
                        f"Refusing to mix. Pass --overwrite to regenerate.")
                # overwrite with different config -> clear stale PNGs
                for p in Path(args.folder).glob("*.png"):
                    p.unlink()
        prov_path.write_text(json.dumps(fp, indent=2))
    dist.barrier()
    # all ranks confirm the on-disk provenance matches their own fingerprint
    assert json.loads(prov_path.read_text())["hash"] == fp["hash"], \
        f"[exp3-sample rank{rank}] provenance mismatch after barrier"

    manifest_path = Path(args.folder) / f"manifest_part_{rank}.csv"
    mf = open(manifest_path, "w")
    mf.write("sample_id,seed,requested_label\n")

    my_indices = list(range(rank, N, world))
    # resume: drop indices already rendered
    if not args.overwrite:
        my_indices = [i for i in my_indices
                      if not (Path(args.folder) / f"{i:06d}.png").exists()]

    bs = args.batch_size
    done = 0
    for s in range(0, len(my_indices), bs):
        chunk = my_indices[s:s + bs]
        zs = []
        for i in chunk:
            g = torch.Generator(device=f"cuda:{device}").manual_seed(int(seeds[i]))
            zs.append(torch.randn(4, latent_size, latent_size,
                                  generator=g, device=device))
        z = torch.stack(zs)
        y = torch.tensor([int(labels[i]) for i in chunk], device=device)
        with torch.no_grad():
            xt = gd_sample(model, z, y, args.stepsize, args.num_sampling_steps,
                           args.sampler, args.mu)
            imgs = vae.decode(xt / VAE_SCALE).sample
            imgs = torch.clamp(127.5 * imgs + 128.0, 0, 255)
            imgs = imgs.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        for j, i in enumerate(chunk):
            Image.fromarray(imgs[j]).save(f"{args.folder}/{i:06d}.png")
            mf.write(f"{i},{int(seeds[i])},{int(labels[i])}\n")
        done += len(chunk)
        if rank == 0 and done % (bs * 20) < bs:
            print(f"[exp3-sample] rank0 {done}/{len(my_indices)}", flush=True)
    mf.close()
    dist.barrier()
    if rank == 0:
        n_png = len(list(Path(args.folder).glob("*.png")))
        print(f"[exp3-sample] DONE arm={args.tag} pngs={n_png}/{N}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EqM-B/2", choices=list(EqM_models.keys()))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-id", default="",
                    help="real checkpoint identity for provenance (the original "
                         "path, since --ckpt is often a /tmp copy)")
    ap.add_argument("--tag", required=True, help="vanilla | anm")
    ap.add_argument("--schedule", required=True, help="shared schedule.json")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--vae", default="ema", choices=["ema", "mse"])
    ap.add_argument("--sampler", default="gd", choices=["gd", "ngd"])
    ap.add_argument("--stepsize", type=float, default=0.003)
    ap.add_argument("--num-sampling-steps", type=int, default=250)
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    assert abs(args.cfg_scale - 1.0) < 1e-9, \
        "Exp3 holds cfg=1.0 (matches the FID-trusted run); cfg>1 path not wired."
    main(args)
