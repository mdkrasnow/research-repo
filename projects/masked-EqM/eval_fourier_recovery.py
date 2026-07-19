# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Fourier-recovery eval for structured start-state EqM checkpoints (Fourier
low-pass corruption family, extended 2026-07-14 alongside blur/downsample
to test whether the masking result generalizes to a spectral corruption).

Preregistered-replication upgrade (2026-07-15): every checkpoint MUST see
the identical 1024-image manifest and the identical per-image encoded
latent (x1) + corrupted latent (z0) at each cutoff, so cross-model deltas
are not confounded by different random draws. This is achieved with a
per-image torch.Generator seeded by (base_seed, image_index) -- NOT by
caching tensors to disk -- so no shared-state file is needed and the
determinism holds regardless of batch composition or run order. The
radial low-pass mask + noise-injection math is copied from
transport.corruption.fourier_corrupt verbatim (not imported) solely to
thread a generator into the noise draw; sampler/model behavior is
unchanged.

Given a checkpoint, corrupts held-out real images with radial Fourier
low-pass, starts the GD/NAG sampler from that corrupted latent instead of
pure noise, and measures full-image recovery MSE + LPIPS vs the original.
Whole-image corruption (no held-out visible region), so no keep_mask/
hard-constrain ceiling arm -- only positive control is
--vae-roundtrip-oracle. Works zero-shot on ANY checkpoint regardless of
training corruption family (pass --cutoff-grid to sweep severities).
"""
import argparse
import json
import os
from copy import deepcopy

import torch

from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from download import find_model
from models import EqM_models

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_ema_model(ckpt_path, model_name, latent_size, num_classes, uncond, ebm, device):
    model = EqM_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        uncond=uncond,
        ebm=ebm,
    ).to(device)
    ema = deepcopy(model).to(device)
    state_dict = find_model(ckpt_path)
    if "ema" in state_dict:
        ema.load_state_dict(state_dict["ema"])
    else:
        ema.load_state_dict(state_dict)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad = False
    return ema


def gd_recover(model_fn, z0, y, num_sampling_steps, stepsize, sampler, mu):
    """Run the GD/NAG sampler starting from z0 instead of pure noise.
    No hard-constrain option here -- fourier corrupts the whole image,
    there is no held-out visible region to reset each step."""
    xt = z0
    t = torch.ones((z0.shape[0],)).to(z0)
    m = torch.zeros_like(xt)
    for _ in range(num_sampling_steps - 1):
        if sampler == "gd":
            out = model_fn(xt, t, y)
            if not torch.is_tensor(out):
                out = out[0]
        else:  # ngd
            x_ = xt + stepsize * m * mu
            out = model_fn(x_, t, y)
            if not torch.is_tensor(out):
                out = out[0]
            m = out
        xt = xt + out * stepsize
        t = t + stepsize
    return xt


def gd_recover_multi(model_fn, z0, y, record_steps, stepsize, sampler, mu):
    """Single trajectory, records xt at every step in record_steps (must
    include 0, ascending). Step label L matches gd_recover's convention
    exactly: gd_recover(..., num_sampling_steps=L, ...) performs L-1 update
    iterations, so recording L here after `L-1` iterations reproduces
    bit-identical output to calling gd_recover(num_sampling_steps=L)
    separately -- this just avoids the redundant re-simulation of shared
    prefix steps across 7 separate calls."""
    record_steps = sorted(set(record_steps))
    assert record_steps[0] == 0, "record_steps must include 0 (the corrupted start state)"
    max_step = record_steps[-1]
    xt = z0
    t = torch.ones((z0.shape[0],)).to(z0)
    m = torch.zeros_like(xt)
    recorded = {0: xt.clone()}
    targets = record_steps[1:]
    next_i = 0
    n_iters = max_step - 1
    for it in range(n_iters):
        if sampler == "gd":
            out = model_fn(xt, t, y)
            if not torch.is_tensor(out):
                out = out[0]
        else:  # ngd
            x_ = xt + stepsize * m * mu
            out = model_fn(x_, t, y)
            if not torch.is_tensor(out):
                out = out[0]
            m = out
        xt = xt + out * stepsize
        t = t + stepsize
        label = it + 2  # L-1 updates done <=> label L (matches gd_recover convention)
        if next_i < len(targets) and label == targets[next_i]:
            recorded[label] = xt.clone()
            next_i += 1
    return recorded


def build_or_load_manifest(dataset, manifest_path, num_images, base_seed):
    """Fixed manifest of dataset indices + labels, shared across every
    checkpoint evaluated for this replication. If manifest_path exists,
    load it verbatim (ignores num_images/base_seed) so re-running never
    silently drifts the image set. Otherwise draw num_images indices with
    a manifest-dedicated RNG and persist immediately."""
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest["indices"], manifest["labels"]

    generator = torch.Generator().manual_seed(base_seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_images].tolist()
    labels = [dataset.samples[i][1] for i in indices]
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump({"indices": indices, "labels": labels, "base_seed": base_seed,
                       "num_images": num_images}, f, indent=2)
    return indices, labels


def radial_lowpass_mask(hw, cutoff, device):
    """Verbatim copy of transport.corruption._radial_lowpass_mask -- kept
    local (not imported) so this file has zero coupling to training-time
    corruption code; any future change there cannot silently alter what
    this eval measures."""
    h, w = hw
    fy = torch.fft.fftfreq(h, device=device).view(h, 1).expand(h, w)
    fx = torch.fft.fftfreq(w, device=device).view(1, w).expand(h, w)
    radius = torch.sqrt(fy ** 2 + fx ** 2)
    max_radius = radius.max()
    return (radius <= cutoff * max_radius).float()


def deterministic_fourier_corrupt(x1_single, cutoff, generator, device):
    """Per-image deterministic version of transport.corruption.fourier_corrupt:
    identical math, but the injected noise spectrum is drawn from a
    per-image torch.Generator (CPU) instead of the global RNG, so the same
    image index always yields the same z0 regardless of which checkpoint
    is being evaluated or what order images are batched in. x1_single is
    a single [C,H,W] latent (no batch dim)."""
    x1 = x1_single.unsqueeze(0)
    x1_fft = torch.fft.fft2(x1, norm="ortho")
    mask = radial_lowpass_mask(x1.shape[-2:], cutoff, x1.device)
    noise = torch.randn(x1.shape, generator=generator).to(device=device, dtype=x1.dtype)
    eps_fft = torch.fft.fft2(noise, norm="ortho")
    z0_fft = mask * x1_fft + (1 - mask) * eps_fft
    return torch.fft.ifft2(z0_fft, norm="ortho").real.squeeze(0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // 8

    ema = load_ema_model(args.ckpt, args.model, latent_size, args.num_classes,
                          args.uncond, args.ebm, device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    lpips_fn = None
    if _HAS_LPIPS and not args.no_lpips:
        lpips_fn = lpips.LPIPS(net="alex").to(device)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    indices, labels = build_or_load_manifest(dataset, args.manifest, args.num_images, args.manifest_seed)

    if args.subset_indices:
        with open(args.subset_indices) as f:
            subset = set(json.load(f))
        pairs = [(i, l) for i, l in zip(indices, labels) if i in subset]
        indices, labels = [p[0] for p in pairs], [p[1] for p in pairs]

    cutoffs = [float(c) for c in args.cutoff_grid.split(",")] if args.cutoff_grid else [args.cutoff]
    record_steps = [int(s) for s in args.record_steps.split(",")] if args.record_steps else None
    save_steps = set(int(s) for s in args.save_steps.split(",")) if args.save_steps else (set(record_steps) if record_steps else None)

    samples_dir = None
    if args.save_images:
        samples_dir = args.out.rsplit(".json", 1)[0] + "_samples"
        os.makedirs(samples_dir, exist_ok=True)

    per_cutoff = {}
    with torch.no_grad():
        for cutoff in cutoffs:
            per_step_image = {s: [] for s in record_steps} if record_steps else None
            per_image = [] if not record_steps else None
            for start in range(0, len(indices), args.batch_size):
                batch_slice = slice(start, start + args.batch_size)
                batch_idx = indices[batch_slice]
                batch_labels = labels[batch_slice]
                xs = [dataset[i][0] for i in batch_idx]
                x = torch.stack(xs).to(device)
                y = torch.tensor(batch_labels).to(device)

                # per-image deterministic VAE sample: same image index ->
                # same x1 for every checkpoint (encode-seed independent of
                # model/order). diffusers' DiagonalGaussianDistribution.sample
                # accepts a generator kwarg for exactly this purpose.
                dist = vae.encode(x).latent_dist
                x1_list = []
                for j, idx in enumerate(batch_idx):
                    gen = torch.Generator().manual_seed(args.encode_seed_offset + idx)
                    x1_j = dist.mean[j:j + 1] + dist.std[j:j + 1] * torch.randn(
                        dist.mean[j:j + 1].shape, generator=gen).to(device=device, dtype=dist.mean.dtype)
                    x1_list.append(x1_j)
                x1 = torch.cat(x1_list, dim=0).mul_(0.18215)

                z0_list = []
                for j, idx in enumerate(batch_idx):
                    gen = torch.Generator().manual_seed(args.corrupt_seed_offset + idx * 1000 + int(cutoff * 1000))
                    z0_j = deterministic_fourier_corrupt(x1[j] / 0.18215, cutoff, gen, device)
                    z0_list.append((z0_j * 0.18215).unsqueeze(0))
                z0 = torch.cat(z0_list, dim=0)

                if record_steps is not None:
                    assert not args.vae_roundtrip_oracle, "oracle mode not supported in --record-steps mode"
                    xt_by_step = gd_recover_multi(
                        ema, z0, y, record_steps, args.stepsize, args.sampler, args.mu,
                    )
                    for step, xt in xt_by_step.items():
                        recovered = vae.decode(xt / 0.18215).sample
                        mse = ((recovered - x) ** 2).mean(dim=[1, 2, 3])
                        lp = None
                        if lpips_fn is not None:
                            lp = lpips_fn(recovered.clamp(-1, 1), x.clamp(-1, 1)).flatten()
                        for j, idx in enumerate(batch_idx):
                            out_path = None
                            if samples_dir is not None and step in save_steps:
                                out_path = os.path.join(samples_dir, f"cutoff{cutoff}_step{step}_idx{idx}.png")
                                save_image(recovered[j].clamp(-1, 1) * 0.5 + 0.5, out_path)
                            per_step_image[step].append({
                                "index": idx,
                                "label": int(batch_labels[j]),
                                "mse": float(mse[j].item()),
                                "lpips": float(lp[j].item()) if lp is not None else None,
                                "out_path": out_path,
                            })
                    continue

                if args.vae_roundtrip_oracle:
                    xt = x1
                else:
                    xt = gd_recover(
                        ema, z0, y, args.num_sampling_steps, args.stepsize,
                        args.sampler, args.mu,
                    )

                recovered = vae.decode(xt / 0.18215).sample
                mse = ((recovered - x) ** 2).mean(dim=[1, 2, 3])
                lp = None
                if lpips_fn is not None:
                    lp = lpips_fn(recovered.clamp(-1, 1), x.clamp(-1, 1)).flatten()

                for j, idx in enumerate(batch_idx):
                    out_path = None
                    if samples_dir is not None:
                        out_path = os.path.join(samples_dir, f"cutoff{cutoff}_idx{idx}.png")
                        save_image(recovered[j].clamp(-1, 1) * 0.5 + 0.5, out_path)
                    per_image.append({
                        "index": idx,
                        "label": int(batch_labels[j]),
                        "mse": float(mse[j].item()),
                        "lpips": float(lp[j].item()) if lp is not None else None,
                        "out_path": out_path,
                    })

            if record_steps is not None:
                per_step = {}
                for step in record_steps:
                    plist = per_step_image[step]
                    mses = [r["mse"] for r in plist]
                    lpipses = [r["lpips"] for r in plist if r["lpips"] is not None]
                    per_step[str(step)] = {
                        "mean_mse": sum(mses) / len(mses),
                        "mean_lpips": (sum(lpipses) / len(lpipses)) if lpipses else None,
                        "num_images": len(mses),
                        "per_image": plist,
                    }
                    print(f"cutoff={cutoff} step={step} mean_mse={per_step[str(step)]['mean_mse']:.5f} "
                          f"mean_lpips={per_step[str(step)]['mean_lpips']}")
                per_cutoff[str(cutoff)] = {"record_steps": record_steps, "per_step": per_step}
                continue

            mses = [r["mse"] for r in per_image]
            lpipses = [r["lpips"] for r in per_image if r["lpips"] is not None]
            per_cutoff[str(cutoff)] = {
                "mean_mse": sum(mses) / len(mses),
                "mean_lpips": (sum(lpipses) / len(lpipses)) if lpipses else None,
                "num_images": len(mses),
                "per_image": per_image,
            }
            print(f"cutoff={cutoff} mean_mse={per_cutoff[str(cutoff)]['mean_mse']:.5f} "
                  f"mean_lpips={per_cutoff[str(cutoff)]['mean_lpips']}")

    result = {
        "ckpt": args.ckpt,
        "manifest": args.manifest,
        "cutoffs": cutoffs,
        "record_steps": record_steps,
        "vae_roundtrip_oracle": args.vae_roundtrip_oracle,
        "has_lpips": lpips_fn is not None,
        "per_cutoff": per_cutoff,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True,
                         help="ImageFolder-format held-out data (e.g. imagenet val)")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--uncond", type=bool, default=True)
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none")
    parser.add_argument("--stepsize", type=float, default=0.0017)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--sampler", type=str, default="gd", choices=["gd", "ngd"])
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--cutoff", type=float, default=0.4181,
                         help="single fourier cutoff for the recovery test (ignored if --cutoff-grid set)")
    parser.add_argument("--cutoff-grid", type=str, default=None,
                         help="comma-separated list of cutoffs for a held-out severity grid, e.g. '0.20,0.30,0.4181,0.55,0.70'")
    parser.add_argument("--vae-roundtrip-oracle", action="store_true",
                         help="positive control: skip fourier/model, just measure VAE encode->decode floor")
    parser.add_argument("--no-lpips", action="store_true", help="skip LPIPS even if installed")
    parser.add_argument("--num-images", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manifest", type=str, default=None,
                         help="path to a fixed index/label manifest JSON; created if missing, "
                              "loaded verbatim (ignoring --num-images/--manifest-seed) if present -- "
                              "MUST be the same file for every checkpoint in a replication so all "
                              "models see identical images")
    parser.add_argument("--manifest-seed", type=int, default=0,
                         help="RNG seed used only when creating a new manifest")
    parser.add_argument("--encode-seed-offset", type=int, default=1_000_000,
                         help="base seed for per-image deterministic VAE latent sampling")
    parser.add_argument("--corrupt-seed-offset", type=int, default=2_000_000,
                         help="base seed for per-image deterministic corruption noise")
    parser.add_argument("--save-images", action="store_true",
                         help="save recovered PNGs per image/cutoff to <out>_samples/")
    parser.add_argument("--record-steps", type=str, default=None,
                         help="comma-separated ascending step counts (must include 0) to record along "
                              "one recovery trajectory, e.g. '0,25,50,100,250,500,1000' -- convergence-"
                              "curve mode. Mutually exclusive with --vae-roundtrip-oracle. When set, "
                              "output is nested per_cutoff[cutoff]['per_step'][step] instead of flat "
                              "per_cutoff[cutoff]['per_image']")
    parser.add_argument("--save-steps", type=str, default=None,
                         help="comma-separated subset of --record-steps at which to save images "
                              "(only used with --save-images); default = all record-steps")
    parser.add_argument("--subset-indices", type=str, default=None,
                         help="path to a JSON file with a flat list of dataset indices to restrict "
                              "evaluation to a subset of --manifest (e.g. for targeted qualitative-grid "
                              "recovery runs); labels looked up from the manifest, order preserved")
    parser.add_argument("--out", type=str, default="eval_results/fourier_recovery.json")
    args = parser.parse_args()
    main(args)
