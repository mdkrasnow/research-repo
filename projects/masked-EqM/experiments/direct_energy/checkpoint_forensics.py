"""Matched field, curvature, and sampling diagnostics for direct checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from download import find_model
from models import EqM_models


def quantiles(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().flatten()
    return {f"p{int(q * 100):02d}": torch.quantile(x, q).item()
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)}


def load_batch(args, device):
    transform = transforms.Compose([
        transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:args.batch_size].tolist()
    images, labels = zip(*(dataset[index] for index in indices))
    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}"
        ).to(device).eval()
        # Fix the latent sample as well as the source-image selection.
        torch.manual_seed(args.seed + 1)
        clean = vae.encode(torch.stack(images).to(device)).latent_dist.sample().mul_(0.18215)
    del vae
    torch.cuda.empty_cache()
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    noise = torch.randn(clean.shape, generator=noise_generator, device=device)
    return clean, noise, torch.tensor(labels, device=device), indices


def field_diagnostics(model, clean, noise, labels, args):
    rows = []
    for scalar_t in torch.linspace(0.1, 0.9, args.num_t, device=clean.device):
        t = torch.full((clean.shape[0],), scalar_t.item(), device=clean.device)
        xt = (t[:, None, None, None] * clean + (1 - t[:, None, None, None]) * noise).detach()
        target = (clean - noise) * torch.minimum(torch.ones_like(t), 5 - 5 * t)[:, None, None, None] * 4
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            field, energy = model(xt, t, labels, get_energy=True, train=False)
        per_loss = (field.detach() - target).square().flatten(1).mean(1)
        field_norm = field.detach().flatten(1).norm(dim=1)
        target_norm = target.flatten(1).norm(dim=1)
        cosine = F.cosine_similarity(field.detach().flatten(1), target.flatten(1), dim=1)
        rows.append({
            "t": scalar_t.item(), "loss_mean": per_loss.mean().item(),
            "loss_quantiles": quantiles(per_loss), "cosine_mean": cosine.mean().item(),
            "cosine_quantiles": quantiles(cosine), "field_norm_mean": field_norm.mean().item(),
            "target_norm_mean": target_norm.mean().item(),
            "norm_ratio_mean": (field_norm / target_norm.clamp_min(1e-12)).mean().item(),
            "energy_mean": energy.detach().mean().item(),
            "energy_std": energy.detach().std(unbiased=False).item(),
            "energy_quantiles": quantiles(energy), "finite": bool(
                torch.isfinite(field).all() and torch.isfinite(energy).all()),
        })
    return rows


def curvature_diagnostics(model, clean, noise, labels, args):
    rows = []
    for scalar_t in args.curvature_t:
        t = torch.full((args.curvature_batch,), scalar_t, device=clean.device)
        xt = (t[:, None, None, None] * clean[:args.curvature_batch]
              + (1 - t[:, None, None, None]) * noise[:args.curvature_batch]).detach()
        xt.requires_grad_(True)
        generator = torch.Generator(device=clean.device).manual_seed(
            args.seed + 10_000 + int(1000 * scalar_t)
        )
        direction = torch.randn(xt.shape, generator=generator, device=clean.device)
        direction = direction / direction.flatten(1).norm(dim=1)[:, None, None, None]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            field = model(xt, t, labels[:args.curvature_batch], train=True)
            hv = -torch.autograd.grad((field * direction).sum(), xt)[0]
        hv_norm = hv.detach().flatten(1).norm(dim=1)
        rows.append({"t": scalar_t, "hessian_vector_norm_mean": hv_norm.mean().item(),
                     "hessian_vector_norm_quantiles": quantiles(hv_norm),
                     "finite": bool(torch.isfinite(hv).all())})
    return rows


def sampling_diagnostics(model, args, device):
    generator = torch.Generator(device=device).manual_seed(args.sample_seed)
    z = torch.randn((args.sample_batch, 4, args.image_size // 8, args.image_size // 8),
                    generator=generator, device=device)
    labels = torch.arange(args.sample_batch, device=device) % args.num_classes
    t = torch.ones(args.sample_batch, device=device)
    rows = []
    start = time.time()
    for step in range(args.sample_steps - 1):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            field, before = model(z, t, labels, get_energy=True, train=False)
        field = field.detach()
        updated = (z + args.step_size * field).detach()
        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            after_fixed_t = model(updated, t, labels, energy_only=True)
            after_next_t = model(updated, t + args.step_size, labels, energy_only=True)
        fixed_t_delta = after_fixed_t.detach() - before.detach()
        scheduled_delta = after_next_t.detach() - before.detach()
        if step % args.sample_log_every == 0 or step == args.sample_steps - 2:
            rows.append({
                "step": step + 1, "latent_norm_mean": updated.flatten(1).norm(dim=1).mean().item(),
                "field_norm_mean": field.flatten(1).norm(dim=1).mean().item(),
                "energy_before_mean": before.detach().mean().item(),
                "energy_after_fixed_t_mean": after_fixed_t.detach().mean().item(),
                "energy_after_next_t_mean": after_next_t.detach().mean().item(),
                "fixed_t_energy_delta_mean": fixed_t_delta.mean().item(),
                "fixed_t_energy_increase_fraction": (fixed_t_delta > 0).float().mean().item(),
                "scheduled_energy_delta_mean": scheduled_delta.mean().item(),
                "scheduled_energy_increase_fraction": (scheduled_delta > 0).float().mean().item(),
                "finite": bool(torch.isfinite(updated).all()
                               and torch.isfinite(after_fixed_t).all()
                               and torch.isfinite(after_next_t).all()),
                "has_grad_fn_after_detach": updated.grad_fn is not None,
            })
        z = updated
        t = t + args.step_size
    return {"runtime_seconds": time.time() - start, "rows": rows}


def main(args):
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    clean, noise, labels, indices = load_batch(args, device)
    state = find_model(args.ckpt)
    output = {"checkpoint": args.ckpt, "checkpoint_label": args.label,
              "seed": args.seed, "dataset_indices": indices, "weights": {}}
    for weight_name in ("model", "ema"):
        model = EqM_models[args.model](input_size=args.image_size // 8,
                                      num_classes=args.num_classes,
                                      uncond=True, ebm="direct").to(device)
        model.load_state_dict(state[weight_name])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        output["weights"][weight_name] = {
            "field_by_t": field_diagnostics(model, clean, noise, labels, args),
            "curvature": curvature_diagnostics(model, clean, noise, labels, args),
            "sampling": sampling_diagnostics(model, args, device),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        del model
        torch.cuda.empty_cache()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="EqM-B/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-t", type=int, default=9)
    parser.add_argument("--curvature-batch", type=int, default=2)
    parser.add_argument("--curvature-t", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--sample-log-every", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--sample-seed", type=int, default=20260803)
    main(parser.parse_args())
