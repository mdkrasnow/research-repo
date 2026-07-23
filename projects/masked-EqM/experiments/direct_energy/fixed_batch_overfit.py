"""Stage-2 fixed encoded-batch learnability test for native direct EqM."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import EqM_models
from experiments.direct_energy.gates import fixed_batch_gate
from experiments.direct_energy.metrics import append_jsonl


def norm(parameters):
    values = [p.grad.detach().norm().square() for p in parameters if p.grad is not None]
    return torch.stack(values).sum().sqrt().item() if values else 0.0


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    latent_size = args.image_size // 8
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    transform = transforms.Compose([
        transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    images, labels = zip(*(dataset[i] for i in range(args.batch_size)))
    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        x1 = vae.encode(torch.stack(images).to(device)).latent_dist.sample().mul_(0.18215)
    del vae
    torch.cuda.empty_cache()

    model = EqM_models[args.model](input_size=latent_size, num_classes=args.num_classes,
                                   uncond=True, ebm="direct").to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    eps = torch.randn(x1.shape, device=device, generator=generator)
    t = torch.linspace(0.1, 0.9, args.batch_size, device=device)
    xt = (t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * eps).detach()
    c_t = torch.minimum(torch.ones_like(t), 5 - 5 * t) * 4
    target = (x1 - eps) * c_t[:, None, None, None]
    y = torch.tensor(labels, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for step in range(args.steps):
            iteration_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            field, energy = model(xt, t, y, get_energy=True, train=True)
            loss = (field - target).square().mean()
            finite = bool(torch.isfinite(loss) and torch.isfinite(field).all() and torch.isfinite(energy).all())
            if not finite:
                append_jsonl(metrics_path, {"step": step, "finite": False})
                raise RuntimeError("non-finite direct-energy fixed-batch metric")
            loss.backward()
            head_grad = norm(model.energy_head.parameters())
            backbone_grad = norm(model.x_embedder.parameters())
            total_grad = norm(model.parameters())
            optimizer.step()
            field_norm = field.detach().flatten(1).norm(dim=1)
            target_norm = target.flatten(1).norm(dim=1)
            record = {
                "step": step, "finite": True, "loss": loss.item(),
                "field_target_cosine": F.cosine_similarity(field.detach().flatten(1), target.flatten(1)).mean().item(),
                "field_norm": field_norm.mean().item(), "target_norm": target_norm.mean().item(),
                "field_target_norm_ratio": (field_norm / target_norm.clamp_min(1e-12)).mean().item(),
                "energy_mean": energy.detach().mean().item(), "energy_std": energy.detach().std(unbiased=False).item(),
                "energy_min": energy.detach().min().item(), "energy_max": energy.detach().max().item(),
                "head_grad_norm": head_grad, "backbone_grad_norm": backbone_grad,
                "total_grad_norm": total_grad, "lr": optimizer.param_groups[0]["lr"],
                "steps_per_sec": 1 / (time.time() - iteration_start),
                "allocated_memory_mb": torch.cuda.memory_allocated(device) / 2**20,
                "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "t_bins": {str(round(v.item(), 2)): (field_norm[i] / target_norm[i].clamp_min(1e-12)).item() for i, v in enumerate(t)},
            }
            append_jsonl(metrics_path, record)
            if step % args.log_every == 0:
                print(json.dumps(record, sort_keys=True), flush=True)

    from experiments.direct_energy.metrics import load_jsonl
    result = fixed_batch_gate(load_jsonl(metrics_path))
    result.update({"steps": args.steps, "runtime_seconds": time.time() - started,
                   "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                   "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    (output / "gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text("# Fixed-batch direct-energy result\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n")
    print(json.dumps({"fixed_batch_gate": result}, sort_keys=True), flush=True)
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="EqM-S/2", choices=list(EqM_models.keys()))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", default="ema", choices=["ema", "mse"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    main(parser.parse_args())
