"""Field/energy diagnostics for fixed checkpoints on one reproducible batch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision import transforms
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from download import find_model
from models import EqM_models


def grad_norm(parameters) -> float:
    vals = [p.grad.detach().norm().square() for p in parameters if p.grad is not None]
    return torch.stack(vals).sum().sqrt().item() if vals else 0.0


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    tfm = transforms.Compose([
        transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds = ImageFolder(args.data_path, transform=tfm)
    g = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(ds), generator=g)[: args.batch_size].tolist()
    images, labels = zip(*(ds[i] for i in indices))
    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        x1 = vae.encode(torch.stack(images).to(device)).latent_dist.sample().mul_(0.18215)
    del vae
    torch.cuda.empty_cache()
    eps = torch.randn(x1.shape, device=device, generator=torch.Generator(device=device).manual_seed(args.seed + 1))
    y = torch.tensor(labels, device=device)
    t_values = torch.linspace(0.1, 0.9, args.num_t, device=device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for spec in args.ckpt:
            label, ckpt = spec.split("=", 1)
            arm = label.split("_seed", 1)[0]
            model = EqM_models[args.model](input_size=args.image_size // 8, num_classes=args.num_classes,
                                           uncond=True, ebm=arm).to(device)
            state = find_model(ckpt)
            model.load_state_dict(state["ema"] if "ema" in state else state)
            model.eval()
            per_t = []
            for t in t_values:
                tv = torch.full((x1.shape[0],), t.item(), device=device)
                xt = (tv[:, None, None, None] * x1 + (1 - tv[:, None, None, None]) * eps).detach()
                target = (x1 - eps) * torch.minimum(1 - 0 * tv, 5 - 5 * tv)[:, None, None, None] * 4
                model.zero_grad(set_to_none=True)
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    field, energy = model(xt, tv, y, get_energy=True, train=True)
                    loss = (field - target).square().mean()
                    loss.backward()
                if not torch.is_tensor(energy):
                    energy = torch.zeros(x1.shape[0], device=device)
                fn = field.detach().flatten(1).norm(dim=1)
                tn = target.flatten(1).norm(dim=1)
                rec = {"t": t.item(), "loss": loss.item(),
                       "cosine": F.cosine_similarity(field.detach().flatten(1), target.flatten(1)).mean().item(),
                       "field_norm": fn.mean().item(), "target_norm": tn.mean().item(),
                       "norm_ratio": (fn / tn.clamp_min(1e-12)).mean().item(),
                       "energy_mean": energy.detach().mean().item(),
                       "energy_std": energy.detach().std(unbiased=False).item(),
                       "head_grad_norm": grad_norm(getattr(model, "energy_head", model.final_layer).parameters()),
                       "backbone_grad_norm": grad_norm(model.x_embedder.parameters())}
                per_t.append(rec)
            handle.write(json.dumps({"label": label, "arm": arm, "ckpt": ckpt, "metrics": per_t}) + "\n")
            handle.flush()
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ckpt", action="append", required=True, help="arm=/path/to/checkpoint")
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--vae", default="ema")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-t", type=int, default=9)
    p.add_argument("--seed", type=int, default=123)
    main(p.parse_args())
