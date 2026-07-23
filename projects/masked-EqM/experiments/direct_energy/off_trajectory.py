"""Matched local off-trajectory field stability probe for direct EqM."""
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
sys.path.insert(0, str(PROJECT_ROOT))
from models import EqM_models


def field(model, x, t, y, ebm):
    with torch.set_grad_enabled(ebm != "none"):
        value = model(x, t, y, train=False)
    return value.detach()


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    if args.ebm != "none":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    transform = transforms.Compose([
        transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    images, labels = zip(*(dataset[i] for i in range(args.batch_size)))
    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        x1 = vae.encode(torch.stack(images).to(device)).latent_dist.sample().mul_(.18215)
    model = EqM_models[args.model](input_size=args.image_size // 8, num_classes=args.num_classes,
                                    uncond=True, ebm=args.ebm).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    eps = torch.randn_like(x1)
    t = torch.full((args.batch_size,), args.t, device=device)
    y = torch.tensor(labels, device=device)
    xt = (t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * eps).detach()
    base = field(model, xt, t, y, args.ebm)
    rows = []
    context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if args.ebm != "none" else __import__("contextlib").nullcontext()
    with context:
        for radius in args.radii:
            direction = torch.randn_like(xt)
            direction = direction / direction.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            perturbed = xt + radius * direction
            shifted = field(model, perturbed, t, y, args.ebm)
            rows.append({
                "radius": radius,
                "finite": bool(torch.isfinite(shifted).all()),
                "field_cosine": F.cosine_similarity(base.flatten(1), shifted.flatten(1)).mean().item(),
                "relative_field_change": ((shifted - base).flatten(1).norm(dim=1) /
                                          base.flatten(1).norm(dim=1).clamp_min(1e-12)).mean().item(),
                "base_field_norm": base.flatten(1).norm(dim=1).mean().item(),
                "perturbed_field_norm": shifted.flatten(1).norm(dim=1).mean().item(),
            })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"ebm": args.ebm, "t": args.t, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ebm", choices=["none", "dot", "direct"], required=True)
    parser.add_argument("--model", default="EqM-S/2", choices=list(EqM_models.keys()))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--t", type=float, default=.5)
    parser.add_argument("--radii", type=float, nargs="+", default=[.1, .5, 1., 2.])
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
