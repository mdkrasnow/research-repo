"""Fixed-step epoch-15 calibration on a deterministic, shared latent bank."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from energy_aware_sampling.core import fixed_sample
from models import EqM_models


def main(args):
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    bank = torch.randn(args.samples, 4, 32, 32, generator=generator)
    labels = torch.arange(args.samples, dtype=torch.long) % 1000
    model = EqM_models["EqM-B/2"](input_size=32, num_classes=1000, uncond=True, ebm="direct").to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["ema"]); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for parameter in vae.parameters(): parameter.requires_grad_(False)
    output = Path(args.output); generated = output / "generated"; generated.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); gradients = 0; finite = 0
    for offset in range(0, args.samples, args.batch_size):
        end = min(args.samples, offset + args.batch_size)
        latent, stats = fixed_sample(model, bank[offset:end].to(device), labels[offset:end].to(device), args.steps, args.step)
        gradients += stats["gradient_evaluations"]
        finite += int(torch.isfinite(latent).all(dim=(1,2,3)).sum())
        with torch.no_grad():
            decoded = vae.decode(latent / 0.18215).sample
        for local_index, image in enumerate(decoded):
            save_image(image, generated / f"{offset + local_index:06d}.png", normalize=True, value_range=(-1, 1))
    fid = calculate_fid_given_paths([str(args.real), str(generated)], batch_size=args.fid_batch_size, device=device, dims=2048)
    report = {"checkpoint": args.checkpoint, "sampler": "fixed", "seed": args.seed, "samples": args.samples,
              "steps": args.steps, "step": args.step, "multiplier": args.multiplier, "fid": fid,
              "finite_samples": finite, "gradient_evaluations": gradients, "energy_forwards": 0,
              "wall_seconds": time.perf_counter() - start}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--real", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--multiplier", type=float, required=True); parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--samples", type=int, default=1024); parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--fid-batch-size", type=int, default=16); parser.add_argument("--seed", type=int, default=20260729)
    main(parser.parse_args())
