# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Energy-ordering sanity check for structured start-state EqM checkpoints
(CLAUDE.md diagnostic #3: E(clean) < E(corrupt) < E(noise)).

These checkpoints are trained with ebm='none' (no explicit scalar energy
head -- see models.py forward(), E stays 0 unless ebm in {l2,dot,mean}).
Per EqM's own framing (README: "learns the equilibrium gradient of an
implicit energy landscape"), the field output f(x) IS the energy gradient,
so field norm ||f(x)|| is the theoretically-justified proxy for distance
from equilibrium: near 0 at the data manifold (a critical point of the
implicit energy), large away from it. This matches the prior finding in
diff-EqM's separability work that raw energy SCALARS were a dead
diagnostic (energy_scalar dot/path probes ~0.6, near chance) while
descent-shape/field-based diagnostics carried real signal (~0.81
de-confound) -- so field norm, not a scalar energy value, is the right
quantity to check here.
"""
import argparse
import json
import os
from copy import deepcopy

import torch
from diffusers.models import AutoencoderKL
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from download import find_model
from models import EqM_models
from transport import corruption

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


def field_norm(model, x, y, device):
    """Mean per-sample L2 norm of the field f(x, t=0, y) -- proxy for
    distance from equilibrium (see module docstring)."""
    t = torch.zeros((x.shape[0],), device=device)
    with torch.set_grad_enabled(model.ebm != "none"):
        out = model(x, t, y)
        if not torch.is_tensor(out):
            out = out[0]
    norms = out.flatten(1).norm(dim=1)
    return norms.mean().item(), norms.std().item()


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // 8

    ema = load_ema_model(args.ckpt, args.model, latent_size, args.num_classes,
                          args.uncond, args.ebm, device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:args.num_images].tolist()

    null_y = args.num_classes

    clean_norms, corrupt_norms, noise_norms = [], [], []
    for start in range(0, len(indices), args.batch_size):
        batch_idx = indices[start:start + args.batch_size]
        xs, _ = zip(*(dataset[i] for i in batch_idx))
        x = torch.stack(xs).to(device)
        with torch.no_grad():
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
        bs = x1.shape[0]
        y = torch.full((bs,), null_y, device=device, dtype=torch.long)

        corrupt = corruption.mask_corrupt(x1, args.mask_prob)
        noise = torch.randn_like(x1)

        clean_mean, _ = field_norm(ema, x1, y, device)
        corrupt_mean, _ = field_norm(ema, corrupt, y, device)
        noise_mean, _ = field_norm(ema, noise, y, device)

        clean_norms.append(clean_mean)
        corrupt_norms.append(corrupt_mean)
        noise_norms.append(noise_mean)

    result = {
        "ckpt": args.ckpt,
        "num_images": len(indices),
        "mask_prob": args.mask_prob,
        "field_norm_clean": sum(clean_norms) / len(clean_norms),
        "field_norm_corrupt": sum(corrupt_norms) / len(corrupt_norms),
        "field_norm_noise": sum(noise_norms) / len(noise_norms),
    }
    result["ordering_holds"] = (
        result["field_norm_clean"] < result["field_norm_corrupt"] < result["field_norm_noise"]
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"clean={result['field_norm_clean']:.4f} corrupt={result['field_norm_corrupt']:.4f} "
          f"noise={result['field_norm_noise']:.4f} ordering_holds={result['ordering_holds']} -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--uncond", type=bool, default=True)
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean", "direct"], default="none")
    parser.add_argument("--mask-prob", type=float, default=0.5)
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="eval_results/energy_ordering.json")
    args = parser.parse_args()
    main(args)
