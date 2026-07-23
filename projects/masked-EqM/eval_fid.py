# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sanity-scale FID for structured start-state EqM checkpoints.

Generates N unconditional samples from pure Gaussian noise via the GD/NAG
sampler (same core loop as sample_gd.py / eval_masked_recovery.py's
gd_recover), saves them as PNGs, and computes FID against a matching-size
sample of real held-out images using pytorch-fid. Single GPU, no DDP --
this is a sanity check (few thousand samples), not the paper-scale
50k-sample FID protocol in sample_gd.py/README.
"""
import argparse
import json
import os
import random

import torch
from diffusers.models import AutoencoderKL
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import save_image

from download import find_model
from models import EqM_models

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_ema_model(ckpt_path, model_name, latent_size, num_classes, uncond, ebm, device):
    from copy import deepcopy
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


def gd_sample(model_fn, x0, y, num_sampling_steps, stepsize, sampler, mu):
    """Same GD/NAG loop as sample_gd.py, starting from pure noise x0."""
    xt = x0
    t = torch.zeros((x0.shape[0],)).to(x0)
    m = torch.zeros_like(xt)
    for _ in range(num_sampling_steps - 1):
        if sampler == "gd":
            model_input = xt
        else:  # ngd
            model_input = xt + stepsize * m * mu

        ebm = getattr(model_fn, 'ebm', getattr(getattr(model_fn, '__self__', None), 'ebm', 'none'))
        with torch.set_grad_enabled(ebm != 'none'):
            out = model_fn(model_input, t, y)
        if not torch.is_tensor(out):
            out = out[0]
        out = out.detach()
        if sampler != "gd":
            m = out
        xt = (xt + out * stepsize).detach()
        t = t + stepsize
    return xt


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // 8

    ema = load_ema_model(args.ckpt, args.model, latent_size, args.num_classes,
                          args.uncond, args.ebm, device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    gen_dir = os.path.join(args.work_dir, "generated")
    real_dir = os.path.join(args.work_dir, "real")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    # --- generate samples from pure noise, unconditional (null class token) ---
    null_y = args.num_classes  # LabelEmbedder null index == num_classes
    idx = 0
    with torch.no_grad():
        while idx < args.num_samples:
            bs = min(args.batch_size, args.num_samples - idx)
            x0 = torch.randn(bs, 4, latent_size, latent_size, device=device)
            y = torch.full((bs,), null_y, device=device, dtype=torch.long)
            xt = gd_sample(ema, x0, y, args.num_sampling_steps, args.stepsize,
                            args.sampler, args.mu)
            samples = vae.decode(xt / 0.18215).sample
            for i in range(bs):
                save_image(samples[i], os.path.join(gen_dir, f"{idx + i:06d}.png"),
                           normalize=True, value_range=(-1, 1))
            idx += bs

    # --- sample matching real images for the reference set ---
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    real_indices = torch.randperm(len(dataset), generator=generator)[:args.num_samples].tolist()
    for i, ridx in enumerate(real_indices):
        img, _ = dataset[ridx]
        img.save(os.path.join(real_dir, f"{i:06d}.png"))

    fid_value = calculate_fid_given_paths(
        [real_dir, gen_dir], batch_size=50, device=device, dims=2048,
    )

    result = {
        "ckpt": args.ckpt,
        "num_samples": args.num_samples,
        "num_sampling_steps": args.num_sampling_steps,
        "fid": fid_value,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"fid={fid_value:.4f} over {args.num_samples} samples -> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True,
                         help="ImageFolder-format held-out data (e.g. imagenet val), used as FID reference")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--uncond", type=bool, default=True)
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean", "direct"], default="none",
                        help="'direct' uses a scalar E_theta(x) and returns -grad_x E_theta(x)")
    parser.add_argument("--stepsize", type=float, default=0.0017)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--sampler", type=str, default="gd", choices=["gd", "ngd"])
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--num-samples", type=int, default=2000,
                         help="sanity-scale sample count (paper-scale FID uses 50000, see sample_gd.py)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=str, default="/tmp/eqm_fid_work")
    parser.add_argument("--out", type=str, default="eval_results/fid.json")
    args = parser.parse_args()
    main(args)
