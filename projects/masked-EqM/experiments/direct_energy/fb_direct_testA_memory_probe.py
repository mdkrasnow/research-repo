"""
One-off memory probe for the Test A OOM (jobs 37688514/37689193): does a
SINGLE jvp_theta_to_cache call already require an enormous amount of GPU
memory on the real B/2 checkpoint, independent of any CGNR loop? If so,
this was never a cross-iteration leak -- the double-backward JVP trick is
simply too expensive per-call at this batch size/model scale and needs a
structural fix (smaller batch, gradient checkpointing, or a cheaper JVP
formulation), not a memory-hygiene fix.

Run (single GPU, seas_gpu):
  python experiments/direct_energy/fb_direct_testA_memory_probe.py \
      --ckpt <epoch40.pt> --data-path <imagenet train dir> --batch-size 8
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import EqM_models
from transport import create_transport
from train import center_crop_arr
from fb_direct.trainer import ForwardBackwardsDirectTrainer
from fb_direct.cache_adjoint import compute_g_exact, compute_g_semi_and_a_star
from fb_direct.adjoint_optimization import vjp_cache_to_theta, jvp_theta_to_cache

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def report(label):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[probe] {label}: allocated={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    latent_size = args.image_size // 8

    raw = torch.load(args.ckpt, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw
    model_fb = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="forward-backwards-direct",
    ).to(device)
    model_fb.load_state_dict(state_dict, strict=False)
    fb_trainer = ForwardBackwardsDirectTrainer(model_fb, device=device)
    fb_trainer.registry.tie_from_forward_()
    theta = fb_trainer.theta
    report("after model load")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    transport = create_transport("Linear", "velocity", None, None, None)

    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)
    t, x0, x1 = transport.sample(x1)
    t = t.to(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    ut = ut * transport.get_ct(t)[:, None, None, None]
    report("after data prep")

    torch.cuda.reset_peak_memory_stats()
    g_exact, loss_exact = compute_g_exact(theta, xt, t, y, ut)
    report("after compute_g_exact (real double-backward, for scale reference)")

    torch.cuda.reset_peak_memory_stats()
    g_semi, a_star, loss_fb, _cache = compute_g_semi_and_a_star(fb_trainer, xt, t, y, ut)
    report("after compute_g_semi_and_a_star")

    active_pairs = [
        (e.forward_name, e.backward_name)
        for e in fb_trainer.registry.entries
        if e.category in ("reverse_active", "recomputed_conditioning")
    ]
    active_names = [fn for fn, _ in active_pairs]
    b_theta = {n: g_exact[n] - g_semi[n] for n in active_names if n in g_exact and n in g_semi}

    torch.cuda.reset_peak_memory_stats()
    w1 = vjp_cache_to_theta(theta, xt, t, y, a_star)
    report("after ONE vjp_cache_to_theta call")
    del w1

    for i in range(3):
        torch.cuda.reset_peak_memory_stats()
        z1 = jvp_theta_to_cache(theta, xt, t, y, b_theta)
        report(f"after JVP call #{i+1} (isolated, no CGNR loop, no cleanup helper)")
        del z1

    print("[probe] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
