"""
Gate 0 (Experimental Decision Tree: Forward-Backward Scalar EqM):
implementation correctness on the REAL locked epoch-40 direct-scalar
checkpoint, real ImageNet-256 data, real SD-VAE latents, real EqM-B/2.

Verifies under synchronized weights (phi = Pi(theta)):
  1. rel_error = ||g_tilde - grad_x E_theta(x)|| / (||grad_x E_theta(x)|| + eps) < 1e-4
  2. L_FB ~= L_exact  (field-matching loss, identical xt/t/y/ut across both arms)

PASS -> Gate 1. FAIL -> stop, do not train (per the decision tree).

Run (single GPU, gpu_test):
  python experiments/direct_energy/fb_direct_gate0_field_check.py \
      --ckpt <epoch40.pt> --data-path <imagenet train dir> --num-batches 2
"""
import argparse
import json
import os
import sys

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import EqM_models
from transport import create_transport
from train import center_crop_arr
from fb_direct.trainer import ForwardBackwardsDirectTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--model", default="EqM-B/2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=2)
    p.add_argument("--vae", default="ema")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.image_size // 8

    raw = torch.load(args.ckpt, map_location="cpu")
    state_dict = raw["model"] if "model" in raw else raw

    model_direct = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="direct",
    ).to(device)
    missing, unexpected = model_direct.load_state_dict(state_dict, strict=False)
    print(f"[gate0] direct model load: missing={missing} unexpected={unexpected}")
    model_direct.eval()

    model_fb = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, ebm="forward-backwards-direct",
    ).to(device)
    missing, unexpected = model_fb.load_state_dict(state_dict, strict=False)
    print(f"[gate0] fb model load: missing={missing} unexpected={unexpected}")
    model_fb.eval()

    fb_trainer = ForwardBackwardsDirectTrainer(model_fb, device=device)
    fb_trainer.registry.tie_from_forward_()
    sync_err = fb_trainer.registry.compute_sync_error()
    print(f"[gate0] phi/theta sync error after tie: {sync_err:.3e}")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    transport = create_transport("Linear", "velocity", None, None, None)

    def mean_flat(x):
        return x.mean(dim=list(range(1, len(x.shape))))

    results = []
    it = iter(loader)
    for b in range(args.num_batches):
        x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x1 = vae.encode(x).latent_dist.sample().mul_(0.18215)

        t, x0, x1 = transport.sample(x1)
        t = t.to(x1)
        t, xt, ut = transport.path_sampler.plan(t, x0, x1)
        ut = ut * transport.get_ct(t)[:, None, None, None]

        # exact arm: real double-backward path, ebm='direct', train=True
        xt_exact = xt.detach().clone().requires_grad_(True)
        field_exact = model_direct(xt_exact, t, y, train=True)
        loss_exact = mean_flat((field_exact - ut) ** 2).mean()

        # FB arm: identical xt/t/y/ut, no double backward
        from fb_direct.forward_cache import forward_energy_with_cache
        E, cache = forward_energy_with_cache(fb_trainer.theta, xt, t, y)
        grad_tilde = fb_trainer.phi(cache)
        prediction_fb = -grad_tilde
        loss_fb = mean_flat((prediction_fb - ut) ** 2).mean()

        audit = fb_trainer.exact_field_audit(xt, t, y)

        rel_err = audit["fb/audit_mean_rel_error"]
        cosine = audit["fb/audit_mean_cosine"]
        loss_rel_diff = abs(float(loss_fb) - float(loss_exact)) / (abs(float(loss_exact)) + 1e-12)

        record = {
            "batch": b,
            "audit_mean_rel_error": rel_err,
            "audit_max_abs_error": audit["fb/audit_max_abs_error"],
            "audit_mean_cosine": cosine,
            "loss_exact": float(loss_exact),
            "loss_fb": float(loss_fb),
            "loss_rel_diff": loss_rel_diff,
            "gate0_field_pass": rel_err < 1e-4,
        }
        results.append(record)
        print(f"[gate0] batch {b}: {json.dumps(record)}")

    overall_pass = all(r["gate0_field_pass"] for r in results)
    summary = {
        "checkpoint": args.ckpt,
        "num_batches": args.num_batches,
        "results": results,
        "gate0_overall_pass": overall_pass,
        "sync_error_after_tie": sync_err,
    }
    print(f"[gate0] SUMMARY: {json.dumps(summary, indent=2)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
