"""CAFM-EqM post-training entry point. Phase 1a Day 2.

Minimal DDP training loop that wraps `training_step.py` with the EqM model
loading, VAE encoding, dataloader, optimizers, EMA, checkpoint persistence,
and the N=16 disc / 1 gen scheduling from Lin's CAFM recipe.

Usage:
    python -m torch.distributed.run --nproc_per_node=4 \
        projects/diff-EqM/experiments/cafm_eqm/train_cafm_eqm.py \
        --config projects/diff-EqM/configs/cafm/eqm_b2_in256_cafm.yaml

Smoke (CPU, no DDP):
    python projects/diff-EqM/experiments/cafm_eqm/train_cafm_eqm.py \
        --config projects/diff-EqM/configs/cafm/smoke_cpu.yaml --smoke

Design notes:
- Reuses EqM model from `projects/diff-EqM/experiments/train_imagenet.py` via
  the existing `EqM_models` registry.
- Discriminator imported from cloned Lin repo via `discriminator_adapter`.
- v10 mining toggled via `--enable-v10`. When False, this is pure CAFM-EqM.
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[3]
EQM_UPSTREAM = REPO_ROOT / "projects" / "diff-EqM" / "eqm-upstream"
DIFF_EQM_EXPERIMENTS = REPO_ROOT / "projects" / "diff-EqM" / "experiments"

# Path setup for EqM upstream + diff-EqM experiments
for p in (str(EQM_UPSTREAM), str(DIFF_EQM_EXPERIMENTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cafm_eqm.generator_wrapper import EqMGeneratorWrapper  # noqa: E402
from cafm_eqm.training_step import (                       # noqa: E402
    cafm_dis_step, cafm_gen_step, cafm_v10_gen_step, prepare_eqm_inputs,
)


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="OmegaConf YAML config")
    p.add_argument("--smoke", action="store_true", help="100-step CPU smoke (no DDP, no FID)")
    p.add_argument("--enable-v10", action="store_true", help="Enable v10 PGD mining in gen step")
    p.add_argument("--ckpt-resume", type=str, default=None, help="Resume from CAFM-EqM ckpt")
    return p.parse_args()


def load_config(path: str):
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def setup_distributed(smoke: bool):
    if smoke:
        return 0, 1, torch.device("cpu")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_generator(cfg, device, smoke: bool):
    """Load vanilla EqM-B/2 80ep checkpoint as generator."""
    # eqm-upstream exposes the model registry as `models.EqM_models`. Already on
    # sys.path via EQM_UPSTREAM injection at module top.
    from models import EqM_models

    image_size = cfg.gen.image_size
    latent_size = image_size // 8
    model = EqM_models[cfg.gen.model](
        input_size=latent_size,
        num_classes=cfg.gen.num_classes,
        uncond=cfg.gen.get("uncond", False),
        ebm=cfg.gen.get("ebm", False),
    ).to(device)

    ckpt_path = cfg.gen.checkpoint
    if ckpt_path and not smoke:
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("model", state)
        # Strip DDP prefix if present.
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"[gen] loaded vanilla EqM ckpt from {ckpt_path}")
    elif not smoke:
        raise ValueError("cfg.gen.checkpoint required in non-smoke mode")

    return EqMGeneratorWrapper(model, mode="gradient")


def build_discriminator(cfg, device):
    """Build γ-conditional discriminator using Lin's CAFM Discriminator class."""
    from cafm_eqm.discriminator_adapter import load_cafm_discriminator_classes
    Discriminator, DiscriminatorJVP = load_cafm_discriminator_classes()

    dis_args = dict(cfg.dis.model_kwargs)
    dis = Discriminator(**dis_args).to(device)

    if cfg.dis.get("checkpoint"):
        state = torch.load(cfg.dis.checkpoint, map_location=device)
        dis.load_state_dict(state.get("model", state))
        print(f"[dis] loaded checkpoint from {cfg.dis.checkpoint}")

    return dis, DiscriminatorJVP(dis)


def build_vae(cfg, device, smoke: bool):
    if smoke:
        # Skip VAE in smoke mode; assume inputs are already latents.
        return None
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae.variant}").to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    return vae


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------


def build_dataloader(cfg, rank, world_size, smoke: bool):
    if smoke:
        # Synthetic micro-batches for smoke.
        class _SmokeDS(torch.utils.data.Dataset):
            def __len__(self):
                return 16
            def __getitem__(self, i):
                return torch.randn(4, 32, 32), int(i % 1000)
        return DataLoader(_SmokeDS(), batch_size=2, shuffle=False)

    def _crop(pil, size):
        from PIL import Image
        while min(*pil.size) >= 2 * size:
            pil = pil.resize(tuple(s // 2 for s in pil.size), resample=Image.Resampling.BOX)
        scale = size / min(*pil.size)
        pil = pil.resize(tuple(round(s * scale) for s in pil.size), resample=Image.Resampling.BICUBIC)
        import numpy as np
        arr = np.array(pil)
        cy = (arr.shape[0] - size) // 2
        cx = (arr.shape[1] - size) // 2
        return Image.fromarray(arr[cy:cy + size, cx:cx + size])

    tfm = transforms.Compose([
        transforms.Lambda(lambda pil: _crop(pil, cfg.data.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True),
    ])
    dataset = ImageFolder(cfg.data.path, transform=tfm)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=cfg.training.seed)
    return DataLoader(
        dataset, batch_size=cfg.training.local_batch_size, sampler=sampler,
        num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
    )


@torch.no_grad()
def vae_encode(vae, images, scale=0.18215):
    if vae is None:
        return images  # smoke: already latents
    return vae.encode(images).latent_dist.sample().mul_(scale)


# ---------------------------------------------------------------------------
# Schedule helpers (Lin's pattern: N disc updates per gen update + warmup)
# ---------------------------------------------------------------------------


def is_dis_step(step: int, N: int, warmup: int) -> bool:
    if step <= warmup:
        return True
    return step % (1 + N) < N


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = load_config(args.config)
    rank, world_size, device = setup_distributed(args.smoke)

    torch.manual_seed(cfg.training.seed + rank)

    gen_wrapper = build_generator(cfg, device, args.smoke)
    dis_model, dis_jvp = build_discriminator(cfg, device) if not args.smoke else (None, None)
    vae = build_vae(cfg, device, args.smoke)
    loader = build_dataloader(cfg, rank, world_size, args.smoke)

    ema = deepcopy(gen_wrapper.eqm).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()

    gen_opt = torch.optim.Adam(
        gen_wrapper.eqm.parameters(),
        lr=cfg.training.gen_lr, betas=(0.0, 0.95),
    )
    dis_opt = None
    if dis_model is not None:
        dis_opt = torch.optim.Adam(
            dis_model.parameters(),
            lr=cfg.training.dis_lr, betas=(0.0, 0.95),
        )

    N = cfg.adversarial.N
    warmup = cfg.adversarial.warmup
    cp_scale = cfg.adversarial.cp_scale
    ot_scale = cfg.adversarial.ot_scale
    ema_decay = cfg.ema.decay
    total_steps = cfg.training.total_steps

    if rank == 0:
        print(f"=== CAFM-EqM training start ===")
        print(f"world_size={world_size}, total_steps={total_steps}, N={N}, warmup={warmup}")
        print(f"v10 enabled: {args.enable_v10}")

    step = 0
    loader_iter = iter(loader)

    while step < total_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        if isinstance(batch, (list, tuple)):
            images, labels = batch
        else:
            images, labels = batch["image"], batch["label"]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        latents = vae_encode(vae, images)
        inputs = prepare_eqm_inputs(latents, labels)

        dis_phase = is_dis_step(step, N, warmup)

        if dis_phase and dis_jvp is not None and dis_opt is not None:
            losses = cafm_dis_step(dis_jvp, gen_wrapper, inputs, cp_scale=cp_scale)
            losses["loss/total_dis"].backward()
            dis_opt.step()
            dis_opt.zero_grad(set_to_none=True)
            gen_opt.zero_grad(set_to_none=True)
        else:
            if args.enable_v10:
                losses = cafm_v10_gen_step(
                    dis_jvp, gen_wrapper, inputs, ot_scale=ot_scale,
                    lambda_v10=cfg.v10.lambda_v10, v10_K=cfg.v10.K,
                    v10_eps_radius=cfg.v10.eps_radius, v10_lr=cfg.v10.lr,
                )
            else:
                losses = cafm_gen_step(dis_jvp, gen_wrapper, inputs, ot_scale=ot_scale)
            losses["loss/total_gen"].backward()
            gen_opt.step()
            gen_opt.zero_grad(set_to_none=True)
            if dis_opt is not None:
                dis_opt.zero_grad(set_to_none=True)

            # EMA update only on gen steps
            with torch.no_grad():
                for tgt, src in zip(ema.parameters(), gen_wrapper.eqm.parameters()):
                    tgt.data.lerp_(src.data.to(tgt), 1.0 - ema_decay)

        if rank == 0 and step % cfg.logging.interval == 0:
            log_keys = sorted(k for k in losses.keys() if k.startswith("loss/"))
            log_str = " ".join(
                f"{k}={(losses[k].item() if hasattr(losses[k], 'item') else losses[k]):.4f}"
                for k in log_keys
            )
            print(f"[step {step:6d} phase={'dis' if dis_phase else 'gen'}] {log_str}")

        if (
            rank == 0 and step > 0 and step % cfg.persistence.interval == 0
            and not args.smoke
        ):
            out_dir = Path(cfg.persistence.dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # (1) Full CAFM-EqM checkpoint for resume.
            ckpt = {
                "step": step,
                "gen": gen_wrapper.eqm.state_dict(),
                "dis": dis_model.state_dict() if dis_model is not None else None,
                "ema": ema.state_dict(),
                "gen_opt": gen_opt.state_dict(),
                "dis_opt": dis_opt.state_dict() if dis_opt is not None else None,
            }
            torch.save(ckpt, out_dir / f"ckpt_{step:08d}.pt")
            # (2) EqM-compatible ckpt (for downstream FID eval via sample_gd.py).
            #    Mirrors vanilla EqM ckpt structure: {model, ema, opt, ...}.
            eqm_compat = {
                "model": ema.state_dict(),     # use EMA for downstream sampling
                "ema": ema.state_dict(),
                "opt": gen_opt.state_dict(),
                "train_steps": step,
                "epoch": step,                 # no epoch tracking in CAFM-EqM; use step
            }
            torch.save(eqm_compat, out_dir / f"eqm_compat_{step:08d}.pt")
            print(f"[ckpt] saved ckpt_{step:08d}.pt + eqm_compat_{step:08d}.pt")

        step += 1

    if rank == 0:
        print("=== training complete ===")

    if not args.smoke:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
