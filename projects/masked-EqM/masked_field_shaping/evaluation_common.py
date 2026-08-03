"""Shared checkpoint, sampler, and image helpers for locked evaluations."""

from __future__ import annotations

from copy import deepcopy

import torch

from models import EqM_models


def load_ema_model(
    checkpoint_path: str,
    *,
    model_name: str,
    image_size: int,
    num_classes: int,
    device,
):
    model = EqM_models[model_name](
        input_size=image_size // 8,
        num_classes=num_classes,
        uncond=True,
        ebm="none",
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"] if "ema" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, checkpoint


def cfg_field(model, x, t, labels, cfg_scale: float, num_classes: int):
    if cfg_scale > 1.0:
        doubled_x = torch.cat([x, x], dim=0)
        doubled_t = torch.cat([t, t], dim=0)
        doubled_y = torch.cat(
            [labels, torch.full_like(labels, num_classes)], dim=0
        )
        output = model.forward_with_cfg(doubled_x, doubled_t, doubled_y, cfg_scale)
        if not torch.is_tensor(output):
            output = output[0]
        output, _ = output.chunk(2, dim=0)
        return output
    output = model(x, t, labels)
    return output if torch.is_tensor(output) else output[0]


def decode_latents(vae, latents):
    return vae.decode(latents / 0.18215).sample
