"""Deterministic pixel-space corruptions for training and locked evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PixelMaskBatch:
    corrupted: torch.Tensor
    keep_mask: torch.Tensor
    requested_missing_ratio: torch.Tensor
    realized_missing_ratio: torch.Tensor


def bernoulli_pixel_corruption(
    images: torch.Tensor,
    min_missing_ratio: float,
    max_missing_ratio: float,
    *,
    generator: torch.Generator,
) -> PixelMaskBatch:
    """Replace independently missing pixels with normalized-space Gaussian noise.

    The spatial keep mask is shared across channels, matching the definition of
    a missing pixel rather than independently corrupting RGB coordinates.
    """
    if images.ndim != 4:
        raise ValueError(f"images must have shape [B,C,H,W], got {tuple(images.shape)}")
    if not 0.0 <= min_missing_ratio <= max_missing_ratio <= 1.0:
        raise ValueError("missing-ratio bounds must satisfy 0 <= min <= max <= 1")
    batch = images.shape[0]
    ratios = torch.rand(
        (batch,), device=images.device, dtype=torch.float32, generator=generator
    )
    ratios = min_missing_ratio + ratios * (max_missing_ratio - min_missing_ratio)
    missing = torch.rand(
        (batch, 1, images.shape[2], images.shape[3]),
        device=images.device,
        dtype=torch.float32,
        generator=generator,
    ) < ratios[:, None, None, None]
    keep = (~missing).to(dtype=images.dtype)
    fill = torch.randn(
        images.shape,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    corrupted = keep * images + (1.0 - keep) * fill
    realized = missing.float().mean(dim=(1, 2, 3))
    return PixelMaskBatch(corrupted, keep, ratios, realized)


@dataclass(frozen=True)
class BlockMask:
    keep_mask: torch.Tensor
    top: int
    left: int
    height: int
    width: int
    requested_area_fraction: float
    realized_area_fraction: float


def rectangular_block_mask(
    height: int,
    width: int,
    area_fraction: float,
    aspect_ratio: float,
    *,
    top_uniform: float,
    left_uniform: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BlockMask:
    """Create one valid rectangle with area as close as possible to target."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not 0.0 < area_fraction <= 1.0:
        raise ValueError("area_fraction must be in (0,1]")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive")
    target = area_fraction * height * width
    block_h = max(1, min(height, int(round(math.sqrt(target / aspect_ratio)))))
    block_w = max(1, min(width, int(round(target / block_h))))
    # Check nearby integer heights and take the closest feasible area, using
    # aspect-ratio error as a deterministic tie breaker.
    candidates = []
    for h in range(max(1, block_h - 3), min(height, block_h + 3) + 1):
        w = max(1, min(width, int(round(target / h))))
        area_error = abs(h * w - target)
        aspect_error = abs(math.log((w / h) / aspect_ratio))
        candidates.append((area_error, aspect_error, h, w))
    _, _, block_h, block_w = min(candidates)
    top = min(int(top_uniform * (height - block_h + 1)), height - block_h)
    left = min(int(left_uniform * (width - block_w + 1)), width - block_w)
    keep = torch.ones((1, height, width), device=device, dtype=dtype)
    keep[:, top : top + block_h, left : left + block_w] = 0
    return BlockMask(
        keep,
        top,
        left,
        block_h,
        block_w,
        area_fraction,
        (block_h * block_w) / float(height * width),
    )


def sample_block_parameters(seed: int) -> tuple[float, float, float]:
    """Return aspect ratio and two placement uniforms from a CPU seed."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.rand(3, generator=generator).tolist()
    aspect = math.exp(math.log(0.5) + values[0] * (math.log(2.0) - math.log(0.5)))
    return aspect, values[1], values[2]
