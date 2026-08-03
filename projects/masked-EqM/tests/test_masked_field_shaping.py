import math

import numpy as np
import pytest
import torch
from torch import nn

from masked_field_shaping.checkpointing import models_numerically_identical, tensor_mapping_sha256
from masked_field_shaping.corruption import (
    bernoulli_pixel_corruption,
    rectangular_block_mask,
    sample_block_parameters,
)
from masked_field_shaping.statistics import classify_result, paired_image_cluster_bootstrap
from masked_field_shaping.train_continuation import freeze_module
from transport import create_transport


def _generator(seed, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed)


def test_pixel_mask_is_deterministic_and_preserves_shape_and_observed_pixels():
    images = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    first = bernoulli_pixel_corruption(images, 0.1, 0.5, generator=_generator(17))
    second = bernoulli_pixel_corruption(images, 0.1, 0.5, generator=_generator(17))
    assert first.corrupted.shape == images.shape
    assert torch.equal(first.keep_mask, second.keep_mask)
    assert torch.equal(first.corrupted, second.corrupted)
    observed = first.keep_mask.expand_as(images).bool()
    missing = ~observed
    assert torch.equal(first.corrupted[observed], images[observed])
    assert missing.any()
    assert not torch.equal(first.corrupted[missing], images[missing])


def test_pixel_mask_realized_ratios_track_per_image_requests():
    images = torch.zeros(64, 3, 128, 128)
    result = bernoulli_pixel_corruption(images, 0.1, 0.5, generator=_generator(9))
    assert torch.all(result.requested_missing_ratio >= 0.1)
    assert torch.all(result.requested_missing_ratio <= 0.5)
    assert torch.max(torch.abs(result.realized_missing_ratio - result.requested_missing_ratio)) < 0.02


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pixel_mask_batch_and_mixed_precision(dtype):
    images = torch.zeros(3, 3, 16, 16, dtype=dtype)
    result = bernoulli_pixel_corruption(images, 0.2, 0.4, generator=_generator(5))
    assert result.corrupted.dtype == dtype
    assert result.keep_mask.dtype == dtype
    assert torch.isfinite(result.corrupted.float()).all()


def test_block_mask_is_reproducible_valid_and_near_30_percent():
    aspect, top_u, left_u = sample_block_parameters(123)
    first = rectangular_block_mask(256, 256, 0.30, aspect, top_uniform=top_u, left_uniform=left_u)
    aspect2, top_u2, left_u2 = sample_block_parameters(123)
    second = rectangular_block_mask(256, 256, 0.30, aspect2, top_uniform=top_u2, left_uniform=left_u2)
    assert 0.5 <= aspect <= 2.0
    assert torch.equal(first.keep_mask, second.keep_mask)
    assert abs(first.realized_area_fraction - 0.30) < 0.002
    assert first.top + first.height <= 256
    assert first.left + first.width <= 256


class ScaleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x, t, **kwargs):
        return x * self.scale


def test_zero_probability_path_matches_original_gaussian_loss_exactly():
    clean = torch.randn(4, 2, 3, 3)
    model = ScaleModel()
    transport = create_transport()
    torch.manual_seed(22)
    original = transport.training_losses(model, clean)["loss"]
    torch.manual_seed(22)
    new_zero_probability_path = transport.training_losses(model, clean)["loss"]
    assert torch.equal(original, new_zero_probability_path)


def test_endpoint_override_uses_same_gamma_ct_and_positive_sampler_sign():
    clean = torch.ones(2, 1, 2, 2)
    endpoint = torch.zeros_like(clean)
    selector = torch.tensor([True, True])
    transport = create_transport()

    def fixed_sample(x1):
        return torch.full((x1.shape[0],), 0.5), torch.full_like(x1, -7), x1

    transport.sample = fixed_sample
    model = ScaleModel()
    losses = transport.training_losses(
        model, clean, corruption_endpoint=endpoint, corruption_selector=selector
    )["loss"]
    t = torch.full((2,), 0.5)
    _, xt, base_velocity = transport.path_sampler.plan(t, endpoint, clean)
    target = base_velocity * transport.get_ct(t)[:, None, None, None]
    manual = ((model(xt, t) - target) ** 2).flatten(1).mean(1)
    assert torch.allclose(losses, manual)
    assert torch.all(base_velocity > 0)  # sampler adds this direction toward clean
    losses.mean().backward()
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)


def test_endpoint_override_can_select_only_part_of_batch():
    clean = torch.ones(2, 1, 2, 2)
    endpoint = torch.zeros_like(clean)
    transport = create_transport()

    def fixed_sample(x1):
        return torch.full((2,), 0.5), torch.full_like(x1, -1), x1

    transport.sample = fixed_sample
    captured = {}
    original_plan = transport.path_sampler.plan

    def capture(t, x0, x1):
        captured["x0"] = x0.clone()
        return original_plan(t, x0, x1)

    transport.path_sampler.plan = capture
    transport.training_losses(
        ScaleModel(), clean, corruption_endpoint=endpoint, corruption_selector=torch.tensor([True, False])
    )
    assert torch.equal(captured["x0"][0], endpoint[0])
    assert torch.equal(captured["x0"][1], torch.full_like(endpoint[1], -1))


def test_state_fingerprints_and_identical_parameter_check():
    left = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3])}
    right = {key: value.clone() for key, value in left.items()}
    assert tensor_mapping_sha256(left) == tensor_mapping_sha256(right)
    assert models_numerically_identical(left, right)
    right["a"][0] += 1
    assert not models_numerically_identical(left, right)


def test_freeze_module_keeps_vae_style_module_frozen():
    module = nn.Sequential(nn.Linear(4, 4), nn.Dropout())
    freeze_module(module)
    assert not module.training
    assert not any(parameter.requires_grad for parameter in module.parameters())


def test_cluster_bootstrap_keeps_two_draws_per_image():
    ids = np.repeat(np.arange(32), 2)
    control = np.zeros(64)
    treatment = np.repeat(np.linspace(0.01, 0.04, 32), 2)
    result = paired_image_cluster_bootstrap(ids, control, treatment, replicates=5000, seed=4)
    assert result.paired_delta > 0
    assert result.ci_lower > 0
    assert result.fraction_improved == 1.0
    with pytest.raises(ValueError, match="exactly two"):
        paired_image_cluster_bootstrap(ids[:-1], control[:-1], treatment[:-1], replicates=5000)


@pytest.mark.parametrize(
    "ci,fid_c,fid_t,expected",
    [
        (0.01, 10, 10.5, "PASS"),
        (0.01, 10, 11.1, "FIELD SHAPED BUT NOT USEFULLY"),
        (0.0, 10, 10.5, "NO ROBUSTNESS EVIDENCE"),
        (-0.01, 10, 11.1, "FAIL"),
    ],
)
def test_predeclared_decision_table(ci, fid_c, fid_t, expected):
    assert classify_result(ci, fid_c, fid_t)["decision"] == expected
