"""
Sanity tests for the ebm='scalar' direct scalar-energy parameterization.

Checks, per the pre-launch test plan:
  1. Field shape, finiteness, and gradient flow into the scalar head at
     zero initialization (and documents the l2 mode's degenerate stationary
     initialization by contrast).
  2. Finite-difference check that the returned field is -grad_x E (sign and
     value), for both forward and forward_with_cfg (full-channel guidance).
  3. Sampling under torch.no_grad() still produces a nonzero-graph-free field.

Run: python tests/test_scalar_ebm.py  (CPU, ~seconds)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import EqM_models

torch.manual_seed(0)


def make_model(ebm):
    model = EqM_models['EqM-S/2'](
        input_size=4,
        in_channels=4,
        num_classes=10,
        learn_sigma=False,
        uncond=True,
        ebm=ebm,
    )
    model.eval()
    return model


def perturb_from_init(model, std=0.02):
    # Zero init gives an identically-zero field; nudge all params so the
    # energy is a nontrivial function of x for the finite-difference test.
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(std * torch.randn_like(p))


def batch(n=2):
    z = torch.randn(n, 4, 4, 4)
    t = torch.rand(n)
    y = torch.randint(0, 10, (n,))
    return z, t, y


def test_shape_and_grad_flow():
    model = make_model('scalar')
    model.train()
    z, t, y = batch()
    target = torch.randn_like(z)

    field = model(z, t, y, train=True)
    assert field.shape == z.shape, field.shape
    assert torch.isfinite(field).all()

    loss = (field - target).square().mean()
    loss.backward()

    grad = model.energy_head.proj.weight.grad
    assert grad is not None
    assert grad.norm() > 0, "scalar head got zero gradient at zero init"
    print(f"PASS shape+gradflow: scalar proj grad norm {grad.norm():.3e} > 0 at zero init")


def test_l2_degenerate_init():
    # Contrast case: the quadratic l2 construction is stationary at the
    # upstream zero initialization of the final layer.
    model = make_model('l2')
    model.train()
    z, t, y = batch()
    target = torch.randn_like(z)

    field = model(z, t, y, train=True)
    loss = (field - target).square().mean()
    loss.backward()

    grad = model.final_layer.linear.weight.grad
    norm = 0.0 if grad is None else grad.norm().item()
    assert norm == 0.0, f"expected degenerate zero gradient for l2 at init, got {norm}"
    print("PASS l2 degeneracy check: final-layer grad is exactly 0 at zero init (as predicted)")


def finite_difference_check(model, forward_fn, z, t, y, eps=1e-3, direction=None):
    if direction is None:
        direction = torch.randn_like(z)
        direction = direction / direction.flatten(1).norm(dim=1)[:, None, None, None]

    field, energy = forward_fn(z, t, y, get_energy=True)
    _, energy_plus = forward_fn(z + eps * direction, t, y, get_energy=True)
    _, energy_minus = forward_fn(z - eps * direction, t, y, get_energy=True)

    fd = (energy_plus - energy_minus) / (2 * eps)
    autodiff = -(field * direction).flatten(1).sum(dim=1)
    torch.testing.assert_close(fd, autodiff.detach(), rtol=1e-2, atol=1e-2)
    return fd, autodiff


def test_field_is_neg_grad_energy():
    model = make_model('scalar')
    perturb_from_init(model)
    z, t, y = batch()

    def fwd(z, t, y, get_energy):
        return model(z, t, y, get_energy=get_energy, train=False)

    fd, ad = finite_difference_check(model, fwd, z, t, y)
    print(f"PASS finite-difference sign: field == -grad_x E "
          f"(fd {fd.tolist()} vs -<field,dir> {[f'{v:.4f}' for v in ad.tolist()]})")


def test_cfg_full_channel_conservative():
    model = make_model('scalar')
    perturb_from_init(model)
    n = 2
    z_half = torch.randn(n, 4, 4, 4)
    z = torch.cat([z_half, z_half], 0)
    t = torch.rand(n).repeat(2)
    y = torch.cat([torch.randint(0, 10, (n,)), torch.randint(0, 10, (n,))])
    cfg_scale = 4.0

    def fwd(z, t, y, get_energy):
        return model.forward_with_cfg(z, t, y, cfg_scale, get_energy=get_energy, train=False)

    field, energy = fwd(z, t, y, get_energy=True)
    assert field.shape == z.shape
    assert torch.allclose(field[:n], field[n:]), "cfg halves must be identical"
    assert torch.allclose(energy[:n], energy[n:])

    # Guided field must be -grad of the guided energy (conservativeness of cfg).
    # forward_with_cfg discards the second input half (duplicates the first),
    # so the perturbation direction must be duplicated across halves too.
    d_half = torch.randn_like(z_half)
    d_half = d_half / d_half.flatten(1).norm(dim=1)[:, None, None, None]
    finite_difference_check(model, fwd, z, t, y, direction=torch.cat([d_half, d_half], 0))
    print("PASS cfg: full-channel guided field == -grad_x E_cfg, halves duplicated")


def test_sampling_under_no_grad():
    model = make_model('scalar')
    perturb_from_init(model)
    z, t, y = batch()

    with torch.no_grad():
        field = model(z, t, y, train=False)

    assert field.shape == z.shape
    assert torch.isfinite(field).all()
    assert field.abs().sum() > 0, "field silently zero under no_grad (grad-context bug)"
    assert not field.requires_grad, "field must not retain graph across sampling steps"
    print(f"PASS no_grad sampling: nonzero field (|f| sum {field.abs().sum():.3e}), no retained graph")


if __name__ == '__main__':
    # Energy modes differentiate through attention (double backward); flash /
    # mem-efficient SDP kernels lack a second derivative — same reason
    # train.py forces math SDP when args.ebm != 'none'.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        test_shape_and_grad_flow()
        test_l2_degenerate_init()
        test_field_is_neg_grad_energy()
        test_cfg_full_channel_conservative()
        test_sampling_under_no_grad()
    print("\nALL SCALAR-EBM SANITY TESTS PASSED")
