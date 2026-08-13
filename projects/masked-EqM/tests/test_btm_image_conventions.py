"""Sign/convention reconciliation between the BTM module and the EqM repo.

These are the assertions that keep the image-stage arms from silently training
the negation of the intended target.  All CPU, all tiny.
"""

import pytest
import torch

from models import EqM
from transport import create_transport
from transport.path import ICPlan
from experiments.btm.image_losses import (
    BTMConfig,
    btm_loss,
    build_image_interpolant,
    phi_closure,
)
from experiments.btm.interpolant import LinearInterpolant, SelfStoppingInterpolant
from experiments.btm.fd import DoubleBackwardViolation


def _tiny(ebm):
    return EqM(input_size=8, patch_size=2, in_channels=4, hidden_size=32,
               depth=2, num_heads=2, num_classes=10, learn_sigma=True,
               uncond=True, ebm=ebm)


def test_repo_ICPlan_matches_btm_linear_interpolant():
    """xt = (1-t)x0 + t x1 and ut = x1 - x0 in BOTH conventions, no sign flip."""
    plan = ICPlan()
    lin = LinearInterpolant()
    x0 = torch.randn(8, 4, 8, 8, dtype=torch.float64)
    x1 = torch.randn(8, 4, 8, 8, dtype=torch.float64)
    t = torch.rand(8, dtype=torch.float64)
    _, xt, ut = plan.plan(t, x0, x1)
    z, zdot = lin.interpolate(t, x0, x1)
    assert torch.allclose(xt, z, atol=1e-12)
    assert torch.allclose(ut, zdot, atol=1e-12)
    assert torch.allclose(ut, x1 - x0, atol=1e-12)


def test_legacy_eqm_target_is_ct_scaled_btm_target():
    """The repo's `ut * get_ct(t)` is exactly eq.(16): the BTM target x c(t)."""
    tr = create_transport("Linear", "velocity", "None", 0, 0)
    x0 = torch.randn(16, 4, 8, 8)
    x1 = torch.randn(16, 4, 8, 8)
    t = torch.rand(16)
    _, xt, ut = tr.path_sampler.plan(t, x0, x1)
    legacy = ut * tr.get_ct(t)[:, None, None, None]
    lin = LinearInterpolant()
    _, zdot = lin.interpolate(t, x0, x1)
    ratio = (legacy / (zdot + 1e-12)).reshape(16, -1).median(dim=1).values
    assert torch.allclose(ratio, tr.get_ct(t), atol=1e-4)


def test_direct_model_field_is_grad_phi_with_phi_equals_minus_E():
    """models.EqM(ebm='direct') returns -grad_z E; we call that grad_z phi."""
    torch.manual_seed(0)
    m = _tiny("direct")
    # zero-init head would make the field identically zero -- perturb first
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    # Label dropout is resampled every forward in train mode; eval() makes the
    # two evaluations comparable (this is exactly the effect that
    # frozen_label_dropout neutralizes inside the real training step).
    m.eval()
    x = torch.randn(4, 4, 8, 8)
    t = torch.rand(4)
    y = torch.randint(0, 10, (4,))
    field = m(x.clone(), t, y)
    phi = phi_closure(m, t, y)
    z = x.clone().detach().requires_grad_(True)
    g = torch.autograd.grad(phi(z).sum(), z)[0]
    assert torch.allclose(field, g, atol=1e-4), (
        float((field - g).abs().max()))


def test_following_the_field_decreases_the_energy():
    """E = -phi, so x <- x + eta * field is gradient DESCENT on E."""
    torch.manual_seed(0)
    m = _tiny("direct")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    x = torch.randn(8, 4, 8, 8)
    t = torch.rand(8)
    y = torch.randint(0, 10, (8,))
    field = m(x.clone(), t, y).detach()
    with torch.no_grad():
        E0 = m(x.clone(), t, y, energy_only=True)
        E1 = m((x + 1e-3 * field).clone(), t, y, energy_only=True)
    assert float((E1 - E0).mean()) < 0


@pytest.mark.parametrize("mode", ["btm_scalar_fd_directional",
                                  "btm_scalar_fd_action"])
def test_image_fd_arms_never_build_an_input_gradient_graph(mode):
    torch.manual_seed(0)
    m = _tiny("direct")
    tr = create_transport("Linear", "velocity", "None", 0, 0)
    x1 = torch.randn(4, 4, 8, 8)
    y = torch.randint(0, 10, (4,))
    cfg = BTMConfig(mode=mode, fd_eps=1e-3, fd_k=2)
    loss, stats = btm_loss(m, m, cfg, tr, x1, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.parameters())
    assert stats["fd_h_mean"] > 0


def test_image_exact_arm_does_use_double_backward():
    """Positive control: Arm G must genuinely take the mixed-derivative path."""
    from experiments.btm.fd import assert_no_double_backward
    torch.manual_seed(0)
    m = _tiny("direct")
    tr = create_transport("Linear", "velocity", "None", 0, 0)
    x1 = torch.randn(4, 4, 8, 8)
    y = torch.randint(0, 10, (4,))
    cfg = BTMConfig(mode="btm_scalar_exact")
    with pytest.raises(DoubleBackwardViolation):
        with assert_no_double_backward():
            btm_loss(m, m, cfg, tr, x1, y)


def test_image_fd_loss_approaches_exact_loss_at_small_h_and_large_K():
    """L_D -> (1/d) E|grad phi - Idot|^2 on the REAL architecture."""
    torch.manual_seed(0)
    # fp32: TimestepEmbedder builds its sinusoid in float32 internally, so the
    # module is not cleanly promotable to fp64.  With fp32 the usable FD window
    # is bounded below by cancellation, hence eps=1e-3 (the calibrated plateau)
    # rather than 1e-6, and a correspondingly looser tolerance.
    m = _tiny("direct")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    tr = create_transport("Linear", "velocity", "None", 0, 0)
    x1 = torch.randn(8, 4, 8, 8)
    y = torch.randint(0, 10, (8,))
    torch.manual_seed(3)
    lg, _ = btm_loss(m, m, BTMConfig(mode="btm_scalar_exact"), tr, x1, y)
    torch.manual_seed(3)
    ld, _ = btm_loss(m, m, BTMConfig(mode="btm_scalar_fd_directional",
                                     fd_eps=1e-3, fd_k=256), tr, x1, y)
    assert abs(float(ld) - float(lg)) / float(lg) < 0.20, (float(ld), float(lg))


def test_vector_arm_requires_a_vector_model():
    torch.manual_seed(0)
    m = _tiny("none")
    tr = create_transport("Linear", "velocity", "None", 0, 0)
    x1 = torch.randn(4, 4, 8, 8)
    y = torch.randint(0, 10, (4,))
    loss, _ = btm_loss(m, m, BTMConfig(mode="btm_vector"), tr, x1, y)
    loss.backward()
    assert torch.isfinite(loss)


def test_self_stopping_target_vanishes_near_data_but_linear_does_not():
    ip = SelfStoppingInterpolant(tc=0.8)
    lin = LinearInterpolant()
    x0 = torch.randn(64, 4, 8, 8)
    x1 = torch.randn(64, 4, 8, 8)
    t = torch.full((64,), 0.999)
    _, d_ss = ip.interpolate(t, x0, x1)
    _, d_lin = lin.interpolate(t, x0, x1)
    assert float(d_ss.norm()) < 0.02 * float(d_lin.norm())


def test_build_image_interpolant_dispatch():
    assert isinstance(build_image_interpolant(BTMConfig(interpolant="linear")),
                      LinearInterpolant)
    assert isinstance(
        build_image_interpolant(BTMConfig(interpolant="self_stopping", tc=0.6)),
        SelfStoppingInterpolant)
