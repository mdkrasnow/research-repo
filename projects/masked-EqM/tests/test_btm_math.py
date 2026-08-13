"""Mathematical unit tests for the corrected-BTM / FD-scalar campaign.

Everything here is cheap and CPU-only; it must pass before ANY cluster compute.
"""

import math

import pytest
import torch

from experiments.btm.fd import (
    DoubleBackwardViolation,
    assert_no_double_backward,
    directional_fd,
    exact_directional_derivative,
    exact_gradient,
    rademacher_directions,
)
from experiments.btm.interpolant import (
    EqMLinearTarget,
    LinearInterpolant,
    SelfStoppingInterpolant,
)
from experiments.btm.models import ScalarMLP
from experiments.btm.objectives import (
    FDConfig,
    loss_action_exact,
    loss_fd_action,
    loss_fd_directional,
    loss_scalar_exact,
)

TCS = [0.5, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------- interpolant
@pytest.mark.parametrize("tc", TCS)
def test_interpolant_endpoints(tc):
    ip = SelfStoppingInterpolant(tc=tc)
    t0 = torch.tensor([0.0], dtype=torch.float64)
    t1 = torch.tensor([1.0], dtype=torch.float64)
    assert torch.allclose(ip.alpha(t0), torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(ip.alpha(t1), torch.tensor([0.0], dtype=torch.float64),
                          atol=1e-14)
    assert torch.allclose(ip.beta(t0), torch.tensor([0.0], dtype=torch.float64))
    assert torch.allclose(ip.beta(t1), torch.tensor([1.0], dtype=torch.float64),
                          atol=1e-14)


@pytest.mark.parametrize("tc", TCS)
def test_alpha_and_alphadot_continuous_at_tc(tc):
    ip = SelfStoppingInterpolant(tc=tc)
    eps = 1e-9
    tm = torch.tensor([tc - eps], dtype=torch.float64)
    tp = torch.tensor([tc + eps], dtype=torch.float64)
    assert abs(float(ip.alpha(tm) - ip.alpha(tp))) < 1e-7
    assert abs(float(ip.alpha_dot(tm) - ip.alpha_dot(tp))) < 1e-7
    # closed-form values at the breakpoint
    assert abs(float(ip.alpha(tm)) - (1 - tc) / (1 + tc)) < 1e-7
    assert abs(float(ip.alpha_dot(tm)) + 2.0 / (1 + tc)) < 1e-7


@pytest.mark.parametrize("tc", TCS)
def test_alpha_dot_matches_numerical_derivative(tc):
    ip = SelfStoppingInterpolant(tc=tc)
    t = torch.linspace(0.01, 0.99, 97, dtype=torch.float64)
    t = t[(t - tc).abs() > 1e-3]                       # skip the kink
    h = 1e-6
    num = (ip.alpha(t + h) - ip.alpha(t - h)) / (2 * h)
    assert torch.allclose(num, ip.alpha_dot(t), atol=1e-6)


@pytest.mark.parametrize("tc", TCS)
def test_Idot_vanishes_at_t1(tc):
    ip = SelfStoppingInterpolant(tc=tc)
    x0 = torch.randn(16, 3, dtype=torch.float64)
    x1 = torch.randn(16, 3, dtype=torch.float64)
    t = torch.ones(16, dtype=torch.float64)
    z, zdot = ip.interpolate(t, x0, x1)
    assert torch.allclose(z, x1, atol=1e-14)
    assert zdot.abs().max() < 1e-14


def test_linear_interpolant_is_the_eqm_ct_equals_one_case():
    """Paper's own statement: eq. (16) == eq. (5) exactly when c_t == 1."""
    lin = LinearInterpolant()
    eqm = EqMLinearTarget(kappa=0.0)                   # c_t = (1-t)^0 = 1
    x0 = torch.randn(32, 2, dtype=torch.float64)
    x1 = torch.randn(32, 2, dtype=torch.float64)
    t = torch.rand(32, dtype=torch.float64)
    z_a, d_a = lin.interpolate(t, x0, x1)
    z_b, d_b = eqm.interpolate(t, x0, x1)
    assert torch.allclose(z_a, z_b)
    assert torch.allclose(d_a, d_b)


# ------------------------------------------------------------------ Rademacher
def test_rademacher_identity():
    torch.manual_seed(0)
    d, K = 8, 200_000
    u = rademacher_directions((1, d), K, dtype=torch.float64)[:, 0]   # [K, d]
    cov = u.T @ u / K
    assert torch.allclose(cov, torch.eye(d, dtype=torch.float64) / d, atol=2e-3)
    # E_u[(u^T e)^2] = |e|^2 / d
    e = torch.randn(d, dtype=torch.float64)
    lhs = ((u @ e) ** 2).mean()
    assert abs(float(lhs) - float(e @ e) / d) < 5e-3 * float(e @ e) / d


# ------------------------------------------------------- finite differences
def test_central_difference_is_second_order_on_analytic_function():
    """phi(x) = sin(a.x) + 0.5 |x|^2 ; error should fall ~ h^2."""
    torch.manual_seed(0)
    d = 4
    a = torch.randn(d, dtype=torch.float64)

    def phi(x):
        return torch.sin(x @ a) + 0.5 * (x ** 2).sum(-1)

    x = torch.randn(64, d, dtype=torch.float64)
    u = rademacher_directions(x.shape, 1, dtype=torch.float64)
    exact = exact_directional_derivative(phi, x, u)

    errs = []
    for eps in (1e-2, 1e-3):
        D, _, _ = directional_fd(phi, x, u, eps)
        errs.append(float((D - exact).abs().mean()))
    # ten-fold smaller h -> ~100x smaller truncation error (fp64, safe regime)
    assert errs[1] < errs[0] / 30.0


def test_directional_fd_converges_to_exact_on_a_network():
    torch.manual_seed(0)
    net = ScalarMLP(d=2, width=64, depth=2).double()
    x = torch.randn(256, 2, dtype=torch.float64)
    u = rademacher_directions(x.shape, 4, dtype=torch.float64)
    exact = exact_directional_derivative(net, x, u)
    D, _, _ = directional_fd(net, x, u, 1e-5)
    rel = (D - exact).norm() / exact.norm()
    assert float(rel) < 1e-6


def test_fd_loss_converges_to_gradient_matching_loss():
    """L_D -> (1/2d) E |grad phi - Idot|^2 as K rises and h falls."""
    torch.manual_seed(0)
    net = ScalarMLP(d=3, width=64, depth=2).double()
    ip = SelfStoppingInterpolant(tc=0.8)
    x0 = torch.randn(4096, 3, dtype=torch.float64)
    x1 = torch.randn(4096, 3, dtype=torch.float64)
    t = torch.rand(4096, dtype=torch.float64)
    z, zdot = ip.interpolate(t, x0, x1)

    lg = float(loss_scalar_exact(net, z, zdot))
    gen = torch.Generator().manual_seed(3)
    ld = float(loss_fd_directional(
        net, z, zdot, FDConfig(eps_fd=1e-6, K=256), generator=gen))
    assert abs(ld - lg) / lg < 0.05


# ------------------------------- exact vs Action/Ritz population identity ----
def test_gradient_matching_equals_action_up_to_phi_independent_constant():
    """L_G - J_A must be the SAME constant for two different networks.

    L_G = (1/2d)E|grad phi|^2 - (1/d)E[grad phi . Idot] + (1/2d)E|Idot|^2
        = J_A + (1/2d) E|Idot|^2   using  E[grad phi . Idot] = E_mu1 phi - E_mu0 phi.
    """
    torch.manual_seed(0)
    ip = SelfStoppingInterpolant(tc=0.8)
    n, d = 400_000, 2
    x0 = torch.randn(n, d, dtype=torch.float64)
    x1 = torch.randn(n, d, dtype=torch.float64)
    t = torch.rand(n, dtype=torch.float64)
    z, zdot = ip.interpolate(t, x0, x1)
    const = float((zdot ** 2).sum(1).mean()) / (2 * d)

    for seed in (0, 1):
        torch.manual_seed(seed)
        net = ScalarMLP(d=d, width=32, depth=2).double()
        # break the zero-ish init so the identity is a real test
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        lg = float(loss_scalar_exact(net, z, zdot))
        ja = float(loss_action_exact(net, z, x0, x1))
        assert abs((lg - ja) - const) < 2e-3 * max(abs(const), 1.0), (
            f"seed {seed}: L_G - J_A = {lg - ja:.6f}, expected {const:.6f}")


def test_fd_action_converges_to_exact_action():
    torch.manual_seed(0)
    net = ScalarMLP(d=3, width=64, depth=2).double()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    ip = SelfStoppingInterpolant(tc=0.8)
    x0 = torch.randn(8192, 3, dtype=torch.float64)
    x1 = torch.randn(8192, 3, dtype=torch.float64)
    t = torch.rand(8192, dtype=torch.float64)
    z, _ = ip.interpolate(t, x0, x1)
    ja = float(loss_action_exact(net, z, x0, x1))
    gen = torch.Generator().manual_seed(5)
    jf = float(loss_fd_action(net, z, x0, x1,
                              FDConfig(eps_fd=1e-6, K=256), generator=gen))
    assert abs(jf - ja) < 0.02 * max(abs(ja), 1e-3)


# ------------------------------------------------------------ autograd guard
def test_guard_blocks_create_graph():
    net = ScalarMLP(d=2, width=16, depth=1)
    x = torch.randn(4, 2)
    with pytest.raises(DoubleBackwardViolation):
        with assert_no_double_backward():
            exact_gradient(net, x, create_graph=True)


def test_fd_arms_do_not_build_input_gradient_graph():
    net = ScalarMLP(d=2, width=16, depth=1)
    ip = SelfStoppingInterpolant(tc=0.8)
    x0, x1 = torch.randn(8, 2), torch.randn(8, 2)
    t = torch.rand(8)
    z, zdot = ip.interpolate(t, x0, x1)
    gen = torch.Generator().manual_seed(0)
    with assert_no_double_backward():
        ld = loss_fd_directional(net, z, zdot, FDConfig(1e-3, 4), generator=gen)
        ld.backward()
        jf = loss_fd_action(net, z, x0, x1, FDConfig(1e-3, 4), generator=gen)
        jf.backward()
    assert all(p.grad is not None for p in net.parameters())


def test_exact_arms_do_use_double_backward():
    """Negative control for the guard itself: G must trip it."""
    net = ScalarMLP(d=2, width=16, depth=1)
    z = torch.randn(8, 2)
    zdot = torch.randn(8, 2)
    with pytest.raises(DoubleBackwardViolation):
        with assert_no_double_backward():
            loss_scalar_exact(net, z, zdot)


# -------------------------------------------------------------------- signs
def test_one_atom_analytic_flow_moves_towards_the_atom():
    """With mu1 = delta_a, the population drift must push samples TO a.

    b*(x) = E[Idot | I_t = x] and Idot = alpha_dot (x0 - x1) with alpha_dot < 0,
    so the target points along (x1 - x0) = (a - x0): towards the atom.  Trained
    briefly, following +b (equivalently descending E = -phi) must reduce the
    distance to a.
    """
    torch.manual_seed(0)
    a = torch.tensor([[2.0, -1.0]])
    ip = SelfStoppingInterpolant(tc=0.8)
    net = ScalarMLP(d=2, width=64, depth=2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(1500):
        x0 = torch.randn(512, 2)
        x1 = a.expand(512, 2)
        t = torch.rand(512)
        z, zdot = ip.interpolate(t, x0, x1)
        loss = loss_scalar_exact(net, z, zdot)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    x = torch.randn(2048, 2)
    d0 = (x - a).norm(dim=1).mean()
    for _ in range(400):
        g = exact_gradient(net, x, create_graph=False).detach()
        x = (x + 0.05 * g).detach()
    d1 = (x - a).norm(dim=1).mean()
    assert float(d1) < 0.35 * float(d0), (float(d0), float(d1))


def test_energy_convention_is_descent():
    """E = -phi, so following +grad phi is gradient DESCENT on E."""
    net = ScalarMLP(d=2, width=16, depth=1)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    x = torch.randn(64, 2)
    g = exact_gradient(net, x, create_graph=False).detach()
    eta = 1e-3
    phi_before = net(x).detach()
    phi_after = net(x + eta * g).detach()
    assert float((phi_after - phi_before).mean()) > 0     # phi increases
    # energy E = -phi therefore decreases
    assert float(((-phi_after) - (-phi_before)).mean()) < 0


def test_atom_geometry_is_well_separated():
    from experiments.btm.toy5 import ATOM_WEIGHTS, atoms
    A = atoms()
    assert abs(sum(ATOM_WEIGHTS) - 1.0) < 1e-12
    d = torch.cdist(A, A)
    d = d + torch.eye(5) * 1e9
    assert float(d.min()) > 2.0        # well separated relative to N(0,I) scale
    assert math.isclose(float(A.norm(dim=1).mean()), 3.0, rel_tol=1e-5)
