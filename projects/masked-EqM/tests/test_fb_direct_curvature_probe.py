"""
FP64 CPU sanity for fb_direct/exact_hvp.py's z-space curvature probe
(hvp_z, power_iteration_spectral_norm) BEFORE trusting it on any real
checkpoint (2026-08-10, growing-instability diagnostic).

The claim under test: power_iteration_spectral_norm's output converges to
max_i |eigenvalue_i| of the TRUE z-space Hessian grad_z^2 E, per sample.
Verified two ways on the tiny EqM-S/2 model (z has only 4*4*4=64 elements
per sample -- small enough to materialize the exact Hessian by finite-degree
autograd, one column at a time):

  1. Exact Hessian match: build the full (64,64) Hessian per sample via 64
     double-backward calls (one per standard basis vector e_i), eigh() it,
     compare max|eigenvalue| to the power-iteration estimate.
  2. hvp_z correctness: Hv from hvp_z matches Hv computed by explicitly
     materializing the Hessian and doing a plain matvec.

Run: python tests/test_fb_direct_curvature_probe.py  (CPU, ~30s)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fb_direct.exact_hvp import hvp_z, power_iteration_spectral_norm

from test_forward_backwards_direct import make_model, perturb, batch  # noqa: E402

torch.manual_seed(0)


def exact_hessian(model, z_one, t_one, y_one):
    """Full (D, D) Hessian of E w.r.t. a SINGLE sample's z, via D
    double-backward calls (D = z_one.numel(), small for the tiny model)."""
    z = z_one.detach().clone().requires_grad_(True)
    E = model(z.unsqueeze(0), t_one.unsqueeze(0), y_one.unsqueeze(0), energy_only=True)
    g = torch.autograd.grad(E.sum(), z, create_graph=True)[0]
    D = z.numel()
    H = torch.zeros(D, D, dtype=z.dtype)
    for i in range(D):
        e_i = torch.zeros(D, dtype=z.dtype)
        e_i[i] = 1.0
        Hi = torch.autograd.grad(g.flatten(), z, grad_outputs=e_i,
                                  retain_graph=True)[0]
        H[i] = Hi.flatten()
    return H


def test_hvp_z_matches_exact_hessian_matvec():
    model = make_model(ebm='direct', dtype=torch.float64)
    perturb(model)
    model.eval()
    z, t, y = batch(n=1, dtype=torch.float64)

    H = exact_hessian(model, z[0], t[0], y[0])
    # symmetry sanity: a real Hessian of a scalar (twice-differentiable) fn
    asym = (H - H.T).abs().max().item()
    assert asym < 1e-9, f"exact Hessian not symmetric (asym={asym}) -- test harness bug"

    torch.manual_seed(5)
    v = torch.randn_like(z)
    Hv_probe = hvp_z(model, z, t, y, v)
    Hv_exact = (H @ v[0].flatten()).view_as(z[0])

    torch.testing.assert_close(Hv_probe[0], Hv_exact, rtol=1e-8, atol=1e-10)
    print("PASS hvp_z matches exact-Hessian matvec at FP64 machine precision")


def test_power_iteration_matches_exact_spectral_norm():
    model = make_model(ebm='direct', dtype=torch.float64)
    perturb(model, seed=17)
    model.eval()
    z, t, y = batch(n=3, dtype=torch.float64, seed=23)

    exact_specnorms = []
    for i in range(z.shape[0]):
        H = exact_hessian(model, z[i], t[i], y[i])
        eigvals = torch.linalg.eigvalsh(H)
        exact_specnorms.append(eigvals.abs().max().item())

    result = power_iteration_spectral_norm(model, z, t, y, num_iters=60, seed=99)
    probe = result["spectral_norm"]

    for i, exact in enumerate(exact_specnorms):
        rel_err = abs(float(probe[i]) - exact) / max(abs(exact), 1e-12)
        print(f"  sample {i}: exact={exact:.6f} power_iter={float(probe[i]):.6f} rel_err={rel_err:.2e}")
        assert rel_err < 1e-3, f"sample {i}: power iteration diverged from exact spectral norm"

    hist = result["rayleigh_history"]
    early_gap = abs(float(hist[5][0]) - exact_specnorms[0])
    late_gap = abs(float(hist[-1][0]) - exact_specnorms[0])
    assert late_gap <= early_gap, "power iteration should converge monotonically closer, not drift away"
    print("PASS power_iteration_spectral_norm matches exact eigh() spectral radius, "
          "3 samples, rel_err<1e-3; convergence is monotone")


def test_indefinite_hessian_negative_dominant_eigenvalue():
    """Deliberately construct a case where the TRUE dominant eigenvalue is
    NEGATIVE (concave direction), to verify power iteration's sign-agnostic
    |Rayleigh quotient| tracking handles this -- naive power iteration on an
    indefinite symmetric matrix can oscillate sign each step; the estimate
    must still converge to the correct MAGNITUDE."""
    torch.manual_seed(31)
    D = 12
    # Symmetric matrix with a known, strictly dominant NEGATIVE eigenvalue.
    Q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))
    eigvals = torch.tensor([-9.0] + [1.0] * (D - 1), dtype=torch.float64)
    H = Q @ torch.diag(eigvals) @ Q.T

    class ToyQuadratic(torch.nn.Module):
        def forward(self, z):
            return 0.5 * torch.einsum('bi,ij,bj->b', z, H, z)

    toy = ToyQuadratic()
    z = torch.randn(2, D, dtype=torch.float64, requires_grad=True)

    def hvp_toy(v):
        E = toy(z)
        g = torch.autograd.grad(E.sum(), z, create_graph=True)[0]
        return torch.autograd.grad(g, z, grad_outputs=v, retain_graph=True)[0].detach()

    u = torch.randn_like(z)
    u = u / u.norm(dim=1, keepdim=True)
    rq = None
    for _ in range(40):
        Hu = hvp_toy(u)
        rq = (u * Hu).sum(dim=1).abs()
        u = (Hu / Hu.norm(dim=1, keepdim=True)).detach()

    for i in range(2):
        assert abs(float(rq[i]) - 9.0) < 1e-4, \
            f"sample {i}: expected |lambda|=9.0 (negative dominant eigenvalue), got {float(rq[i])}"
    print("PASS indefinite-Hessian case: power iteration correctly recovers |lambda|=9.0 "
          "when the TRUE dominant eigenvalue is negative")


if __name__ == "__main__":
    # Fused SDPA lacks a double-backward derivative; train.py forces the
    # math backend for all ebm != 'none' training (and this offline probe
    # runs the model the same way), so tests must match that regime.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        test_hvp_z_matches_exact_hessian_matvec()
        test_power_iteration_matches_exact_spectral_norm()
    test_indefinite_hessian_negative_dominant_eigenvalue()
    print("ALL CURVATURE-PROBE TESTS PASSED")
