"""
CPU/FP64 sanity tests for fb_direct/adjoint_optimization.py (Test A of
Yilun's post-decomposition decision tree, 2026-08-07), BEFORE spending any
GPU time on the real epoch-40 checkpoint.

1. Adjoint-consistency: for random theta-space direction v and random
   cache-space cotangent u, <vjp_cache_to_theta(u), v> == <u,
   jvp_theta_to_cache(v)>. This is THE correctness check for a JVP
   implementation validated against an already-verified VJP (the identity
   <A^T u, v> = <u, A v> holds for ANY linear operator A -- if this fails,
   the JVP's double-backward trick has a bug, full stop, independent of
   anything else in this module).

2. CGNR sanity: on the tiny synthetic model, does a few iterations of CGNR
   reduce the least-squares residual monotonically and produce a solution
   a_best whose reconstruction J_C^T a_best has HIGHER cosine similarity to
   a target b than a naive/random a would.

Run: python tests/test_fb_direct_adjoint_optimization.py  (CPU, ~tens of seconds)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fb_direct import ForwardBackwardsDirectTrainer
from fb_direct.adjoint_optimization import (
    vjp_cache_to_theta,
    jvp_theta_to_cache,
    cgnr_solve_optimal_adjoint,
    _active_theta_params,
)
from fb_direct.forward_cache_grad import forward_energy_with_cache_grad
from transport import create_transport

from test_forward_backwards_direct import make_model, perturb  # noqa: E402

torch.manual_seed(0)


def _make_fixed_batch(dtype=torch.float64, seed=7):
    # Wrap in ForwardBackwardsDirectTrainer (not a bare model) so
    # cache_only params (t_embedder/y_embedder -> the `c` cache tensor) are
    # frozen exactly as they are for `theta` in the real experiment script
    # -- job 37687761 crashed on GPU precisely because this fixture
    # originally used a bare, unfrozen model and never exercised that path.
    model = make_model(ebm='forward-backwards-direct', dtype=dtype, seed=0)
    perturb(model, seed=101)
    model.eval()
    ForwardBackwardsDirectTrainer(model, lr=1e-4, device=torch.device('cpu'))

    transport = create_transport("Linear", "velocity", None, None, None)
    g = torch.Generator().manual_seed(seed)
    x1 = torch.randn(2, 4, 4, 4, generator=g).to(dtype)
    y = torch.randint(0, 10, (2,), generator=g)

    t, x0, x1 = transport.sample(x1)
    t = t.to(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    ut = ut * transport.get_ct(t)[:, None, None, None]
    return model, xt, t, y, ut


def test_adjoint_consistency_fp64():
    model, xt, t, y, ut = _make_fixed_batch()
    theta_names, theta_params = _active_theta_params(model)

    torch.manual_seed(42)
    v = {n: torch.randn_like(p) for n, p in zip(theta_names, theta_params)}

    with torch.no_grad():
        _, cache = forward_energy_with_cache_grad(model, xt.clone(), t, y)
    cache_shapes = dict(cache.flatten())
    u = {n: torch.randn_like(c) for n, c in cache_shapes.items()}

    Av = vjp_cache_to_theta(model, xt, t, y, u)       # A u = J_C^T u (theta-space)
    ATu = jvp_theta_to_cache(model, xt, t, y, v)      # A^T v = J_C v (cache-space)

    lhs = sum((Av[n].reshape(-1) * v[n].reshape(-1)).sum() for n in v if n in Av)
    rhs = sum((u[n].reshape(-1) * ATu[n].reshape(-1)).sum() for n in u if n in ATu)

    rel_err = float((lhs - rhs).abs() / max(abs(float(lhs)), abs(float(rhs)), 1e-30))
    print(f"[adjoint consistency FP64] <VJP(u),v>={float(lhs):.6e} "
          f"<u,JVP(v)>={float(rhs):.6e} rel_err={rel_err:.3e}")
    assert rel_err < 1e-8, (float(lhs), float(rhs), rel_err)
    print("PASS adjoint consistency: <A^T u, v> == <u, A v> to FP64 precision")


def test_cgnr_reduces_residual_and_improves_cosine():
    """Uses a PLANTED-solution target b = J_C^T a_true (guaranteed to lie
    exactly in Range(J_C^T)), not an arbitrary random theta-space vector.
    This toy model's theta-active dimension (~tens of millions, driven by
    hidden_size^2 weight matrices) vastly exceeds its cache dimension
    (~hundreds of thousands, driven by only 4 tokens at input_size=4) --
    an unrelated random b would be almost entirely orthogonal to the tiny
    range of A=J_C^T by dimension-counting alone, making "does CG reduce
    the residual a lot" the wrong question for THIS toy config (that
    imbalance is specific to the tiny synthetic model's tiny image size;
    the real B/2 checkpoint has a much larger cache dimension relative to
    active-parameter count). Planting b in-range isolates what we actually
    want to check here: does the CG *mechanics* work.
    """
    model, xt, t, y, ut = _make_fixed_batch()
    theta_names, theta_params = _active_theta_params(model)

    torch.manual_seed(43)
    with torch.no_grad():
        _, cache = forward_energy_with_cache_grad(model, xt.clone(), t, y)
    cache_shapes = dict(cache.flatten())
    a_true = {n: torch.randn_like(c) for n, c in cache_shapes.items()}
    b = vjp_cache_to_theta(model, xt, t, y, a_true)  # planted: b in Range(A) exactly

    a_best, history = cgnr_solve_optimal_adjoint(model, xt, t, y, b, num_iters=15)

    residuals = [h["residual_norm"] for h in history]
    print(f"[cgnr] residual norms: {['%.4e' % r for r in residuals]}")
    assert residuals[-1] < residuals[0] * 0.5, residuals

    recon = vjp_cache_to_theta(model, xt, t, y, a_best)
    recon_vec = torch.cat([recon[n].reshape(-1) for n in theta_names])
    b_vec = torch.cat([b[n].reshape(-1) for n in theta_names])
    cos_best = float(torch.nn.functional.cosine_similarity(
        recon_vec.unsqueeze(0), b_vec.unsqueeze(0)).item())

    torch.manual_seed(44)
    a_random = {n: torch.randn_like(c) * 1e-3 for n, c in cache_shapes.items()}
    recon_rand = vjp_cache_to_theta(model, xt, t, y, a_random)
    recon_rand_vec = torch.cat([recon_rand[n].reshape(-1) for n in theta_names])
    cos_random = float(torch.nn.functional.cosine_similarity(
        recon_rand_vec.unsqueeze(0), b_vec.unsqueeze(0)).item())

    print(f"[cgnr] cos(J_C^T a_best, b)={cos_best:.4f} vs cos(J_C^T a_random, b)={cos_random:.4f}")
    assert cos_best > 0.9 and cos_best > cos_random
    print("PASS CGNR: residual decreases substantially toward a planted in-range "
          "solution, and reconstruction beats a random adjoint")


def test_cgnr_warm_start_never_worse_than_start():
    """The v2 correction (job 37692043 postmortem): warm-starting CGNR at a
    feasible point must yield a best-iterate rho >= the starting point's rho
    -- the property whose violation proved the v1 run invalid. Planted case:
    warm-starting AT the exact solution must return rho ~ 1 immediately and
    never degrade the returned iterate."""
    model, xt, t, y, ut = _make_fixed_batch()
    theta_names, theta_params = _active_theta_params(model)

    torch.manual_seed(47)
    with torch.no_grad():
        _, cache = forward_energy_with_cache_grad(model, xt.clone(), t, y)
    cache_shapes = dict(cache.flatten())
    a_true = {n: torch.randn_like(c) for n, c in cache_shapes.items()}
    b = vjp_cache_to_theta(model, xt, t, y, a_true)

    a_best, history = cgnr_solve_optimal_adjoint(
        model, xt, t, y, b, num_iters=3, x0=a_true,
    )
    rhos = [h["rho"] for h in history]
    print(f"[cgnr-warm] rho trajectory from exact warm start: {['%.6f' % r for r in rhos]}")
    assert rhos[0] > 0.999999, f"warm start at exact solution should give rho ~ 1, got {rhos[0]}"

    recon = vjp_cache_to_theta(model, xt, t, y, a_best)
    recon_vec = torch.cat([recon[n].reshape(-1) for n in theta_names])
    b_vec = torch.cat([b[n].reshape(-1) for n in theta_names])
    cos_returned = float(torch.nn.functional.cosine_similarity(
        recon_vec.unsqueeze(0), b_vec.unsqueeze(0)).item())
    assert cos_returned >= rhos[0] - 1e-9, \
        f"returned iterate ({cos_returned}) worse than warm start ({rhos[0]})"
    print("PASS CGNR warm start: exact-solution start gives rho ~ 1; returned "
          "best-iterate never degrades below the starting point")


if __name__ == '__main__':
    test_adjoint_consistency_fp64()
    test_cgnr_reduces_residual_and_improves_cosine()
    test_cgnr_warm_start_never_worse_than_start()
    print("\nALL ADJOINT-OPTIMIZATION (TEST A) SANITY TESTS PASSED")
