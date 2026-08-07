"""
CPU/FP64 sanity tests for the cache-adjoint decomposition (Section 5 of the
learned-cache-adjoint proposal, 2026-08-07) BEFORE spending any GPU time on
the real epoch-40 checkpoint.

Validates, on the tiny EqM-S/2 synthetic model already used by
tests/test_forward_backwards_direct.py:

  1. forward_energy_with_cache_grad produces the SAME energy as
     forward_energy_with_cache / ebm='direct' energy_only=True (the
     gradient-carrying rewrite didn't change the forward numerics).
  2. The mandatory decomposition test itself: cosine(g_cache_vjp,
     g_cache_direct) > 0.999 and low relative norm error, i.e. a*=dL/dC is
     SUFFICIENT (given the true a*, not a learned one) to reconstruct the
     term g_semi omits, via an ordinary (non-second-order) VJP.

Run: python tests/test_fb_direct_cache_adjoint.py  (CPU, seconds)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fb_direct import ForwardBackwardsDirectTrainer
from fb_direct.forward_cache import forward_energy_with_cache
from fb_direct.forward_cache_grad import forward_energy_with_cache_grad
from fb_direct.cache_adjoint import decomposition_test
from transport import create_transport

from test_forward_backwards_direct import make_model, perturb, batch  # noqa: E402

torch.manual_seed(0)


def test_forward_cache_grad_matches_direct_energy():
    model = make_model(ebm='direct', dtype=torch.float64)
    perturb(model)
    model.eval()
    z, t, y = batch(n=2, dtype=torch.float64)
    with torch.no_grad():
        E_direct = model(z, t, y, energy_only=True)
    E_cache, _ = forward_energy_with_cache(model, z, t, y)
    z_req = z.clone().requires_grad_(True)
    E_grad, _ = forward_energy_with_cache_grad(model, z_req, t, y)
    torch.testing.assert_close(E_grad.detach(), E_direct, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(E_cache, E_direct, rtol=1e-10, atol=1e-10)
    print("PASS forward_energy_with_cache_grad energy matches ebm='direct' and forward_energy_with_cache")


def test_cache_adjoint_decomposition_fp64():
    model = make_model(ebm='forward-backwards-direct', dtype=torch.float64)
    perturb(model, seed=101)
    model.eval()
    trainer = ForwardBackwardsDirectTrainer(model, lr=1e-4, device=torch.device('cpu'))
    trainer.phi.to(torch.float64)
    trainer.registry.tie_from_forward_()
    active_pairs = [
        (e.forward_name, e.backward_name)
        for e in trainer.registry.entries
        if e.category in ("reverse_active", "recomputed_conditioning")
    ]

    transport = create_transport("Linear", "velocity", None, None, None)
    x1 = torch.randn(3, 4, 4, 4, dtype=torch.float64)
    y = torch.randint(0, 10, (3,))

    t, x0, x1 = transport.sample(x1)
    t = t.to(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    ut = ut * transport.get_ct(t)[:, None, None, None]

    result = decomposition_test(trainer, active_pairs, xt, t, y, ut)

    print(
        "[decomposition test FP64] "
        f"cosine(g_cache_vjp, g_cache_direct)={result['cosine_g_cache_vjp_vs_direct']:.6f} "
        f"rel_norm_error={result['rel_norm_error_g_cache']:.3e} "
        f"cosine(g_hat, g_exact)={result['cosine_g_hat_vs_exact']:.6f} "
        f"cosine(g_semi, g_exact)={result['cosine_g_semi_vs_exact']:.6f}"
    )
    assert result["cosine_g_cache_vjp_vs_direct"] > 0.999, result
    assert result["rel_norm_error_g_cache"] < 0.05, result
    # At FP64 with the true a*, g_hat = g_semi + g_cache_vjp should
    # reconstruct g_exact almost exactly (this is the ceiling any learned
    # corrector is trying to approach).
    assert result["cosine_g_hat_vs_exact"] > 0.999, result
    print("PASS cache-adjoint decomposition: a* is SUFFICIENT to reconstruct g_cache via ordinary VJP")


if __name__ == '__main__':
    test_forward_cache_grad_matches_direct_energy()
    test_cache_adjoint_decomposition_fp64()
    print("\nALL CACHE-ADJOINT DECOMPOSITION TESTS PASSED")
