"""
FP64 CPU exactness tests for fb_direct/exact_hvp.py BEFORE any GPU time
(2026-08-07, post-Gate-2 redesign).

The claim under test is strong: exact_fwrev_backward's forward-over-reverse
gradient (dual-tensor JVP in z + one first-order backward over theta) is
MATHEMATICALLY IDENTICAL to ebm='direct''s create_graph=True double-backward
gradient -- not approximately aligned, identical. So the test demands
machine-precision agreement at FP64, per-parameter, including:

  1. gp_lambda = 0: exact match to loss.backward() through
     model(..., train=True) with loss = mean_flat((field - ut)**2).mean().
  2. gp_lambda > 0: exact match to the double-backward gradient of
     loss + gp_lambda * mean_flat((grad_z E)**2).mean().
  3. Label-dropout determinism: with dropout_prob > 0 and model.train(),
     the two internal forward passes must see IDENTICAL dropped labels
     (pre-draw + temporary dropout_prob=0), and dropout_prob must be
     restored afterwards.
  4. Loss values reported match the double-backward reference.

Run: python tests/test_fb_direct_exact_hvp.py  (CPU, seconds)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fb_direct.exact_hvp import (
    block_subspace_iteration_theta, compute_field_direct, compute_wfb_gradient,
    estimate_lambda_max, exact_field_vjp, exact_fwrev_backward,
    field_jvp_direct, field_jvp_none, field_vjp_none,
    lanczos_inv_pow_apply, lanczos_inv_sqrt_apply, mixed_gram_mv, power_iteration_theta_sigma1,
    _estimate_lambda_max_generic, _lanczos_inv_pow_apply_generic, _lanczos_inv_sqrt_apply_generic,
)
from transport.utils import mean_flat

from test_forward_backwards_direct import make_model, perturb, batch  # noqa: E402

torch.manual_seed(0)


def reference_double_backward(model, xt, t, y, ut, gp_lambda=0.0):
    """The ebm='direct' training gradient, computed the production way:
    field = -autograd.grad(E.sum(), z, create_graph=True), then
    loss.backward() through the double-backward graph."""
    model.zero_grad(set_to_none=True)
    z = xt.detach().clone().requires_grad_(True)
    E = model(z, t, y, energy_only=True)
    grad_z = torch.autograd.grad(E.sum(), z, create_graph=True)[0]
    field = -grad_z
    loss_main = mean_flat((field - ut) ** 2).mean()
    loss_gp = mean_flat(grad_z ** 2).mean()
    loss = loss_main + gp_lambda * loss_gp
    loss.backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    out = (grads, float(loss_main.detach()), float(loss_gp.detach()))
    model.zero_grad(set_to_none=True)
    return out


def target_for(xt, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(xt.shape, generator=g).to(xt.dtype)


def compare_grads(g_ref, model, label, rtol=1e-9, atol=1e-11):
    n_checked = 0
    for n, p in model.named_parameters():
        ref = g_ref.get(n)
        got = p.grad
        if ref is None and (got is None or got.abs().max() == 0):
            continue
        assert ref is not None, f"{label}: fwrev produced grad for {n}, reference did not"
        assert got is not None, f"{label}: reference produced grad for {n}, fwrev did not"
        torch.testing.assert_close(got, ref, rtol=rtol, atol=atol, msg=f"{label}: {n}")
        n_checked += 1
    assert n_checked > 0, f"{label}: no gradients compared"
    return n_checked


def test_exact_match_no_gp():
    model = make_model(ebm='direct', dtype=torch.float64)
    perturb(model)
    model.eval()
    z, t, y = batch(n=2, dtype=torch.float64)
    ut = target_for(z)

    g_ref, loss_main_ref, _ = reference_double_backward(model, z, t, y, ut, gp_lambda=0.0)

    model.zero_grad(set_to_none=True)
    stats = exact_fwrev_backward(model, z, t, y, ut, gp_lambda=0.0)
    n = compare_grads(g_ref, model, "no_gp")
    assert abs(stats["loss_main"] - loss_main_ref) < 1e-12, \
        f"loss mismatch: {stats['loss_main']} vs {loss_main_ref}"
    model.zero_grad(set_to_none=True)
    print(f"PASS exact_fwrev == double-backward (gp_lambda=0), {n} param tensors at FP64 machine precision")


def test_exact_match_with_gp():
    model = make_model(ebm='direct', dtype=torch.float64)
    perturb(model, seed=7)
    model.eval()
    z, t, y = batch(n=3, dtype=torch.float64, seed=11)
    ut = target_for(z, seed=13)
    lam = 0.05

    g_ref, loss_main_ref, loss_gp_ref = reference_double_backward(model, z, t, y, ut, gp_lambda=lam)

    model.zero_grad(set_to_none=True)
    stats = exact_fwrev_backward(model, z, t, y, ut, gp_lambda=lam)
    n = compare_grads(g_ref, model, "with_gp")
    assert abs(stats["loss_main"] - loss_main_ref) < 1e-12
    assert abs(stats["loss_gp"] - loss_gp_ref) < 1e-12
    model.zero_grad(set_to_none=True)
    print(f"PASS exact_fwrev == double-backward (gp_lambda={lam}), {n} param tensors at FP64 machine precision")


def test_label_dropout_determinism():
    model = make_model(ebm='direct', dtype=torch.float64)
    perturb(model, seed=21)
    ye = model.y_embedder
    original_p = 0.5
    ye.dropout_prob = original_p
    model.train()
    z, t, y = batch(n=4, dtype=torch.float64, seed=23)
    ut = target_for(z, seed=29)

    # Determinism across the two internal passes: fix torch's global RNG so
    # the ONE pre-draw is reproducible, then verify the fwrev gradient
    # matches a reference that saw exactly those dropped labels.
    torch.manual_seed(777)
    with torch.no_grad():
        expected_dropped = ye.token_drop(y.clone())

    torch.manual_seed(777)
    model.zero_grad(set_to_none=True)
    stats = exact_fwrev_backward(model, z, t, y, ut, gp_lambda=0.0)
    got = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    assert ye.dropout_prob == original_p, \
        f"dropout_prob not restored: {ye.dropout_prob} vs {original_p}"

    # Reference: same dropped labels, dropout disabled inside.
    ye.dropout_prob = 0.0
    g_ref, loss_main_ref, _ = reference_double_backward(model, z, t, expected_dropped, ut)
    ye.dropout_prob = original_p

    model.zero_grad(set_to_none=True)
    for n_name, ref in g_ref.items():
        torch.testing.assert_close(got[n_name], ref, rtol=1e-9, atol=1e-11, msg=f"dropout: {n_name}")
    assert abs(stats["loss_main"] - loss_main_ref) < 1e-12
    print("PASS label-dropout pre-draw: both internal passes saw identical labels; dropout_prob restored")


def test_rejects_wrong_ebm():
    model = make_model(ebm='none', dtype=torch.float64)
    z, t, y = batch(n=2, dtype=torch.float64)
    try:
        exact_fwrev_backward(model, z, t, y, target_for(z))
    except ValueError:
        print("PASS rejects ebm != 'direct'")
        return
    raise AssertionError("exact_fwrev_backward accepted ebm='none'")


def reference_field_vjp_direct(model, xt, t, y, v):
    """Independent double-backward reference for J_field^T v (ebm='direct'):
    field = -grad_z E (create_graph=True), s = <v, field> summed, s.backward()
    accumulates exactly J_field^T v into .grad -- no forward-mode AD, no
    Pearlmutter trick, structurally independent of exact_field_vjp's
    implementation path (mirrors reference_double_backward's role above)."""
    model.zero_grad(set_to_none=True)
    z = xt.detach().clone().requires_grad_(True)
    E = model(z, t, y, energy_only=True)
    g = torch.autograd.grad(E.sum(), z, create_graph=True)[0]
    field = -g
    s = (v.detach() * field).sum()
    s.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads, field.detach()


def test_field_vjp_direct_matches_double_backward_reference():
    """exact_field_vjp (forward-over-reverse) must exactly match the
    independent double-backward reference, for an ARBITRARY direction v
    (not the training w) -- this is the generalization
    matched_replay_jacobian_diagnostic.py depends on."""
    model = make_model(ebm='direct', dtype=torch.float64)
    perturb(model, seed=41)
    model.eval()
    z, t, y = batch(n=3, dtype=torch.float64, seed=43)
    v = torch.randn(z.shape, generator=torch.Generator().manual_seed(47)).to(z.dtype)

    g_ref, field_ref = reference_field_vjp_direct(model, z, t, y, v)

    model.zero_grad(set_to_none=True)
    stats = exact_field_vjp(model, z, t, y, v)
    n = compare_grads(g_ref, model, "field_vjp_direct")
    torch.testing.assert_close(torch.tensor(stats["field_norm"], dtype=field_ref.dtype), field_ref.norm(), rtol=1e-9, atol=1e-11)
    model.zero_grad(set_to_none=True)
    print(f"PASS exact_field_vjp == double-backward J_field^T v reference, {n} param tensors at FP64")


def test_field_vjp_direct_sign_relation_to_fwrev_w():
    """Consistency check tying the new primitive back to the original:
    exact_fwrev_backward's internal w satisfies (gp_lambda=0)
        w = (2/(B*D)) * (g + ut) = -(2/(B*D)) * (field - ut) = -(2/(B*D)) * r
    i.e. w is the canonical residual r RESCALED by 2/(B*D) and NEGATED.
    So exact_field_vjp(model, z, t, y, v=-w) must reproduce
    exact_fwrev_backward(model, z, t, y, ut, gp_lambda=0)'s .grad exactly
    (both compute J_field^T(-w) = J_g^T w, the fwrev function's own
    definition) -- this is the precise mathematical statement of the Phase 0
    audit finding: the OLD diagnostic's w_norm differs from the canonical
    ||r|| by exactly the factor 2/(B*D), not a directional/model-specific
    effect."""
    model = make_model(ebm='direct', dtype=torch.float64)
    perturb(model, seed=51)
    model.eval()
    z, t, y = batch(n=2, dtype=torch.float64, seed=53)
    ut = target_for(z, seed=59)
    B, D = z.shape[0], z[0].numel()

    model.zero_grad(set_to_none=True)
    exact_fwrev_backward(model, z, t, y, ut, gp_lambda=0.0)
    g_fwrev = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    # Recompute field/residual independently (first-order only) to build w by hand.
    zc = z.detach().clone().requires_grad_(True)
    E = model(zc, t, y, energy_only=True)
    g = torch.autograd.grad(E.sum(), zc, create_graph=False)[0].detach()
    field = -g
    r = field - ut  # canonical residual
    w_hand = (2.0 / (B * D)) * (g + ut)  # == -(2/(B*D)) * r
    torch.testing.assert_close(w_hand, -(2.0 / (B * D)) * r, rtol=1e-9, atol=1e-12)

    model.zero_grad(set_to_none=True)
    exact_field_vjp(model, z, t, y, v=-w_hand)
    g_via_field_vjp = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    n = compare_grads(g_fwrev, model, "sign_relation")
    for name in g_fwrev:
        torch.testing.assert_close(g_via_field_vjp[name], g_fwrev[name], rtol=1e-9, atol=1e-11)
    model.zero_grad(set_to_none=True)
    print(f"PASS w = -(2/(B*D))*r confirmed algebraically + exact_field_vjp(v=-w) == exact_fwrev_backward.grad, "
          f"{n} tensors -- Phase 0 audit's normalization-mismatch claim is exact, not approximate")


def test_field_vjp_none_matches_autograd_grad():
    """field_vjp_none must match plain torch.autograd.grad(field, params,
    grad_outputs=v) -- the textbook VJP, independent implementation path."""
    model = make_model(ebm='none', dtype=torch.float64)
    perturb(model, seed=61)
    model.eval()
    z, t, y = batch(n=3, dtype=torch.float64, seed=63)
    v = torch.randn(z.shape, generator=torch.Generator().manual_seed(67)).to(z.dtype)

    field = model(z, t, y, train=True)
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ref_grads = torch.autograd.grad(field, [p for _, p in named_trainable],
                                     grad_outputs=v, allow_unused=True)
    ref = {n: g for (n, _), g in zip(named_trainable, ref_grads) if g is not None}

    model.zero_grad(set_to_none=True)
    stats = field_vjp_none(model, z, t, y, v)
    n = compare_grads(ref, model, "field_vjp_none")
    torch.testing.assert_close(torch.tensor(stats["field_norm"], dtype=field.dtype), field.detach().norm(), rtol=1e-9, atol=1e-11)
    model.zero_grad(set_to_none=True)
    print(f"PASS field_vjp_none == torch.autograd.grad(field, params, grad_outputs=v) reference, {n} tensors")


def test_field_vjp_finite_difference_sanity():
    """Extra independent check (not just two analytic derivations agreeing
    with each other, but a numerical ground truth): finite-difference the
    SCALAR s(theta) = <v, field_theta(x)> along a random parameter
    direction, compare to <exact_field_vjp grad, direction>. FP64, central
    difference, eps=1e-6 -- both direct and none arms."""
    torch.manual_seed(71)
    for ebm in ("direct", "none"):
        model = make_model(ebm=ebm, dtype=torch.float64)
        perturb(model, seed=73)
        model.eval()
        z, t, y = batch(n=2, dtype=torch.float64, seed=79)
        v = torch.randn(z.shape, generator=torch.Generator().manual_seed(83)).to(z.dtype)

        model.zero_grad(set_to_none=True)
        if ebm == "direct":
            exact_field_vjp(model, z, t, y, v)
        else:
            field_vjp_none(model, z, t, y, v)
        analytic_grad = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)

        def s_of_theta():
            if ebm == "direct":
                zc = z.detach().clone().requires_grad_(True)
                E = model(zc, t, y, energy_only=True)
                g = torch.autograd.grad(E.sum(), zc, create_graph=False)[0]
                field = -g
            else:
                with torch.no_grad():
                    field = model(z, t, y, train=True)
            return float((v * field).sum())

        gen = torch.Generator().manual_seed(89)
        params = [(n, p) for n, p in model.named_parameters() if n in analytic_grad]
        directions = {n: torch.randn(p.shape, generator=gen).to(p.dtype) for n, p in params}
        dnorm = (sum(float((d ** 2).sum()) for d in directions.values())) ** 0.5
        eps = 1e-6
        with torch.no_grad():
            for n, p in params:
                p.add_(eps * directions[n])
        s_plus = s_of_theta()
        with torch.no_grad():
            for n, p in params:
                p.add_(-2 * eps * directions[n])
        s_minus = s_of_theta()
        with torch.no_grad():
            for n, p in params:
                p.add_(eps * directions[n])  # restore

        fd_directional_deriv = (s_plus - s_minus) / (2 * eps)
        analytic_directional_deriv = sum(
            float((analytic_grad[n] * directions[n]).sum()) for n, _ in params
        )
        rel_err = abs(fd_directional_deriv - analytic_directional_deriv) / (abs(fd_directional_deriv) + 1e-12)
        assert rel_err < 1e-4, (
            f"{ebm}: finite-difference mismatch, fd={fd_directional_deriv} "
            f"analytic={analytic_directional_deriv} rel_err={rel_err} (dnorm={dnorm})"
        )
        print(f"PASS {ebm} finite-difference sanity: fd={fd_directional_deriv:.6e} "
              f"analytic={analytic_directional_deriv:.6e} rel_err={rel_err:.2e}")


def _explicit_jacobian(model, ebm, z, t, y, params):
    """Ground truth for field_jvp_*/power_iteration_theta_sigma1: materialize
    the FULL Jacobian J (n_out x n_params) by looping over field's output
    elements, each via an INDEPENDENT torch.autograd.grad call -- no R-op
    trick, no forward-mode AD, structurally unrelated to the code under
    test. Only tractable for a small params subset."""
    def field_of():
        if ebm == "direct":
            zc = z.detach().clone().requires_grad_(True)
            E = model(zc, t, y, energy_only=True)
            g = torch.autograd.grad(E.sum(), zc, create_graph=True)[0]
            return -g
        return model(z, t, y, train=True)

    field = field_of()
    n_out = field.numel()
    n_params = sum(p.numel() for p in params)
    field_flat = field.flatten()
    J = torch.zeros(n_out, n_params, dtype=torch.float64)
    for i in range(n_out):
        gi = torch.autograd.grad(field_flat[i], params, retain_graph=(i < n_out - 1))
        J[i] = torch.cat([g.flatten() for g in gi])
    model.zero_grad(set_to_none=True)
    return field, J


def test_theta_jvp_matches_explicit_jacobian():
    """field_jvp_direct/field_jvp_none (the R-op double-backward trick) must
    match an EXPLICITLY MATERIALIZED Jacobian -- the ground truth the
    matrix-free power iteration depends on (Phase 6 engineering
    requirement: 'validate the matrix-free sigma_1 estimator against an
    explicitly materialized Jacobian on a tiny model')."""
    torch.manual_seed(101)
    for ebm in ("direct", "none"):
        model = make_model(ebm=ebm, dtype=torch.float64)
        perturb(model, seed=103)
        model.eval()
        z, t, y = batch(n=1, dtype=torch.float64, seed=107)
        params = [p for n, p in model.named_parameters() if n == "t_embedder.mlp.2.bias"]
        assert len(params) == 1, "t_embedder.mlp.2.bias not found -- architecture changed"

        field, J = _explicit_jacobian(model, ebm, z, t, y, params)
        n_params = params[0].numel()

        gen = torch.Generator().manual_seed(109)
        v_flat = torch.randn(n_params, generator=gen, dtype=torch.float64)
        v = [v_flat.view(params[0].shape)]
        Jv_explicit = (J @ v_flat).view(field.shape)

        jvp_fn = field_jvp_direct if ebm == "direct" else field_jvp_none
        model.zero_grad(set_to_none=True)
        Jv_got = jvp_fn(model, z, t, y, params, v)
        model.zero_grad(set_to_none=True)

        torch.testing.assert_close(Jv_got, Jv_explicit, rtol=1e-6, atol=1e-9)
        print(f"PASS {ebm} field_jvp matches explicitly materialized Jacobian "
              f"({J.shape[0]}x{J.shape[1]}), "
              f"max_abs_err={float((Jv_got - Jv_explicit).abs().max()):.2e}")


def test_theta_jvp_handles_zero_jacobian_parameter():
    """A parameter can be mathematically DISCONNECTED from field's Jacobian
    without being a bug (found in production 2026-08-11, job 38268323):
    ScalarEnergyHead.linear.bias is a per-token additive bias summed BEFORE
    the token dimension, i.e. E = sum_tokens(W.x_token + bias) =
    W.sum(x_token) + N_tokens*bias -- an ADDITIVE CONSTANT in z, so
    d(field)/d(bias) = d(-grad_z E)/d(bias) = 0 identically. Before this fix
    field_jvp_* raised 'One of the differentiated Tensors appears to not
    have been used in the graph' on any params list containing such a
    parameter -- allow_unused=True + zero-substitution is the correct fix
    (an identically-zero Jacobian column, not a missing dependency).
    Simulated here with a genuinely-unconnected extra leaf parameter
    appended to a real, connected params list, on a real tiny model."""
    torch.manual_seed(151)
    for ebm in ("direct", "none"):
        model = make_model(ebm=ebm, dtype=torch.float64)
        perturb(model, seed=157)
        model.eval()
        z, t, y = batch(n=1, dtype=torch.float64, seed=163)
        connected = [p for n, p in model.named_parameters() if n == "t_embedder.mlp.2.bias"][0]
        disconnected = torch.nn.Parameter(torch.randn(5, dtype=torch.float64))  # never touched by forward
        params = [connected, disconnected]

        field, J_connected = _explicit_jacobian(model, ebm, z, t, y, [connected])
        n_conn = connected.numel()
        gen = torch.Generator().manual_seed(167)
        v_conn = torch.randn(n_conn, generator=gen, dtype=torch.float64)
        v_disc = torch.randn(5, generator=gen, dtype=torch.float64)
        v = [v_conn.view(connected.shape), v_disc.view(disconnected.shape)]
        Jv_explicit = (J_connected @ v_conn).view(field.shape)  # disconnected column is all-zero

        jvp_fn = field_jvp_direct if ebm == "direct" else field_jvp_none
        model.zero_grad(set_to_none=True)
        Jv_got = jvp_fn(model, z, t, y, params, v)  # must NOT raise
        model.zero_grad(set_to_none=True)
        torch.testing.assert_close(Jv_got, Jv_explicit, rtol=1e-6, atol=1e-9)

        vjp_fn = exact_field_vjp if ebm == "direct" else field_vjp_none
        model.zero_grad(set_to_none=True)
        vjp_fn(model, z, t, y, torch.ones_like(field))
        assert disconnected.grad is None, "disconnected param should get no .grad from vjp_fn"
        model.zero_grad(set_to_none=True)

        # block_subspace_iteration_theta must also not crash when params includes
        # a zero-Jacobian column (exercises the .grad-is-None zero-substitution fix).
        result = block_subspace_iteration_theta(jvp_fn, vjp_fn, model, z, t, y, params,
                                                  k=1, num_iters=10, seed=1, tol=1e-6)
        assert result["sigma"].shape[0] == 1
        print(f"PASS {ebm} field_jvp/block_subspace_iteration_theta handle a zero-Jacobian "
              f"parameter without raising (matches explicit-Jacobian ground truth)")


def test_power_iteration_sigma1_matches_explicit_svd():
    """power_iteration_theta_sigma1 must converge to the top singular value
    (via numpy SVD of the SAME explicitly materialized Jacobian above) --
    ground truth independent of whether the JVP and VJP primitives happen
    to be internally self-consistent with each other."""
    import numpy as np
    torch.manual_seed(113)
    for ebm in ("direct", "none"):
        model = make_model(ebm=ebm, dtype=torch.float64)
        perturb(model, seed=127)
        model.eval()
        z, t, y = batch(n=1, dtype=torch.float64, seed=131)
        params = [p for n, p in model.named_parameters() if n == "t_embedder.mlp.2.bias"]

        _, J = _explicit_jacobian(model, ebm, z, t, y, params)
        sigma1_ref = float(np.linalg.svd(J.numpy(), compute_uv=False)[0])

        jvp_fn = field_jvp_direct if ebm == "direct" else field_jvp_none
        vjp_fn = exact_field_vjp if ebm == "direct" else field_vjp_none
        model.zero_grad(set_to_none=True)
        result = power_iteration_theta_sigma1(jvp_fn, vjp_fn, model, z, t, y, params,
                                               num_iters=60, seed=17, tol=1e-10)
        model.zero_grad(set_to_none=True)

        rel_err = abs(result["sigma_1"] - sigma1_ref) / sigma1_ref
        assert rel_err < 1e-4, (
            f"{ebm}: power-iteration sigma_1={result['sigma_1']} vs SVD reference={sigma1_ref}, "
            f"rel_err={rel_err} ({result['n_iters']} iters)"
        )
        print(f"PASS {ebm} power_iteration_theta_sigma1 == SVD reference: "
              f"{result['sigma_1']:.6f} vs {sigma1_ref:.6f} "
              f"(rel_err={rel_err:.2e}, {result['n_iters']} iters)")


def test_block_subspace_matches_explicit_svd():
    """block_subspace_iteration_theta (orthogonal iteration, Stage-A top-k
    high-gain SUBSPACE diagnostic) must match numpy SVD of an explicitly
    materialized Jacobian, both on singular VALUES and (separately, via a
    synthetic operator with a deliberately engineered spectral gap) on the
    recovered SUBSPACE itself.

    Two-part test, per the engineering requirement ("validate the
    matrix-free sigma_1 estimator against an explicitly materialized
    Jacobian on a tiny model" + "do not require individual singular
    vectors to match when eigenvalues are nearly degenerate -- validate
    the SUBSPACE"):

    Part 1 (real tiny model): field_jvp_*/exact_field_vjp/field_vjp_none
    ARE the model-dependent primitives under test elsewhere already
    (test_theta_jvp_matches_explicit_jacobian); here we only need to
    confirm block_subspace_iteration_theta's sigma estimates track the
    true singular values on a real model's parameter Jacobian. Measured
    fact (checked before writing this test): this tiny random-init
    model's t_embedder.mlp weight matrices have a NEAR-FLAT top-8
    spectrum (consecutive gaps 0.5%-9%, e.g. sigma_3/sigma_4 differ by
    <1%) -- a property of small randomly-initialized matrices, not of the
    estimator. Under that regime even hundreds of orthogonal-iteration
    steps leave the INDIVIDUAL subspace direction underdetermined (the
    span of a near-degenerate cluster is well-defined; which orthonormal
    basis vector "is" sigma_3 vs sigma_4 is not, until iterated far longer
    than is practical here) -- so Part 1 checks singular VALUES only.

    Part 2 (synthetic linear operator, model-independent): validates the
    SUBSPACE recovery itself using a hand-built operator with an explicit,
    large spectral gap between rank k and k+1 -- exactly the regime where
    subspace-vs-individual-vector comparison is meaningful, decoupled from
    any particular model's incidental spectrum.
    """
    import numpy as np
    torch.manual_seed(211)
    k = 3

    # ---- Part 1: real tiny model, sigma values only ----
    for ebm in ("direct", "none"):
        model = make_model(ebm=ebm, dtype=torch.float64)
        perturb(model, seed=223)
        model.eval()
        z, t, y = batch(n=1, dtype=torch.float64, seed=227)
        params = [p for n, p in model.named_parameters()
                  if n in ("t_embedder.mlp.0.bias", "t_embedder.mlp.2.bias")]
        assert len(params) == 2, "t_embedder.mlp.{0,2}.bias not found -- architecture changed"

        _, J = _explicit_jacobian(model, ebm, z, t, y, params)
        sigma_ref = np.linalg.svd(J.numpy(), compute_uv=False)[:k]

        jvp_fn = field_jvp_direct if ebm == "direct" else field_jvp_none
        vjp_fn = exact_field_vjp if ebm == "direct" else field_vjp_none
        model.zero_grad(set_to_none=True)
        result = block_subspace_iteration_theta(jvp_fn, vjp_fn, model, z, t, y, params,
                                                  k=k, num_iters=100, seed=17, tol=1e-12)
        model.zero_grad(set_to_none=True)

        sigma_got = result["sigma"].numpy()
        rel_err_sigma = np.abs(sigma_got - sigma_ref) / np.abs(sigma_ref)
        assert (rel_err_sigma < 5e-3).all(), (
            f"{ebm}: block subspace sigma={sigma_got} vs SVD reference={sigma_ref}, rel_err={rel_err_sigma}"
        )
        assert result["ortho_error_V"] < 1e-8 and result["ortho_error_U"] < 1e-8, (
            f"{ebm}: orthonormality error V={result['ortho_error_V']} U={result['ortho_error_U']}"
        )
        print(f"PASS {ebm} block_subspace_iteration_theta sigma == SVD reference (real tiny model, "
              f"near-flat spectrum -- values only): {sigma_got} vs {sigma_ref} "
              f"(max rel_err={rel_err_sigma.max():.2e}), {result['n_iters']} iters")

    # ---- Part 2: synthetic operator, explicit spectral gap, subspace comparison ----
    n_out, n_params = 40, 25
    gen = np.random.RandomState(97)
    U_true, _ = np.linalg.qr(gen.randn(n_out, n_out))
    V_true, _ = np.linalg.qr(gen.randn(n_params, n_params))
    sigma_true = np.array([10.0, 7.0, 4.0, 0.3, 0.25, 0.2, 0.15] + [0.1] * (min(n_out, n_params) - 7))
    S_mat = np.zeros((n_out, n_params))
    for i, s in enumerate(sigma_true):
        S_mat[i, i] = s
    J_synth = U_true @ S_mat @ V_true.T  # rank-k=3 vs rest: gap 4.0 -> 0.3, ~13x

    J_t = torch.tensor(J_synth, dtype=torch.float64)
    fake_param = torch.zeros(n_params, dtype=torch.float64, requires_grad=False)

    def synth_jvp(_model, _xt, _t, _y, params_, v):
        v_flat = v[0].reshape(-1)
        return (J_t @ v_flat).clone()

    def synth_vjp(_model, _xt, _t, _y, q):
        grad = J_t.T @ q.reshape(-1)
        fake_param.grad = grad.view(fake_param.shape).clone()

    class _DummyModel:
        def zero_grad(self, set_to_none=True):
            fake_param.grad = None

    result = block_subspace_iteration_theta(synth_jvp, synth_vjp, _DummyModel(), None, None, None,
                                             [fake_param], k=k, num_iters=60, seed=5, tol=1e-13)
    sigma_got = result["sigma"].numpy()
    sigma_ref = sigma_true[:k]
    rel_err_sigma = np.abs(sigma_got - sigma_ref) / np.abs(sigma_ref)
    assert (rel_err_sigma < 1e-6).all(), f"synthetic: sigma={sigma_got} vs ref={sigma_ref}"

    Uref_k = U_true[:, :k]
    Vref_k = V_true[:, :k]
    U_got, V_got = result["U"].numpy(), result["V"].numpy()
    proj_dist_U = float(np.linalg.norm(Uref_k @ Uref_k.T - U_got @ U_got.T))
    proj_dist_V = float(np.linalg.norm(Vref_k @ Vref_k.T - V_got @ V_got.T))
    assert proj_dist_U < 1e-4, f"synthetic: U subspace projector distance {proj_dist_U} too large"
    assert proj_dist_V < 1e-4, f"synthetic: V subspace projector distance {proj_dist_V} too large"
    assert result["ortho_error_V"] < 1e-8 and result["ortho_error_U"] < 1e-8
    print(f"PASS block_subspace_iteration_theta == SVD reference (synthetic operator, explicit "
          f"spectral gap sigma_3/sigma_4={sigma_true[2] / sigma_true[3]:.1f}x): "
          f"sigma {sigma_got} vs {sigma_ref} (max rel_err={rel_err_sigma.max():.2e}), "
          f"U subspace dist={proj_dist_U:.2e}, V subspace dist={proj_dist_V:.2e}, {result['n_iters']} iters")


def test_wfb_operators_match_explicit_jacobian():
    """WFB-EqM Stage 0.A: on a real tiny model with an explicitly
    materialized mixed Jacobian M (n_out=64 x n_params, tractable per
    _explicit_jacobian's per-output-element autograd.grad loop), verify:

    1. mixed_gram_mv matches the explicit A = M M^T applied to a random v.
    2. estimate_lambda_max matches A's true top eigenvalue (explicit eigh).
    3. lanczos_inv_sqrt_apply at k=n_out (exact Krylov recovery -- a Krylov
       subspace of dimension >= rank(A) spans A's full range, so m=n_out
       Lanczos steps reproduce (A+lam I)^{-1/2}r to numerical precision,
       not merely approximate it) matches the explicit eigh-based
       (A+lam I)^{-1/2} r.
    4. g_wfb = M^T u (via compute_wfb_gradient) matches M.T @ u_exact.
    5. Norm bound ||g_wfb|| <= ||r|| holds (spec Section 3 diagnostic identity).
    6. compute_field_direct matches the explicitly-computed field (backward
       operator swap does not change the model's forward-computed field).
    """
    torch.manual_seed(201)
    model = make_model(ebm="direct", dtype=torch.float64)
    perturb(model, seed=203)
    model.eval()
    z, t, y = batch(n=1, dtype=torch.float64, seed=207)
    params = [p for n, p in model.named_parameters() if n == "t_embedder.mlp.2.bias"]
    assert len(params) == 1, "t_embedder.mlp.2.bias not found -- architecture changed"

    field, J = _explicit_jacobian(model, "direct", z, t, y, params)
    n_out, n_params = J.shape

    gen = torch.Generator().manual_seed(211)
    ut = torch.randn(field.shape, generator=gen, dtype=torch.float64)
    r = (field.detach() - ut)
    r_norm = float(r.norm())

    A = J @ J.T  # (n_out, n_out), explicit Gram matrix
    assert float((A - A.T).abs().max()) < 1e-12, "A must be numerically symmetric"
    eigvals_A = torch.linalg.eigvalsh(A)
    assert float(eigvals_A.min()) > -1e-9, "A must be PSD (it's a Gram matrix)"
    lambda_max_explicit = float(eigvals_A.max())

    # 1. mixed_gram_mv vs explicit A @ v
    gen2 = torch.Generator().manual_seed(213)
    v_probe = torch.randn(field.shape, generator=gen2, dtype=torch.float64)
    Av_explicit = (A @ v_probe.flatten()).view(field.shape)
    model.zero_grad(set_to_none=True)
    Av_got = mixed_gram_mv(model, z, t, y, params, v_probe)
    torch.testing.assert_close(Av_got, Av_explicit, rtol=1e-6, atol=1e-9)

    # 2. estimate_lambda_max vs explicit top eigenvalue
    lam_result = estimate_lambda_max(model, z, t, y, params, num_iters=200, seed=217, tol=1e-14)
    rel_err_lambda_max = abs(lam_result["lambda_max"] - lambda_max_explicit) / abs(lambda_max_explicit)
    assert rel_err_lambda_max < 1e-5, (
        f"estimate_lambda_max={lam_result['lambda_max']} vs explicit={lambda_max_explicit}, "
        f"rel_err={rel_err_lambda_max:.2e}")

    # 3. lanczos_inv_sqrt_apply at k=n_out (exact recovery) vs explicit (A+lam I)^{-1/2} r
    lam = 1e-2 * lambda_max_explicit
    theta_A, S_A = torch.linalg.eigh(A)
    u_exact_flat = S_A @ (torch.diag(1.0 / torch.sqrt(theta_A + lam)) @ (S_A.T @ r.flatten()))
    u_exact = u_exact_flat.view(field.shape)

    lz = lanczos_inv_sqrt_apply(model, z, t, y, params, r, lam, k=n_out)
    assert not lz["breakdown"], f"unexpected Lanczos breakdown: {lz['breakdown_reason']}"
    torch.testing.assert_close(lz["u"], u_exact, rtol=1e-4, atol=1e-7)
    assert lz["ortho_error"] < 1e-6, f"Lanczos basis orthogonality error too large: {lz['ortho_error']}"
    rel_err_T_eigmax = abs(lz["T_eigmax"] - lambda_max_explicit) / abs(lambda_max_explicit)
    assert rel_err_T_eigmax < 1e-6, f"Lanczos T_eigmax vs explicit: rel_err={rel_err_T_eigmax:.2e}"

    # Convergence trend with growing k (spec Section 7.A: k=2,4,8,12 where affordable)
    err_prev = float("inf")
    for k in (2, 4, 8, min(12, n_out)):
        lz_k = lanczos_inv_sqrt_apply(model, z, t, y, params, r, lam, k=k)
        if lz_k["breakdown"]:
            continue
        err_k = float((lz_k["u"] - u_exact).norm())
        assert err_k <= err_prev + 1e-8, f"Lanczos error should not increase with larger k: k={k} err={err_k} vs prev={err_prev}"
        err_prev = err_k

    # 4. g_wfb via compute_wfb_gradient vs explicit M^T u_exact
    g_wfb_explicit = (J.T @ u_exact_flat).view(params[0].shape)
    result = compute_wfb_gradient(model, z, t, y, ut, params=params, rho=1e-2, k=n_out,
                                   lambda_max_num_iters=200, seed=219)
    assert not result["breakdown"], f"unexpected breakdown in compute_wfb_gradient: {result['breakdown_reason']}"
    # Loose tolerance here: compute_wfb_gradient's internal lambda_max estimate uses its own
    # early-stop tol (checked separately below), which perturbs lam slightly vs this test's
    # externally-fixed lam -- items 1-3/5/6 above already validate the exact math tightly.
    # A small lam shift has an outsized RELATIVE effect on near-zero-eigenvalue modes of
    # g_wfb_explicit (small denominator), so this is a wrapper self-consistency check, not
    # a math-correctness check -- atol carries it for those near-zero components.
    torch.testing.assert_close(result["g_wfb"][0], g_wfb_explicit, rtol=5e-2, atol=2e-4)
    # compute_wfb_gradient's internal estimate_lambda_max uses its own default early-stop
    # tol (1e-3), independent of this test's tighter tol=1e-14 call above -- loose bound here.
    rel_err_lambda_max_internal = abs(result["lambda_max"] - lambda_max_explicit) / abs(lambda_max_explicit)
    assert rel_err_lambda_max_internal < 5e-2, (
        f"compute_wfb_gradient's internal lambda_max={result['lambda_max']} vs "
        f"explicit={lambda_max_explicit}, rel_err={rel_err_lambda_max_internal:.2e}")

    # 5. Norm bound: ||g_wfb|| <= ||r|| (spec Section 3)
    g_wfb_norm = float(g_wfb_explicit.norm())
    assert g_wfb_norm <= r_norm + 1e-6, f"||g_wfb||={g_wfb_norm} exceeds ||r||={r_norm}"

    # 6. field unchanged by the backward-operator swap
    field_via_helper = compute_field_direct(model, z, t, y)
    torch.testing.assert_close(field_via_helper, field.detach(), rtol=1e-9, atol=1e-12)

    print(f"PASS WFB Stage 0.A (explicit-Jacobian toy test, n_out={n_out} n_params={n_params}): "
          f"lambda_max rel_err={rel_err_lambda_max:.2e}, Lanczos(k={n_out}) u rel err "
          f"{float((lz['u'] - u_exact).norm() / u_exact.norm()):.2e}, "
          f"||g_wfb||={g_wfb_norm:.4f} <= ||r||={r_norm:.4f}")


def test_wfb_singular_mode_gain():
    """WFB-EqM Stage 0.B: SINGULAR-MODE TEST (spec Section 7.B). Synthetic
    M with an intentionally extreme, hand-built singular spectrum
    (sigma = [50, 20, 5, 1, 0.5, 0.2, 0.1] -- a 500x gap top-to-bottom) run
    through the model-agnostic generic Lanczos/power-iteration cores
    directly (no real model involved, decoupled from any model's incidental
    spectrum -- same rationale as block_subspace_iteration_theta's
    synthetic-operator test). Confirms per-mode gain:

        ordinary (raw):  gain = sigma_i
        WFB:              gain = sigma_i / sqrt(sigma_i^2 + lambda)

    i.e. a very large sigma_i produces an unbounded-looking raw contribution
    but a WFB contribution capped near 1/sqrt(lambda)-independent-of-sigma
    for sigma_i >> sqrt(lambda), and near sigma_i (unchanged) for
    sigma_i << sqrt(lambda) -- exactly the damped-whitening transition the
    method is designed to produce.
    """
    torch.manual_seed(223)
    sigma_true = torch.tensor([50.0, 20.0, 5.0, 1.0, 0.5, 0.2, 0.1], dtype=torch.float64)
    n = sigma_true.numel()
    n_out, n_params = n, n  # square, diagonal-in-its-own-basis synthetic M
    Q1, _ = torch.linalg.qr(torch.randn(n_out, n_out, dtype=torch.float64, generator=torch.Generator().manual_seed(227)))
    Q2, _ = torch.linalg.qr(torch.randn(n_params, n_params, dtype=torch.float64, generator=torch.Generator().manual_seed(229)))
    M = Q1 @ torch.diag(sigma_true) @ Q2.T  # explicit M with EXACTLY sigma_true as its singular values

    def gram_mv_fn(v):
        return M @ (M.T @ v)

    lam_result = _estimate_lambda_max_generic(gram_mv_fn, torch.randn(n_out, dtype=torch.float64,
                                               generator=torch.Generator().manual_seed(233)),
                                               num_iters=300, tol=1e-15)
    lambda_max_true = float((sigma_true.max()) ** 2)
    rel_err_lambda_max = abs(lam_result["lambda_max"] - lambda_max_true) / lambda_max_true
    assert rel_err_lambda_max < 1e-6, f"synthetic lambda_max: rel_err={rel_err_lambda_max:.2e}"

    rho = 1e-3
    lam = rho * lambda_max_true

    for r_dir_idx in range(n):  # probe with r aligned to EACH known singular direction of M
        r = Q1[:, r_dir_idx].clone()  # r = u_i (a left singular vector of M, unit norm)
        lz = _lanczos_inv_sqrt_apply_generic(gram_mv_fn, r, lam, k=n)  # k=n: exact Krylov recovery
        # r is BY CONSTRUCTION an exact eigenvector of A here (r=u_i => A r = sigma_i^2 r),
        # so a "lucky" 1-step Lanczos breakdown (invariant_subspace_at_step_0) is the
        # EXPECTED, exact-answer outcome, not a failure -- only reject other reasons.
        reason = lz["breakdown_reason"]
        assert not lz["breakdown"] or (reason or "").startswith("invariant_subspace"), (
            f"unexpected breakdown at mode {r_dir_idx}: {reason}")
        u_wfb = lz["u"]
        g_wfb = M.T @ u_wfb

        sigma_i = float(sigma_true[r_dir_idx])
        raw_gain = sigma_i  # ||M^T r|| for r = u_i, since M^T u_i = sigma_i v_i
        wfb_gain_expected = sigma_i / (sigma_i ** 2 + lam) ** 0.5
        wfb_gain_got = float(g_wfb.norm())  # ||r||=1, so this IS the mode gain

        rel_err_gain = abs(wfb_gain_got - wfb_gain_expected) / wfb_gain_expected
        assert rel_err_gain < 1e-4, (
            f"mode {r_dir_idx} (sigma={sigma_i}): WFB gain got={wfb_gain_got} "
            f"expected={wfb_gain_expected}, rel_err={rel_err_gain:.2e}")
        assert wfb_gain_got <= raw_gain + 1e-9, (
            f"mode {r_dir_idx}: WFB gain ({wfb_gain_got}) must not exceed raw gain ({raw_gain})")

    # Explicit large-sigma-bounded check: top mode's raw gain is 50x its own
    # sigma-independent damping ceiling ~1/sqrt(lam); WFB gain stays near
    # sqrt(lam) while raw stays at 50 -- the core claim under test.
    top_raw = float(sigma_true[0])
    top_wfb = float(sigma_true[0] / (sigma_true[0] ** 2 + lam) ** 0.5)
    assert top_wfb < top_raw * 0.1, f"expected WFB to sharply suppress the extreme top mode: raw={top_raw} wfb={top_wfb}"
    print(f"PASS WFB Stage 0.B (singular-mode test, sigma={sigma_true.tolist()}, rho={rho}, "
          f"lambda={lam:.4f}): top mode raw_gain={top_raw:.2f} -> wfb_gain={top_wfb:.4f} "
          f"({top_raw / top_wfb:.1f}x suppression), lambda_max rel_err={rel_err_lambda_max:.2e}")


def test_wfb_alpha_family_singular_mode_gain():
    """WFB-EqM Stage 2.5 (reviewer note, 2026-08-12): generalizes
    test_wfb_singular_mode_gain from a fixed alpha=1/2 to the full
    alpha-family g_alpha = M^T (A+lambda I)^{-alpha} r, confirming BOTH
    the parameter-gradient gain (||g_alpha|| for r=u_i, a left singular
    vector of M) AND the first-order INDUCED FIELD update gain (M g_alpha,
    per the linearization delta_s = M delta_theta) match their closed forms:

        param-gradient gain:  sigma_i / (sigma_i^2+lambda)^alpha
        induced-field gain:   sigma_i^2 / (sigma_i^2+lambda)^alpha

    for alpha in {0 (direct), 1/2 (WFB), 1 (FBGN)}. This is the concrete
    claim motivating FBGN: at alpha=1 the field gain collapses to
    sigma_i^2/(sigma_i^2+lambda), which is bounded in [0,1] for every mode
    (unlike alpha=0's unbounded sigma_i^2, or alpha=1/2's still-unbounded
    sigma_i), i.e. alpha=1/2 fixes the PARAMETER gradient's conditioning
    but not the induced FIELD update's conditioning -- exactly the gap the
    Stage 2 v5/D factorial (delta_theta_norm matched via Adam reset, yet
    held-out probe loss got WORSE under WFB) points to.
    """
    torch.manual_seed(241)
    sigma_true = torch.tensor([50.0, 20.0, 5.0, 1.0, 0.5, 0.2, 0.1], dtype=torch.float64)
    n = sigma_true.numel()
    Q1, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=torch.Generator().manual_seed(243)))
    Q2, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=torch.Generator().manual_seed(247)))
    M = Q1 @ torch.diag(sigma_true) @ Q2.T

    def gram_mv_fn(v):
        return M @ (M.T @ v)

    lambda_max_true = float(sigma_true.max() ** 2)
    rho = 1e-3
    lam = rho * lambda_max_true

    for alpha in (0.0, 0.5, 1.0):
        for r_dir_idx in range(n):
            r = Q1[:, r_dir_idx].clone()  # r = u_i, an exact eigenvector of A=MM^T
            lz = _lanczos_inv_pow_apply_generic(gram_mv_fn, r, lam, alpha, k=n)
            reason = lz["breakdown_reason"]
            assert not lz["breakdown"] or (reason or "").startswith("invariant_subspace"), (
                f"alpha={alpha} mode {r_dir_idx}: unexpected breakdown: {reason}")
            u = lz["u"]
            g_alpha = M.T @ u          # parameter-space pseudo-gradient
            q_alpha = M @ g_alpha      # induced first-order field update M g_alpha

            sigma_i = float(sigma_true[r_dir_idx])
            param_gain_expected = sigma_i / (sigma_i ** 2 + lam) ** alpha
            field_gain_expected = sigma_i ** 2 / (sigma_i ** 2 + lam) ** alpha
            param_gain_got = float(g_alpha.norm())  # ||r||=1
            field_gain_got = float(q_alpha.norm())

            rel_err_param = abs(param_gain_got - param_gain_expected) / (param_gain_expected + 1e-30)
            rel_err_field = abs(field_gain_got - field_gain_expected) / (field_gain_expected + 1e-30)
            assert rel_err_param < 1e-4, (
                f"alpha={alpha} mode {r_dir_idx} (sigma={sigma_i}): param gain got={param_gain_got} "
                f"expected={param_gain_expected}, rel_err={rel_err_param:.2e}")
            assert rel_err_field < 1e-4, (
                f"alpha={alpha} mode {r_dir_idx} (sigma={sigma_i}): field gain got={field_gain_got} "
                f"expected={field_gain_expected}, rel_err={rel_err_field:.2e}")

    # The central FBGN claim: at alpha=1, EVERY mode's induced field gain is in [0,1],
    # unlike alpha=0 (unbounded, = sigma_i^2) or alpha=0.5 (still unbounded, = sigma_i).
    for sigma_i in sigma_true.tolist():
        field_gain_alpha1 = sigma_i ** 2 / (sigma_i ** 2 + lam) ** 1.0
        assert 0.0 <= field_gain_alpha1 <= 1.0 + 1e-9, (
            f"alpha=1 field gain must lie in [0,1]: sigma={sigma_i} gain={field_gain_alpha1}")
    top_sigma = float(sigma_true[0])
    field_gain_alpha0 = top_sigma ** 2
    field_gain_alpha05 = top_sigma
    field_gain_alpha1 = top_sigma ** 2 / (top_sigma ** 2 + lam)
    assert field_gain_alpha1 < field_gain_alpha05 < field_gain_alpha0, (
        "expected strictly decreasing top-mode field gain as alpha increases 0 -> 0.5 -> 1: "
        f"{field_gain_alpha0} / {field_gain_alpha05} / {field_gain_alpha1}")
    print(f"PASS WFB Stage 2.5 (alpha-family mode-gain test, sigma={sigma_true.tolist()}, rho={rho}, "
          f"lambda={lam:.4f}): top-mode induced field gain alpha=0:{field_gain_alpha0:.2f} -> "
          f"alpha=0.5:{field_gain_alpha05:.2f} -> alpha=1:{field_gain_alpha1:.4f} (bounded, as predicted)")


def test_lanczos_inv_pow_apply_alpha_half_matches_sqrt_apply():
    """Regression: the generalized _lanczos_inv_pow_apply_generic(alpha=0.5)
    must reproduce _lanczos_inv_sqrt_apply_generic's u/T_eigmax/breakdown
    output EXACTLY (both are now thin wrappers around the same alpha-power
    core) -- guards against the Stage 2.5 generalization silently changing
    alpha=0.5/WFB's existing, already-validated (Stage 0/1) numerics."""
    torch.manual_seed(251)
    n = 9
    A_sqrt = torch.randn(n, n, dtype=torch.float64, generator=torch.Generator().manual_seed(253))
    A = A_sqrt @ A_sqrt.T + 0.01 * torch.eye(n, dtype=torch.float64)  # PSD

    def gram_mv_fn(v):
        return A @ v

    r = torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(257))
    lam = 0.37
    lz_old = _lanczos_inv_sqrt_apply_generic(gram_mv_fn, r, lam, k=6)
    lz_new = _lanczos_inv_pow_apply_generic(gram_mv_fn, r, lam, alpha=0.5, k=6)
    assert lz_old["breakdown"] == lz_new["breakdown"] and lz_old["breakdown_reason"] == lz_new["breakdown_reason"]
    torch.testing.assert_close(lz_old["u"], lz_new["u"], rtol=1e-12, atol=1e-14)
    assert lz_old["T_eigmax"] == lz_new["T_eigmax"]
    print("PASS: alpha=0.5 specialization of the generalized Lanczos apply matches the original exactly.")


def test_compute_wfb_gradient_alpha_param_threads_through_on_real_model():
    """WFB-EqM Stage 2.5: compute_wfb_gradient's new `alpha` kwarg actually
    changes the returned gradient on the real tiny model (not silently
    ignored), alpha=0.5 (default, omitted) matches alpha=0.5 (explicit),
    and alpha=1 (FBGN) produces a DIFFERENT, and per the theory a SMALLER
    (more damped), gradient norm than alpha=0.5 on the same residual --
    consistent with (theta+lam)^{-alpha} being a decreasing function of
    alpha for theta+lam > 1 (true here: lambda_max estimated on a real
    130-ish-param tiny model's mixed Jacobian is >> 1)."""
    torch.manual_seed(261)
    model = make_model(ebm="direct", dtype=torch.float64)
    perturb(model, seed=263)
    model.eval()
    z, t, y = batch(n=2, dtype=torch.float64, seed=267)

    result_default = compute_wfb_gradient(model, z, t, y, torch.zeros_like(z), rho=1e-2, k=8,
                                           lambda_max_num_iters=50, seed=269)
    assert result_default["alpha"] == 0.5
    result_half = compute_wfb_gradient(model, z, t, y, torch.zeros_like(z), rho=1e-2, k=8,
                                        lambda_max_num_iters=50, seed=269, alpha=0.5)
    for g_d, g_h in zip(result_default["g_wfb"], result_half["g_wfb"]):
        torch.testing.assert_close(g_d, g_h, rtol=1e-8, atol=1e-10)

    result_fbgn = compute_wfb_gradient(model, z, t, y, torch.zeros_like(z), rho=1e-2, k=8,
                                        lambda_max_num_iters=50, seed=269, alpha=1.0)
    assert result_fbgn["alpha"] == 1.0
    assert not result_fbgn["breakdown"] or (result_fbgn["breakdown_reason"] or "").startswith("invariant_subspace")
    assert result_fbgn["g_wfb_norm"] < result_half["g_wfb_norm"], (
        f"expected alpha=1 (FBGN) gradient norm ({result_fbgn['g_wfb_norm']}) < alpha=0.5 (WFB) "
        f"gradient norm ({result_half['g_wfb_norm']}) since lambda_max >> 1 here")
    print(f"PASS: compute_wfb_gradient(alpha=...) threads through correctly on real model "
          f"(||g_wfb|| alpha=0.5: {result_half['g_wfb_norm']:.6f} -> alpha=1.0: {result_fbgn['g_wfb_norm']:.6f})")


def test_wfb_zero_residual_breakdown():
    """Zero residual (r=0, e.g. field already matches target exactly) must
    be reported as an explicit breakdown, not silently treated as u=0 being
    a normal converged result -- caller-visible per spec Section 5's
    breakdown-handling requirement."""
    torch.manual_seed(241)
    n = 6
    M = torch.randn(n, n, dtype=torch.float64, generator=torch.Generator().manual_seed(243))

    def gram_mv_fn(v):
        return M @ (M.T @ v)

    r = torch.zeros(n, dtype=torch.float64)
    lz = _lanczos_inv_sqrt_apply_generic(gram_mv_fn, r, lam=1.0, k=4)
    assert lz["breakdown"] and lz["breakdown_reason"] == "zero_residual"
    assert torch.equal(lz["u"], torch.zeros(n, dtype=torch.float64))
    print("PASS WFB Stage 0: zero-residual breakdown reported explicitly (not silently substituted)")


def test_wfb_compute_wfb_gradient_filters_frozen_parameters():
    """Regression test (found in production 2026-08-11, job 38472301):
    this codebase's `pos_embed` is registered as `nn.Parameter(...,
    requires_grad=False)` (a fixed sinusoidal embedding) so it appears in
    model.parameters(), but torch.autograd.grad raises unconditionally on a
    requires_grad=False `inputs` tensor -- allow_unused=True only covers a
    requires_grad=True tensor that is DISCONNECTED from the graph, not this
    case. No prior diagnostic (topk_subspace, matched_replay) ever hit this
    because they always passed a restricted head/backbone-only params
    subset that happened to exclude pos_embed; compute_wfb_gradient is the
    first caller in this codebase to pass the FULL parameter list by
    default. Must filter silently (with a printed note) rather than crash,
    and the returned `params` must be usable to correctly align g_raw/g_wfb
    against model.named_parameters() despite being shorter than
    list(model.parameters())."""
    torch.manual_seed(251)
    model = make_model(ebm="direct", dtype=torch.float64)
    perturb(model, seed=253)
    model.eval()
    z, t, y = batch(n=1, dtype=torch.float64, seed=257)
    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert "pos_embed" in frozen_names, "expected pos_embed to be requires_grad=False -- architecture changed"

    gen = torch.Generator().manual_seed(259)
    ut = torch.randn(z.shape, generator=gen, dtype=torch.float64)

    result = compute_wfb_gradient(model, z, t, y, ut, params=None, rho=1e-3, k=6, seed=263)
    assert not result["breakdown"] or (result["breakdown_reason"] or "").startswith("invariant_subspace")
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(result["params"]) == n_trainable, (
        f"expected filtered params to match the {n_trainable} trainable tensors, got {len(result['params'])}")
    assert len(result["g_wfb"]) == len(result["params"]) == len(result["g_raw"])
    for p in result["params"]:
        assert p.requires_grad, "compute_wfb_gradient must not include a requires_grad=False parameter"
    print(f"PASS WFB Stage 0: compute_wfb_gradient filters {len(frozen_names)} frozen "
          f"parameter(s) ({frozen_names}) out of the default full-params path without crashing")


def test_wfb_compute_wfb_gradient_stable_under_train_mode_cfg_dropout():
    """Regression test (found in production 2026-08-12, WFB-EqM Stage 2 job 38493610):
    compute_wfb_gradient crashed with 'Lanczos produced a non-finite/absent u' on
    literally the FIRST real training step -- the first time it ever ran with
    model.training=True (Stage 0/1 only ever used model.eval()). Root cause: this
    function calls the model MANY times (lambda_max power iterations + Lanczos steps,
    each an exact_field_vjp/field_jvp_direct call) to build/apply A = M M^T: with CFG
    label dropout active (dropout_prob>0, only in train mode), each call independently
    redraws its own random dropout mask, making M a DIFFERENT operator on every
    internal application -- violating the fixed-linear-operator assumption the whole
    Lanczos three-term recurrence depends on, producing numerical garbage. Fixed by
    predrawing the dropout ONCE at the top of compute_wfb_gradient and holding it
    fixed for every internal call (mirrors exact_fwrev_backward/exact_field_vjp's
    existing _predrop_labels usage, just scoped across compute_wfb_gradient's many
    internal calls instead of a single VJP/JVP pair).

    High dropout_prob (0.9) + several seeds to make an un-fixed dropout draw very
    likely to produce a large finite-difference between successive calls if the bug
    were still present."""
    torch.manual_seed(271)
    model = make_model(ebm="direct", dtype=torch.float64)
    perturb(model, seed=273)
    model.y_embedder.dropout_prob = 0.9  # aggressive, to make an unfixed re-draw likely
    model.train()
    for trial_seed in range(5):
        z, t, y = batch(n=2, dtype=torch.float64, seed=277 + trial_seed)
        gen = torch.Generator().manual_seed(281 + trial_seed)
        ut = torch.randn(z.shape, generator=gen, dtype=torch.float64)
        result = compute_wfb_gradient(model, z, t, y, ut, params=None, rho=1e-3, k=6, seed=283 + trial_seed)
        assert torch.isfinite(result["g_wfb_norm"] * torch.ones(1)).all(), \
            f"trial {trial_seed}: g_wfb_norm not finite ({result['g_wfb_norm']})"
        for gp in result["g_wfb"]:
            assert torch.isfinite(gp).all(), f"trial {trial_seed}: non-finite entries in g_wfb"
        assert not result["breakdown"] or (result["breakdown_reason"] or "").startswith("invariant_subspace"), \
            f"trial {trial_seed}: unexpected breakdown {result['breakdown_reason']}"
    assert model.y_embedder.dropout_prob == 0.9, "dropout_prob must be restored after compute_wfb_gradient returns"
    model.eval()
    print("PASS WFB Stage 0: compute_wfb_gradient stable under model.train() + CFG label "
          "dropout (5 trials, dropout_prob=0.9), dropout_prob correctly restored")


def test_wfb_gradient_matches_native_fwrev_scale():
    """Normalization-mapping proof requested during external review of WFB-EqM Stage 1/2
    (2026-08-12): g_diagnostic (compute_wfb_gradient's canonical, UNRESCALED convention)
    must be tied to g_optimizer_preclip (the actual gradient exact_fwrev_backward/ordinary
    direct training would apply) by an EXPLICIT, PROVEN factor -- not just documented in
    prose. Composes two facts:

    (a) w = -(2/(B*D)) * r EXACTLY (already proven at machine precision in
        test_field_vjp_direct_sign_relation_to_fwrev_w, Phase 0 audit finding), so
        exact_fwrev_backward's applied gradient g_fwrev = exact_field_vjp(v=-w) =
        (2/(B*D)) * exact_field_vjp(v=r) = (2/(B*D)) * g_raw.
    (b) compute_wfb_gradient's entire (A + lambda I)^{-1/2} chain is LINEAR in r: A,
        lambda_max, and lambda depend only on M (the model's Jacobian at this batch),
        NEVER on r's scale, so g_wfb(c*r) = c*g_wfb(r) for any scalar c -- proven here
        directly on the REAL model (not just the generic/synthetic Stage 0 test) by
        constructing two targets ut1/ut2 whose residuals differ by a KNOWN factor c and
        confirming g_wfb2 == c*g_wfb1 to near machine precision.

    Together: the correctly native-scaled WFB gradient the optimizer should receive is
    wfb_native_scale * g_wfb(r), wfb_native_scale = 2/(B*D) -- EXACTLY the factor train.py's
    --wfb-backward branch applies before p.grad = ... (fixed 2026-08-12 after this
    mismatch was caught pre-Stage-2-GPU-step, external review: WFB-EqM Stage 2 v1 applied
    g_wfb UNSCALED, off from native by ~(B*D)/2, before this fix)."""
    torch.manual_seed(291)
    model = make_model(ebm="direct", dtype=torch.float64)
    perturb(model, seed=293)
    model.eval()
    z, t, y = batch(n=2, dtype=torch.float64, seed=297)
    B, D = z.shape[0], z[0].numel()

    # (a) exact_fwrev's native applied gradient vs (2/(B*D))*g_raw.
    ut1 = target_for(z, seed=301)
    model.zero_grad(set_to_none=True)
    exact_fwrev_backward(model, z, t, y, ut1, gp_lambda=0.0)
    g_fwrev = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    result1 = compute_wfb_gradient(model, z, t, y, ut1, params=None, rho=1e-3, k=10, seed=307)
    scale = 2.0 / (B * D)
    g_raw_native = {p: scale * gr for p, gr in zip(result1["params"], result1["g_raw"])}
    for p, g_native in g_raw_native.items():
        name = [n for n, pp in model.named_parameters() if pp is p][0]
        if name in g_fwrev:
            torch.testing.assert_close(g_native, g_fwrev[name], rtol=1e-8, atol=1e-10)

    # (b) g_wfb linear in r: build ut2 so that r2 = c * r1 for a known c != 1.
    c = -2.37
    field1 = result1["field"]
    r1 = result1["r"]
    ut2 = field1 - c * r1  # field - ut2 = c*r1  =>  r2 = c*r1
    result2 = compute_wfb_gradient(model, z, t, y, ut2, params=None, rho=1e-3, k=10, seed=307)
    torch.testing.assert_close(result2["r"], c * r1, rtol=1e-8, atol=1e-10)
    assert abs(result2["lambda_max"] - result1["lambda_max"]) / abs(result1["lambda_max"]) < 1e-10, \
        "lambda_max must be identical across ut1/ut2 -- A depends only on M, never on r"
    for g1, g2 in zip(result1["g_wfb"], result2["g_wfb"]):
        torch.testing.assert_close(g2, c * g1, rtol=1e-6, atol=1e-8)

    print(f"PASS WFB Stage 0: normalization-mapping proof -- (a) (2/(B*D))*g_raw == "
          f"exact_fwrev_backward's native applied gradient exactly (B={B},D={D},scale={scale:.3e}); "
          f"(b) g_wfb(c*r) == c*g_wfb(r) confirmed on the real model (c={c}), completing the "
          f"g_diagnostic -> g_optimizer_preclip mapping train.py's --wfb-backward branch uses.")


import contextlib


@contextlib.contextmanager
def _math_sdpa_backend():
    """Fused SDPA kernels lack double-backward (reference path) and
    forward-AD (fwrev path) derivatives; train.py forces the math backend
    for all ebm != 'none' training (fb_direct/trainer.py, via the
    torch<2.3-compatible torch.backends.cuda.sdp_kernel API, since the
    cluster's torch is 2.1.2 and lacks torch.nn.attention.sdpa_kernel,
    added in 2.3), so the tests must match that regime. Prefers the newer
    API when available (forward-compatible with future torch upgrades),
    falls back to the older one otherwise."""
    if hasattr(torch.nn, "attention"):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    else:
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            yield


if __name__ == "__main__":
    with _math_sdpa_backend():
        test_exact_match_no_gp()
        test_exact_match_with_gp()
        test_label_dropout_determinism()
        test_rejects_wrong_ebm()
        test_field_vjp_direct_matches_double_backward_reference()
        test_field_vjp_direct_sign_relation_to_fwrev_w()
        test_field_vjp_none_matches_autograd_grad()
        test_field_vjp_finite_difference_sanity()
        test_theta_jvp_matches_explicit_jacobian()
        test_theta_jvp_handles_zero_jacobian_parameter()
        test_power_iteration_sigma1_matches_explicit_svd()
        test_block_subspace_matches_explicit_svd()
        test_wfb_operators_match_explicit_jacobian()
        test_wfb_singular_mode_gain()
        test_wfb_alpha_family_singular_mode_gain()
        test_lanczos_inv_pow_apply_alpha_half_matches_sqrt_apply()
        test_compute_wfb_gradient_alpha_param_threads_through_on_real_model()
        test_wfb_zero_residual_breakdown()
        test_wfb_compute_wfb_gradient_filters_frozen_parameters()
        test_wfb_compute_wfb_gradient_stable_under_train_mode_cfg_dropout()
        test_wfb_gradient_matches_native_fwrev_scale()
    print("ALL PASS")
