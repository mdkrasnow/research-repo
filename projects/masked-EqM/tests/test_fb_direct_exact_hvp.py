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
    block_subspace_iteration_theta, exact_field_vjp, exact_fwrev_backward,
    field_jvp_direct, field_jvp_none, field_vjp_none,
    power_iteration_theta_sigma1,
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
    print("ALL PASS")
