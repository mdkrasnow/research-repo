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

from fb_direct.exact_hvp import exact_fwrev_backward
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


if __name__ == "__main__":
    # Fused SDPA kernels lack double-backward (reference path) and
    # forward-AD (fwrev path) derivatives; train.py forces the math backend
    # for all ebm != 'none' training, so the tests must match that regime.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        test_exact_match_no_gp()
        test_exact_match_with_gp()
        test_label_dropout_determinism()
        test_rejects_wrong_ebm()
    print("ALL PASS")
