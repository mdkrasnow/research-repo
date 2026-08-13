"""
FP64 CPU correctness tests for the STACKED (microbatched) Gauss-Newton operator
added to fb_direct/exact_hvp.py (2026-08-13, Stage 3B's H2 repair primitive).

The point of the stacked operator is to build a GN model over a LARGER
stochastic batch than the 8 images every FBGN direction has used so far. The
Stage 3 directive is explicit that this must not be faked:

  "Do NOT approximate a large-batch GN direction by separately solving four GN
   systems and averaging the final parameter directions unless you explicitly
   label that as a different algorithm."

So these tests pin the operator to its definition rather than to itself:

  r     = [r_1; ...; r_J]
  M p   = [M_1 p; ...; M_J p]
  M^T v = sum_j M_j^T v_j
  A v   = M M^T v

  1. AGAINST THE EXPLICIT JACOBIAN. The stacked A is built by concatenating the
     per-microbatch explicit Jacobians M_j into one tall M and forming M M^T
     densely, then compared to the matrix-free stacked_mixed_gram_mv on random
     probe vectors. This is ground truth, not a self-consistency check.

  2. J = 1 REDUCES TO THE EXISTING OPERATOR. With one microbatch the stacked
     operator must equal mixed_gram_mv bit-for-bit-ish; if it did not, the new
     path would be a different algorithm from the validated one.

  3. THE SOLVE SOLVES THE STACKED SYSTEM. compute_fbgn_gradient_cg_microbatched
     must return g = M^T u with (A + lambda I) u = r to the requested tolerance,
     verified by re-applying the operator (true residual, never CG's recursive
     one).

  4. IT IS NOT THE AVERAGE-OF-SOLVES. The decisive test: the stacked direction
     must DIFFER materially from sum_j M_j^T (A_j + lambda I)^{-1} r_j, the
     block-diagonal approximation that ignores the cross-microbatch coupling
     through the shared theta. If these agreed, the whole construction would be
     pointless and the cheap version should be used instead. (A is not block
     diagonal precisely because M^T v sums over j.)

  5. SYMMETRY / PSD. A must be symmetric (<u, Av> == <Av, u> to FP64) and PSD
     (<v, Av> >= 0), the properties CG's applicability rests on.

NOTE ON COST: the dense-Jacobian test is run over a SMALL PARAMETER SUBSET
(_small_params). Materializing M over all ~33M params of the FP64 toy is ~50 GB
and OOM-killed SLURM 39024136. The CG tests use n=1 microbatch rows for the same
reason -- each CG iteration is a VJP+JVP over full theta, per microbatch.

Run: python tests/test_stacked_microbatch_gn.py   (CPU, ~1-2 min)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from fb_direct.exact_hvp import (  # noqa: E402
    _cg_solve_shifted_system_generic, compute_field_direct, exact_field_vjp,
    mixed_gram_mv, stacked_estimate_lambda_max,
    stacked_mixed_gram_mv, compute_fbgn_gradient_cg_microbatched,
)
from test_forward_backwards_direct import make_model, batch, perturb  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def _setup(J=3, n=2):
    model = make_model(ebm="direct", dtype=torch.float64, seed=0)
    perturb(model, std=0.02, seed=1)   # escape the linear-at-init degeneracy
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    micro = []
    for j in range(J):
        xt, t, y = batch(n=n, dtype=torch.float64, seed=300 + j)
        g = torch.Generator().manual_seed(700 + j)
        ut = torch.randn(xt.shape, generator=g, dtype=torch.float64)
        micro.append((xt, t, y, ut))
    return model, params, micro


def _small_params(model, max_elems=2500):
    """A SUBSET of theta small enough to materialize a dense Jacobian against.

    The full EqM-S/2 toy still has ~33M parameters, so an explicit
    M of shape (numel(field), n_params) is ~50 GB in FP64 -- it OOM-kills the
    job (observed: SLURM 39024136, oom_kill on this very test). The operator
    identity A = M M^T is defined per parameter LIST, so restricting to a subset
    is a valid ground truth, not a weakened one: stacked_mixed_gram_mv is called
    with the same subset, and the dense reference is built from the same subset.

    Smallest tensors first, until the budget is spent. The caller asserts the
    resulting A is nonzero so the test cannot pass vacuously on parameters the
    field does not depend on.
    """
    cand = sorted((p for p in model.parameters() if p.requires_grad),
                  key=lambda p: p.numel())
    out, tot = [], 0
    for p in cand:
        if tot + p.numel() > max_elems:
            continue
        out.append(p)
        tot += p.numel()
    if not out:
        raise RuntimeError("no parameter tensor fits the dense-Jacobian budget")
    return out


def _explicit_M(model, xt, t, y, params):
    """Dense M_j = d(field)/d(theta), shape (numel(field), n_params), built one
    field component at a time via the validated VJP. Only ever called with the
    _small_params subset -- see the OOM note there."""
    field = compute_field_direct(model, xt, t, y)
    n_out = field.numel()
    n_par = sum(p.numel() for p in params)
    M = torch.zeros(n_out, n_par, dtype=torch.float64)
    for i in range(n_out):
        e = torch.zeros_like(field).reshape(-1)
        e[i] = 1.0
        model.zero_grad(set_to_none=True)
        exact_field_vjp(model, xt, t, y, e.reshape(field.shape))
        row = torch.cat([(p.grad.detach() if p.grad is not None
                          else torch.zeros_like(p)).reshape(-1) for p in params])
        M[i] = row
    model.zero_grad(set_to_none=True)
    return M


def test_stacked_operator_matches_explicit_jacobian():
    model, _full, micro = _setup(J=2, n=1)
    params = _small_params(model)
    triples = [(xt, t, y) for (xt, t, y, _) in micro]
    M = torch.cat([_explicit_M(model, xt, t, y, params) for (xt, t, y) in triples], dim=0)
    A_dense = M @ M.T
    assert float(A_dense.abs().max()) > 0.0, (
        "A is identically zero on this parameter subset -- the test would pass "
        "vacuously; widen _small_params to tensors the field actually depends on")

    shape = (micro[0][0].shape[0] * len(micro),) + tuple(micro[0][0].shape[1:])
    worst = 0.0
    for s in range(3):
        g = torch.Generator().manual_seed(900 + s)
        v = torch.randn(shape, generator=g, dtype=torch.float64)
        got = stacked_mixed_gram_mv(model, triples, params, v)
        want = (A_dense @ v.reshape(-1)).reshape(shape)
        rel = float((got - want).norm() / (want.norm() + 1e-30))
        worst = max(worst, rel)
        assert rel < 1e-9, f"stacked A v mismatch vs explicit M M^T: rel={rel:.3e}"
    print(f"PASS stacked A == explicit M M^T on random probes (worst rel={worst:.2e})")


def test_J1_reduces_to_single_batch_operator():
    model, params, micro = _setup(J=1, n=1)
    xt, t, y, _ = micro[0]
    g = torch.Generator().manual_seed(11)
    v = torch.randn(xt.shape, generator=g, dtype=torch.float64)
    a = stacked_mixed_gram_mv(model, [(xt, t, y)], params, v)
    b = mixed_gram_mv(model, xt, t, y, params, v)
    rel = float((a - b).norm() / (b.norm() + 1e-30))
    assert rel < 1e-12, f"J=1 stacked operator differs from mixed_gram_mv: rel={rel:.3e}"
    print(f"PASS J=1 stacked operator reduces to mixed_gram_mv (rel={rel:.2e})")


def test_microbatched_solve_solves_the_stacked_system():
    model, params, micro = _setup(J=3, n=1)
    triples = [(xt, t, y) for (xt, t, y, _) in micro]
    tol = 1e-6
    res = compute_fbgn_gradient_cg_microbatched(model, micro, params=params, rho=1e-3,
                                                cg_tol=tol, cg_max_iters=300, seed=0)
    assert res["n_micro"] == 3

    # Certify the ACTUAL returned solution, not a fresh solve: re-apply the
    # operator to res["u"] and recompute the TRUE residual. (A second solve
    # would only certify that CG is reproducible, which is not the claim.)
    r = res["r"]
    lam = res["lam"]
    u = res["u"]
    Au = stacked_mixed_gram_mv(model, triples, params, u)
    true_ratio = float((r - (Au + lam * u)).norm() / r.norm())
    assert true_ratio < 10 * tol, f"true residual {true_ratio:.3e} exceeds requested tol {tol:.3e}"

    # and g must equal M^T u = sum_j M_j^T u_j
    u_chunks = torch.chunk(u, 3, dim=0)
    manual = [torch.zeros_like(p) for p in params]
    for (xt, t, y), uj in zip(triples, u_chunks):
        model.zero_grad(set_to_none=True)
        exact_field_vjp(model, xt, t, y, uj.contiguous())
        for acc, p in zip(manual, params):
            if p.grad is not None:
                acc.add_(p.grad.detach())
    model.zero_grad(set_to_none=True)
    num = sum(float(((a - b) ** 2).sum()) for a, b in zip(res["g_wfb"], manual)) ** 0.5
    den = sum(float((b ** 2).sum()) for b in manual) ** 0.5
    assert num / den < 1e-8, f"returned g != M^T u (rel={num/den:.3e})"
    print(f"PASS microbatched solve satisfies (A+lam I)u=r (true residual {true_ratio:.2e}) and g=M^T u")


def test_stacked_is_not_the_average_of_separate_solves():
    """The decisive anti-shortcut test: the stacked direction must differ
    materially from the block-diagonal average-of-solves."""
    model, params, micro = _setup(J=3, n=1)
    triples = [(xt, t, y) for (xt, t, y, _) in micro]
    lam_res = stacked_estimate_lambda_max(model, triples, params, num_iters=30, seed=0)
    lam = 1e-3 * lam_res["lambda_max"]

    stacked = compute_fbgn_gradient_cg_microbatched(model, micro, params=params,
                                                    cg_tol=1e-7, cg_max_iters=400,
                                                    lam_override=lam, seed=0)
    g_stacked = stacked["g_wfb"]

    # block-diagonal alternative: solve each microbatch separately, sum M_j^T u_j
    g_blockdiag = [torch.zeros_like(p) for p in params]
    for (xt, t, y, ut) in micro:
        field = compute_field_direct(model, xt, t, y)
        rj = (field - ut).detach()
        cg = _cg_solve_shifted_system_generic(
            lambda v: mixed_gram_mv(model, xt, t, y, params, v), rj, lam,
            tol=1e-7, max_iters=400)
        model.zero_grad(set_to_none=True)
        exact_field_vjp(model, xt, t, y, cg["u"])
        for acc, p in zip(g_blockdiag, params):
            if p.grad is not None:
                acc.add_(p.grad.detach())
    model.zero_grad(set_to_none=True)

    ns = sum(float((a ** 2).sum()) for a in g_stacked) ** 0.5
    nb = sum(float((a ** 2).sum()) for a in g_blockdiag) ** 0.5
    cos = sum(float((a * b).sum()) for a, b in zip(g_stacked, g_blockdiag)) / (ns * nb)
    rel = sum(float(((a - b) ** 2).sum()) for a, b in zip(g_stacked, g_blockdiag)) ** 0.5 / ns
    assert rel > 1e-3, (f"stacked and average-of-solves are indistinguishable (rel={rel:.3e}, "
                        f"cos={cos:.6f}) -- the cross-microbatch coupling would be doing nothing")
    print(f"PASS stacked != average-of-separate-solves (rel diff={rel:.3e}, cosine={cos:.6f}) "
          f"-- the block-diagonal shortcut is a genuinely different algorithm")


def test_stacked_operator_is_symmetric_psd():
    model, params, micro = _setup(J=2, n=1)
    triples = [(xt, t, y) for (xt, t, y, _) in micro]
    shape = (micro[0][0].shape[0] * len(micro),) + tuple(micro[0][0].shape[1:])
    g = torch.Generator().manual_seed(4242)
    u = torch.randn(shape, generator=g, dtype=torch.float64)
    v = torch.randn(shape, generator=g, dtype=torch.float64)
    Au = stacked_mixed_gram_mv(model, triples, params, u)
    Av = stacked_mixed_gram_mv(model, triples, params, v)
    lhs, rhs = float((v * Au).sum()), float((u * Av).sum())
    assert abs(lhs - rhs) / (abs(lhs) + 1e-30) < 1e-10, f"A not symmetric: {lhs} vs {rhs}"
    assert float((u * Au).sum()) >= 0.0, "A not PSD"
    print("PASS stacked A is symmetric and PSD (CG's applicability holds)")


if __name__ == "__main__":
    test_J1_reduces_to_single_batch_operator()
    test_stacked_operator_is_symmetric_psd()
    test_stacked_operator_matches_explicit_jacobian()
    test_microbatched_solve_solves_the_stacked_system()
    test_stacked_is_not_the_average_of_separate_solves()
    print("\nALL STACKED MICROBATCH GN TESTS PASS")
