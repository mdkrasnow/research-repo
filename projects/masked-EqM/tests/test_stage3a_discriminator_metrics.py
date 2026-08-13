"""
FP64 CPU correctness tests for the Stage 3A mechanism-discriminator metrics
(2026-08-13), run BEFORE any GPU time per this project's Stage-0 discipline.

Stage 3A's whole job is to decide between "the Gauss-Newton LOCAL MODEL is
wrong" (H1) and "the local model is right for the WRONG MINIBATCH" (H2). That
decision rests entirely on three quantities, so each is validated here against
an independent ground truth on the FP64 CPU toy model:

  1. apply/revert EXACTNESS. Every candidate step in Stage 3A is applied,
     evaluated, and reverted against a frozen checkpoint. If revert leaked any
     drift, every measurement after the first would be silently contaminated.
     Asserted bitwise (max|diff| == 0), not approximately.

  2. LINEARIZATION DEFECT D_B(eta) -- the H1 discriminator. By Taylor,
         r(theta + eta p) = r(theta) + eta M p + O(eta^2),
     so the absolute defect ||r(theta+eta p) - [r + eta M p]|| must scale as
     O(eta^2), and hence D_B = defect / ||eta M p|| must scale as O(eta) --
     i.e. D_B must HALVE when eta halves. If D_B did not vanish with eta, the
     metric would be measuring an operator bug (a wrong M), not curvature, and
     any H1 verdict drawn from it would be an artifact. Tested over a decade
     of eta by checking the observed convergence order is ~1.

  3. REDUCTION RATIO R_B(eta) -> 1 as eta -> 0. The local Gauss-Newton model
     m(eta) = ||r + eta M p||^2/(BD) is exact to second order in eta along p,
     so actual/predicted must converge to 1 as the step shrinks. This is what
     makes a LOW R_B at a LARGE eta meaningful evidence of local-model failure
     rather than a constant offset from a normalization mistake.

  4. INFINITESIMAL TRANSFER d_V = grad L_V . p -- the H2 discriminator, and
     the only Stage 3A quantity that is eta-INDEPENDENT. Validated against a
     central finite difference of the trust-bank loss along p:
         [L_V(theta + h p) - L_V(theta - h p)] / (2h)  ->  d_V.
     If d_V's sign were wrong, Stage 3A would draw exactly the opposite
     conclusion about whether FBGN directions transfer, so the sign and
     magnitude are both checked. grad_bank's manual per-batch harvesting (see
     its docstring: exact_field_vjp negates the whole accumulated .grad
     buffer, so naive accumulation across batches double-negates earlier
     terms) is precisely what this test guards.

Run: python tests/test_stage3a_discriminator_metrics.py   (CPU, seconds)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments", "direct_energy"))

from fb_direct.exact_hvp import field_jvp_direct  # noqa: E402
from test_forward_backwards_direct import make_model, batch, perturb  # noqa: E402

from wfb_stage3a_mechanism_discriminator import (  # noqa: E402
    apply_step, assert_exact_restore, bank_loss, clone_params, direct_direction,
    dot, grad_bank, pnorm, residual_and_loss, restore_params,
)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def _setup(seed=0, n_banks=3):
    """Frozen toy banks: (xt, t, y, ut) tuples materialized once, exactly the
    shape of Stage 3A's real frozen banks. ut is an arbitrary FIXED target --
    every metric under test is a property of the residual map r = field - ut
    and its Jacobian, not of ut's provenance."""
    model = make_model(ebm="direct", dtype=torch.float64, seed=seed)
    # perturb() is REQUIRED, not cosmetic: at EqM's default (DiT-style) init the
    # adaLN-zero gates and zeroed output projection make the residual map LINEAR
    # in theta to machine precision, so the linearization defect sits at the FP64
    # roundoff floor (~6.6e-16, constant in eta) and the O(eta^2) test below would
    # be measuring nothing but rounding. Perturbing off that degenerate point
    # gives the toy genuine theta-curvature to detect.
    perturb(model, std=0.02, seed=1)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    banks = []
    for i in range(n_banks):
        xt, t, y = batch(n=2, dtype=torch.float64, seed=100 + i)
        g = torch.Generator().manual_seed(500 + i)
        ut = torch.randn(xt.shape, generator=g, dtype=torch.float64)
        banks.append((xt, t, y, ut))
    return model, params, banks


def test_apply_revert_is_bitwise_exact():
    model, params, banks = _setup()
    xt, t, y, ut = banks[0]
    d = direct_direction(model, xt, t, y, ut, params)
    saved = clone_params(params)
    for eta in (1e-3, 1e-1, 7.5):
        apply_step(params, d["p"], eta)
        moved = max(float((p - s).abs().max()) for p, s in zip(params, saved))
        assert moved > 0.0, "apply_step did not actually move the parameters"
        restore_params(params, saved)
        worst = assert_exact_restore(params, saved)
        assert worst == 0.0
    print("PASS apply/revert is bitwise exact (and apply genuinely moves theta)")


def test_linearization_defect_is_first_order_in_eta():
    """D_B = ||r(theta+eta p) - (r + eta M p)|| / ||eta M p|| must be O(eta)."""
    model, params, banks = _setup()
    xt, t, y, ut = banks[0]
    d = direct_direction(model, xt, t, y, ut, params)
    # scale p to ||theta|| so eta reads directly as a RELATIVE parameter
    # perturbation (eta=1e-2 <=> a 1% move), which is what puts the defect
    # safely above the roundoff floor while staying in the Taylor regime.
    scale = pnorm(params) / pnorm(d["p"])
    p = [pi * scale for pi in d["p"]]
    r0 = d["r"]
    q = field_jvp_direct(model, xt, t, y, params, p)
    q_norm = float(q.norm())

    saved = clone_params(params)
    etas = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
    defects, ratios = [], []
    for eta in etas:
        apply_step(params, p, eta)
        r_new, _ = residual_and_loss(model, xt, t, y, ut)
        restore_params(params, saved)
        assert_exact_restore(params, saved)
        defect = float((r_new - (r0 + eta * q)).norm())
        defects.append(defect)
        ratios.append(defect / (eta * q_norm))

    # absolute defect ~ O(eta^2): halving eta must quarter it (order ~2)
    for a, b, ea, eb in zip(defects, defects[1:], etas, etas[1:]):
        order = torch.log(torch.tensor(a / b)) / torch.log(torch.tensor(ea / eb))
        assert 1.7 < float(order) < 2.3, f"absolute defect order {float(order):.3f} not ~2 (defects={defects})"
    # normalized D_B ~ O(eta): must halve when eta halves
    for a, b in zip(ratios, ratios[1:]):
        assert 1.7 < a / b < 2.3, f"D_B did not scale as O(eta): ratios={ratios}"
    # D_B ~ O(eta), so over an 8x eta range it must shrink ~8x (never 10x --
    # asserting that would be demanding better than the exact expected order).
    expected = etas[-1] / etas[0]
    assert ratios[-1] < 1.6 * expected * ratios[0], f"D_B failed to shrink with eta: {ratios}"
    print(f"PASS linearization defect D_B is O(eta) and -> 0 (D_B={[round(x, 6) for x in ratios]})")


def test_reduction_ratio_converges_to_one():
    """R_B = actual/predicted -> 1 as eta -> 0 (the GN model is exact to 2nd order)."""
    model, params, banks = _setup()
    xt, t, y, ut = banks[0]
    d = direct_direction(model, xt, t, y, ut, params)
    scale = pnorm(params) / pnorm(d["p"])
    p = [pi * scale for pi in d["p"]]
    r0, L0 = d["r"], d["L"]
    B, D = xt.shape[0], xt[0].numel()
    BD = B * D
    q = field_jvp_direct(model, xt, t, y, params, p)

    saved = clone_params(params)
    Rs = []
    for eta in (1e-2, 1e-3, 1e-4):
        m_eta = float(((r0 + eta * q) ** 2).sum()) / BD
        pred = m_eta - L0
        apply_step(params, p, eta)
        _, L_new = residual_and_loss(model, xt, t, y, ut)
        restore_params(params, saved)
        assert_exact_restore(params, saved)
        Rs.append((L_new - L0) / pred)
    assert abs(Rs[-1] - 1.0) < 1e-3, f"R_B did not converge to 1: {Rs}"
    assert abs(Rs[-1] - 1.0) < abs(Rs[0] - 1.0), f"R_B not converging monotonically: {Rs}"
    print(f"PASS reduction ratio R_B -> 1 as eta -> 0 (R={[round(x, 6) for x in Rs]})")


def test_grad_bank_matches_finite_difference_transfer():
    """d_V = grad L_V . p must match a central finite difference of L_V along p.

    This is the H2 discriminator; a sign error here would invert Stage 3A's
    entire conclusion. Also guards grad_bank's manual per-batch harvesting
    against exact_field_vjp's whole-buffer .grad negation."""
    model, params, banks = _setup(n_banks=4)
    V_batches, src = banks[:3], banks[3]
    xt, t, y, ut = src
    d = direct_direction(model, xt, t, y, ut, params)
    scale = pnorm(params) / pnorm(d["p"])
    p = [pi * scale for pi in d["p"]]

    g_V = grad_bank(model, V_batches, params)
    d_V = dot(g_V, p)

    saved = clone_params(params)
    h = 1e-5
    apply_step(params, p, h)
    L_plus = bank_loss(model, V_batches)[0]
    restore_params(params, saved)
    apply_step(params, p, -h)
    L_minus = bank_loss(model, V_batches)[0]
    restore_params(params, saved)
    assert_exact_restore(params, saved)

    fd = (L_plus - L_minus) / (2 * h)
    rel = abs(fd - d_V) / (abs(fd) + 1e-30)
    assert rel < 1e-5, f"d_V={d_V:.12e} vs finite-difference {fd:.12e} (rel={rel:.3e})"
    # and the direction built from the source batch must actually descend ITS own loss
    assert dot(grad_bank(model, [src], params), p) < 0, "p_direct is not a descent direction on its own batch"
    print(f"PASS d_V matches central finite difference (d_V={d_V:.6e}, fd={fd:.6e}, rel={rel:.2e})")


def test_grad_bank_is_not_naively_accumulated():
    """Regression guard: the bank gradient must equal the mean of the
    per-batch gradients. A naive accumulate-across-calls implementation would
    double-negate earlier batches and fail this."""
    model, params, banks = _setup(n_banks=3)
    g_all = grad_bank(model, banks, params)
    per = [grad_bank(model, [b], params) for b in banks]
    manual = [sum(gs) / len(per) for gs in zip(*per)]
    worst = max(float((a - b).abs().max()) for a, b in zip(g_all, manual))
    denom = max(pnorm(g_all), 1e-30)
    assert worst / denom < 1e-10, f"grad_bank != mean of per-batch grads (worst={worst:.3e})"
    print("PASS grad_bank equals the mean of per-batch gradients")


if __name__ == "__main__":
    test_apply_revert_is_bitwise_exact()
    test_linearization_defect_is_first_order_in_eta()
    test_reduction_ratio_converges_to_one()
    test_grad_bank_matches_finite_difference_transfer()
    test_grad_bank_is_not_naively_accumulated()
    print("\nALL STAGE 3A METRIC TESTS PASS")
