"""Finite-difference step-size calibration (mandatory before expensive runs).

For a frozen scalar model and a held-out batch, compare the FD directional
derivative against the exact autograd directional derivative over a ladder of
relative step sizes h = eps_fd * ||z||_2.  Reported per eps:

  rel_rmse        sqrt(E[(D-g)^2] / E[g^2])
  median_abs_rel  median |D-g| / (|g| + tiny)
  corr            Pearson correlation across examples
  cosine          <D, g> / (|D| |g|) over the flattened (K,B) estimates
  bias            mean(D - g)
  var             var(D - g)
  nonfinite_frac  fraction of non-finite estimates
  gap_abs         mean |phi(z+hu) - phi(z-hu)|          (cancellation numerator)
  cancel_ratio    gap_abs / mean|phi(z)|                (cancellation severity)

The correct choice is the STABLE PLATEAU: small h is dominated by cancellation
(gap_abs falls below the precision of phi), large h by O(h^2) truncation.  Never
take the smallest h.  `pick_plateau` implements that rule mechanically.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from .fd import directional_fd, exact_directional_derivative, rademacher_directions

DEFAULT_EPS_LADDER = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


@torch.no_grad()
def _stats(D: torch.Tensor, g: torch.Tensor, phi_z: torch.Tensor,
           gap: torch.Tensor):
    D64, g64 = D.double().flatten(), g.double().flatten()
    finite = torch.isfinite(D64) & torch.isfinite(g64)
    nonfinite_frac = float(1.0 - finite.double().mean())
    D64, g64 = D64[finite], g64[finite]
    err = D64 - g64
    denom = (g64 ** 2).mean().sqrt()
    out = {
        "rel_rmse": float((err ** 2).mean().sqrt() / (denom + 1e-30)),
        "median_abs_rel": float((err.abs() / (g64.abs() + 1e-30)).median()),
        "bias": float(err.mean()),
        "var": float(err.var()),
        "nonfinite_frac": nonfinite_frac,
        "gap_abs": float(gap.double().abs().mean()),
        "cancel_ratio": float(gap.double().abs().mean()
                              / (phi_z.double().abs().mean() + 1e-30)),
        "exact_rms": float(denom),
    }
    if D64.numel() > 2:
        dc, gc = D64 - D64.mean(), g64 - g64.mean()
        out["corr"] = float((dc * gc).mean()
                            / (dc.std(unbiased=False) * gc.std(unbiased=False)
                               + 1e-30))
        out["cosine"] = float((D64 * g64).sum()
                              / (D64.norm() * g64.norm() + 1e-30))
    return out


def calibrate_fd(
    phi_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    K: int = 4,
    eps_ladder: Sequence[float] = DEFAULT_EPS_LADDER,
    fp32_subtract: bool = True,
    seed: int = 0,
    chunk: int | None = None,
):
    """Run the full eps ladder on a frozen model / held-out batch."""
    gen = torch.Generator(device=z.device).manual_seed(seed)
    u = rademacher_directions(z.shape, K, device=z.device, dtype=z.dtype,
                              generator=gen)
    g = exact_directional_derivative(phi_fn, z, u).detach()
    with torch.no_grad():
        phi_z = phi_fn(z).detach()

    rows = []
    for eps in eps_ladder:
        with torch.no_grad():
            D, h, gap = directional_fd(phi_fn, z, u, eps,
                                       fp32_subtract=fp32_subtract, chunk=chunk)
        row = {"eps_fd": eps, "h_mean": float(h.mean()), "K": K}
        row.update(_stats(D, g, phi_z, gap))
        rows.append(row)
    return rows


def pick_plateau(rows, tol: float = 1.5):
    """Choose eps from the stable plateau, not the smallest h.

    Rule: take the minimum rel_rmse over the ladder, then among all eps whose
    rel_rmse <= tol * min_rel_rmse choose the LARGEST eps (largest h wins ties:
    it maximizes the FD numerator and therefore the cancellation headroom that
    matters once training changes the energy scale).
    """
    good = [r for r in rows if r["nonfinite_frac"] == 0.0]
    if not good:
        raise RuntimeError("every eps produced non-finite FD estimates")
    best = min(r["rel_rmse"] for r in good)
    cands = [r for r in good if r["rel_rmse"] <= tol * best]
    chosen = max(cands, key=lambda r: r["eps_fd"])
    return chosen["eps_fd"], {"best_rel_rmse": best, "chosen": chosen}
