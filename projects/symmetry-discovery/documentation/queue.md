# Experiment Queue — Symmetry Discovery

## DONE
- Rungs 1–9 (see `documentation/findings.md`, `results/`).

## READY (pending user go-ahead — do not stack mechanisms)
### Rung 10: frozen anchor + operator coherence
- Hypothesis: rung-9 frozen anchor keeps T on-manifold but distributional matching permits a random
  shuffle; adding a COHERENCE constraint (one mechanism) yields a consistent flow = the symmetry.
- Two minimal variants (pick ONE):
  - (a) parametrize T as a single Lie-generator flow `T = decode(exp(t·ξ)·enc(x))` (or observed-space
    `exp(t·ξ)`), ξ learned, t fixed → structurally a coherent one-parameter flow.
  - (b) keep residual T but add `λ · Var(Δφ)` penalty (shift-inconsistency) to the frozen-anchor loss.
- Success: FROZEN_ANCHOR recall_arc → ORACLE level, shift_std small (coherent), FROZEN ≫ VISIBLE.
- Kill: if coherence constraint still doesn't fill arc with T_onman high → distributional anchor is
  fundamentally insufficient; consider a frozen score/discriminator anchor instead.

## BLOCKED
- GPU/scaled confirmation: needs cluster 2FA re-auth.

## PARKED
- Scaling any positive toy result to real EqM/IN-1K — only after a clean toy win; not before.
