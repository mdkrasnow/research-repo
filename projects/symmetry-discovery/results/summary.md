# Results Summary — Symmetry Discovery

## Research question
Can a learned manifold-preserving operator (anchored to data, not the live field) discover a
nonlinearly-hidden symmetry and fill held-out manifold regions? vs hard-negative mining and
known-symmetry constraints.

## Key findings
1. **Hard-negative mining (v10 mechanism) installs no missing manifold structure** — dead in every rung.
2. **Symmetry constraints generalize only when the symmetry is KNOWN.**
3. **Unsupervised discovery is blocked by a 3-layer stack:** identity attractor → field co-adaptation
   [FIXED rung 9] → operator coherence [OPEN].
4. **Frozen data anchor (rung 9) fixes the co-adaptation** (operator on-manifold rate 0.00 → 0.70) but
   distributional matching alone yields a random on-manifold shuffle, not a coherent flow — so it does
   not yet fill held-out regions.

## Headline numbers
| rung | metric | treatment | positive ctrl | floor |
|---|---|---|---|---|
| 1 (arithmetic) | test acc on held-out orderings | EQUIV 1.00 | — | hard-neg 0.00 |
| 2 (ring) | recall@held-out modes | discovered 0.19 | known 0.19 | base 0.00 |
| 9 (frozen anchor) | recall_arc / T_onman | FROZEN 0.009 / 0.70 | ORACLE 0.068 | FIELD_CLOSURE 0.002 / 0.00 |

## Completed experiments
Rungs 1–9 (see `documentation/findings.md` for full tables + diagnoses). Result JSONs in this dir.

## Next steps
- Rung 10 (pending user go-ahead): frozen anchor + operator-coherence constraint (Lie-generator flow or
  shift-consistency penalty). One mechanism, do not stack.
- For the parent diff-EqM project: prefer known-symmetry equivariance/augmentation over more
  hard-negative mining.
