# Symmetry Discovery vs Hard-Negative Mining for Generative Models (EqM)

## Research Question
For a regression-target generative model (EqM), what lets it **generalize onto held-out manifold
regions** it was never trained on? Specifically: can a learned **manifold-preserving operator** —
anchored to data rather than the live field — discover a symmetry that is **hidden by a nonlinear
observation map**, and use it to transfer supervision into untrained regions? How does this compare to
**hard-negative mining** (the v10 mechanism) and to **known-symmetry constraints**?

## TL;DR findings (toy ladder, rungs 1–9)
- **Hard-negative / v10-style mining installs NO missing manifold structure** in any setting — it can
  sharpen an existing boundary but never fills held-out regions.
- **Symmetry constraints / augmentation generalize — but only when the symmetry is KNOWN** (provided as
  the true group, an equivariance tie, or a known rotation).
- **Unsupervised discovery of a nonlinearly-hidden symmetry is blocked by a 3-layer stack:**
  1. **Identity attractor** — every prior-free structural constraint (on-manifold, finite-order, recon)
     is satisfied by the identity; soft anti-identity penalties can't escape it.
  2. **Field co-adaptation** — discovering the operator via live EqM-field closure fails because the
     field just learns to model whatever the operator emits (no anchor to the data manifold).
     **FIXED in rung 9** by a frozen, non-co-adapting data anchor (operator now stays on-manifold).
  3. **Operator coherence [OPEN]** — distributional anchoring (energy-distance/MMD) keeps the operator
     on-manifold but allows a *random distribution-preserving shuffle*, not a coherent flow, so it still
     doesn't transfer the symmetry. Needs a flow/coherence constraint (candidate rung 10).

Full detail: `documentation/findings.md`.

## The ladder (each rung diagnosed the next)
| rung | setup | result |
|---|---|---|
| 1 | arithmetic (classification): equivariance vs hard-neg | constraint perfect; **mining dead** |
| 2 | 2D ring, symmetry linear in observed space | known + discovered rotation both work |
| 3 | symmetry hidden by nonlinear decoder, free observed operator | fails (off-manifold) |
| 4 | enc→linear→dec, reconstruction-frozen latent | fails (recon latent ≠ linearizing latent) |
| 5 | ORACLE_LATENT control (perfect latent) | recipe collapses M→identity even so |
| 6 | hinge anti-identity | insufficient |
| 7 | continuous manifold | cyclic term now pulls M→identity |
| 8 | hard identity-exclusion | M drifts to non-symmetry; closure can't anchor |
| 9 | **frozen data anchor** (vs live field) | **co-adaptation FIXED (on-manifold); coherence still missing** |

## Methodology that made it work
Every rung ships **positive + negative controls in the same run** (ORACLE = given the true symmetry;
ORACLE_LATENT = given a clean latent; DISC_LINEAR / VISIBLE_ANCHOR = should fail). Treatment is read
ONLY inside the control band. These controls are what isolated each root cause (e.g. ORACLE_LATENT
exposed the recipe bug; FIELD_CLOSURE vs FROZEN_ANCHOR isolated co-adaptation). See the
positive/negative-control rule added to the repo CLAUDE.md.

## Project structure
- `experiments/` — the rung scripts: `latent_symmetry_rung{4..9}.py`, plus early toys
  (`exp2.py` arithmetic, `eqm_toy*.py` ring, `latent_symmetry.py` rung-3 nonlinear-hidden).
- `documentation/findings.md` — full per-rung results, diagnoses, verdicts.
- `results/` — saved result JSONs per rung.
- `.state/pipeline.json` — current phase + ladder status + next action.

## Run (CPU, seconds–minutes; auto-uses GPU if present)
```bash
python experiments/latent_symmetry_rung9.py --quick   # fast smoke
python experiments/latent_symmetry_rung9.py            # full (3 seeds)
```

## Relationship to diff-EqM
This started as an investigation of the diff-EqM **v10** premise (adaptive hard-negative mining). It
informs that project: mining doesn't install structure; prefer **known-symmetry constraints**. Scripts
and findings are mirrored from `projects/diff-EqM/experiments/symmetry_toys/`.

## Status
Exploratory; CPU-toy scale. **NOT publishable as-is** — these are filters/diagnostics, not paper-scale
results. Open next step: rung 10 (frozen anchor + operator-coherence constraint).
