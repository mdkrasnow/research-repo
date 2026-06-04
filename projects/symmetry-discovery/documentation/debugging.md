# Debugging / Methodology Lessons — Symmetry Discovery

## Methodology lessons (load-bearing)
- **Positive + negative controls in every run.** ORACLE (true symmetry) and ORACLE_LATENT (clean latent)
  bracketed every treatment; FIELD_CLOSURE / VISIBLE_ANCHOR / DISC_LINEAR are the negatives. These
  controls isolated each root cause — without ORACLE_LATENT, rung-4 looked like a thesis negative when
  it was a recipe bug. (Rule added to repo CLAUDE.md.)
- **Verify the AE reconstructs BEFORE judging a symmetry result.** Rung-4 semi-freeze corrupted a good
  AE (recon 0.03 → 0.44); the symmetry result was then meaningless. A corrupted AE masquerades as a
  thesis negative.
- **Smoke = run end-to-end + sanity-read M/recon, not just "no crash."** Under-trained smoke (300 steps)
  routinely shows M≈identity; do not conclude from smoke.

## Diagnosed failure modes (the identity-collapse stack)
- **Identity attractor:** identity satisfies on-manifold + finite-order + recon. Soft anti-identity
  penalties fail — squared-move has zero gradient at identity (rung 5); hinge too weak (rung 6); cyclic
  term itself prefers identity (rung 7); even hard exclusion lets M drift to a non-symmetry (rung 8).
- **Field co-adaptation:** `closure = eqm(f, T(x))` with a live `f` is satisfiable by any T (the field
  models whatever T emits). Diagnostic: FIELD_CLOSURE T_onman → 0.00. **Fix:** frozen data anchor
  (rung 9) → T_onman 0.70.
- **Operator coherence [OPEN]:** distributional anchor (energy-distance/MMD) keeps T on-manifold but
  permits a random distribution-preserving shuffle (shift_std ≈ 95°), not a coherent flow. Symptom:
  T_onman high but recall ~0 and FROZEN ≈ VISIBLE.

## Interpretation key (which arm pattern means what)
- ORACLE fails → harness broken (stop, fix).
- BASE succeeds at held-out → split too easy / leakage (redesign).
- FIELD_CLOSURE fails but FROZEN_ANCHOR on-manifold → co-adaptation confirmed + fixed.
- FROZEN_ANCHOR on-manifold but recall ~0 and shift inconsistent → operator not a reusable direction.
- FROZEN_ANCHOR ≈ VISIBLE_ANCHOR → no real symmetry learned (no full-manifold advantage).

## Resolved
- Cluster submit path needs ABSOLUTE sbatch path + case-correct repo root (`desktop` vs `Desktop`).
- Cluster `tee` logs are block-buffered; set `PYTHONUNBUFFERED=1` to see live progress.
- Local CPU reproduces cluster (cuda) results identically for these toys.

## Open
- Cluster SSH session needs interactive 2FA re-auth (`scripts/cluster/ssh_bootstrap.sh`) for any
  GPU/scaled confirmation.
