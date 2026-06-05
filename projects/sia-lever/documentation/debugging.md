# Debugging — SIA-Lever

## Active Issues
(none yet)

## Resolved
- **Muddy shortcut trap (Phase 0)**: unit-vector shortcut leak left stage-1 neg_control ~0.23
  (not clearly "cheating"); model partially learned rotation. Fix: leak the FULL target vector y
  (dir+magnitude) → prediction-only cheater solves the broken control (neg_control ~0.06), crisp
  shortcut_win. `data.py` shortcut = y.clone().
- **bad_verifier oracle picked H_THEN_W not H (Phase 3)**: an already-honest model still gains
  ~0.05 true-score from extra structural training, so HW edged H. Fixes: (1) train mode-C model
  near ceiling (1600 steps); (2) add documented W_COST=0.05 — an unnecessary weight update is not
  free (compute + regression risk; the SIA thesis). Then H ≈ best for mode C, selector regret ~0.
  W_COST is a transparent additive per-W-retrain constant, NOT a transition table — all true-scores
  still come from real reruns.

---
## Toy-design risks (watch these — they kill the phenomenon)
- [ ] **Neg-control not trapping**: if prediction-only model ALSO gets high neg_control_mse,
      the shortcut isn't strong enough / leak too weak. Phase-0 gate fails. Fix: make shortcut
      channel the dominant signal (cheaper to read than computing rotation).
- [ ] **Stage-4 doesn't recover**: structural objective too weak (lam_short/lam_comp too low) or
      too strong (clean_mse blows up). Tune lam_short, lam_comp. Want clean low AND neg-ctrl high.
- [ ] **Equivariance metric ill-posed**: encode-of-rotated-point comparison is a proxy; if noisy,
      lean on composition_error + neg_control + shortcut_sensitivity as primary tells.
- [ ] **Circular-eval trap (the cardinal sin)**: never score a stage against a hardcoded expected
      table. Every number must come from a real forward pass on real data. Stages 2 vs 4 must be
      real reruns of the SAME pipeline, only the objective/harness differing.

## Common failure modes (preemptive)
- [ ] Import path (run from project root or experiments/; modules import siblings by name)
- [ ] Determinism: make_batch seeds per-call; verifier uses fixed seeds for comparable metrics
- [ ] CPU slowness: keep steps ~2000, bs ~512; full episode should be < ~1 min
