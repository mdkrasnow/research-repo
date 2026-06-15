# MNIST inpainting rung — RESULT: weak/null (an honest scope boundary)

Reuses real MNIST + standard RePaint-style clamp masks + a classifier-consistency
oracle (an inpaint is valid iff the completed digit still classifies as the true
label). Small unconditional EqM (651K, faithful EqM target, loss 2.66), classifier
acc 0.967. Multi-seed metacognition sweep (3 masks × 3 seeds, R=4, n=400):

| mask | invalid | probe AUROC (de-conf, mean±CI) | probe−random gap (mean±CI) | all seeds + |
|---|---|---|---|---|
| center 0.40 | 0.33 | 0.600 ± 0.019 | +0.010 ± 0.025 | no (CI straddles 0) |
| half 0.55 | 0.28 | 0.629 ± 0.010 | +0.005 ± 0.007 | no |
| center 0.55 | 0.64 | 0.572 ± 0.007 | +0.031 ± 0.014 | yes (barely) |

Pooled gap **+0.015 ± 0.011** over 9 runs (positive 7/9). VERDICT: **MIXED / weak —
effectively null.** The probe is barely above chance (AUROC ~0.60) and the restart
gain is negligible (~+0.015 valid-rate) — far below the maze rung (AUROC 0.67–0.76,
gap +0.17). Full-field vs masked-region dynamics both tried; neither helps (full was
−0.045 on center0.55).

## Why — the scope boundary (the actual finding)
Trajectory-metacognition keys on **descent INSTABILITY** — the probe flags samples whose
relaxation is rough/oscillatory/struggling. That cleanly catches:
- image generation collapse (garbage samples relax differently), and
- maze planning failures (a broken/disconnected path is a genuine spurious minimum
  with disturbed descent).

It does NOT catch MNIST inpainting failures, because those are **confidently wrong**, not
struggling: with most pixels clamped, the EqM descends *cleanly* to a plausible-but-wrong
digit (e.g. a masked 4 completed as a 9). The descent looks healthy; only the *identity*
is wrong. A dynamics probe has nothing to grip. This is a real limit, not a tuning failure.

## Honest takeaways
- The metacognition mechanism is **failure-mode-specific**: it rescues *instability/
  collapse/broken-structure* failures (generation, planning), NOT *confident-semantic-error*
  failures (ambiguous inpainting). This sharpens the paper's claim rather than weakening it.
- Caveat: the MNIST EqM is small/mediocre (loss 2.66); a stronger inpainting model *might*
  surface more instability-type failures. The image-scale RePaint-on-IN-1K-EqM test
  (`separability_diagnostic/REPAINT_EQM_SPEC.md`, June-18 cluster) will check whether a
  strong EqM's inpainting failures are any more dynamics-detectable. Prediction: still weak,
  for the same reason.
- Code: `mnist_eqm.py` (EqM+classifier), `mnist_inpaint.py` (RePaint + metacognition,
  `--dyn-region full|masked`), `mnist_sweep.py`. Data/models gitignored (reproducible).
