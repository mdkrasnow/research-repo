# Maze-EqM — metacognition aids OOD generalization (2026-06-18, job 23234577)

**Question (user):** does trajectory-metacognition help *more* as you go OOD — does
probe-restart's advantage over vanilla grow with distribution shift, and does an
in-distribution-trained probe still transfer to harder OOD?

**Design.** One c7-trained maze-EqM. Eval on tiers c5…c13 (increasing grid size =
increasing OOD). **Fixed descent budget** (steps 80, η0.02) across all tiers → equal NFE
everywhere; vanilla degrades naturally. The metacognition probe is trained ONCE on the
in-dist tier (c7) and **frozen**, then applied to every tier (tests transfer, not refit).
Arms equal-NFE, exact BFS labels: vanilla / random-restart (neg) / probe-restart / oracle.

## Result — s1 model (vanilla spans 0.80→0.37, the informative regime)

| tier | OOD dist | invalid | frozen-probe AUROC (de-conf) | vanilla | random | **probe** | oracle | Δ(probe−vanilla) | Δ(probe−random) |
|---|---|---|---|---|---|---|---|---|---|
| c5 | −2 | 0.20 | 0.788 | 0.805 | 0.790 | 0.883 | 0.927 | +0.078 | +0.093 |
| c7 | 0 (in-dist) | 0.29 | 0.792 | 0.727 | 0.682 | 0.813 | 0.862 | +0.087 | +0.132 |
| c9 | +2 | 0.41 | 0.808 | 0.593 | 0.590 | 0.728 | 0.790 | +0.135 | +0.138 |
| c11 | +4 | 0.49 | 0.807 | 0.507 | 0.500 | 0.672 | 0.727 | +0.165 | +0.172 |
| c13 | +6 | 0.62 | 0.792 | 0.365 | 0.397 | 0.547 | 0.615 | +0.182 | +0.150 |

- **H1 ✓ vanilla degrades with OOD** (0.805 → 0.365): the task genuinely gets harder.
- **H2 ✓ the probe-restart advantage GROWS with OOD**, monotonically — corr(OOD, Δprobe−vanilla)
  = **0.98** (+0.078 in-easier → **+0.182** at +6 OOD). The further OOD, the more metacognition buys.
- **H3 ✓ the frozen in-dist (c7) probe transfers to far-OOD** — de-confounded AUROC holds at
  **~0.79–0.81 across every tier**, including c13 (6 cells larger than training). The failure-
  signature in the descent dynamics is distribution-shift-invariant.
- probe-restart > random at **every** tier.

## Control — s0 model (near solve-ceiling, vanilla 0.998→0.730)
Vanilla barely fails even OOD, so there is almost nothing to rescue: Δ tiny (+0.002…+0.010),
advantage can't grow (corr −0.87). The frozen probe's AUROC also fades OOD (1.0→0.57). This is
the expected boundary: **the gain scales with the failure rate**, which is ~0 for a near-perfect
model. Metacognition pays off precisely when the model is operating below ceiling — which is what
OOD shift induces.

## Bottom line
**Trajectory-metacognition is an OOD-generalization aid.** On a model with headroom (s1), the
advantage over vanilla rises monotonically with distribution shift (corr 0.98), and a probe
trained only on in-distribution data still catches failures far OOD (AUROC ~0.8 at +6 tiers).
The mechanism: OOD shift pushes more solves into spurious minima (instability), which the
descent-shape probe detects and restarts. Caveat: clean on the headroom model; a near-ceiling
model (s0) has little to gain — the effect is real but conditional on the model actually failing.
