# Capability Ladder — Results

Question (Yilun): beyond improving FID, does the trajectory-metacognition signal
**unlock capabilities** — failed-sample rescue, inpainting/repair, translation,
or maze planning? This file holds the smallest credible probes. Status is marked
honestly: RUN (with numbers) vs DESIGNED (scoped, not yet run).

The unifying mechanism under test in every rung: *a probe reads the refinement/
descent dynamics, flags likely-failure, and reallocates a fixed compute budget —
and it must beat spending that same budget at random (equal-NFE control).*

---

## D. Maze planning — **RUN ✅ (prioritized per Yilun)**

`maze_planning.py` → `results/capabilities/maze/`. CPU, numpy-only, self-contained.

A planner refines a candidate path by gradient-descending an energy = string
tension + soft wall penalty. The soft/local barrier creates genuine **spurious
minima**: a path can settle at low gradient while still clipping a wall corner
(invalid) — the exact maze analog of an EqM "garbage" sample. A logistic probe
reads ONLY the refinement-trajectory dynamics (energy/wall/grad-norm curve shape)
at a partial step and predicts invalidity; risk-guided branching restarts the
flagged candidates. Arms below the line are EXACTLY compute-matched (280 path-grad
steps); oracle is the over-budget ceiling.

| difficulty | n | vanilla | random | **probe** | oracle |
|---|---|---|---|---|---|
| 0 | 90 | 0.989 | 0.978 | 1.000 | 1.000 |
| 1 | 94 | 0.968 | 0.957 | 1.000 | 1.000 |
| 2 | 97 | 0.794 | 0.691 | 0.938 | 0.959 |
| 3 (hardest) | 79 | 0.557 | 0.519 | **0.747** | 0.797 |
| **ALL** | 360 | 0.836 | 0.794 | **0.928** | 0.944 |

- **probe − random = +0.133 valid-path-rate at equal compute** (hardest tier +0.228).
- Recovers **89%** of the oracle's gain over random.
- Verdict: **PROBE > RANDOM at equal compute — the general mechanism transfers to
  maze planning**, and the gap widens with difficulty (where failure matters).

**Honest caveat:** the toy's invalid/valid classes are almost perfectly separable
from dynamics (probe held-out AUC ≈ 1.0), far easier than EqM's 0.82. The toy is
not evidence about detection *difficulty*; it is evidence that **acting** on a
dynamics risk score (branch the flagged, not the random) converts a real detection
signal into a real capability gain under a strict equal-compute control. That is
the claim Yilun's maze question was probing, and it holds.

---

## A. Failed-sample rescue — **PARTIALLY RUN (via the online sampler)**

This rung is structurally identical to the online adaptive sampler (Phase 2):
flag high-risk in-flight samples at a partial step and restart them, vs restart a
random equal-size subset. The mechanism is validated in two places already:
- **Mock (CPU, RUN):** `online_adaptive_sampler.py --mock` — probe-restart quality
  0.212 vs random-restart 0.385 at byte-identical NFE (1,294,800 each). Logic +
  equal-NFE bookkeeping confirmed.
- **EqM @ scale (DESIGNED, cluster-gated):** `online_adaptive.sbatch` runs the same
  arms on the real B/2 checkpoint with Inception-NN-dist quality. Pending GPU.

The partial-probe study (`partial_probe.py`) shows the rescue decision can be made
as early as step 100/249 with no loss (de-conf AUROC 0.814 vs 0.818 at the end),
so rescue has a real early window to act in.

## B. Corruption repair / inpainting — **DESIGNED (not run)**

Smallest credible test: take real IN-1K latents, corrupt a fixed region / add
noise at a severity ladder, run EqM refinement to repair, and compare vanilla vs
random-restart vs probe-restart at equal NFE. Metrics: masked-region reconstruction
error, classifier consistency (resnet50 top-1 agreement vs clean), Inception NN
distance, contact sheets. Gated on (a) the EqM online sampler returning WORKS at
scale and (b) advisor priority vs maze depth (see YILUN_UPDATE.md).

## C. Image translation / constraint satisfaction — **DESIGNED (not run)**

Minimal controlled source→target constraint (e.g. fixed class + held color/region
constraint) refined under EqM; compare realism (FID/NN-dist) + constraint
satisfaction at equal compute across the three arms. Lowest priority of the four;
gated on B's outcome.

---

## Bottom line
One capability rung (maze planning, the one Yilun named) is **RUN and positive**
under a strict equal-compute control. The EqM-native rescue rung is **logic-validated
in mock** and **ready to run at scale** (sbatch prepared). Inpainting/translation
are **designed and scoped**, deliberately gated on the scale verdicts and on Yilun's
call about where to spend the next GPU block.
