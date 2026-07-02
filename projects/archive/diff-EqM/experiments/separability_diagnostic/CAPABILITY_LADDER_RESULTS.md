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
signal into a real capability gain under a strict equal-compute control.

### D′. SAME task on a REAL trained EqM — RUN ✅ (`experiments/maze_eqm/`)
The toy is now superseded by a genuine **trained conditional EqM** that solves grid
mazes (maze layout → shortest-path grid; 653K params, faithful EqM target + GD
sampler). It solves c5 in-dist at **0.99 valid** and generalizes to 4× larger OOD
mazes (0.88–0.93). Trajectory-metacognition on it, exact BFS labels, equal NFE:

| tier | invalid | probe AUROC | vanilla | random | **probe** | oracle | Δ | %oracle |
|---|---|---|---|---|---|---|---|---|---|
| c7 (15²) | 0.78 | 0.76 | 0.21 | 0.22 | **0.44** | 0.59 | +0.22 | 59% |
| c10 (21², 2 seeds) | 0.40 | 0.67–0.68 | 0.61 | 0.58–0.62 | **0.71–0.72** | 0.83 | +0.10–0.13 | 46–52% |

**PROBE>RANDOM on all 3 runs.** Harder tier → higher AUROC + bigger rescue. This is
the EqM-native planning result Yilun asked for: metacognition rescues a real trained
EqM planner under exact-label, equal-compute controls. Detail: `maze_eqm/STEP3_RESULTS.md`.

### D″. GPU scale-up (3.7× wider EqM, native c7 train) — RUN ✅ (`maze_eqm/STEP3_GPU_RESULTS.md`)
Wider EqM (width 128, ~2.4M params), GPU-trained natively on c7, metacog on c10 OOD:

| seed | invalid | AUROC (de-conf) | vanilla | random | **probe** | oracle | Δ | %oracle |
|---|---|---|---|---|---|---|---|---|
| **s1** (in-band) | 0.42 | **0.872** | 0.583 | 0.584 | **0.759** | 0.785 | **+0.175** | **87%** |
| s0 (near-ceiling) | 0.25 | 0.645 | 0.756 | 0.753 | 0.775 | 0.846 | +0.022 | 24% |

Both probe>random. The in-band seed gives the **strongest maze result to date** (AUROC
0.872 > CPU 0.67–0.76; +0.175; 87% oracle) — the mechanism tightens at a wider, GPU-
trained EqM. Caveats: width-128/80ep training was **unstable** (s2 diverged, loss 20 —
recipe issue, excluded), and the strong number is one in-band seed, so this **corroborates**
the CPU multi-seed (+0.174±0.034), not replaces it. Root-caused a false-negative first run
(sampler budget 0.25 < manifold threshold; fixed at η≥0.02 via `maze_sampler_probe.py`).

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

## B. Inpainting — **RUN on MNIST: weak/null (scope boundary)** ⚠️

Real MNIST + standard RePaint clamp masks + classifier-consistency oracle, small
trained EqM (`experiments/mnist_eqm/`). 3 masks × 3 seeds: probe AUROC ~0.60 (barely
above chance), probe−random gap **+0.015±0.011** (pooled) — effectively null, far below
maze (+0.17). Both full-field and masked-region dynamics fail.

**Finding (not a tuning failure):** trajectory-metacognition keys on descent
*instability*. Inpainting failures are *confidently wrong* — with most pixels clamped
the EqM descends cleanly to a plausible-but-wrong digit (masked 4→9); the dynamics look
healthy, only the identity is wrong, so the probe can't grip. The method rescues
collapse/broken-structure failures (generation, planning), NOT confident-semantic-error
failures (ambiguous inpainting). Detail: `mnist_eqm/MNIST_RESULTS.md`. Image-scale
RePaint-on-IN-1K-EqM (`REPAINT_EQM_SPEC.md`, June-18 cluster) will confirm; predicted weak.

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
