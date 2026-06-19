# Maze-EqM GPU scale-up — metacognition results (2026-06-18)

Wider EqM (width 128, ~2.4M params, vs CPU 64-wide/653K), trained **natively on c7
(15×15)** mazes on GPU, metacognition evaluated on **c10 (21×21) OOD**. Purpose: tighten
the CPU maze rung at a larger model + native harder-tier training.

## Training stability (honest)
width-128 / 80-epoch training is **unstable across seeds**:

| seed | final loss | in-dist c7 valid | train gate |
|---|---|---|---|
| s0 | 0.073 | 0.975 | PASS |
| s1 | 2.065 | 0.68 | PASS |
| s2 | 20.42 | 0.000 | **FAIL (diverged)** |

s2 diverged (loss blew up) — a training-recipe issue (LR too high for the wider model,
no warmup/grad-clip), **not** a metacognition issue. Excluded. Fix (warmup/clip) deferred:
the CPU multi-seed maze result is the primary evidence; this GPU run is confirmatory.

## Sampler-budget note (root-caused a false-negative)
The first GPU metacog run used `eta=0.01, steps=25` (descent budget 0.25) — **below the
threshold** where these models reach the manifold, so vanilla c10-valid ≈ 0 (no invalid-
band, oracle also ≈0). `maze_sampler_probe.py` swept the budget: at `eta≥0.02` the
trainable models solve c10 (s0 0.76–0.84, s1 0.31–0.61). Real metacognition needs the
0.3–0.7 band; results below use it.

## Metacognition @ proper budget (R=4, equal NFE, exact BFS labels, n=800)

| seed | budget | invalid | AUROC (de-conf) | vanilla | random | **probe** | oracle | Δ(probe−rand) | %oracle |
|---|---|---|---|---|---|---|---|---|---|
| **s1** | steps80 η0.02 | 0.42 | **0.872** | 0.583 | 0.584 | **0.759** | 0.785 | **+0.175** | **87%** |
| s0 | steps25 η0.02 | 0.25 | 0.645 | 0.756 | 0.753 | 0.775 | 0.846 | +0.022 | 24% |

- **Both seeds: probe-restart > random at equal NFE.**
- **s1 (in the right difficulty band) is the strongest maze result to date:** de-confounded
  AUROC **0.872** (> CPU 0.67–0.76), Δ **+0.175** valid-rate, recovering **87%** of the
  oracle gain — at a 3.7× wider, GPU-trained EqM. Confirms the mechanism scales.
- s0 sits near the solve-ceiling (vanilla 0.76, few failures), so there is little to
  rescue — small but positive Δ, as expected near ceiling.

## Bottom line
The GPU scale-up **confirms and tightens** the maze rung: a wider, GPU-trained EqM in the
right difficulty band gives a cleaner, stronger probe>random result (AUROC 0.872, +0.175,
87% oracle) than the CPU model. Caveats: only 2/3 seeds were trainable (width-128
instability — recipe fix deferred), and the strongest number is a single in-band seed, so
this **corroborates** the CPU multi-seed result (+0.174±0.034, 5 seeds × 2 tiers), it does
not replace it.
