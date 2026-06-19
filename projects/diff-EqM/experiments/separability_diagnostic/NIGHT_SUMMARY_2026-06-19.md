# Overnight research summary — morning of 2026-06-19

You asked (AFK): does metacognition generalize better than vanilla? improve tasks beyond
planning+image-gen? can inpainting be made positive? I set a goal (`NIGHT_PLAN_2026-06-18.md`),
ran 3 controlled threads, and got a result on each. All pushed.

## TL;DR — the scope boundary is now a *tested law*
The probe detects **instability in the EqM's relaxation**. That single mechanism explains all
three new results:
- It **transfers across tasks for DETECTION** (image, maze, CSP, inpaint-when-structural).
- **Restart-ACTION** pays off only where failures are *both* instability-type *and* stochastic.
- Because OOD shift manufactures exactly those failures, metacognition is an **OOD aid**.

## Thread 1 — Generalizes better than vanilla? YES (strong)
Maze-EqM, frozen in-dist (c7) probe applied across c5→c13 OOD at fixed NFE.
- vanilla degrades with OOD (0.81→0.37); **probe-restart advantage over vanilla GROWS
  monotonically with shift — corr 0.98** (+0.078 → +0.182).
- The **in-distribution-trained probe transfers to +6-tier OOD** (de-conf AUROC ~0.8 throughout).
- Near-ceiling control (s0): little gain — expected, nothing to rescue when the model rarely fails.
- → Metacognition is a generalization aid; the worse the shift, the more it buys. (`maze_eqm/OOD_SCALING_RESULTS.md`)

## Thread 2 — A third task type (Sudoku, constraint reasoning): DETECTION transfers, restart doesn't
- 9×9 too hard for a small GD-EqM (learns, loss 2.4→0.99, but never solves a full 81-cell grid —
  not a metacog failure, a capacity one).
- 4×4 works (solve 0.955). Metacog: **probe DETECTS constraint-violation from dynamics, AUROC
  0.84–0.90** — the signal transfers to a non-spatial CSP. **But best-of-R restart gives ~0 gain**
  because the oracle≈random too: CSP failures are deterministic/puzzle-intrinsic, restarts are
  correlated, so there's no lucky restart to select.
- → New distinction: **detection ≠ actionability**. Restart needs *stochastic* failure diversity.
  (`sudoku_eqm/SUDOKU_RESULTS.md`)

## Thread 3 — Can inpainting be made positive? YES, and it proves the mechanism
Dual oracle on MNIST RePaint, mask 0.3→0.9:
- **Classifier (semantic / confident-wrong) oracle: probe ≈ chance (0.56–0.61) at every mask** —
  reproduces the prior null.
- **Structural (instability / coherence) oracle: at extreme mask 0.90, AUROC 0.84 + restart +0.18**;
  random-mask structural AUROC rises monotonically with mask (corr 0.88).
- → Inpainting flips positive **exactly when extreme masking turns failures from confident-wrong
  into structural collapse (instability)**. The June-15 null and this positive are the *same law*
  at two mask sizes. (`mnist_eqm/INPAINT_POSITIVE_RESULT.md`)

## What it means for the paper
- Strengthens the central claim from "works on gen+planning" to a **mechanism with a stated,
  tested boundary**: detect+rescue instability failures; transfers across tasks; restart-action
  needs instability+stochasticity; aids OOD.
- Inpainting moves from "limitation" to "controlled confirmation."
- Sudoku adds a 3rd task for the *detection* claim and motivates a non-restart action (future work).

## Logistics / open
- SSH ControlMaster dropped ~23:00 (2FA expired). **No GPU jobs lost** — all GPU work (online 50k,
  maze GPU) had already finished. 4×4 sudoku + MNIST inpaint are tiny, ran on **local CPU**.
- To resume GPU work: `! scripts/cluster/ssh_bootstrap.sh` (2FA).
- Open, SSH-gated: 9×9 sudoku with a bigger arch / more inference steps; image-scale
  RePaint-on-IN-1K (predict structural-failure regime positive, same as MNIST); a non-restart
  action for CSP (spend extra compute on probe-flagged puzzles instead of restarting).
- All three night threads: committed + pushed (latest 2b33929). Nothing running now.
