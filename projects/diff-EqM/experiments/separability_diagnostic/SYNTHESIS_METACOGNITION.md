# Trajectory-Metacognition for EqM — cross-task synthesis + positioning

One narrative tying the whole line of work, for the paper. Updated 2026-06-15.

## Thesis
An Equilibrium-Matching model's *energy scalar does not predict whether a sample is
good or garbage* — but the **shape of its gradient-descent trajectory does**, and acting
on that signal at inference (restart the likely-failures) improves generation **at equal
compute**, across image generation, planning, and inpainting, **without retraining**.

## Evidence chain (each link controlled: positive + negative + de-confound)

| # | claim | testbed | result | status |
|---|---|---|---|---|
| 1 | endpoint energy predicts quality | IN-1K EqM-B/2 | dot/path-integral 0.61 de-conf, **below** a trivial geometry baseline 0.627 | **DEAD** |
| 2 | descent **dynamics** predict failure | IN-1K EqM-B/2 | learned shape probe **0.82** held-out, de-confounded from grad-norm (5 seeds) | SUPPORTED |
| 3 | acting improves FID (consistent) | IN-1K EqM-B/2 | best-of-R restart, **50k × 3 seeds Δ1.87±0.11 FID** (CI excl 0) | SUPPORTED |
| 4 | online equal-NFE sampler beats random | IN-1K EqM-B/2 | restart probe-flagged mid-flight: **26.90 < random 27.9–28.0 @ equal NFE, 50k, 3 control-draws** (Δ~1.05, all draws; 15k pilot 28.51<29.76) | SUPPORTED |
| 5 | mechanism transfers — **planning** | trained maze-EqM | probe-restart > random, CPU 5 seeds × 2 tiers **+0.174±0.033** (CI excl 0); GPU-scaled (3.7× wider) in-band seed **AUROC 0.872, Δ+0.175, 87% oracle** | SUPPORTED |
| 6 | mechanism transfers — **inpainting** | MNIST EqM (RePaint) | confident-wrong failures: probe ~chance (0.60). **Extreme-mask structural (instability) failures: AUROC 0.84, restart +0.18** | SCOPE-CONFIRMING |
| 8 | **detection vs action** — constraint reasoning | Sudoku-EqM 4×4 (CSP) | probe DETECTS constraint-violation (AUROC **0.84–0.90**) but restart can't rescue (oracle≈random — deterministic/correlated failures) | DETECTION TRANSFERS |

| 7 | aids **OOD generalization** | trained maze-EqM | frozen in-dist probe across c5→c13 OOD: probe−vanilla advantage **grows monotonically with shift, corr 0.98** (+0.078→+0.182); probe transfers (AUROC ~0.8 at +6 tiers) | SUPPORTED |

The signal is the **same probe over the same descent-shape features** (oscillation, log-
decay slope, magnitude-normalized norm/dot curves) in every testbed. Exact labels where
available (BFS for maze, classifier for MNIST) corroborate the noisy image-FID labels.

## The scope boundary (what #6 taught us — a feature, not a bug)
Trajectory-metacognition keys on **descent INSTABILITY**. It rescues failures that *show
up in the relaxation* — image-generation collapse, broken/disconnected maze paths (genuine
spurious minima with disturbed descent). It does NOT rescue **confidently-wrong** failures:
MNIST inpainting with most pixels clamped descends *cleanly* to a plausible-but-wrong digit
(masked 4 → 9); the dynamics look healthy, only the identity is wrong, so the probe has
nothing to grip (AUROC ~0.60, negligible gap; full & masked dynamics both fail). This is a
real limit that sharpens the claim: **the method detects instability/collapse/broken-
structure failures, not confident-semantic-error failures.**

**The boundary is now a tested law, not just an observation (2026-06-18 night runs):**
- **Inpainting is positive when failures are instability-type.** Pushing the MNIST mask to 0.90
  forces the model to hallucinate whole digits; its failures become *structural collapse*
  (broken/incoherent completions). A structural-coherence oracle then gives AUROC **0.84** and
  restart **+0.18** — while the *classifier* (semantic) oracle stays at chance (0.56–0.61) at
  every mask. Same probe; the only thing that changed is whether the failure perturbs the
  descent. (`mnist_eqm/INPAINT_POSITIVE_RESULT.md`)
- **Detection ≠ actionability (Sudoku 4×4 CSP).** The probe *detects* constraint-violation from
  dynamics (AUROC 0.84–0.90 — the signal transfers to a third task type), but best-of-R restart
  can't rescue because CSP failures are deterministic/puzzle-intrinsic (oracle≈random; restarts
  correlated). Restart needs *stochastic* failure diversity (present in gen & maze, absent in
  CSP). (`sudoku_eqm/SUDOKU_RESULTS.md`)
- **Aids OOD generalization** (link #7): on a model with headroom the restart advantage grows
  monotonically with distribution shift (corr 0.98), frozen in-dist probe transfers to +6 OOD.

Net: the signal is **instability in the relaxation**; it transfers across tasks for DETECTION,
and restart-ACTION pays off wherever failures are (a) instability-type and (b) stochastic.

## What's load-bearing methodologically
- **De-confounding from gradient/score norm** (matched-norm-bin AUROC). Without it, a
  "quality signal" is often just magnitude-in-disguise. This is how we can claim the
  signal is the *shape*, not the energy — and it's what kills the energy-scalar story (#1).
- **Equal-NFE controls.** Every treatment is bracketed by a random arm at *identical*
  compute and an oracle ceiling. Probe gains are read only inside that band.
- **Exact-label corroboration.** Maze (BFS) and MNIST (classifier) give uncontaminated
  labels, so the probe AUROC isn't an artifact of noisy image-quality oracles.

## Positioning vs concurrent work
- **CFG-Rejection — "Diffusion Sampling Path Tells More"** (arXiv:2505.23343, May 2025):
  also links trajectory to quality, but the signal is *Accumulated Score Differences*
  (conditional−unconditional score divergence) — **requires CFG**; undefined for our EqM
  (cfg=1.0, no guidance). Post-hoc filtering only; images only; no norm de-confounding;
  no energy-scalar-failure analysis. → independent support for "trajectory→quality", but
  our EqM/energy-model setting, de-confounded shape signal, online equal-NFE restart, and
  planning/inpainting transfer are all distinct. Cite as concurrent; differentiate.
- **Geometric Regularity in Deterministic Sampling Dynamics** (arXiv:2506.10177, Jun 2025):
  characterizes trajectory regularity of diffusion sampling; descriptive, not a quality
  predictor / inference-time intervention. Background, not a threat.
- General inference-time-scaling / best-of-N: known that early reward scores correlate
  poorly with final quality — our **partial-probe (0.814 de-conf at step 100/249)** is a
  direct positive answer to that gap, in the EqM setting.

## Honest scope (what we do NOT claim)
- Energy is dead *as a quality axis* — it remains a valid density/OOD signal.
- Image results are EqM-B/2 (one checkpoint); maze + MNIST are small EqMs we trained
  (real models, not the image checkpoint). Claim = "metacognition works on real trained
  EqMs across tasks", not "the image EqM plans/inpaints".
- Probe AUROC is task-dependent (0.82 image / 0.67–0.76 maze / MNIST pending); the
  *action* (probe-restart > random at equal NFE) is what transfers, and it does.

## Paper shape (workshop → ICLR)
1. EqM energy ≠ quality (the negative that motivates everything).
2. Descent-shape probe (de-confounded) — the positive detection result.
3. Acting on it: best-of-R (consistent 50k) + online equal-NFE restart.
4. It's a *mechanism*, not an image trick: planning (maze) + inpainting (MNIST/RePaint).
5. Concurrent CFG-Rejection differs (CFG-specific, post-hoc, images). Limitations + scope.
