# Capability ladder v2 — what behavior changed? (2026-06-07)

> **PURPOSE / REUSE (added 2026-06-08): this ladder is the shared CAPABILITY BENCHMARK
> for ANM vs SYMMETRY-DISCOVERY.** The v10/ANM numbers below are the BASELINE BAR.
> The new symmetry-discovery method (see symmetry project: morphism-gym / latent-symmetry
> line) must be run through the SAME Rungs A–F, SAME frozen-ckpt comparison protocol,
> SAME metrics/seeds/sampler, and compared head-to-head. Question: **does symmetry
> discovery beat ANM on these behavioral axes** (per-class gain localization, hard-class
> ratio, sample-efficiency D) — and/or does it light up the rungs ANM left NULL
> (C rescue, E splice-localization, F counterfactual steering)? A symmetry method that
> installs FAR-FROM-MANIFOLD structure could plausibly win exactly where ANM is null.
> To compare: swap the ANM ckpt for the symmetry-method ckpt as the second arm; keep
> vanilla-s0 as the shared control; reuse eval_capabilities.py + eval_trajectory.py + exp1.

Re-opens the ANM/v10 capability question after the v1 ladder NULL. v1 only
falsified **naive zero-shot clamped restoration** (denoise/inpaint/transfer).
v2 asks: does the −3.83 FID gain correspond to ANY real behavioral change?

Frozen checkpoints. Sampler / seeds / labels / VAE / EMA / eval held fixed.
Arms: vanilla seed0 vs ANM v10 λ0.3 seed0 (matches Exp 3). Pre-registered gates,
nulls reported honestly, no long random-control train unless a checkpoint signal
appears.

Gate to even proceed past A: a checkpoint-level signal beyond aggregate FID.
**PASSED at Rung A** (class-adherence + hard-class localization, below).

---

## Rung A — gain localization  [DONE, from Exp 3, single seed λ0.3]
Source: `results/exp3_metrics_out/` (49996 matched ids, parity-controlled).

| axis | Δ (v10 − vanilla) | reads as |
|---|---|---|
| FID | −4.38 (CIs disjoint) | gain real |
| KID | −0.0057 | confirms |
| precision | +0.023 | fidelity up |
| recall | +0.0008 | **diversity FLAT — no tax** |
| density | +0.044 | local density up |
| coverage | +0.072 | mode coverage up |
| conditional top-1 | +0.050 | **more on-class** |
| classifier TV→requested | −0.019 | better class balance |
| frac classes improved (feat-dist) | 0.913 | broad |

**Verdict: the FID gain is QUALITY + CLASS-ADHERENCE + COVERAGE, not mode-dropping.**
Better conditional top-1 = a behavioral change (images land in requested class more
often), independent of FID.

## Rung B — hard-class improvement  [DONE, from Exp 3, single seed λ0.3]
Classes ranked by vanilla difficulty (per-class feature distance; per-class FID is
50-sample-noisy and not used for ranking).

| quartile | Δfeat-dist | Δcond-top1 |
|---|---|---|
| EASY (lowest van feat-dist) | −0.417 | +0.033 |
| HARD (highest van feat-dist) | −0.557 | +0.062 |
| **HARD/EASY ratio** | **1.34×** | **~1.9×** |

Also: bottom-quartile weak-class FID −5.61 vs mean −4.38 (Exp 3). 91.3% classes
improve feat-dist; 72.9% improve top-1.

**Verdict: v10 preferentially fixes the model's weak regions** — the hard-example
mining mechanism prediction, confirmed at checkpoint level.

## Rung C — failed-generation rescue  [DONE — NULL] (job 19846918, n=64, v10 s1)
Bad = bottom-quartile vanilla classifier conf (16/64, conf≤0.048). Branch their
matched t=0.5 mid-latent under vanilla vs v10 continuation.
- vanilla_cont conf 0.0344 top5 0.375 ; v10_cont conf 0.0363 top5 0.375.
- v10 conf gain +0.0018, top5 identical. **v10 does NOT rescue failed trajectories.**
Mechanism: trajectory is committed by the midpoint — c(γ)→0 near the manifold means
the late field is too weak to repair structure set early, for BOTH arms.

## Rung E — trajectory swap  [DONE — NULL/weak] (job 19846918)
vanilla/v10 spliced early↔late at t∈{.25,.5,.75}, plus pure arms.
- pure_vanilla top1 0.422 / top5 0.719 ; pure_v10 0.438 / 0.719 (+0.016 top1, exp3
  direction but tiny at n=64; top5 flat).
- EVERY splice config ≈ the pure arms (top5 0.703–0.719, conf 0.113–0.118).
**v10's contribution is NOT localizable** to early global structure or late detail —
it is diffuse and small. No timestep where swapping in v10 helps measurably.

## Rung F — class-guided counterfactual edit  [DONE — INCONCLUSIVE] (job 19846918)
Switch class A→B at t∈{.25,.5,.75}, measure target-B success + source-A retention.
- target_B_top5 = 0.0 at EVERY switch, BOTH arms; source_A_top5 unchanged (~0.71);
  LPIPS-to-pure-A ~1e-4 (output essentially identical to no-switch).
**Label switch has ~no effect for either arm** → base EqM-B/2 is too weakly
class-conditional in this GD/cfg=1.0 regime (cond-top1 only 0.43) and c(γ) decay
kills late switches. A base-model ceiling, NOT a v10-vs-vanilla differentiator.
Counterfactual editing not supported by either checkpoint.

## Rung D — sampler robustness (NFE × step_mult)  [DONE — WEAK-POSITIVE] (job 19948994)
Full 96-cell grid: {gd,ngd} × nfe{5,10,25,50,100,250} × step_mult{0.5,1,1.5,2.0},
vanilla-s0 vs v10-s1 λ0.3, 5K samples/cell. (3 prior attempts failed: --export
comma-split truncated NFE list → 4 cells; then home03 100%-full exit-53; fixed both.)

- **No collapse anywhere:** 0 nan, 0 divergence across all 96 cells, both arms.
- **Converged regime (nfe≥100, step≥1.0):** v10 consistently −2.5 to −3.2 FID; holds
  the edge under step-size overshoot (step_mult 2.0).
- **Sample-efficiency (modest, real):** v10 @ nfe100 (FID 43.0–44.9) ≤ vanilla @
  nfe250 (45.3/45.1) for BOTH gd and ngd → v10 reaches vanilla's converged quality
  at ~2.5× fewer steps.
- **Starved budgets (nfe≤25, or nfe50–100 @ step0.5):** v10 ≈ vanilla or marginally
  WORSE (+0.5 to +2.6). NOT a brittleness fix; no advantage when the trajectory
  hasn't reached the data manifold (FID 220–392, both garbage at nfe≤25).

**Verdict: v10 is more sample-efficient and overshoot-tolerant in the near-converged
regime, but NOT more robust at starved budgets.** The gain lives where the trajectory
arrives at the data manifold — nothing gained far from it.

## FINAL SYNTHESIS (A–F)
The v10 FID gain (−3.83 at full budget) corresponds to a REAL behavioral change, but
a specific and bounded one:

**What changed (positive):**
- A: gain = quality + class-adherence + mode-coverage, diversity FLAT, 91% classes.
- B: concentrated on HARD classes (1.34× feat-dist, ~1.9× class-adherence vs easy).
- D: ~2.5× more sample-efficient + overshoot-tolerant in the converged regime.

**What did NOT change (null):**
- C: no failed-trajectory rescue (+0.0018 conf on bad mid-states).
- E: contribution not splice-localizable to early-structure or late-detail.
- F: no counterfactual class-steerability (base model too weakly conditional; inert
  both arms — uninformative, not a true negative).

**Unifying mechanism:** v10 (PGD hard-example mining on the EqM target) **sharpens the
velocity field near the data manifold**, preferentially in the model's weak (hard-class)
regions. This buys better final-sample quality, broad per-class gains, and modest
sample-efficiency — all "near-manifold" effects. It installs NO new far-from-manifold
behavior: no repair of broken states (v1 null + C), no steering (F), no early-trajectory
restructuring (E). The c(γ)→0 decay near the manifold is consistent: mining acts where
the field is still informative, not in the near-zero-field late/edit regime.

**Paper framing (unchanged, now better-evidenced):** v10 = "adaptive hard-negative
mining that improves regression-target generative quality, concentrated on hard classes,
with sample-efficiency gains." Do NOT claim conditional-editing / repair / robustness-
to-low-NFE capabilities — A–F bound those out.

Caveats: A/B single-seed λ0.3 seed0 (cached; ckpt since pruned); C–F single-seed n=64
on λ0.3 seed1 (seed0 ckpt gone); D single-seed seed1. 3-seed (27.58±0.36 exists)
would tighten but the broad per-class signal (91%/73%) and full-grid consistency make
sign-flips unlikely.

(Inpainting/outpainting/translation revisited ONLY after A–F.)

## Caveats
- Rung A/B = single seed λ0.3. 3-seed (27.58±0.36 exists) would tighten; per-class
  signal is broad (91%/73%) so unlikely to vanish.
- Per-class FID noisy at 50 samples/class → ranking uses feature distance.
