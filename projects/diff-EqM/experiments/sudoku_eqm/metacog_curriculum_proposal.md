# Variant proposal — v11 "Metacognitive Curriculum" (probe-guided hard-example mining for EqM training)

Status: PROPOSAL (mechanism-first, no code yet — per CLAUDE.md "prove it deserves to exist
before code"). Drafted 2026-06-19 from a user idea. First testbed = real Sudoku (Stage 3+ of
REAL_SUDOKU_PLAN.md), because it has exact labels and is where inference-restart provably can't
fix failures (the night's "detection≠actionability" gap).

## One-line idea
Periodically run the current model's GD sampling on already-seen training examples, flag the
ones it currently FAILS on, and upweight those in continued training — a self-generated
curriculum of the model's own failure modes, with the **metacognition probe as the selector**.
Turns the probe from an *inference* tool into a *training* signal.

---

## Variant Proposal Template (mandatory)

**Variant name:** v11 metacognitive-curriculum (mc-curriculum).

**Hypothesis:** Reweighting the base EqM loss toward examples whose *descent dynamics are
currently unstable* (probe-flagged failures) reshapes the field where it actually breaks,
improving solve-rate/FID more than uniform training at equal compute — and, on labeled tasks,
probe-mining ≈ true-failure mining (so the method generalizes to the no-label image case).

**Failure mode addressed:** Deterministic, on-distribution failures that inference-time restart
CANNOT fix (night finding: Sudoku failures are detectable but puzzle-intrinsic; oracle≈random).
The correct action for those is *training on them*, not restarting. Also: instability-type
failures in image gen that uniform training under-samples.

**EqM compatibility argument:** Mining = the **base EqM loss** `MSE(f(x_γ), target)` evaluated on
*real seen training examples*, only **reweighted** by a per-example weight w_i ∈ [1, 1+λ]. No new
loss geometry, no auxiliary objective fighting c(γ). This is the CLAUDE.md "preferred low-risk"
class (aux loss = base loss on a mined input; provably bounded by the base loss since it IS the
base loss). Contrast with high-risk losses (cosine/hinge/Jacobian) — none of that here.

**Loss definition:**
  L = Σ_i w_i · ||f(x_{γ,i}) − target_i||²,  with
  w_i = 1 + λ · s_i,  s_i ∈ [0,1] = current failure score of example i,
  refreshed every K epochs by a mining pass (below). λ small (≤1) so mining never dominates the
  uniform base loss (CLAUDE.md aux/base-ratio rule).

**Mining pass (every K epochs):** for a pool P of seen examples, run GD sampling (the inference
sampler), then set s_i by one of two selectors:
  - **Oracle selector (A):** s_i = 1 if the sampled output is wrong (exact label: BFS/board-check/
    classifier), else 0. Strong baseline = classic hard-example mining.
  - **Probe selector (B):** s_i = P(unstable) from the trajectory-shape metacognition probe over
    the mining-pass descent (no label needed). The novel arm — targets *instability* failures.
  Soft option: s_i = probe score directly (continuous), to avoid hard 0/1 thresholding.

**Expected diagnostics if working:**
  - solve-rate / FID at equal total compute: mc-curriculum > uniform, and > random-reweight control.
  - the mined-set hardness DROPS over rounds (model fixes its failures → fewer flagged next pass).
  - probe-selector (B) tracks oracle-selector (A): corr(s_B, s_A) rises; B's solve-rate ≈ A's.
  - field diagnostics on previously-failed examples: descent stabilizes (lower oscillation, the
    probe's own features move toward the "good" regime).

**Expected diagnostics if failing:**
  - mined set keeps growing / hardness flat → model can't fix them (unsolvable noise; cap or drop).
  - catastrophic forgetting: solve-rate on EASY examples drops while hard improves (net wash) →
    mixing ratio wrong; increase uniform fraction.
  - B ≠ A (probe-mining picks different examples than true failures) AND B underperforms → the
    probe isn't a useful training selector (still fine as an inference tool; kills only the v11 claim).
  - aux/base weight ratio explodes (λ too big) → mining dominates, base task degrades.

**Minimal test (Sudoku, exact labels — ideal first testbed):**
  1. Train SudokuEqM to a *mediocre* checkpoint (board-acc in a workable band, NOT saturated).
  2. Continue training 3 arms at EQUAL added compute:
       - uniform (negative/floor): keep training, no mining.
       - random-reweight (negative control): upweight a random equal-size subset.
       - oracle-mine (positive/ceiling): upweight true-failed puzzles.
       - probe-mine (TREATMENT): upweight probe-flagged puzzles.
  3. Read board-acc gain on a held-out test set, in the band between the controls.
  Cheap (Sudoku trains in minutes), exact labels, and directly attacks the restart-can't-fix gap.

**Promotion rule:** probe-mine board-acc gain ≥ random-reweight + a real margin AND probe-mine
≈ oracle-mine (within noise) on ≥2 seeds → promote; then test on image gen (no-label, the real
payoff: hard-mining without labels) and report as v11.

**Kill rule:** probe-mine ≤ random-reweight (mining the probe's picks is no better than random),
OR oracle-mine itself doesn't beat uniform (then the *whole* hard-mining premise is dead here,
independent of the probe). Max 1 retune (λ, K, mixing ratio) per CLAUDE.md.

---

## Controls (mandatory, per CLAUDE.md positive+negative)
- **Negative (floor):** uniform training (no mining) AND random-reweight (mine random, not failures).
  Two floors: uniform isolates "does reweighting help at all"; random isolates "is it the FAILURE
  selection or just upweighting *something*".
- **Positive (ceiling):** oracle-mine (true failures). If oracle-mine doesn't beat the floors, the
  task/harness can't show mining value — fix that before reading the probe arm.
- **Treatment:** probe-mine. Read ONLY in the [random, oracle] band.

## Why this is more than classic hard-example mining
1. **Selector = trajectory-metacognition probe**, unique to this project — and it targets a SPECIFIC
   failure type (descent instability), not raw loss. The claim is "mine the *dynamically-fixable*
   failures."
2. **No-label generalization:** if probe-mine ≈ oracle-mine on labeled tasks, the probe enables
   hard-mining on UNLABELED generation (image FID), where you can't check correctness — that's the
   real contribution, not the mining mechanism itself.
3. **Operationalizes the night's law:** deterministic failures that inference-restart can't fix
   ARE fixable by training on them. v11 is the "right action" for that failure class.
4. **Relation to v10:** v10 mines *synthetic* hard negatives via PGD input perturbation (off-manifold).
   v11 mines *real on-distribution* seen examples by failure. Complementary; could combine.

## Risks / open questions
- Forgetting → reweight (mix), never replace; tune uniform:hard ratio. Track easy-set acc as a guard.
- Unsolvable examples dominating → cap s_i contribution; use probe to keep *instability* (fixable)
  over confident-wrong (maybe not) — a place the probe's failure-type specificity earns its keep.
- Cost of the mining pass (full sampling on the pool every K epochs) — amortize with a subsampled pool.
- Honesty: must show the equal-COMPUTE accounting (mining pass + extra weighted steps counted against
  the uniform baseline's compute), else gains are just "more training".

## Sequencing
Slots in AFTER the Stage-1/2 Sudoku gates (need a trainable-but-imperfect SudokuEqM to have failures
to mine). Then port the winning arm to image-gen EqM-B/2 for the no-label payoff. Does NOT block the
current RRN-parity work — it's the natural Stage-3-prime once a workable-band Sudoku model exists.
