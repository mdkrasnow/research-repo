# Handoff — D1 diagnostic answers (post-G2 repositioning node)

Audience: next agent. Everything below is machine-extracted
(results/d1_diagnostics.json, pre-registered branch rules in
documentation/preregistration-d1-diagnostics.md) plus prior gate verdicts in
results/*.json. Read documentation/cpu-stage-summary.md first for context.

## The eight answers

### 1. Headroom accounting — THE decisive split
- **Contrastive toy: K3 was NEVER TESTABLE.** Headroom (SupCon-oracle −
  no-mine) at high concentration = **0.0249**, below the 2×margin
  threshold 0.0304. "No utility gain" at G1 is a property of the toy, not
  the method. The G1 utility kill is downgraded to UNTESTABLE.
- **EqM bridge: the kill STANDS.** Capacity floor of the same MLP
  (direct supervised fit of the reference field) = **0.282** MSE vs vanilla
  0.837 → available headroom 0.555, >> threshold 0.172. The oracle-pairs null
  happened with real headroom on the table → genuine no-channel, not a
  ceiling effect. Do not reopen arm A at this scale.

### 2. Certification telemetry — mining was neutered by its own machinery
Certified-negative fraction in the TOP similarity decile: low 3.3%, med
17.3%, high: calibration **aborted** in this probe. Worse: among the top
decile that IS certified, the loss weight v⁻ = (1−ρ̂) averages **0.032** at
low — a double throttle (scarce certs × near-zero weight). RC-HPM at the
hard tail effectively WAS no-mine. Supply-side explanation confirmed:
gains require regimes where (1−ρ̂(s)) stays high at large s — measurable
per-domain with the E1.0 machinery before any compute commit.

### 3. Damage autopsy — FN-push, trivially and structurally
Attribution: **100% FN-push by construction** (the naive arm mines negatives
only; no FP-pull channel exists in it). Structure of the damage: same-class
spread (alignment 1.386 vs 1.221 no-mine) plus mild dimensional collapse
(effective rank 27.1 vs 29.5); uniformity unchanged. **H-B (certify
negatives only) is licensed**: drops half the machinery and half the label
demand while keeping all demonstrated protection.

### 4. Training curves — no hidden sample-efficiency positive
RC AUC 1126.7 vs no-mine 1129.7 (RC slightly SLOWER early: 0.933 vs 0.938 at
step 300). SupCon converges faster and higher. `hidden_positive: false`.

### 5. RINCE / visible-vs-silent failure — half demonstrated
Under gate corruption, RC fails **visibly**: 3/3 corrupted-gate runs ABORT
(loud signal), train un-mined, probe preserved (0.92–0.94). But RINCE does
NOT silently degrade in this regime (0.948 ≈ no-mine) — the toy never
produces conditions that poison robust losses. The strongest-figure contrast
(RC abstains visibly / RINCE rots silently) is **untested, not confirmed**.
Caveat: label corruption has no entry point into unsupervised RINCE; the
foil needs a regime where pair errors actually transfer into RINCE's loss.

### 6. α-grid — NOT over-throttled
α=0.05 and 0.10: aborts in this probe (fallback to un-mined, probe 0.944).
α=0.20: certifies and is WORSE (0.917). Monotone non-increasing in α →
loosening the safety dial does not unlock utility. Combined with #2: the
supply is thin AND opening the valve doesn't help — consistent with #1
(no headroom to win regardless).

### 7. Certified-random-k — hardness contributed NOTHING within certified sets
Gap RC − cert-random-k = 0.000 (high), −0.006 (med). Next hypothesis should
target **certification/curriculum-by-confidence**, not hardness.

### 8. Spectral structure — PROBE ARTIFACT, redo before relying on it
All measured K=10 eigengaps ≈ 1e-15 (toy AND CIFAR) — numerically degenerate
(row-normalized exp-affinity is near-rank-collapsed at this τ). The
`toy_too_easy` flag fired on the fallback criterion, not a clean γ
measurement. **Next agent: recompute with a kNN-graph normalized Laplacian
and per-K eigengap sweep before using γ in any design decision.** Treat the
H-A spectral condition as provisionally satisfied via #1, not via #8.

## Pre-registered branch outcome
**H-A fires (primary): difficulty ladder**, with the **H-B arm folded in**
(rc-negatives-only variant), per results/d1_diagnostics.json `branch`.
H-C does not fully fire (supply thin but nonzero at med).

## What the next experiment is (H-A ladder, CPU)
Same toy generator with controlled knobs: class count K, overlap σ,
tail imbalance; pre-registered prediction: RC-minus-baseline gap grows as
headroom opens and certified-hard supply (#2 metric) rises. Arms per rung:
no-mine / RC / RC-neg-only (H-B) / cert-random-k / SupCon ⊕ / naive ⊖.
Plus one "semi-real" rung: small heads on frozen CIFAR rn18 embeddings
(brings the measured 14.3× damage concentration into CPU budget).
GATE BEFORE BUILDING: rung is valid only if (a) headroom > 2×margin
(the #1 check) and (b) certified-hard supply > 10% top-decile (the #2
check) — both computable from the generator + gate BEFORE training arms.

## Standing cautions for the next agent
- Preregistration discipline: thresholds/margins in
  documentation/preregistration.md (+A1–A4) are binding; deviations →
  documentation/deviations.md.
- EqM arm A (contrastive-on-activations) is dead with headroom proven (#1b).
  EqM endpoint mining lives at scale via diff-EqM v10 (FID 27.58 vs 31.41,
  IN-1K) — certification-on-v10 is a Stage-2/GPU question, human-gated.
- rc_hpm/core.py (LTT/HB pipeline) is validated by G-1/G0 — reuse, don't
  rewrite. Monitor units are per-BATCH statistics (pair-level KS is invalid).
