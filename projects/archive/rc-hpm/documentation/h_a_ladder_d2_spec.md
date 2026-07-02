# D2 — H-A Difficulty Ladder (pre-registration spec, CPU)

Successor to D1 branch `H-A (primary) + H-B (folded)`. Inherits all standing
protocol: P-rules from decision-tree v2.1, preregistration.md + A1–A4 binding,
deviations → deviations.md, rc_hpm/core.py reused not rewritten, monitor units
per-batch.

## Hypothesis (pre-registered, replaces "toy too easy")

**H-A′ (band hypothesis):** the RC-minus-no-mine utility gap is non-monotone in
tail error rate ρ_tail, peaking in a moderate band where
  (i) headroom H = (SupCon-oracle − no-mine) exceeds 2× the rung noise floor,
  (ii) certified-hard supply S = certified fraction of the top similarity
       decile exceeds 10%,
  (iii) ρ_tail is high enough that the naive ⊖ arm measurably damages.
Below the band: gap ≈ 0 with no naive damage (nothing at stake).
Above the band: gap ≈ 0 with RC degenerating toward no-mine (abstention),
while naive damage is large — the safety regime.
**H-B′ (folded):** RC-negatives-only (mined positives DROPPED, not just
uncertified) retains ≥ 80% of RC's protection (naive-damage prevented) at
≤ 50% of the label/computation cost of the full gate.

D1 corrections carried in:
- #3 was attribution-by-construction → FP-pull harm is UNKNOWN; one rung runs
  a naive-positive-mining ⊖ probe to measure it.
- #7 is a corollary of thin supply (#2), not evidence against hardness; the
  ladder re-tests hardness ONLY at rungs passing the supply gate.
- #6 reframed: the α-sweep at the peak-band rung draws the safety–utility
  frontier (the "risk dial" figure).

## Generator knobs and the 2-factor design (anti-confounding)

Knobs: class count K, cluster overlap σ, tail imbalance π.
These jointly move H, S, ρ_tail, and γ — a 1-D "ladder" cannot attribute.
Design: a small 2-factor grid targeting (H, ρ_tail) cells:
    H ∈ {low, med, high} × ρ_tail ∈ {~0.05, ~0.2, ~0.4, ~0.6}
with knob settings found by search on the GENERATOR ONLY (no training) —
H, S, ρ_tail, γ are all computable from the generator + frozen teacher + gate
before any arm trains. Rungs that cannot hit their target cell are reported
as unreachable (that's a finding about the geometry, not a deviation).

PRE-FLIGHT GATE per rung (before training arms):
  (a) H > 2× rung noise floor (≥10-seed no-mine variance at toy cost)
  (b) S > 10% top-decile certified
  (c) measured ρ_tail within ±0.05 of cell target
Rungs failing (a) or (b) train only {no-mine, naive ⊖} (cheap; they still
contribute the "nothing at stake / safety-only" ends of the band curve).

## Arms per full rung (seeds: 10)

no-mine / RC-HPM (α₀) / RC-neg-only [H-B′: mined positives DROPPED] /
cert-random-k / RINCE / SupCon-oracle ⊕ / naive-negative ⊖
+ ONE designated rung adds: naive-positive ⊖ (closes D1-#3) and the
  α ∈ {α₀/2, α₀, 2α₀, 4α₀} frontier sweep (D1-#6 figure).

## RINCE foil regime (D1-#5 fix)

Gate corruption never reaches RINCE's loss — pair noise must live in the DATA.
New knob: boundary-crossing view noise — augmentations whose probability of
crossing a class boundary increases with similarity-to-boundary (concentrated)
vs uniform crossing probability (diffuse), matched marginal noise rate.
Pre-registered prediction: diffuse → RINCE ≥ RC; concentrated → RC > RINCE
with RC's abstention rising VISIBLY while RINCE has no observable signal.
If concentrated noise still doesn't separate them, the silent-vs-visible
figure is dead and the safety claim stands on the abort behavior alone
(already demonstrated 3/3 in D1-#5).

## Semi-real rung (A1/A1′ guard)

Frozen encoder for CIFAR embeddings must be label-free w.r.t. CIFAR:
self-supervised on CIFAR or supervised on ImageNet — NEVER CIFAR-label-
supervised (label leakage through the teacher fakes utility). Calibration and
training examples must be symmetrically in/out of the encoder's pretraining
corpus (A1′). Pre-flight gate applies as on synthetic rungs; D1's 14.3×
concentration suggests this rung lands in the upper band (safety-dominant) —
that is itself the prediction.

## Endpoints (P3 discipline)

PRIMARY (one): pre-registered regression of gap(RC − no-mine) on (H, S) with
band shape in ρ_tail — sign and ordering constraints declared in
preregistration-d2.md before any arm trains. Per-rung wins are SECONDARY,
descriptive, never gate-bearing (multiplicity).
SECONDARY: H-B′ retention ratio; frontier monotonicity at the α-sweep rung;
RINCE separation under concentrated view noise; γ (recomputed per D1-#8:
kNN graph, symmetric normalized Laplacian, Lanczos, eigengap at K) as a
MEASURED covariate only — no design decision hangs on γ until the probe is
validated against the known-K synthetic rungs.

## Branch outcomes (pre-registered)

B1 band found (primary regression confirms): → GPU Stage-2 proposal for the
   contrastive-standalone track, with the pre-flight metrics (H, S, ρ_tail)
   promoted to a DOMAIN-SELECTION test run on candidate real datasets before
   any GPU commit. This pre-flight is the product of D2.
B2 no band anywhere (gap ≈ 0 across all reachable cells incl. semi-real):
   utility claim retired at CPU scale honestly; paper = safety/guarantee +
   risk-dial + abort-visibility; GPU only if the RINCE foil separated.
B3 H-B′ retention < 80%: full gate is load-bearing → positives channel
   matters; revisit #3 conclusion, keep full RC.
Any result matching no branch: STOP + postmortem (P0), no improvised branch.
