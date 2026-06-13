# D1 — Post-G2 Diagnostic Node (pre-registered 2026-06-12, BEFORE extraction)

Single gated node. Eight diagnostics; branches H-A / H-B / H-C chosen by the
decision table below, written before any new probe runs.

## Honesty ledger — what is already observed vs fresh
OBSERVED PRIOR (in results/e0_2_verdict.json, read at G1): #1a contrastive
headroom (oracle 0.9705 − no_mine 0.9456 = 0.0249 vs K3 margin 0.0152 at
high); #7 cert-random-k gap (0.0 at high). These are recorded, not re-derived;
their thresholds below are therefore DESCRIPTIVE, not predictive.
FRESH (no one has looked): #1b, #2, #3-decomposition, #4, #5, #6, #8.

## Diagnostics and pre-registered thresholds

D1.1a Contrastive headroom: H_c = probe(SupCon ⊕) − probe(no_mine), high
  level. UNTESTABLE-K3 declared iff H_c < 2 × K3 margin (0.0304).
  [observed: 0.0249 → untestable; recorded for the table.]
D1.1b EqM headroom: capacity floor F = field MSE of an identical Field MLP
  distilled DIRECTLY on the MC reference f* (3 seeds, same steps/opt).
  Vanilla gap G_v = 0.837 − F. Bridge verdict stays "no channel" iff
  G_v > 2 × margin_mse (0.172) AND oracle failed to claim it; softens to
  "untestable (ceiling effect)" iff G_v ≤ 0.172.
D1.2 Certified-hard supply: at λ* per ladder level, among pairs in the TOP
  teacher-similarity decile: certified-negative fraction, ambiguous fraction,
  (1−ρ̂(s)) mean. SUPPLY-STARVED declared iff certified-neg fraction < 0.10
  in the top decile at ALL three levels.
D1.3 Damage autopsy (naive arm vs no_mine, high level, 3 seeds): Δalignment
  (mean same-class embedding distance), Δuniformity (log mean exp(−2·pdist²)),
  Δeffective rank (covariance spectrum). Attribution: 100% FN-push BY
  CONSTRUCTION (naive arm mines negatives only — no positive mining exists in
  that arm). H-B asymmetry condition: declared supported (it is, structurally);
  decomposition tells WHAT dies, reported descriptively.
D1.4 Sample efficiency: probe accuracy at steps {300, 600, 900, 1200, 1500}
  for {no_mine, rc_hpm, supcon}, high level, 5 seeds. HIDDEN-POSITIVE declared
  iff AUC(rc) > AUC(no_mine) + 2×SD_seed(AUC(no_mine)) OR steps-to-95%-of-final
  (rc) < no_mine by ≥ 300 steps at matched final accuracy (within margin).
D1.5 Silent-vs-visible failure contrast: corrupt gate (30% label flips in
  D_gate) for RC; same data for RINCE. CONTRAST CONFIRMED iff (i) RC abstention
  rises ≥ 3× its clean value OR RC aborts (visible), AND (ii) RINCE probe drops
  below its clean value by ≥ K3 margin with NO loss-curve signature (KS on
  loss-curve windows vs clean seed, p > 0.05). If RINCE does not drop, the
  foil is weak — record and lean on the guarantee alone.
D1.6 α monotonicity: rc_hpm probe at α ∈ {0.05, 0.10, 0.20}, high level,
  5 seeds. THROTTLED declared iff probe(0.20) > probe(0.05) + 2×SD_seed.
  FLAT iff all within 2×SD band.
D1.7 Hardness-within-certification: [observed prior: gap 0.0 → hardness
  contributed nothing within certified sets at this toy; points at
  certification/curriculum, not hardness, as the active ingredient.]
D1.8 Spectral gap γ: normalized-Laplacian algebraic connectivity (λ₂) of the
  k=10 mutual-kNN graph on teacher embeddings: toy levels {low, med, high}
  (2,000 pts each) vs CIFAR-10 rn18 (2,000-pt subsample, γ=1 bin).
  TOY-TOO-EASY declared iff γ_toy(high) > 3 × γ_cifar.

## Pre-registered branch table (evaluate in order; first match = primary)

1. IF D1.2 SUPPLY-STARVED at all levels AND D1.6 ≠ THROTTLED
   → **H-C primary**: reposition fully onto the safety product (risk dial,
     matched asymptote, bounded harm, D1.5 contrast as the headline figure;
     gains question explicitly abandoned). No new mechanism experiment.
2. ELSE IF (D1.1a untestable) AND (D1.8 TOY-TOO-EASY)
   → **H-A primary**: difficulty ladder — same generator, knobs = class count
     K ∈ {10, 30, 100}, overlap σ, tail imbalance; semi-real arm = small heads
     on frozen CIFAR rn18 embeddings (brings the 14.3× concentration into CPU
     budget). PRE-REGISTERED PREDICTION: RC-minus-no_mine gap increases as γ
     shrinks and headroom opens; kill if gap ≤ 0 at the smallest-γ rung with
     headroom ≥ 3× margin. H-B ARM INCLUDED in the ladder (rc-negatives-only,
     justified by D1.3 structural attribution) — if it matches full RC, the
     simplified method is the paper's method.
3. ELSE IF D1.6 THROTTLED
   → supply-side variant of H-A: ladder run at the loosest certified α; same
     prediction and kill rule.
4. D1.5 contrast is quantified and reported under EVERY branch (it is the
   safety paper's figure regardless).
5. D1.1b decides only the WORDING of the EqM-bridge postmortem (kill vs
   untestable); it does not gate a new EqM experiment in this node.

## Budget
Log extraction + ~2–3 CPU-hours of probes (D1.3/4/5/6 are short re-trains,
≤ 25 runs of the 1,500-step student). No GPU. Single verdict JSON:
results/d1_diagnostics.json; branch recorded in pipeline.json.
