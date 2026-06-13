# Pre-registration — D2 H-A′ Difficulty Ladder (binding numbers)

Written 2026-06-12, BEFORE any D2 code runs. Supersedes the H-A ladder sketch
in handoff-diagnostics.md; implements documentation/h_a_ladder_d2_spec.md.
Inherits preregistration.md pinned functions (w, w′, v, ω, c_amb=0.5, HB
p-value, fixed-sequence path, isotonic directions) and A1–A4. Deviations →
deviations.md with date + reason.

## Constants (inherited unless noted)
- α₀ = 0.10, δ_r = 0.05, m_test = 250, m_fit = 40, n_batch = 64,
  k⁺ = 2, k⁻ = 8, τ = 0.5, n_fold = 32,000, n_gate = 2,000.
- Student/probe protocol EXACTLY as E0.2′: MLP d→64→32 unit-norm, Adam 1e-3,
  1,500 steps, bs 64; probe = multinomial logistic, 1,000 train / 2,000 test
  fresh draws, 300 steps.
- Seeds: 10 per arm on full rungs; 10 no-mine + 10 SupCon for pre-flight H.

## Generator (rc_hpm/ladder.py)
x | class k ~ N(μ_k, σ²I) in R^16, μ_k ~ N(0, 2²I) (seeded per rung-family,
shared across arms/seeds within a rung). Knobs:
- K ∈ {4, 10, 20, 40}
- σ ∈ {0.4, 0.8, 1.2, 1.6, 2.0, 2.4}
- imbalance π ∈ {uniform, mild, heavy}: class weights ∝ rank^(−a),
  a ∈ {0, 0.8, 1.6}, normalized.
Teacher: frozen random linear R^16→R^16 + unit-norm (as toy.py).
Aug: x + N(0, 0.25²I) (unchanged).

## Pre-flight metrics (computable before arm training)
- ρ_tail := same-class fraction among the top similarity decile of 200,000
  random distinct pairs (teacher sims, fresh population draw, seed 0).
- S := certified-negative fraction of the top similarity decile under the
  standard gate + LTT calibration at α₀ (one calibration, seed 0).
  ABORT ⇒ S := 0.
- H := mean(SupCon probe acc) − mean(no-mine probe acc), 10 seeds each.
  Noise floor := SD(no-mine probe acc across its 10 seeds).
- γ := eigengap probe (below) — MEASURED COVARIATE ONLY.

## Grid targets (12 cells)
ρ_tail targets {0.05, 0.20, 0.40, 0.60} × H targets {low, med, high} where
  H-low: H < 0.02; H-med: 0.02 ≤ H < 0.06; H-high: H ≥ 0.06 (probe-acc units).
Knob search (D2 step 2) is GENERATOR-ONLY for ρ_tail; H requires the cheap
training pre-flight, so the search procedure is:
 1. For every (K, σ, a) combo: compute ρ_tail (no training). Keep combos
    within ±0.05 of a ρ_tail target.
 2. Among those, predict H ordering by K and a (more classes / heavier tail →
    larger expected headroom); select up to 3 candidates per ρ_tail target
    spanning the predicted H range; measure H for candidates only.
 3. Assign each measured candidate to its (H-bin, ρ_tail) cell. Cells left
    empty are reported UNREACHABLE (a geometry finding, not a deviation).

## Pre-flight gate (per rung, BEFORE full arms)
 (a) H > 2 × noise floor
 (b) S > 0.10
 (c) |ρ_tail − target| ≤ 0.05
Fail (a) or (b) → rung trains {no-mine, naive-negative ⊖} only (10 seeds).

## Arms (full rungs, 10 seeds)
no-mine / RC-HPM(α₀) / RC-neg-only / cert-random-k / RINCE (q=0.5, λ=0.025) /
SupCon ⊕ / naive-negative ⊖ (teacher top-k⁻, no cert).
RC-neg-only (H-B′): mined positives DROPPED entirely (positive set = aug view
only); negative side identical to RC-HPM (certified + ω-ambiguous). No ρ̂⁺,
no v⁺, no β⁺-mining. Cost claim measured as wall-time ratio vs RC-HPM.

## Designated rung
The PASSING rung closest to (H-high, ρ_tail = 0.20); ties → larger H. Adds:
- naive-positive ⊖ probe: top-k⁺ = 2 teacher-most-similar non-self as
  UNCERTIFIED positives in a multi-positive InfoNCE (FP-pull channel).
- α frontier sweep: α ∈ {0.05, 0.10, 0.20, 0.40}, RC-HPM, 10 seeds each.

## RINCE foil rung (boundary-crossing view noise)
Run at the designated rung. View-noise knob: with probability p(x), replace
the second view with a draw from the NEAREST other class (by μ distance from
x). Marginal crossing rate pinned at 0.15 for both conditions:
- concentrated: p(x) ∝ softmax margin — p(x) = c · sigmoid(−(d₂(x) − d₁(x))/σ)
  scaled so the population mean is 0.15 (d₁, d₂ = distances to nearest and
  second-nearest class means).
- diffuse: p(x) = 0.15 constant.
Arms: no-mine / RC-HPM / RINCE, 10 seeds per condition.
PASS for the foil figure: under concentrated, RC − RINCE > 2× pooled seed-SD
AND RC's monitored abstention (or ABORT) visibly exceeds its clean-rung
reference (Δabstention > 0.05 or ABORT), while under diffuse
|RC − RINCE| ≤ 2× pooled seed-SD. Anything else → foil figure dead; safety
claim stands on D1's 3/3 abort behavior (already banked).

## Semi-real rung (A1′ guard)
Encoder: torchvision resnet18 IMAGENET1K_V1 (label-free w.r.t. CIFAR ✓).
Data: 20,000-example CIFAR-10 train subset (seed 0); embeddings 512-d
unit-norm; "x" for the student = the frozen embedding; aug view = embedding +
N(0, (0.1·per-dim SD)²). Student: MLP 512→64→32. Same probe protocol
(1,000/2,000 fresh from held-out remainder). Pre-flight gate applies
unchanged. Pre-registered prediction: lands upper-band (safety-dominant).

## γ probe (D1-#8 repair; covariate only)
kNN graph (k = 15, cosine), symmetric normalized Laplacian L_sym,
smallest 50 eigenvalues via scipy.sparse.linalg.eigsh; estimated cluster
count K̂ = argmax_{i≤49} (λ_{i+1} − λ_i); γ̂ = λ_{K+1} − λ_K at TRUE K.
VALIDATION (before any use): on synthetic rungs with known K ∈ {4,10,20,40}
(σ = 0.8, uniform), K̂ must equal K for ≥ 3 of 4. Fail → γ reported as
"uninstrumented", excluded from all analysis (no redesign within D2).

## PRIMARY endpoint (one; everything else descriptive)
Per full rung r and seed s: gap_{rs} = acc_RC − acc_no-mine (paired by seed).
Regression over full rungs:
  gap = b0 + b1·H + b2·S + q1·ρ_tail + q2·ρ_tail² + ε
Pre-registered constraints for B1 (band found) — ALL must hold:
  (C1) q2 < 0 with bootstrap 95% CI excluding 0 (2,000 resamples over seeds
       within rungs, rung-level cluster bootstrap).
  (C2) fitted peak ρ* = −q1/(2q2) ∈ (0.10, 0.50).
  (C3) the observed best cell's mean gap > 2 × its seed-SE.
  (C4) b1 ≥ 0 and b2 ≥ 0 (point estimates; descriptive if CIs cross 0).
B2 (no band): max cell mean gap ≤ 2 × its seed-SE across ALL reachable full
rungs including semi-real.
B3 (H-B′ retention fails): retention = (acc_RCneg − acc_naive) /
(acc_RC − acc_naive) on the designated rung, valid only if
(acc_no-mine − acc_naive) > 2× pooled SD there; retention < 0.80 → B3.
B1/B3 can co-fire (band found AND full gate load-bearing). A result matching
no branch → STOP + postmortem; no improvised branch.

## Multiplicity & reporting
Per-rung comparisons: descriptive only, reported with seed-SDs, no stars.
The regression is the single confirmatory test. Post-pre-flight rung
exclusions are reported in full (which cells unreachable, which failed (a)/(b)).

## Compute budget (declared)
Knob search: generator-only, minutes. Pre-flight H: ≤ 12 candidates × 20 runs
× ~45 s. Full rungs: expected 4–6 × 7 arms × 10 seeds. Designated extras +
foil + semi-real. All CPU, est. 6–10 h wall; run sequentially in background.
