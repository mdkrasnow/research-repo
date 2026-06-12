# RC-HPM CPU-Stage Pre-registration (Stages 0–1)

Written BEFORE any experiment runs (2026-06-12). Binding per decision tree v2
(P2, P3). Any deviation must be logged in `documentation/deviations.md` with
date + reason. Constants here are pinned; "nuisance knobs" eligible for the
single retune budget are marked [nuisance].

## Environment
- Local CPU, python 3.11.8, torch 2.7.1, numpy 2.4.6, scipy 1.15.3.
- sklearn broken in env → own PAV isotonic implementation (rc_hpm/isotonic.py).
- Seeds: experiment base seed 0; seed i = base + i.

## Pinned functions (define the controlled functionals; never tuned after this)
- w_ij (repulsive surrogate)  = exp(s_ij / τ)
- w′_ip (attractive surrogate)= 1 − σ(s_ip / τ_w), τ_w = τ
- hardness_weight(s) ≡ 1  (hardness lives in selection utility only)
- v⁻_ij = (1 − ρ̂(s_ij)) · hardness_weight = (1 − ρ̂(s_ij))
- v⁺_ip = (1 − ρ̂⁺(s_ip))
- ω_ij  = (1 − ρ_amb(q_ij)) · c_amb,  c_amb = 0.5
- c_syn = 0 (no synthetic mixing anywhere in CPU stages — theorem-grade arms)
- Degenerate batch: L ≜ 0 (founding spec v1.4 convention)
- HB p-value: p = min(1, exp(−m·KL(min(μ̂,α)‖α)), e·BinCDF(⌈m·μ̂⌉; m, α))
- Fixed-sequence path: λ grid sorted ascending by
  max(mean_fit(L⁻)/α⁻, mean_fit(L⁺)/α⁺) computed on D_fit batches.
- λ* among VALID: argmax D_fit utility throughput = mean over D_fit batches of
  Σ_i [Σ_{j∈N_i} u⁻(j) + Σ_{p∈P_i} u⁺(p)].
- Isotonic directions: ρ̂(s)=P(y=1|s) increasing in s; ρ̂⁺(s)=P(y=0|s)
  decreasing in s; ρ_amb(q)=P(y=1|q) increasing in q.

## Shared hyperparameters (CPU toys)
- τ = 0.5 (toy scale; embeddings unit-normalized)
- k⁺ = 2, k⁻ = 8
- Training/calibration batch size n = 64 (P1a; all arms share)
- m = 20 calibration batches per fold half (D_test side); D_fit/D_test 50/50
- M = 2 folds (toys; recalibration rarely exercised)
- δ = 0.1, δ_r = δ/M = 0.05
- α grid (P2): α₀ = 0.05 primary, 2α₀ = 0.10 secondary, 4α₀ = 0.20 tertiary;
  α⁺ = α⁻ = α at each grid point. Full grid runs at Stages 0–1.
- (β⁺, β⁻) grid: β⁺ ∈ {0.55, 0.65, 0.75, 0.85, 0.92, 0.97},
  β⁻ ∈ {0.03, 0.08, 0.15, 0.25, 0.35, 0.45}, constraint β⁻ < β⁺ (36 pts max).
- Calibration batches sampled INDEPENDENTLY with replacement at the batch level
  from fold data (i.i.d. batch losses ⇒ HB applies as stated). Dependence
  checker verifies the declared mode.
- Realized-risk evaluation: 200 fresh batches from the true generative law
  (synthetic data ⇒ population access); report mean ± SE per risk.

## AMENDMENT A1 (2026-06-12, PRE-RUN — feasibility arithmetic, before E0.1)
Tree v2 P1: "the per-bin arithmetic must be checked before E1.1, not
discovered during it." Checked during pipeline smoke (no gate evaluated):
- HB certification needs m·KL(μ̂‖α) ≥ ln(1/δ_r) (Hoeffding) or the Bentkus
  binomial tail. At m=20, even μ̂=0 gives p ≥ 0.36 at α=0.05 — the original
  m=20 can certify NOTHING. Original E0.1 budget was arithmetically void.
- Measured toy risk floor (smoke, seed 0, strict λ): L⁻ ≈ 0.037–0.042,
  dominated by ω-weighted ambiguous same-class mass (per-spec behavior).
- AMENDED constants (derived, not outcome-tuned):
  m_test = 250 disjoint batches, m_fit = 40 pool batches,
  n_fold = 2×250×64 = 32,000; α grid RE-ANCHORED: α₀ = 0.10 PRIMARY,
  0.05 stretch endpoint (ABORT expected on some seeds, P7 applies),
  0.20 loose endpoint. δ_r = 0.05 unchanged.
  Feasibility at the primary: μ̂=0.05, m=250 → Bentkus p ≈ 0.018 ≤ δ_r ✓.
- G0 informativeness condition (added): ≥ 15/20 seeds must certify (non-ABORT)
  at α₀ for the exceedance test to be read; ABORTs counted and reported.
Deviation also logged in documentation/deviations.md.

## E0.0 bug-injection suite — G-1
Injections (each must be DETECTED: realized/true risk inflated beyond the
clean envelope, or caught loudly by a checker):
1. fold_reuse        — ρ̂/ρ̂⁺/ρ_amb AND path fit on D_test. Detection:
                       statistical (false-certification rate).
2. a1prime_asymmetry — teacher noisier on training-side draws than calibration
                       draws (σ_extra = 0.5). Detection: symmetry checker (KS
                       on score distributions) OR risk inflation.
3. partition_bentkus — partitioned (without-replacement) calibration batches
                       with Bentkus kept. Detection: config checker raises.
4. wprime_mismatch   — w′ changed between calibration and deployment.
                       Detection: pinned-function hash checker raises.
5. skip_degenerate   — remove the L≜0 convention. Detection: NaN guard raises
                       on strict-λ grid points.
6. no_bonferroni     — monitor at raw per-test level on a stationary stream.
                       Detection: false-alarm count > Bonferroni-implied bound.
7. live_student      — mining scores from a drifting encoder instead of frozen
                       teacher. Detection: drift monitor alarm OR risk inflation.
8. gamma_pooled      — calibrate under γ-law A, train under γ-law B with a
                       γ-conditional gate truth. Detection: γ-law checker
                       (histogram mismatch) OR risk inflation.
9. overlapping_batches — calibration batches share examples while declared
                       i.i.d. Detection: example-id overlap checker raises.
Gate G-1: ALL 9 detected AND clean run passes (no checker fires; true risk of
chosen λ ≤ α envelope). No retune budget — correctness.

AMENDMENT A2 (2026-06-12, during E0.0 test engineering — detector changes,
no science thresholds touched):
- Injection 1 (fold reuse): measured realized-risk signature of 1-D isotonic
  fold reuse at toy scale ≈ 0 (knife-edge α landed below the m-certifiability
  floor; both arms abort ⇒ statistical test vacuous). Detection mechanism
  changed to a DATA-FLOW guard inside calibrate_ltt (rho/path fit indices
  must not intersect the LTT test split). Documented limitation: realized
  inflation from 1-D isotonic reuse is not empirically demonstrable in this
  toy; the violation is structural and is checked structurally.
- Injection 7 (live student): selected-SCORE CDF is selection-invariant under
  drift (mining re-picks top-k) — weak signal, 2/8 detected. Monitor extended
  to label-free per-batch statistics {s_mean, q_mean of mined sets, certified
  count} + RELATIVE abstention tripwire (kappa_a = max(0.05, 0.5·a_ref));
  Bonferroni across checks AND statistics. Measured: drift=1.0 doubles
  realized L⁻ (0.035→0.073, still < α=0.10 — slack absorbs it) and is now
  caught by the q_mean statistic (5 alarms + A1' checker).

## E0.1 synthetic toy — G0
- Data: K = 10 Gaussian clusters in R^16, means ~ N(0, I·2.0), σ_cluster = 1.0
  (moderate overlap), teacher = frozen random linear R^16→R^16 + unit-norm.
- Augmentation law: view = x + N(0, 0.25²·I).
- D_lab per seed: gate 2,000 / cal 2×(2×20×64)=5,120 / holdout 1,000.
- 20 seeds. ⊕ oracle gate (q from true labels, ±0.02 margin),
  ⊖ random gate (q ~ U(0,1)).
- G0(a): exceedance = seed where [realized mean − 2·SE] > α for EITHER risk.
  PASS iff #exceedances ≤ 4 (smallest c with P(Bin(20, 0.1) ≥ c) < 0.05 is 5).
- G0(a′): pooled across-seed mean of realized means ≤ α + δ·(1−α) for each
  risk, evaluated at pooled-mean + 2·SE_pooled (upper CI must satisfy bound).
- G0(b): ⊖ random gate: per-seed outcome must be ABORT or abstention ≥ 0.95;
  whenever any mining occurred, realized risk ≤ α (same exceedance rule).
- Evaluated at all three α grid points; gate decision on α₀ = 0.05 primary.

## E0.2′ damage/utility probe — G1
- Same toy family; concentration ladder pre-registered, ALL levels run:
  σ_cluster ∈ {0.6 (low), 1.2 (med), 1.8 (high)} with K = 10, mean scale 2.0
  (overlap ⇒ hard pairs disproportionately same-class).
- Arms (same data, same steps, same optimizer):
  1. no-mine (uniform InfoNCE, in-batch negatives)
  2. naive hard-mine (top-k⁻ by similarity, no certification)
  3. RC-HPM (full pipeline, α₀)
  4. RINCE (q = 0.5, λ_rince = 0.025) at matched compute
  5. certified-random-k (gate certification, random k among certified)
  6. ⊕ SupCon oracle (true labels)   7. ⊖ = arm 1 is the floor reference
- Student: MLP 16→64→32, unit-norm output; Adam 1e-3; 1,500 steps; bs 64.
- Primary metric (P3a): linear-probe accuracy (multinomial logistic on frozen
  embeddings, 1,000 labeled probe-train / 2,000 probe-test, fresh draws).
- 5 seeds per arm × level. Noise floor: bootstrap SD of arm-1 probe accuracy
  across its 5 seeds; margin = 2× that SD (pre-registered formula; numeric
  value recorded in results before treatment arms are read).
- G1: (K2) naive < no-mine by margin at high concentration;
      (K3) RC-HPM > no-mine by margin at high concentration;
      (e) RC-HPM > RINCE by margin at high concentration;
      (f) certified-random-k recorded (descriptive).
  Branching per tree v2 (no escalation retune; no α retune).

## E1.0 target-domain premise — G1.5
- Teacher (pre-registered substitute; no trained EqM EMA exists at CPU stage):
  torchvision resnet18 ImageNet-pretrained penultimate features. DEVIATION
  NOTE: spec says "EMA embeddings"; substitute justified as frozen
  representative encoder; logged as deviation.
- Data: CIFAR-10 train subset, 10,000 examples (seed-0 uniform subsample).
- γ-bins: γ ∈ {0.9, 0.6, 0.3} via x_γ = γ·x + (1−γ)·ε, ε ~ N(0, I) per-pixel
  after normalization (P4 preview); plus clean bin γ=1.
- Pairs: 200,000 uniformly sampled distinct pairs per bin.
- ρ(s): isotonic P(cross-class | s) — for REPULSIVE damage, error = same-class
  at high s. Damage density d(s) = w(s)·ρ_same(s) with w = exp(s/τ), τ = 0.5.
- G1.5 (primary, density form): mean damage density over pairs in top hardness
  (similarity) decile ≥ 3× mean density over bottom half, in the γ=1 bin.
  Other bins descriptive. No retune.
- Fallback if downloads unavailable: defer E1.0 until network, proceed to
  Stage 1 (E1.0 informs FRAMING, not machinery); log as deviation.

## E1.1 / E1.2 2D EqM toys — G2
- Data: K = 4 Gaussian clusters in R², means at radius 1.5 (square corners),
  σ = 0.12. Classes = cluster id. EqM field: MLP 2(+γ-feature for none —
  noise-unconditional per paper) → 128 → 128 → 2, penultimate = 128-d act.
- c(γ): EXACT get_ct port (truncated decay interp=0.8, start=1, ×4).
- Training: γ ~ U(0,1), ε ~ N(0, I), x_γ = γx + (1−γ)ε, target (x−ε)·c(γ)
  [paper Eq. 3 with ut convention as transport.py], Adam 1e-3, 4,000 steps,
  bs 128. 5 seeds.
- ARM A arms: vanilla / +disp uniform (coeff 0.5 as upstream) / +RC-HPM
  InfoNCE (coeff 0.5) / +RC-repulsive-only (coeff 0.5) / ⊕ oracle-pair
  InfoNCE / (⊖ = vanilla floor). Gate features γ-conditional (P4): gate input
  = [e_i ⊙ e_j features at γ-bin], per-bin ρ̂ with bins γ ∈ {[0,.33),[.33,.67),[.67,1]}.
- ARM B arms: vanilla / ANM-frozen-teacher certified (LTT-calibrated eps_ball
  ∈ {0.1,...,1.2} grid × cert threshold) / ⊖ live-student ANM eps×10
  uncertified. Teacher = EMA(0.999) snapshot frozen at step 1,000 (calibration
  time). Basin labeling: 2D analytic — descend TRUE mixture score field;
  attractor = nearest cluster mean; exact (P5: toy validates procedure).
- Primary metrics (P3a, ONE each): ARM A = field MSE vs Monte-Carlo reference
  field on a 40×40 grid over [−3,3]² at γ-slice set {0.3, 0.6, 0.9}
  (reference: kernel regression of targets from 2×10⁶ (x,ε,γ) draws,
  bandwidth 0.15); ARM B = attractor purity (fraction of 2,048 GD samples,
  η=0.05, 400 steps, landing within 0.3 of any true mean AND assigned mean
  matches conditioning class when conditional; unconditional toy → fraction
  within 0.3 of any mean) — higher better.
- Noise floor: bootstrap/seed SD of vanilla arm primary metric (5 seeds);
  margin = 2× SD.
- G2: (a) loss finite all steps; (b) aux/base ratio ∈ [0.05, 20] entire run
  AND no monotone collapse to <0.01 (saturation, v02 lesson); (c) arm beats
  vanilla on its primary by margin; (d) ⊖ damage arm WORSE than vanilla by
  margin on the same primary. Branching per v2.
- [nuisance] eligible for the single G2 retune: c(γ)-reweighting of the
  contrastive coefficient, γ-window restriction of mining.

## E1.3 MNIST EqM-mini — G3
- MNIST 28×28, pixel space scaled to [−1,1]. Field: small ConvNet
  (3 conv layers 32/64/64 + 2-layer head), noise-unconditional. 3,000 steps,
  bs 128, Adam 1e-3. Winning Stage-1 arm(s) vs vanilla. 3 seeds.
- Sample probe (MANDATORY): 64 images via GD sampler (η = 0.05, 300 steps,
  init N(0,I)); tiny-FID on 2,000 samples vs 2,000 held-out test images;
  feature extractor = small MNIST CNN classifier trained to ≥ 97% test acc
  (penultimate 64-d features); save grid PNG.
- G3: ε = 2× bootstrap SD of vanilla tiny-FID (resampled 2K-subsets, 50
  bootstrap reps) — NUMERIC VALUE recorded from the vanilla run BEFORE any
  treatment arm is sampled. PASS: tiny-FID(arm) ≤ tiny-FID(vanilla) + ε AND
  visual sanity (digit-like strokes ≥ majority of grid by eyeball, logged)
  AND realized risk ≤ α throughout (monitor log) AND abstention < 50%.
- [nuisance] eligible for G3 retune: k⁺/k⁻, contrastive coefficient.

## Gate statistics policy (P3 instantiation)
- Per-gate false-kill target ≤ 3%: achieved via 2×SD margins + ≥5 seeds at
  toys, 20 seeds at G0, binomial-correct G0 test.
- Post-retune passes flagged secondary-confidence in all reporting.
- Every gate's evaluation script writes a JSON verdict to results/ — gates are
  computed by code, not eyeballed (except E1.3 visual sanity, logged with PNG).
