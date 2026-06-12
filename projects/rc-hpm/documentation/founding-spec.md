# Risk-Controlled Hard-Pair Mining for Contrastive Learning (v1.1)
## Full pseudocode specification

Changes vs. the document's "fixed version":
- Exchangeable unit = **batch**, not pair (restores validity under top-k selection;
  selection rule is symmetric ⇒ batch-level CRC/LTT is valid).
- Joint thresholds (β⁺, β⁻) calibrated by **Learn-then-Test** (no monotonicity
  assumption; coupled risks handled by FWER control).
- Selection is **fully teacher-driven** (stationary pipeline ⇒ calibration does not
  expire as θ drifts). Student similarity is used only inside the loss.
- Drift monitor + fresh-fold recalibration as a safety valve.
- Synthetic mixed negatives are down-weighted and explicitly OUTSIDE the guarantee.
- All hardness/confidence weights are stop-gradiented.
- Loss is L_out multi-positive InfoNCE; ambiguous pairs get debiased soft weights,
  never deletion.

Notation:
  f_θ   student encoder + projector;  z = f_θ(x), unit-normalized
  g     frozen teacher encoder (EMA snapshot or pretrained); e = g(x)
  A_ψ   frozen bilinear pair-scorer:  q(x, x') = σ(e^T A_ψ e')  ∈ (0,1)
        (q ≈ P(same semantic class); trained on D_gate, then FROZEN)
  τ     InfoNCE temperature
  k⁺,k⁻ number of mined positives / negatives per anchor
  α⁺    target risk: gradient-weighted false-positive-pull damage per batch
  α⁻    target risk: gradient-weighted false-negative-push damage per batch
  δ     FWER level for LTT (overall guarantee: P(R⁺≤α⁺ and R⁻≤α⁻) ≥ 1−δ)

--------------------------------------------------------------------------------
## 0. Data splits  (guarantee dies if these leak into each other)
--------------------------------------------------------------------------------

```
SPLIT labeled pool D_lab into:
    D_gate        # train the pair-scorer A_ψ           (~40%)
    D_cal[1..M]   # M disjoint calibration folds        (~50%)
                  # one fold consumed per (re)calibration; never reused
    D_holdout     # final report of realized risk        (~10%)
D_unlab = the large unlabeled corpus (training data)

# n labeled examples ⇒ O(n²) labeled pairs. Pair labels:
#   y(x,x') = 1[class(x) == class(x')]  at a FIXED, declared granularity G.
# The guarantee is relative to G. Report this. (Multi-granularity eval in §6.)

# ASSUMPTION A1 (load-bearing, irreducible): D_lab must be a UNIFORMLY RANDOM
# labeled subsample of the SAME corpus as D_unlab, labeled at granularity G.
# Using an external curated labeled dataset (the common shortcut) breaks
# exchangeability between calibration batches and training batches at step one,
# and every guarantee downstream is void. Operationally: draw random examples
# from D_unlab and label them. Budget is the cost of validity.

# ASSUMPTION A1′ (v1.3 — in-sample/out-of-sample SYMMETRY): the frozen teacher g
# and gate A_ψ must treat calibration and training examples symmetrically.
# Failure mode: g pretrained on D_unlab while D_cal examples were held out of
# that pretraining ⇒ training-pair scores are in-sample for g, calibration-pair
# scores are out-of-sample ⇒ the score distributions differ ⇒ exchangeability
# of the LOSSES is broken even though the examples are exchangeable.
# Enforce: (a) sample D_lab FROM D_unlab and keep those examples inside any
# corpus used to train/EMA the teacher; (b) exclude D_gate examples from
# training batches (they are in-sample for ψ; calibration examples are not) —
# or report their fraction as a quantified exchangeability leak.

# Calibration batches must be constructed by the SAME sampler discipline as
# training batches (same batch size, same with/without-replacement scheme,
# same augmentation law). If training samples without replacement per epoch,
# use Hoeffding–Serfling bounds in §3 or sample calibration batches identically.
# Note: within-epoch batches are mutually dependent (no shared examples), but
# each batch's MARGINAL law matches; the cumulative-bias bound needs only
# linearity of expectation, so this dependence is harmless for §7(i).
```

--------------------------------------------------------------------------------
## 1. Gate training (once, then frozen)
--------------------------------------------------------------------------------

```
function TRAIN_GATE(D_gate, g):
    P ← all pairs (e_i, e_j, y_ij) from teacher embeddings of D_gate
    ψ ← minimize binary cross-entropy of σ(e_i^T A_ψ e_j) on P
        # cheap: bilinear head only; raw inputs never touched at train time
    FREEZE ψ
    return A_ψ
```

Note: calibration (§3) is valid for ANY gate, however bad — a bad gate only
inflates abstention, never risk. Do not tune ψ against D_cal.

--------------------------------------------------------------------------------
## 2. The mining pipeline (pure function — identical at calibration & training)
--------------------------------------------------------------------------------

All selection decisions use ONLY frozen quantities (teacher embeddings, gate
scores, calibration-fold statistics). θ appears nowhere here. This is what makes
the pipeline stationary and the calibration permanent (until deliberate refresh).

```
function MINE_BATCH(B, β⁺, β⁻, ρ̂):
    # B = {x_1..x_n} drawn by the standard sampler (same law as training batches)
    e_i ← g(x_i)  for all i                      # teacher embeddings
    s_ij ← e_i^T e_j                             # teacher similarity (selection!)
    q_ij ← σ(e_i^T A_ψ e_j)                      # gate score
    for each anchor i:
        CertPos_i ← { j : q_ij ≥ β⁺ }            # certified positive
        CertNeg_i ← { j : q_ij ≤ β⁻ }            # certified negative
        Amb_i     ← { j : β⁻ < q_ij < β⁺ }       # ambiguous (soft, never hard)

        # risk-weighted utility; interior max NOT assumed — argmax may be boundary
        u⁻(j) ← exp(s_ij / τ) · (1 − ρ̂(s_ij))          # for negatives
        u⁺(j) ← exp(−s_ij / τ) · (1 − ρ̂⁺(s_ij))        # for positives (hard = far)

        N_i ← top-k⁻ of CertNeg_i by u⁻          # mined hard certified negatives
        P_i ← top-k⁺ of CertPos_i by u⁺          # mined hard certified positives
    return {(P_i, N_i, Amb_i, s, q)}_i
```

ρ̂(s): empirical P(false pair | teacher similarity = s), fit by isotonic
regression on the CURRENT calibration fold (Stage-1 estimate). Held fixed
between recalibrations.

--------------------------------------------------------------------------------
## 3. Joint threshold calibration via Learn-then-Test
--------------------------------------------------------------------------------

Risk functionals (per batch, normalized to [0,1] — required by Hoeffding–Bentkus):

```
# v1.3 FIX: v1.2's L⁻ counted only mined certified negatives N_i. But the loss
# ALSO pushes ambiguous pairs with soft weight ω (§4) — false negatives in Amb_i
# were receiving uncertified repulsive gradient. The risk must cover ALL real
# repulsive mass the loss applies, and the normalization must be pinned.
# Definition: the risk is the FRACTION OF THE GRADIENT BUDGET spent on errors.

# surrogate weights (teacher s; see §3 scope caveat):
#   repulsive:  w_ij  ≜ exp(s_ij/τ)            (push force ∝ softmax weight)
#   attractive: w'_ip ≜ 1 − σ(s_ip/τ_w)        (pull force ∝ how UNaligned the
#               positive is — v1.4 fix: this was previously UNDEFINED, which
#               made L⁺ uncomputable; any fixed monotone-decreasing-in-s choice
#               works, but it must be pinned BEFORE calibration and never tuned
#               afterward, since it defines the controlled functional)
#   v⁻_ij, v⁺_ip, ω_ij : the SAME weights the loss uses (§4)

L⁻(B; λ) =  Σ_i [ Σ_{j∈N_i: y_ij=1} v⁻_ij·w_ij  +  Σ_{j∈Amb_i: y_ij=1} ω_ij·w_ij ]
           ─────────────────────────────────────────────────────────────────────
            Σ_i [ Σ_{j∈N_i}         v⁻_ij·w_ij  +  Σ_{j∈Amb_i}         ω_ij·w_ij ]
    ∈ [0,1]  # "≤ α⁻ of the total repulsive (push) budget hits same-class pairs"

L⁺(B; λ) =  Σ_i Σ_{p∈P_i: y_ip=0} v⁺_ip·w'_ip  /  Σ_i Σ_{p∈P_i} v⁺_ip·w'_ip
    ∈ [0,1]  # "≤ α⁺ of the total attractive (pull) budget hits cross-class pairs"
             # augmentation-view positives are true by construction: they enter
             # the denominator (budget) with numerator contribution zero.

# DEGENERATE-BATCH CONVENTION (v1.4): if a denominator is 0 (no certified or
# ambiguous mass — e.g. very strict λ, all-abstain batch), define L ≜ 0:
# no gradient applied ⇒ no damage. Without this the ratio is undefined and
# Hoeffding–Bentkus boundedness fails on exactly the strict-λ grid points.

# Dimensionless, physically interpretable, bounded — α now has a meaning that
# survives into the bias bound without an arbitrary (n·k) constant.
# Both risks depend on BOTH thresholds through the coupled top-k competition
# AND are ratios (no monotonicity in λ even threshold-wise) ⇒ LTT mandatory.
# Synthetic mixed negatives remain OUTSIDE these functionals: unlabeled,
# uncertifiable, hence c_syn = 0 in the theorem-grade arm (§7).
```

```
function CALIBRATE_LTT(D_cal_fold, α⁺, α⁻, δ):
    # --- BUG FIX (v1.2): the fold must be split. ρ̂, ρ̂⁺, ρ_amb, and the grid
    # ordering enter the DEFINITION of the loss/selection rule. Fitting them on
    # the same data that computes the p-values makes the loss data-dependent and
    # the LTT p-values invalid — the same sin, one level down, as training the
    # gate on calibration data. ---
    (D_fit, D_test) ← split D_cal_fold 50/50

    fit on D_fit ONLY:
        ρ̂, ρ̂⁺       (isotonic regression of 1[error] on teacher similarity s)
        ρ_amb(q)     (P(same class | gate score q), for ambiguous soft weights §4)
        the fixed-sequence PATH through the (β⁺,β⁻) grid
                     (a 2D grid needs a pre-specified test order; choosing the
                      path on D_fit preserves validity and improves power)

    construct m calibration batches B_1..B_m from D_test:
        DEFAULT: sample each batch INDEPENDENTLY by one training-batch draw
        (i.i.d. at the batch level ⇒ batch losses i.i.d. ⇒ Hoeffding–Bentkus
        applies as stated). If instead partitioning D_test without replacement,
        batch losses are negatively dependent: keep Hoeffding–Serfling, DROP
        the Bentkus component.
    for λ = (β⁺,β⁻) along the D_fit-chosen path (in order):
        compute losses ℓ⁻_t = L⁻(B_t; λ), ℓ⁺_t = L⁺(B_t; λ)  for t = 1..m
        p⁻(λ) ← HoeffdingBentkus_pvalue(mean(ℓ⁻), m, α⁻)
        p⁺(λ) ← HoeffdingBentkus_pvalue(mean(ℓ⁺), m, α⁺)
        p(λ)  ← max(p⁻(λ), p⁺(λ))        # both risks must hold
        if p(λ) ≤ δ_r:  mark λ VALID
        else:           STOP (fixed-sequence testing)     # FWER ≤ δ_r
            # δ budget (v1.3): the NUMBER of recalibrations is random (drift-
            # triggered), so budget over the fold CAP, not a plan: δ_r = δ/M.
    if no VALID λ: ABORT — report "labeled fold too small or gate too weak;
                   train un-mined baseline"
        # honest framing: the fallback is STANDARD, not SAFE — plain contrastive
        # training has uncontrolled pair-label damage; ABORT surrenders the
        # guarantee rather than faking one.
    λ* ← among VALID λ, argmax expected utility throughput
         # validity unaffected (all VALID λ are certified); the throughput
         # estimate itself is optimistically biased — report it from D_holdout
    record drift reference statistics (monitoring only — reuse is harmless):
         F_ref ← empirical CDF of selected-pair teacher scores {s_ij : j ∈ N_i ∪ P_i}
         a_ref ← mean abstention rate |Amb| / n²
    return λ*, ρ̂, ρ̂⁺, ρ_amb, F_ref, a_ref
```

Guarantee delivered (state as the paper's Theorem, part i):
  With prob ≥ 1−δ_r over the calibration fold, for a FRESH batch drawn from the
  same law and pushed through the same frozen pipeline,
      E[L⁺] ≤ α⁺  and  E[L⁻] ≤ α⁻ .
  Valid for any gate; degrades only through abstention. Because the pipeline is
  θ-free, "fresh batch" includes every future training batch — no drift caveat
  EXCEPT changes to the sampler/augmentations (monitored in §5).

SCOPE CAVEAT (v1.2, do not hide this): L± weight errors by the TEACHER-similarity
surrogate exp(s/τ). The realized gradient damage at step t is weighted by the
STUDENT's softmax under ŝ_θt. The theorem controls the surrogate exactly and the
realized damage only insofar as the two correlate; the gap grows as the student
diverges from the teacher. Controlling realized damage directly would require a
θ-dependent loss and destroy stationarity. Mitigation: (a) report the empirical
surrogate↔realized damage correlation over training (E10); (b) refresh the
teacher (with mandatory recalibration on a fresh fold) when correlation decays.

Optional simplification: if monotonicity of each risk in its own threshold is
EMPIRICALLY verified on a scout fold, replace LTT with two independent CRC runs
at levels α⁺·(m/(m+1)) − 1/(m+1) etc. Verify; do not assume.

--------------------------------------------------------------------------------
## 4. Loss (L_out multi-positive InfoNCE, debiased ambiguous mass)
--------------------------------------------------------------------------------

```
function LOSS(batch, mined, θ):
    z_i ← f_θ(x_i); normalize                       # STUDENT embeddings
    ŝ_ij ← z_i^T z_j / τ                            # student logits (these get grads)

    for each anchor i:
        # --- denominator: certified negs (weighted) + ambiguous (debiased soft) ---
        # ambiguous soft weight: ω_ij = stopgrad( (1 − ρ_amb(q_ij)) · c_amb )
        #   where ρ_amb(q) = P(same class | gate score q), fit on D_fit ONLY (§3);
        #   importance-weight treatment (PU/debiased correction), NOT deletion.
        # mixed synthetic negatives (MoCHi over CertNeg only): weight c_syn < 1,
        #   OUTSIDE the guarantee — excluded from L± risk accounting.
        D_i = Σ_{j∈N_i}  stopgrad(v⁻_ij) · exp(ŝ_ij)
            + Σ_{j∈Amb_i} stopgrad(ω_ij)  · exp(ŝ_ij)
            + Σ_{mixed m} stopgrad(c_syn) · exp(ŝ_im)
          where v⁻_ij = (1 − ρ̂(s_ij)) · hardness_weight(s_ij)   # TEACHER s, stopgrad

        # --- L_out form: log OUTSIDE the average over positives ---
        L_i = − (1/|P_i|) Σ_{p∈P_i} stopgrad(v⁺_ip) · log( exp(ŝ_ip) / (exp(ŝ_ip) + D_i) )

    return mean_i L_i
```

Hard rules:
  R1. Every weight that depends on similarity is stop-gradiented (otherwise the
      model is rewarded for pushing positives away to raise their hardness).
  R2. Weights computed from TEACHER similarity s, never student ŝ
      (keeps the selection/weighting pipeline θ-free).
  R3. Ambiguous pairs never appear as positives and never get full negative
      weight; report Σω (effective negative pool size) so the partition-function
      shift is visible in ablations, not silent.

--------------------------------------------------------------------------------
## 5. Training loop with drift monitor and recalibration
--------------------------------------------------------------------------------

```
function TRAIN(D_unlab, D_cal[1..M], α⁺, α⁻, δ):
    A_ψ ← TRAIN_GATE(D_gate, g)
    (λ*, ρ̂, ρ̂⁺, F_ref, a_ref) ← CALIBRATE_LTT(D_cal[1], α⁺, α⁻, δ);  fold ← 2

    for step t = 1..T:
        B ← sample batch from D_unlab
        mined ← MINE_BATCH(B, λ*, ρ̂)
        θ ← θ − η ∇_θ LOSS(B, mined, θ)

        # ---- monitoring (the guarantee's tripwires) ----
        log: abstention rate a_t, selected-score CDF F_t, Σω_t, utility throughput
        every T_check steps:
            # v1.4: the monitor runs T/T_check tests over training — without
            # multiplicity control, false alarms are near-certain and each one
            # BURNS a calibration fold. Bonferroni the monitor: alarm level
            # κ set for per-test level δ_mon/(T/T_check). Under a truly θ-free,
            # fixed-sampler pipeline F_t cannot drift, so alarms should be rare;
            # additionally, any DELIBERATE pipeline change (sampler, augmenta-
            # tions, teacher refresh) triggers recalibration unconditionally,
            # without spending the statistical monitor.
            if KS(F_t_window, F_ref) > κ  or  |a_t_window − a_ref| > κ_a:
                if fold ≤ M:
                    (λ*, ρ̂, ρ̂⁺, F_ref, a_ref) ← CALIBRATE_LTT(D_cal[fold], α⁺, α⁻, δ)
                    fold ← fold + 1          # FRESH fold: adaptivity never
                                             # contaminates a reused fold
                else:
                    # v1.4 HONESTY FIX: drift detected means the old validation
                    # is VOID — there is no "certified-conservative" fallback;
                    # any λ from a voided calibration is uncertified, however
                    # strict. The only honest options:
                    #   (a) STOP mining: revert to un-mined training (standard,
                    #       not safe — but makes no false guarantee), or
                    #   (b) continue mining at the last λ* with the guarantee
                    #       EXPLICITLY WITHDRAWN in all logs and reports.
                    # Pick (a) by default. Never report a risk level after this.
    return θ
```

Why drift can occur at all despite a θ-free pipeline: curriculum changes to the
sampler, augmentation schedule changes, or a deliberate teacher refresh. If the
teacher g is EVER updated (e.g., periodic EMA snapshot to improve mining), that
is a NEW pipeline: mandatory recalibration on a fresh fold. Never update g
between recalibrations.

--------------------------------------------------------------------------------
## 6. Evaluation harness (the experiments that make or break the paper)
--------------------------------------------------------------------------------

```
E1  Motivation figure: on D_holdout, plot ρ(s) and gradient-weighted damage
    density w(s)·ρ(s) vs s. The claim "mining concentrates errors where they
    hurt most" must be visible here or the paper has no premise.
E2  Baselines, every stage: SimCLR/MoCo base; RINCE; τ-sweep (adaptive-temperature
    attack); NNCLR / soft-neighbor; debiased-CL; FN-cancellation; SupCon oracle
    (upper bound). Pre-register the prediction: robust losses win under diffuse
    noise, certification wins when errors concentrate in the hard tail
    (fine-grained / near-duplicate corpora).
E3  Guarantee verification: realized E[L⁺], E[L⁻] on D_holdout vs (α⁺, α⁻),
    across seeds. This is the table reviewers check first.
E4  Gate-corruption stress test: corrupt A_ψ (label noise in D_gate, ablated
    capacity). Show: abstention inflates, realized risk stays ≤ α (LTT arm) while
    μ±κσ heuristic arm silently exceeds its nominal risk. Load-bearing experiment.
E5  Selection-conditioning ablation: calibrate marginally (pair-level, pre-
    selection) vs batch-level post-selection. Show marginal calibration UNDER-
    covers on mined pairs — this empirically justifies the paper's central move.
E6  Granularity sweep: relabel pairs at multiple hierarchy levels (e.g., WordNet
    depths); rerun E3. The guarantee should hold at the declared granularity and
    measurably break at finer ones — honest scoping, stated as a feature.
E7  Theory arm: spectral contrastive loss variant (HaoChen) trained with the same
    mining, so Theorem part (ii) — subspace error O(α/(δγ)) via Davis–Kahan +
    Markov — is exact for a trained model, not just motivational for InfoNCE.
E8  Ablations: utility selection vs fixed hardness window; debiased-soft vs
    delete-ambiguous (track Σω); mixed synthetics on/off; L_out vs L_in;
    stop-grad on/off (expect visible degeneracy without it).
E9  Validity–power tradeoff: teacher-driven mining (stationary, certified) vs
    student-driven mining (stale-prone guarantee, sharper hardness) vs
    student-driven + per-epoch recalibration. Measures what the θ-free design
    costs in accuracy — the honest price of the guarantee.
E10 Surrogate fidelity: correlation over training between teacher-surrogate
    damage exp(s/τ) and realized student-gradient damage exp(ŝ/τ)/Z on labeled
    pairs from D_holdout_monitor. If this decays, the §3 scope caveat is biting;
    triggers teacher refresh + recalibration.
    # v1.4: split D_holdout into D_holdout_monitor (repeated use during
    # training, E10) and D_holdout_report (touched ONCE, for E3's final risk
    # table) — otherwise repeated monitoring adaptively contaminates the
    # number you publish.
```

--------------------------------------------------------------------------------
## 7. Theorem, restated honestly (v1.3)
--------------------------------------------------------------------------------

(0)  [Assumptions, all load-bearing] A1 (calibration data is a random labeled
     subsample of the training corpus at granularity G); A1′ (teacher and gate
     treat calibration and training examples in-sample/out-of-sample
     symmetrically); frozen teacher and gate between recalibrations;
     calibration/training batches share one sampling law;
     for part (ii): spectral contrastive loss, sufficient capacity, and global
     optimization (HaoChen-style idealization); synthetic mixing OFF (c_syn = 0)
     in the theorem-grade arm.

(i)  [Risk control — finite sample, distribution-free] With probability ≥ 1−δ_r
     over the calibration fold: for a fresh training batch, the expected
     fraction of the SURROGATE attractive gradient budget spent on cross-class
     pairs is ≤ α⁺, and of the repulsive budget spent on same-class pairs is
     ≤ α⁻ — covering certified-mined AND soft-weighted ambiguous mass, i.e.
     ALL real-pair gradient the loss applies (synthetics excluded by c_syn = 0).
     Valid for any gate. PRECISION FIX (v1.4): there is no conditioning event
     in this statement — the selection (trichotomy + top-k) is INTERNAL to the
     loss functional, so the guarantee is MARGINAL over batch draws, with
     selection effects priced in. Earlier "conditional on the selection event"
     phrasing conflated this construction with selective-inference conditioning
     (Jin–Ren style) and would draw a correct objection from that community:
     we do not condition on selection, we integrate over it inside the risk.
     Union over at most M recalibrations holds at level δ = M·δ_r.

(ii) [Representation — spectral-loss arm] Hard mining itself reweights the
     augmentation graph even with ZERO label errors: it defines an intended
     mined graph Ã (edges importance-weighted by the selection rule). The
     correct comparison is perturbed-Ã vs intended-Ã, NOT vs the unmined graph.
     Then: label errors perturb Ã by expected relative (surrogate-)weighted edge
     mass ≤ α⁺+α⁻; with probability ≥ 1−δ−δ′ (Markov factor stated, not
     hidden), and by Davis–Kahan the representation subspace of the mined-graph
     embedding rotates by O((α⁺+α⁻)/(δ′·γ̃)), where γ̃ is the spectral gap of
     the INTENDED mined graph. CAVEAT (v1.3): the step from relative edge-mass
     to the operator norm ‖ΔL̃‖₂ involves norm conversions (entrywise-L1 → 
     Frobenius → operator) that can hide graph-volume factors; the O(·) is
     order-of-magnitude for the normalized Laplacian under relative-mass
     perturbation, and the constant must be tracked in the formal proof, not
     assumed to be 1. Excess linear-probe risk is controlled relative to the
     clean-mined representation. Whether the clean-mined representation beats
     the unmined one is a separate, EMPIRICAL question (E2/E9) — no theorem
     here claims mining helps; the theorem says mining cannot silently hurt
     beyond the certified budget. Claimed for the spectral-loss arm; reported
     empirically for InfoNCE.

(iii)[Graceful degradation] Gate quality affects abstention rate and utility
     throughput — both observable — never the risk level.

What remains UNPROVABLE in this framework (state in the limitations section):
  - A1 cannot be verified from data alone; it is a design discipline.
  - The surrogate↔realized damage gap (§3 caveat) is monitored, not bounded.
  - Granularity G is a modeling choice; "false negative" has no G-free meaning.
  - No guarantee of benefit — only certified bounded harm plus empirical benefit.
