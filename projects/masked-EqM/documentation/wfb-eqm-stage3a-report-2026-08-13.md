# Stage 3A report — why certified CG-FBGN optimizes every minibatch while worsening the held-out EqM field

**Date**: 2026-08-13
**Job**: 38985448 (gpu_test, COMPLETED, 01:08:54, exit 0), git_sha `27c227a`
**Rows**: `runs/wfb_stage3a_38985448/stage3a_rows.jsonl` (336)
**Analysis**: `experiments/direct_energy/analyze_stage3a_discriminator.py`
**Pre-registration**: `wfb-eqm-stage3bc-preregistration-2026-08-13.md` (written before these rows existed)
**Audit it follows**: `wfb-eqm-stage3-audit-2026-08-13.md`

---

## 1. Question

Explain why certified CG-FBGN can optimize each minibatch while worsening the held-out
EqM field, and discriminate **H1** (Gauss–Newton local-model failure) from **H2**
(stochastic minibatch over-solving) with a controlled intervention.

## 2. Protocol actually run

No training, no optimizer, no Adam, no parameter persistence. Every candidate update
was **apply → evaluate → revert**, and every revert was verified to restore θ *bitwise*
(`assert_exact_restore`, max|Δ| required to be exactly `0.0`; it was, on all 336 rows).

Three frozen checkpoints:

| tag | checkpoint | role |
|---|---|---|
| `start` | `fwrev_ep80_lambda0_job37780076/.../2825000.pt` | the healthy model FBGN training actually began from — **the decisive one** |
| `fbgn100` | `wfb_stage3_full_alpha1p0_cg_step100.pt` | already-damaged, 100 FBGN steps in |
| `fbgn300` | `wfb_stage3_full_alpha1p0_cg_step300.pt` | already-damaged, end of the 300-step run |

Three predetermined banks drawn from one deterministic pool (`build_pool(seed=0,
pool_size=324)`), disjoint by construction, corruption frozen once and reused:

- **B** — 8 model minibatches, batch size 8 (`pool[308:316]`), the systems GN is built on;
- **V** — 64 independent examples (`pool[316:324]`), the *trust bank* for transfer;
- **P** — deterministic global probe (`pool[300:308]`), **evaluation only** — never used
  for acceptance, damping, step selection, or retries.

`pool[0:300]` are the exact training batches of the 300-step runs and were untouched here.

**Probe integrity**: re-evaluated twice per checkpoint, delta exactly `0.000e+00`
(10.624368667602539 / 14.115535736083984 / 14.881022453308105). No repair needed.

**Direction certification**: matrix-free CG on `(A + λI)u = r`, `A = MMᵀ`, with the
**true** residual recomputed from a fresh operator application (never CG's recursive
one). All 8 systems at `start` converged, 195–229 iterations, true relative residual
0.0115–0.0159 (≤ 0.02 as required). The reported CG residual understated the true one
by ~1.9× — which is exactly why the true one is the certificate.

## 3. Headline numbers at `start` (the checkpoint that matters)

`L_P = 10.6244`, `L_V = 10.9232`, `‖g_V‖ = 1.2040`, `λ = ρ·λ_max = 68.05`,
`‖Mp‖/‖r‖ = 0.656–0.735`, `η* = 1.215–1.343`, `‖p‖ = 22.8–32.2`.

### 3.1 η-scan, certified FBGN (medians over the 8 model batches)

| η/η* | R_B | D_B | median ΔL_V | worsened V | median ΔL_P | worsened P |
|---|---|---|---|---|---|---|
| 1     | **−10.685** | **3.3967** | +44.42 | 8/8 | +51.22 | 8/8 |
| 1/2   | −0.848 | 2.0585 | +6.233 | 8/8 | +5.824 | 8/8 |
| 1/4   | +0.617 | 0.9618 | +0.469 | 8/8 | +0.454 | 8/8 |
| 1/8   | +0.910 | 0.4593 | **+0.0629** | **8/8** | +0.0610 | 8/8 |

### 3.2 η-scan, raw-direct negative control (same batches, native scale)

| η/η* | R_B | D_B | median ΔL_V | worsened V |
|---|---|---|---|---|
| 1     | +0.973 | 0.2260 | +0.2026 | 8/8 |
| 1/2   | +0.989 | 0.1252 | +0.0345 | 7/8 |
| 1/4   | +0.998 | 0.0651 | +0.0046 | 6/8 |
| 1/8   | +0.999 | 0.0361 | +0.0003 | 4/8 |

### 3.3 η-independent transfer `d_V = g_V·p` and normalized alignment `C_V`

| checkpoint | ‖g_V‖ | FBGN C_V | FBGN descent | direct C_V | direct descent |
|---|---|---|---|---|---|
| `start`   | 1.204   | **+0.00108** | **4/8** | +0.03954 | 5/8 |
| `fbgn100` | 74.588  | +0.01679 | 8/8 | +0.97884 | 8/8 |
| `fbgn300` | 192.173 | +0.00978 | 8/8 | +0.88241 | 8/8 |

### 3.4 Damping sweep (the pre-registered H1 repair, folded into this job)

| checkpoint | λ×1 C_V / desc | λ×10 C_V / desc | λ×100 C_V / desc |
|---|---|---|---|
| `start`   | +0.00108 / 4/8 | **−0.00314 / 1/4** | **−0.00829 / 1/4** |
| `fbgn100` | +0.01679 / 8/8 | +0.05756 / 4/4 | +0.18698 / 4/4 |
| `fbgn300` | +0.00978 / 8/8 | +0.04206 / 4/4 | +0.14862 / 4/4 |

At `start`, λ×100 does repair the local model (R_B −10.685 → +0.150, D_B 3.397 → 0.907
at η_frac = 1) — and simultaneously drives median `d_V` from −0.0367 to **+0.0177** and
`C_V` **negative**. It fixes the linearization and destroys the transfer.

## 4. Classification

The mechanical rule in the analyzer labels `start` **H3** (both `R_B`/`D_B` bad at η*
*and* `d_V ≥ 0` on half the batches). That label is correct in the letter, but the two
halves are not equally load-bearing, and the η-scan separates them cleanly:

- **The H1 component is real but is a step-length artifact, and it is curable by η.**
  D_B falls 3.397 → 0.459 and R_B rises −10.685 → 0.910 monotonically as η → η*/8,
  which is the textbook O(η) behaviour of a *correct* linearization evaluated too far
  out. Nothing is wrong with the Gauss–Newton model itself; η* ≈ 1.3 is simply far
  outside its trust region. This is not a mystery and not the cause of the 300-step
  failure.

- **The H2 component is η-irreducible and survives its own repair.** At η*/8, where the
  local model is now decent (R_B = 0.91, D_B = 0.46), **all 8 batches still worsen the
  trust bank**, ΔL_V = +0.0629 median, paired SE 0.013–0.020 per batch → ~3–6.5σ each,
  with only 23–30% of the 64 trust examples improving. Halving η again would shrink the
  damage but never change its sign, because `d_V` — which is η-independent — is already
  positive on half the batches. Per §8 of the directive that is H2 in its decisive form:
  *no step size and no damping can rescue a direction that is not a descent direction
  for the population objective.*

**Verdict: H2 is the cause. H1 is present, is a trust-region artifact, and its
pre-registered repair (damping) FAILED its own pre-registered falsifier** — "stronger
damping … kills independent-batch descent." It did exactly that.

## 5. Mechanism

The 8-image GN system at `start` contains almost no population signal, and GN whitening
removes what little there is.

1. **The raw B=8 minibatch gradient is already nearly orthogonal to the population
   gradient at this checkpoint**: `C_V = 0.0395`, i.e. cosine ≈ 4%, descent on only 5/8
   batches, and a *finite* step along it worsens V on 8/8 batches at η* and 4/8 even at
   η*/8. This is not an FBGN defect. It is a property of a converged scalar-energy EqM
   checkpoint: at `‖g_V‖ = 1.204` the population gradient is small, so per-minibatch
   gradient noise dominates the signal at B = 8.

2. **Whitening by `(A + λI)^{-1}` is an anti-filter for generalization here.** GN
   multiplies each field direction by ≈ 1/(σ² + λ). The small-σ directions of `MMᵀ` are
   precisely the field directions that *only these 8 images constrain* — the ones with
   no population support — and the solve amplifies them most. Measured consequence:
   alignment drops from `C_V = 0.0395` (raw gradient) to `C_V = 0.00108` (certified
   FBGN), a **36× loss of alignment**, while ‖Mp‖/‖r‖ = 0.66–0.74 confirms the direction
   is doing real work on its own batch. That is the entire phenomenon in one pair of
   numbers: **FBGN removes 66–74% of the source-batch residual while retaining 0.1% of
   the population gradient direction.**

3. **Certification makes it worse, not better.** Solving to a true residual of 0.013
   guarantees fidelity to the *minibatch* Gauss–Newton model. Since that model is ~99.9%
   batch-idiosyncratic at this checkpoint, higher solve accuracy buys more faithful
   fitting of noise. This is why 100% Armijo acceptance and monotone same-batch
   reduction coexisted with monotone probe damage for 300 steps: both statements were
   true, about different objectives.

4. **Why the damaged checkpoints look "healthy".** At `fbgn100`/`fbgn300`, `‖g_V‖` is
   74.6/192.2 and `direct C_V` is 0.98/0.88 — the model is broken enough that one common
   error direction dominates the landscape and almost anything descends. Their apparent
   H1 signature (`R_B` ≈ 0.44–0.52, `d_V` < 0 on 8/8) is a consequence of being far from
   any optimum, not evidence about the mechanism. **Do not read the mechanism off
   `fbgn100`/`fbgn300`.** Only `start` answers the question, because only `start` is the
   checkpoint the failure began from. Notably, damping *does* behave as theory predicts
   there (`C_V` 0.017 → 0.187, 10×) — it recovers gradient alignment when there is
   alignment to recover. At `start` there is none, so damping merely walks toward a
   direction that is itself orthogonal.

## 6. Claims explicitly NOT made

Per §19 of the directive, and none of these are supported by these rows:

- *"FBGN is curvature blind."* It is not; `D_B → 0` as O(η) proves the model is correct.
- *"λ_max growth caused divergence."* Retracted in the audit; nothing here revives it.
- *"Backbone sharpening is the root cause."* Not tested, not claimed, not the next thread.
- *"100% Armijo acceptance proves the local model is accurate."* §3.1 refutes this
  directly: at η*, R_B = −10.7 while Armijo accepted every step.
- *"Adaptive λ fixes stochastic batch overfitting."* §3.4 shows the opposite at `start`.

## 7. What follows, and what would kill the thread

The only pre-registered intervention the classification supports is the **H2** branch:
enlarge the stochastic model batch, via **one stacked solve**, never an average of
separate solves (`stacked_mixed_gram_mv` / `compute_fbgn_gradient_cg_microbatched`,
validated in `tests/test_stacked_microbatch_gn.py` against an explicit dense `MMᵀ` and
*against* the block-diagonal shortcut it must not match).

The mechanism in §5 makes a sharp, cheap, falsifiable prediction that should be tested
**before** paying for any stacked CG solve:

> If B = 8 is signal-starved, then `C_V(B)` for the **plain** minibatch gradient must
> rise materially with B at `start`. Pure noise-averaging gives `C_V ∝ √B`, i.e.
> 0.0395 → ~0.079 at B = 32 → ~0.158 at B = 128 — nowhere near enough for GN whitening
> to survive a 36× alignment loss.

That is a gradient-only measurement, minutes of GPU, no CG. Its outcome sets the branch:

- `C_V(B)` rises **faster than √B** and reaches a usable level → run the stacked GN
  solve at the smallest such B, per the pre-registered escalation rule (32 first; 64
  only if 32 materially helps but is clearly not enough).
- `C_V(B)` tracks √B → the minibatch and population objectives are decoupled at this
  checkpoint at every batch size FBGN can afford, and **the pre-registered kill
  condition fires.** The honest result is then:

  > Mixed-Jacobian whitening explains and suppresses the gradient-spike mechanism, but
  > Gauss–Newton optimization of the scalar EqM field does not provide a useful
  > stochastic training trajectory at a converged checkpoint: the B = 8 minibatch GN
  > model retains 0.1% of the population gradient direction, and damping trades local
  > model fidelity against transfer rather than improving both.

Nothing else is authorized: no 300-step run, no Krylov reuse, no low-rank curvature, no
backbone localization.
