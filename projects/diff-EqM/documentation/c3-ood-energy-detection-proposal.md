# C3 — OOD Detection via EqM Energy — Capability Proposal

Per CLAUDE.md "Research Process Rules for EqM / ANM". Mechanism check filled before code.
Capability experiment. **Frozen checkpoints only — no retraining.**

## Name
`c3_ood_energy_detection`

## Hypothesis
EqM is an *explicit* energy-based model: `forward(..., get_energy=True)` returns a per-sample scalar
energy `E` (models.py:257-272; `E = -½‖h‖²` for `ebm='l2'`, or `⟨h, x⟩` for `ebm='dot'`), with the
sampling field `f = ∇_{x0} E`. A well-shaped energy should be **low on in-distribution data and high
off-manifold** — making `E` (or `‖∇E‖`) a ready OOD score, a *discriminative* use of a *generative*
model.

Vanilla EqM only sees on/near-trajectory points during training, so its energy is uncalibrated where
it matters for OOD (far from data). v10 (ANM) explicitly trains the field — hence the energy's
gradient — to be correct at adversarially-mined off-manifold points (Exp 2 CONFIRMED). Prediction:
**v10's energy separates in-distribution from OOD better than vanilla's** (higher AUROC), and the
margin grows with mining dose (λ=0.3 > λ=0.1 > vanilla).

This is the cleanest "new feature" framing: a metric axis (OOD AUROC) entirely orthogonal to FID,
where the mechanism Exp 2 confirmed should produce a categorical, quantitative win.

## Failure mode addressed
FID and the current 4-experiment suite never query the energy as a *scalar discriminator*. Yet the
energy is the object ANM most directly reshapes (field = its gradient). C3 reads out the quantity the
mechanism most directly improves, on a task (OOD detection) with a standard, legible metric.

## EqM compatibility argument
Pure read-out of the model's own scalar energy at fixed inputs — no objective change, no sampler,
no new loss. We evaluate `E(x)` (and `‖∇E‖`, and a few-step descent ΔE) on in-dist vs OOD batches
and compute AUROC. Native EBM functionality already wired (`get_energy=True`, used by v04 EBM
variant and the capability eval). This is the lowest-risk, lowest-cost of the three probes.

## Measurement definition
Arms: {vanilla 31.41, v10 λ=0.1 29.01, v10 λ=0.3 27.09}, frozen EMA. Latent space (encode through
the same VAE used in training). Class-conditional energy: score with the true label where available,
else marginalize / use the unconditional token.

In-distribution: held-out IN-1K val (N≥4000, class-balanced).
OOD sets (standard battery, encoded to the same latent space):
- Near-OOD: IN-1K classes held out from a chosen subset (leave-out-class), or iNaturalist.
- Far-OOD: SVHN, Textures/DTD, Gaussian/Perlin noise, uniform noise.

Scores per sample (evaluate several, report best + all):
1. `E(x)` at the data point (t→1 / equilibrium time).
2. `‖∇E(x)‖ = ‖f(x)‖` — equilibrium residual (should be ~0 on-manifold, large off).
3. ΔE over k short descent steps (how much energy the point can still shed — in-dist already low).
4. Energy at a fixed intermediate flow time (sweep t, pick most separating).

Metric: **AUROC** (and AUPR, FPR@95%TPR) in-dist vs each OOD set, per arm. Headline:
**ΔAUROC(v10 − vanilla)** per OOD set, dose-ordered by λ.

## Expected diagnostics if working
- v10 AUROC > vanilla AUROC on far-OOD (largest gap), positive on near-OOD; dose-ordered
  λ=0.3 ≥ λ=0.1 ≥ vanilla.
- `‖∇E‖` (residual) is the strongest score and most improved by ANM — directly the Exp 2 quantity
  (off-manifold field magnitude), now used discriminatively.
- In-dist energy histograms similar across arms (ANM didn't distort in-dist); OOD energy pushed
  **higher** by ANM → cleaner separation.

## Expected diagnostics if failing
- ΔAUROC ≈ 0 everywhere → energy reshaping too small at B/2 to discriminate (consistent with NULL
  capability eval / small Exp 2 effect) → escalate λ / scale before kill.
- Vanilla already saturates AUROC (≈1.0) on far-OOD → no headroom; restrict claim to near-OOD where
  headroom exists.
- v10 distorts in-dist energy (in-dist histogram shifts) → mining harmed calibration; report.

## Minimal test
`E(x)` + `‖∇E‖` scores, in-dist IN-1K-val (N=2000) vs 2 OOD sets (SVHN far + leave-out-class near),
all 3 arms. AUROC table. **<0.5 GPU-day, no sampler, no decode.** If v10 beats vanilla on ≥1 set with
dose-ordering → expand to full OOD battery + score variants (~0.5-1 GPU-day).

## Promotion rule
PROMOTE ("ANM yields a better OOD detector from EqM's energy") if v10 AUROC exceeds vanilla by a
CI-separated margin on ≥2 OOD sets, with dose-ordering λ=0.3 ≥ λ=0.1 ≥ vanilla, AND in-dist energy
calibration is preserved (no in-dist histogram shift). Report against standard generative-OOD
baselines (likelihood, ‖score‖) for context.

## Kill rule
KILL (1-retune cap on score-choice / time t) if the minimal test shows ΔAUROC ≈ 0 and one battery
extension confirms no separation. On kill: energy is not a B/2 differentiator → log postmortem;
energy mechanism still supports the Exp 2 field-robustness story but not a standalone OOD claim.

## Anti-laundering note
An OOD-detection claim is made ONLY with proper AUROC vs standard baselines and dose-ordering as the
mechanism control — not from a single cherry-picked OOD set. Beating vanilla on noise-OOD but not
semantic-OOD is reported as exactly that, not as "OOD detection."
