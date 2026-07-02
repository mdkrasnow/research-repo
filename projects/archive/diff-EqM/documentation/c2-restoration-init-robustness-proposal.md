# C2 — Restoration / Robustness to Corrupted & OOD Initialization — Capability Proposal

Per CLAUDE.md "Research Process Rules for EqM / ANM". Mechanism check filled before code.
Capability experiment. **Frozen checkpoints only — no retraining.**

## Name
`c2_restoration_init_robustness`

## Hypothesis
EqM samples by gradient-descending the scalar energy `E` to an equilibrium (field `f = ∇E → 0`).
The standard sampler starts from Gaussian noise — always inside the trajectory tube where the field
was trained. **Image restoration / editing** instead starts from an *off-manifold* point (a noisy,
blurred, masked, or arbitrary real image) and asks EqM to descend to the nearest clean mode. That
init lives exactly in the region where Exp 2 showed vanilla's field is least accurate and v10's is
most improved — the v10-**mined** δ (Exp 2: largest gap, dMSE −0.0368 at the perturbation ANM trains
on; λ=0.3: −0.0565).

Prediction: **vanilla EqM, descended from a corrupted init, converges to a worse / wrong mode (its
field is unreliable there) while v10 restores faithfully — and the v10 advantage GROWS with
corruption severity** (deeper off-manifold = where the field gap widens). This is a capability the
paper-recipe inpaint eval (job 17086244, NULL at B/2) did not stress: that eval used a single mild
mask. C2 sweeps severity to find the separation.

## Failure mode addressed
The NULL B/2 capability eval tested one mild corruption and concluded "no gap." But Exp 2 proves the
field gap is real and **widens with off-manifold distance**. A single mild operating point cannot see
a gap that only opens up under stress. C2 measures the severity axis the NULL eval collapsed.

## EqM compatibility argument
Restoration is the native EqM inference mode — gradient descent on `E` (the EqM paper's §5 editing /
inpainting use exactly this). We change only the **initialization** (corrupted image instead of noise)
and, for inpainting, project known pixels each step (standard EqM masked-sampling). No objective
change, no new loss. `get_energy=True` gives both the descent direction and the per-step energy for
convergence monitoring. Reuse `experiments/eval_capabilities.py` (already implements EqM
denoise/inpaint/compose).

## Measurement definition
Arms: {vanilla 31.41, v10 λ=0.1 29.01, v10 λ=0.3 27.09}, frozen EMA. Held-out IN-1K val images
(class-balanced, N≥2000), latent space, matched seeds/latents across arms.

Corruption families, each swept over a **severity ladder**:
1. **Gaussian noise** init: x_corrupt = x_clean + σ·ε, σ ∈ {0.1, 0.25, 0.5, 1.0, 2.0}.
2. **Blur** init: Gaussian blur radius r ∈ {1,2,4,8 px} (pixel space, then re-encode).
3. **Mask / inpaint**: square hole fraction ∈ {0.1, 0.25, 0.5, 0.75}; descend with known-pixel
   projection.
4. **OOD init**: start descent from a *different-class* val image (semantic restoration / nearest
   in-class mode) and from pure structured noise (e.g. Perlin) — pure basin-of-attraction probe.

Per (arm × family × severity) record:
- **Restoration fidelity**: PSNR + LPIPS + DINO-cosine to the clean target.
- **Convergence**: steps to `‖f‖ < τ`; final energy `E`; fraction reaching equilibrium vs diverging.
- **Basin**: fraction of inits that land in the correct class (probe with a pretrained classifier on
  the decoded result) / land in a high-quality mode.

Headline curve: **Δ(v10 − vanilla) restoration metric vs severity** — predicted to be ≥0 everywhere
and **increasing** with severity, dose-ordered by λ.

## Expected diagnostics if working
- At mild severity: vanilla ≈ v10 (reproduces the NULL B/2 eval — important consistency check).
- As severity ↑: v10 holds PSNR/LPIPS while vanilla degrades faster → Δ widens monotonically.
- v10 reaches lower final `E` and higher correct-class rate from OOD inits (bigger basin).
- Dose-ordering: λ=0.3 widest gap (matches Exp 2 dose-response).
- Largest separation when the init perturbation resembles the v10-mined δ family (Exp 2's strongest
  cell) — mechanistic fingerprint.

## Expected diagnostics if failing
- Δ flat (≈0) across the whole severity ladder → field gap, though real (Exp 2), is too small to
  change restoration outcomes at B/2 → consistent with NULL eval; escalate to λ↑ / XL-2 before kill.
- Both arms diverge together at high severity → corruption past both fields' validity; no
  differentiation.
- v10 worse (over-smooths / collapses to mean) → mining hurt expressivity; report honestly.

## Minimal test
One family (Gaussian-noise init), 3 severities (σ ∈ {0.25, 0.5, 1.0}), N=512, both v10 arms +
vanilla, PSNR/LPIPS + energy trace. ~0.5 GPU-day. If Δ widens with σ → expand to all families and the
full ladder (~1-2 GPU-days). If Δ flat → stop, log, escalate or kill.

## Promotion rule
PROMOTE to a capability claim ("ANM gives EqM corruption-robust restoration vanilla lacks") if, on
≥2 of 3 arms: Δ(v10−vanilla) restoration metric is ≥0 at every severity AND strictly increasing
(monotone) across the ladder for ≥2 corruption families, with dose-ordering λ=0.3 ≥ λ=0.1 ≥ vanilla.
Basin (correct-class rate from OOD init) must also favor v10 by a CI-separated margin.

## Kill rule
KILL (1-retune cap) if the minimal test shows flat Δ AND one severity-ladder extension confirms no
widening. On kill: restoration is not a differentiator at B/2 → fall back to mechanism + FID; write
postmortem. Do NOT reinterpret a flat result as "robust because equal."

## Anti-laundering note
"Restoration capability" requires the Δ to **grow with severity** (the mechanism's signature), not
just a single operating point where v10 happens to win. A flat-but-positive Δ is reported as "small
uniform restoration edge," not a new capability.
