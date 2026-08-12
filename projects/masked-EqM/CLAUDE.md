# masked-EqM — Project Goal

## Origin

Prior work (diff-EqM, archived `projects/archive/diff-EqM/`) applied adversarial negative mining (ANM/v10) to EqM. Worked, not novel enough. Discussed next direction w/ Yilun Du (Slack, 2026-07-02):

> Yilun: training to make EqM have more robust energy landscape — interpolate not just Gaussian noise → image, but also *partially masked images* and other structured starting points → ground truth image.
> Matthew: proposed Fourier-space corruption as alt/addition to Bernoulli pixel masking.
> Yilun: Fourier corruption could be good, but start simpler — partially masked images first.

## Research question

> Does broadening EqM's corruption/start distribution from pure Gaussian noise to structured corruptions produce a more robust and informative energy landscape?

Hypothesis: Gaussian starts teach EqM to descend from unstructured noise to data. Many bad inference states are structured-but-wrong (partial info, coarse structure w/ missing detail). Training from structured corruptions may make learned energy gradients useful in a wider neighborhood around the data manifold.

EqM already claims to naturally support partially-noised denoising, OOD detection, image composition (arXiv 2510.02300) — masking/structured-start training is aligned with the paper's own framing, not a random bolt-on.

**Method name: Structured Start-State EqM** (Fourier variant = Spectral Start-State EqM).

## Scope discipline (Yilun's steer — do not overcomplicate v1)

Single question per experiment. Do NOT start with phase corruption, band-drop, amplitude/phase separation, or adaptive hard negatives — those make the first result uninterpretable. Build order is strict:

### Step 1 — reproduce baseline
Get masked-EqM (fresh clone raywang4/EqM) running unmodified, reproduce official numbers. No modifications before this.

### Step 2 — Bernoulli pixel/patch masking (build this first, per Yilun explicit)
Start distribution:
```
z0 = m ⊙ x + (1-m) ⊙ ε        m = random mask, ε ~ N(0,I)
zt = (1-t) z0 + t x
```
Train w/ same EqM target as baseline. Lowest-risk first experiment.

### Step 3 — Fourier low-pass corruption (only after step 2 shows signal)
```
x̂ = F(x)
ẑ0 = M_ρ ⊙ x̂ + (1-M_ρ) ⊙ ε̂      M_ρ = radial low-pass mask
z0 = F⁻¹(ẑ0)
```
Coarse structure kept, high-freq detail missing → model learns gradients from structured-but-degraded state back to clean image.

### Step 4 — mixture ablation (Gaussian retained throughout, never fully replaced)
```
q(z0|x) = λ_G q_G + λ_M q_mask + λ_F q_fourier
```
Arms:
| Arm | Start distribution |
|---|---|
| A | Gaussian only (baseline) |
| B | Gaussian + mask |
| C | Gaussian + Fourier low-pass |
| D | Gaussian + mask + Fourier |

First mixture to try: 50% Gaussian / 25% mask / 25% Fourier (or simpler 50/50 Gaussian/structured).

### Step 5 — scale
CIFAR-level sanity only until real signal + diagnosed. No jump to IN-1K without a passing gate (same discipline as diff-EqM: proxy-scale results are filters, not publishable on their own).

## What to measure (not just FID)

1. **FID** — generation quality.
2. **Convergence / restart rate** — fewer samples falling into bad trajectories.
3. **Energy ordering** — sanity check learned energy ranks states correctly:
   `E(clean) < E(fourier-corrupt) < E(gaussian-noisy)`, and `E(clean) < E(good sample) < E(bad sample)`.
4. **Denoising/inpainting recovery** — masking/Fourier corruption are denoising-style tasks; test recovery quality vs baseline EqM.
5. **Trajectory diagnostics** — reuse prior shape-vector probe (from diff-EqM separability work) as a *diagnostic*, not the method: does new training reduce bad-shape trajectory rate? Story: prior probe found a failure mode, this training tries to remove it.

## Plan status

No formal phase gates / summer plan written yet. Write `documentation/summer-2026-plan.md` once step 2 (masking) shows signal at CIFAR scale.

## Second track: fb_direct scalar-energy training stability (active, parallel to Structured Start-State EqM)

Separate research thread within the same `masked-EqM` codebase/repo scope, investigating why
`--ebm direct` (scalar energy head, field = ∇_z E) trains less stably than `--ebm none`
(direct field prediction) at IN-1K B/2 scale. Not part of the Structured Start-State EqM
question above — orthogonal axis (backward/optimization mechanics vs. corruption/start
distribution) on the same codebase. Runs concurrently; does not block or get blocked by
Steps 1-5 above.

History (chronological, full detail in `documentation/`):
- 2026-08-01/05: gradient-guard calibration (`max_grad_norm=6.87141`); forensic finding that
  `direct` has a real, growing (~24x), backbone-dominated gradient-clip-event signature
  invisible in the training loss, mechanistically tied to double-backward-through-Hessian
  training (`direct-energy-gradient-guard-2026-08-01.md`).
- 2026-08-06/08: `forward-backwards-direct`/`exact_hvp.py` (exact forward-over-reverse,
  proven mathematically identical to double-backward) reproduces the same signature — rules
  out autograd-implementation artifact, confirms the instability is structural
  (`forward-backwards-direct.md`, `fb_direct/exact_hvp.py`).
- 2026-08-10/11: mechanism-check sweep, each falsified: gradient penalty on `‖∇_z E‖²`,
  z-space Hessian curvature, weight/logit spectral-norm drift, gauge-safe sensitivity,
  weight decay, z-loss (`memory/2026-08-10.md`, `memory/2026-08-11.md`).
- 2026-08-11: Stage A top-k head-Jacobian subspace confirmation — **gate FAILED**; energy
  head is not the dominant contributor on spike batches; redirect finding: backbone
  spike/control amplification scales monotonically with tail severity (2.50x/3.56x/4.45x)
  (`documentation/postmortem-stageA-topk-subspace-2026-08-11.md`).
- 2026-08-11 (current): implementing/testing **Whitened Forward-Backward EqM (WFB-EqM)** —
  mixed input-parameter Jacobian whitening on the backward operator, staged Stage 0
  (operator correctness) through Stage 5 (full run) with explicit gates per stage. See
  session record for full spec; artifacts land in `documentation/wfb-eqm-*.md` and
  `experiments/direct_energy/` as stages complete.

Checkpoints (true root is netscratch, not holylabs — see `documentation/longitudinal_jacobian_audit.md`):
`CKPT_DIRECT_LATE=/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/fwrev_ep80_lambda0_job37780076/000-EqM-B-2-Linear-velocity-None-ebm-direct/checkpoints/2825000.pt`
`CKPT_NONE_LATE=/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer80_none_seed0_ckpt50k_job36632776/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/2800000.pt`

## Scope

Only active project in this repo (root `AGENTS.md`). Do not touch `projects/archive/*`.
Two active tracks within this project: Structured Start-State EqM (primary, per Origin
above) and fb_direct scalar-energy stability (secondary/parallel, per above). Both share
this codebase and `.state/pipeline.json` job tracking.
