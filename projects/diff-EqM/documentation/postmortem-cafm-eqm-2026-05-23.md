# Postmortem — CAFM-on-EqM Port Failure (Phase 1b)

**Date:** 2026-05-23
**Outcome:** Phase 1 gate FAILED catastrophically. CAFM post-training of vanilla EqM-B/2 80ep collapsed the generator. FID 341.25 vs vanilla 31.41 (10× worse). Branch B-Both retired.
**Cost:** ~33 GPU-h (smoke iterations + Phase 1b 18h + FID 1.5h + diagnostics 1.5h).

## What happened

1. Phase 0.3 PASS (v10 PGD mining on CIFAR-10 vs vanilla EqM: FID 13.40 vs 14.17) unblocked Phase 1.
2. Branch B-Both plan locked: combine v10 mining with CAFM-style adversarial post-training (Lin et al. 2026, ICML 2026) on EqM-B/2 IN-1K-256.
3. Phase 1a: ported Lin's CAFM discriminator/JVP machinery to wrap EqM. 14 unit tests passed locally. Smoke v14 on cluster passed (4×A100-80GB, 500 steps, exit 0, dis loss 2.0→0.03, gen loss converged to 1.7-2.3, ckpts persisted).
4. Phase 1b: full 10ep CAFM-only post-training, 50K steps, 17h50m on holygpu8a16304. Exit 0. Loss curves looked superficially healthy (dis 0.9-2.1 oscillating, gen 1.7-2.3 stable).
5. FID eval on eqm_compat_00045000.pt (50K samples, sample_gd.py + pytorch-fid): **FID 341.25**.
6. Diagnostic at ckpt_00005000.pt (~250 gen updates past warmup, 10K samples): **FID 369.64**.
7. Diagnostic at vanilla 0380000.pt (10K samples, noise-matched): **FID 37.09**.
8. Conclusion: gen collapsed essentially the moment the warmup ended. Drift was instant, not cumulative.

## Root cause

CAFM is a discriminator-based post-training method. It works when the generator's output distribution is already adversarially-trained-compatible (Lin's setup: SiT/DiT post-trained from flow-matching scratch). Our generator was a vanilla EqM-B/2 trained purely by regression on `(ε − x)·c(γ)` — no adversarial signal in its history.

When we plugged a freshly-initialized discriminator on top of this generator:
- After 1000-step disc-only warmup, the disc had learned to discriminate vanilla-EqM-output vs real ImageNet latents. The vanilla EqM output occupies a specific narrow distribution.
- The first generator update (lr 1e-5, adversarial gradient through `torch.func.jvp`) pushed the generator's output **away** from that exact distribution to fool the disc.
- "Away from vanilla EqM distribution" = "away from the EqM regression target manifold" = field collapse.
- With N=16 disc updates per gen update, the disc kept relearning the (now-collapsing) distribution and the gen kept being pushed further out. Positive feedback loop.

The dis loss curve confirms this in hindsight: dropping 2.0 → 0.03 in 500 smoke steps = disc is winning crushingly, not oscillating. Lin CAFM expects oscillation. One-sided crush = trivial discrimination shortcut found = gen forced into unrecoverable suppression.

## Why we did not catch it earlier

Three compounding gaps:

**1. Smoke validated wiring, not behavior.**
We checked: exit code, loss finite, checkpoints saved. We did NOT: sample from the smoke checkpoint and look at the resulting images. A 5-minute sample + visual probe at smoke time would have shown collapse already in progress and saved 20+ GPU-h.

In our own Phase 0.3 v10 CIFAR work we DID evaluate FID every 30 epochs during the run. We dropped that habit when porting CAFM, partly because IN-1K sampling is more expensive (~3 min per 1K samples). We rationalized the omission. We were wrong.

**2. Misread the smoke loss curve.**
Dis loss collapsing one-sided to near zero is a classic GAN failure mode (mode collapse / discriminator winning), not "healthy convergence." We logged it as "healthy adversarial" because the gen loss was finite and the dis loss was decreasing monotonically. Adversarial training is the one regime where monotonic loss decrease on either side is bad news.

**3. Design doc reasoned about geometry, not failure modes.**
`documentation/cafm-eqm-port-design.md` reasoned carefully about the JVP, the γ-conditional discriminator, the wrapper modes, and the EqM target geometry. It did NOT enumerate "what cheap shortcut might a freshly-initialized discriminator find against a non-adversarially-trained generator." That class of failure was outside our explicit threat model.

CLAUDE.md research-process template requires "expected diagnostics if failing" — but our v10 template was for a regression-target loss, not a discriminator-based loss. We never adapted the template. Discriminator-based losses have a specific failure pattern (one-sided dis loss collapse) that should be on a standard checklist.

## What was on paper that flagged this risk (in retrospect)

Re-reading the SYNTHESIS.md after the fact, two notes warned us:

- **Briglia 2025 (`briglia2025_at_for_diffusion.md`):** "Standard adversarial training fails on diffusion models that are not equivariance-trained." We had a v11 equivariant fallback for v10 PGD, but did not transfer this caution to CAFM.
- **Lin 2026 AFM/CAFM context:** Lin's CAFM is post-training of a model that was ALREADY adversarially compatible (SiT trained from scratch as part of an AFM pipeline). The "post-training" framing obscured that the generator was never naive to the adversarial geometry. We treated "post-training" as "drop-in on any pretrained model" — Lin never claimed that.

We had the warnings. We didn't operationalize them into a smoke-time test.

## Branch B-Both decision

Branch B-Both = v10 PGD mining × CAFM discriminator post-training. CAFM half is broken in a non-retunable way (mechanism bug, not hyperparameter). One-side retune (cp_scale or warmup) wouldn't fix it because the gradient direction itself is wrong.

Options considered:
- **A. Fix CAFM port:** scale gen output by c(γ), restrict CAFM to γ ∈ [0, 0.5], or change the adversarial coupling. Estimated 3-5 days to design + smoke + retest. Risk: still fails, because the fundamental issue is that a non-adversarially-trained model has no compatible adversarial structure.
- **B. Pivot to v10-only:** drop CAFM, build the paper around v10 alone. Phase 0.3 PASS evidence intact (CIFAR FID 13.40 vs 14.17). VeCoR future-work cite ("adaptive hard-negative mining for regression-target generative models") still positions v10 as a novel contribution. Submit-able for workshop with single-method results.
- **C. Kill v10+CAFM entirely, propose v12 new branch:** too expensive given workshop deadline 2026-08-29.

**Decision: B (pivot to v10-only).** Per CLAUDE.md gating discipline: Phase 1 gate failed, mechanism is not retunable, "kill" is the prescribed action. v10 half has independent evidence at Phase 0.3 and a clear path to Phase 1 IN-1K test.

Workshop story becomes: "First adaptive hard-negative mining for regression-target generative models. PGD-mined hard examples plug into EqM training and improve FID at CIFAR scale; we verify the gain at ImageNet-1K scale on EqM-B/2."

## Action items

1. ✅ Postmortem written (this doc).
2. Update `summer-2026-plan.md`: Branch B-Both → v10-only branch. Adjust Phases 1-5 timeline.
3. Update `CLAUDE.md` research-process rules: add "mid-training sample-quality probe required for any new loss on generative models" and "discriminator-based losses require oscillation check, not monotonic-decrease check".
4. Submit Phase 1 v10-only run on cluster: `cafm_eqm_b2_in256.sbatch` with `ENABLE_V10=1` AND a flag disabling CAFM dis updates entirely. Need a new sbatch / config path. Estimated 18h per seed.
5. PI update: drafted in `pi-updates.md` (NEW DRAFT — branch pivot trigger).
6. Archive CAFM-EqM port code: keep in `experiments/cafm_eqm/` for record but mark unused.
