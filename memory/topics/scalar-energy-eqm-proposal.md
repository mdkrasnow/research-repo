---
name: scalar-energy-eqm-proposal
description: ChatGPT conversation proposing a scalar explicit-energy variant of EqM (vs vector-field target) plus an evaluation plan, for masked-EqM
status: active
---

External conversation (ChatGPT, not yet implemented/discussed with user's Claude-side
collaborator) proposing a candidate direction for masked-EqM. Recorded here so it can
be evaluated against the project's compatibility-check process before any code is
written (per `AGENTS.md` "Variant Proposal Template" and "no literature laundering").

## The proposal

Instead of training EqM's vector field `f(x_γ)` to match the local target direction
`(ε − x) · c(γ)`, train a scalar energy `E(x)` directly:

- Derive a scalar energy target per corrupted sample by integrating the existing EqM
  vector-field target along the corruption path (clean → corrupted).
- Regress `E(x)` onto that scalar target with a standard regression loss.
- At sample time, descend `∇E(x)` instead of following a directly-predicted vector
  field — the field is then guaranteed conservative (gradient of one scalar), since it
  literally is one.

Intuition: vector-field EqM learns "which way to step" locally; this proposal learns
"how bad is this point" globally, and derives the step from that landscape.

Known prior art / risk flagged in the conversation itself: EqM paper's existing
explicit-energy variants reportedly underperformed the implicit vector model. The open
question is whether this specific direct-scalar-regression construction (attributed to
"Yilun" in the source conversation) avoids that known failure mode — not just whether
it's more interpretable.

## Proposed benefit (hypothesis)

A scalar energy must explain all observed states through one shared function, which
may regularize behavior between/beyond trained corruption trajectories (Gaussian,
mask) better than a field trained only on local directions. Three claimed upsides:
generalization to unseen corruptions, more diagnosable sampling (can check energy is
actually decreasing), and a usable plausibility score `E(x)`.

## Proposed evaluation plan

**Primary metric — unseen-corruption LPIPS.** Train vector-EqM and scalar-EqM with
matched architecture/data/compute/sampler budget on the same corruptions currently
used; evaluate LPIPS(x_recovered, x_clean) on corruptions NOT in training (unseen mask
structures, unseen Gaussian severity, Fourier corruption, composed Gaussian+mask,
downsampling/blur).

**Secondary metrics:**
1. In-distribution LPIPS/MSE on trained corruptions — distinguishes real generalization
   gain from just "trains better overall" or "underfits everything." Track the
   generalization gap `G = L_unseen − L_seen` (lower is better, only if seen-corruption
   performance stays competitive).
2. Generation quality: FID + precision + recall (FID alone insufficient — could improve
   FID while losing diversity).
3. Energy monotonicity during sampling: fraction of steps with `E(x_{k+1}) < E(x_k)`,
   plus average descent rate `(E(x_0) − E(x_K))/K` and magnitude of any upward steps.
4. Convergence efficiency: steps to reach a gradient-norm threshold
   `K_ε = min{k : ‖∇E(x_k)‖ < ε}`, final gradient norm, divergence rate, sensitivity to
   step size/optimizer.
5. Energy ranking: does `E(clean) < E(mild corruption) < E(heavy corruption)` hold?
   Report pairwise ranking accuracy / AUROC, e.g. `P(E(clean) < E(corrupted))`.

**Predefined success criterion:** scalar-EqM significantly lowers LPIPS on unseen
corruptions across multiple seeds, without materially degrading FID/diversity/seen-
corruption recovery. Energy monotonicity + ranking are secondary mechanistic evidence,
not the headline result — the headline is generalization or sampling robustness.

## Status / next step

Not yet run through this project's mandatory pre-code process: baseline check,
mechanism note, EqM-compatibility argument, minimal implementation + diagnostics,
smoke test, decision (promote/retune/kill), postmortem if killed (`AGENTS.md`
"Research process rules"). Also needs a written compatibility argument given it's an
explicit-energy objective and the project's prior (diff-EqM era) experience flagged
EBM-style energy losses not tied to the EqM target as high-risk. No masked-EqM summer
plan / phase gates exist yet (per `AGENTS.md` 2026-07-02 note), so this can't be gated
until that plan exists — record the idea now, evaluate formally once the plan is
re-derived.
