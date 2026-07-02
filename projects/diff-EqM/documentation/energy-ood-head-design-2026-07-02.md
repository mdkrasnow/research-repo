# Post-hoc energy/OOD head — design note (2026-07-02)

## Motivation (Yilun's question)
Can E(x) itself be made informative for ID/OOD, instead of relying on the
descent-shape probe? Backwards logic: EqM's `f(x_γ) ≈ target(x, ε, γ)` trains
transport/descent *behavior*, not a semantic energy ranking, and
`c(γ) → 0` near the data manifold flattens any endpoint energy signal there.
That's exactly why endpoint dot / path-integral dot cap at ~0.60-0.61 AUROC
(below the trivial kNN-distance baseline 0.627) — see
`experiments/separability_diagnostic/SYNTHESIS_METACOGNITION.md` link #1.

## Why current E(x) is uninformative
- `c(γ)` decay near γ→1 kills endpoint energy resolution exactly where ID/OOD
  separation would need to happen (near-manifold points).
- The EqM objective never sees a real/OOD contrastive signal — it only ever
  regresses to `(ε - x)·c(γ)`, so nothing pushes `E(good) < E(bad)` at the
  scale that matters.
- Confirmed dead empirically: `dynamics_probe.py` / `learned_probe.py` on
  `runs/b2_vanilla` — MAG-only / endpoint_only / scalar_only all sit
  0.60-0.67 within-norm AUROC, while SHAPE-only (de-confounded descent
  dynamics) reaches ~0.81-0.82.

## What hard negatives are mined
Source: `runs/b2_vanilla/logs/traj_rank0.npz` (3000 IN-1K-256 B/2 vanilla
samples, T=249 steps, cached x0/x_final VAE latents + norm/dot/l2/step_dot
traces) + `labels.csv` (750 good / 750 garbage / 1500 ambiguous, from
nn_dist + max_softmax thresholds, see `thresholds.json`).

Hard negatives = top-K by the existing SHAPE-only metacog probe's p_bad
(`learned_probe.py` logistic probe over magnitude-normalized descent
dynamics), restricted to the `garbage` ∪ `ambiguous` pool (not paired with
an image/kNN filter for "OOD garbage" — these are near-manifold: they come
from the *same* generative chain population, so trivially-far outliers are
already excluded by construction, satisfying the "near-manifold but likely
bad" requirement without needing a separate kNN gate).

Positives = `good`-labeled `x_final` latents (endpoint of a descent the
probe scores as low p_bad and nn_dist/softmax mark clean).

## What E_ψ is trained to predict
A small MLP over the flattened VAE latent `x_final` (4×32×32 = 4096-d),
`E_ψ: R^4096 -> R`, trained with a margin ranking loss so that
`E_ψ(good) < E_ψ(hard_negative)`:

```
L_rank = E[ max(0, m + E_ψ(x_pos) - E_ψ(x_neg)) ]
```

Base EqM model is frozen throughout — this is a post-hoc head on cached
latents, no EqM retraining, no change to the deployed probe-gated selector.

## Success criterion
`E_ψ` AUROC (on held-out good vs garbage, standard labels, not the mined
negatives) should substantially beat the dead energy baselines (~0.605-0.609)
and ideally approach the SHAPE probe (~0.81). Read as the answer to: how
much of the metacognitive trajectory signal can be folded into a single
scalar energy-like score trained post-hoc on endpoints alone (no descent
dynamics as input)?

## Next step if it works
Distill probe p_bad into `E_ψ` directly (regression to p_bad, not just
rank-loss on hard mined pairs), or add descent-trajectory features as
auxiliary input to `E_ψ` and re-test — that's already most of the way to
"probe" and should be reported as such, not re-labeled "energy".

## Result (2026-07-02, `experiments/separability_diagnostic/energy_ood_head.py`)

Run on `runs/b2_vanilla` (750 good / 750 garbage / 1500 ambiguous, IN-1K B/2
vanilla). 5-seed MLP `E_ψ` over flattened `x_final` latent, margin-ranking
loss (m=1.0) vs mined hard negatives (top-50% p_bad of garbage+ambiguous
pool, 1125 samples, 41% already true-garbage — confirms mining pulls
real near-manifold failures, not noise).

| method | AUROC |
|---|---|
| endpoint_dot (dead energy) | 0.505 |
| path_integral_dot (dead energy) | 0.568 |
| final_norm_magnitude | 0.548 |
| **E_ψ post-hoc energy head (5 seeds)** | **0.753 ± 0.009** |
| SHAPE-only descent probe (reference ceiling) | 0.815 |

(`knn_dist` excluded as a baseline — labels.csv derives good/garbage by
thresholding `nn_dist` itself, so it is circular, AUROC=1.0 by construction.)

**VERDICT: PARTIAL.** `E_ψ` clears the dead energy baselines by +0.19 to
+0.25 AUROC — hard-negative mining recovers real endpoint-only signal that
plain endpoint dot/norm cannot see. But it stays ~0.06 below the SHAPE
probe ceiling: most of the metacognitive signal still requires trajectory
dynamics, not just the endpoint, however the endpoint is scored. Failure
mode is representation-limit, not bad negatives (mining pool is 41% true
garbage, i.e. informative) or lack of endpoint information outright (0.75
is well above the endpoint-info floor of ~0.55).

## Recommendation
Do not chase a pure-endpoint `E_ψ` further as an energy story. The result
says the manifold-relative signal in `x_final` alone recovers ~60% of the
probe's gap over baseline (0.753 vs 0.815 ceiling from a 0.55 floor).
Next low-risk step: give `E_ψ` the same descent-shape features as input
(`shape_feats`, not just `x_final`) — this is a strict superset test, not
new risk, and should converge to ~SHAPE-probe AUROC by construction. If
Yilun's ask is specifically "a *scalar function of x*" (not of the chain),
report this 0.753 number as the honest ceiling for that framing and state
explicitly: **an energy function of the endpoint alone cannot fully replace
the trajectory-shape probe; the informative signal is inherently a
function of the descent, not the point.**

## Next step if it fails
Diagnose failure mode explicitly:
- If `E_ψ` on `x_final` alone stays near baseline (~0.60) → confirms the
  signal genuinely lives in trajectory *shape*, not in any static function of
  the endpoint, however parameterized (representation-limit story, not a
  training-recipe problem).
- If mined negatives look indistinguishable from held-out garbage (check
  overlap of mined-neg sample_ids vs true garbage sample_ids) → negative
  mining is not adding information beyond the label split already used.
