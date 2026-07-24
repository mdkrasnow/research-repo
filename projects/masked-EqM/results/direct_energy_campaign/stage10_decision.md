# Stage 10 — Full-campaign launch decision

Decision: **do not launch the result-bearing multi-seed training campaign.**

The direct scalar-energy arm has passed implementation, learnability, and
sampling-viability checks. It is not, however, scientifically promising on a
central hypothesis metric at the matched pilot budget:

* Off-trajectory local field stability is slightly worse than `none` and
  `dot` at the largest tested perturbation.
* Direct has the same measured training and sampling cost class as `dot`,
  roughly twice the vanilla field cost, without a compensating advantage.
* It has a finite practical sampling interval through eta=0.006 but begins
  overshooting at eta=0.012; this is not evidence of a wider descent regime.
* Long-NFE pilot trajectories remain finite but their latent norm increases,
  so the available evidence does not show more reliable convergence.

The requested unseen-corruption comparison is intentionally not promoted to
an expensive evaluation: these are 1,000-update pilot checkpoints trained on
a 256-latent pool, and a recovery score would not be a scientifically valid
generalization result. It remains a possible small technical smoke only, not
evidence sufficient to reverse this decision.

Verdict: **NEGATIVE at pilot scale** — technically valid scalar-energy EqM,
but no demonstrated practical or off-trajectory advantage over existing EqM
or derived dot-energy EqM.
