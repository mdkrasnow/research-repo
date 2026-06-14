# Online Adaptive Sampler — Summary

The true "metacognition sampler": during generation, read a **partial-trajectory**
risk score at a decision step `k_dec < N` and reallocate a fixed extra compute
budget toward the flagged (likely-garbage) slots — restart them — while the control
spends the identical budget on a random subset. Equal NFE is the load-bearing
control: if probe ≈ random, the gain was compute, not the probe.

## Prerequisite (RUN, cached) — early signal exists
`partial_probe.py` on the cached 3000-sample pool: held-out de-confounded AUROC vs
decision step.

| k_frac | k_dec | within-norm AUROC (de-conf) |
|---|---|---|
| 0.40 | 100 | 0.814 ± 0.013 |
| 0.50 | 124 | 0.814 ± 0.011 |
| 0.60 | 149 | 0.812 ± 0.015 |
| 0.70 | 174 | 0.818 ± 0.014 |
| 1.00 | 249 | 0.818 ± 0.012 |

**Failure is detectable at step 100/249 (40% in) as well as at the end.** Verdict:
**ONLINE-VIABLE** — there is enough early signal to act causally. Deploy artifact
`partial_probe_k100.npz`.

## Logic validation (RUN, mock CPU)
`online_adaptive_sampler.py --mock --slots 4000`:

| arm | quality (lower=better) | NFE |
|---|---|---|
| vanilla | 0.388 | 996,000 |
| random-restart | 0.385 | 1,294,800 |
| **probe-restart** | **0.212** | 1,294,800 |
| oracle-restart | 0.177 | 1,294,800 |

probe-restart and random-restart NFE **byte-identical**; probe beats random by
0.173 at equal compute; oracle ceiling 0.177. The orchestration + equal-NFE
accounting are correct.

## Scale (DESIGNED, cluster-gated)
`slurm/jobs/online_adaptive.sbatch` runs the same four arms on the EqM-B/2
checkpoint (restart = fresh noise, same class; quality = Inception-NN-dist), FID
via `fid_online_agg.py` vs the trusted 50k reference. Smoke 512 → 5k → 15k.

**Status:** logic green in mock; real-scale FID pending a GPU block. The decisive
number for the paper is whether probe-restart < random-restart at equal NFE on the
real checkpoint — that run is prepared and one `sbatch` from launching.
