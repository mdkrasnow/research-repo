# SHAPE probe nn_dist skepticism control — 2026-07-02

## Why this check
The endpoint `E_ψ` head (`energy-ood-head-design-2026-07-02.md`) collapsed
under an nn_dist residual check: raw AUROC 0.753 dropped to 0.524 once the
part of the score correlated with `nn_dist` was regressed out — `E_ψ` was
mostly re-deriving the distance-derived label, not finding new signal.

Same question applies to the **main claimed result** of this whole line of
work: the descent-SHAPE probe (`learned_probe.py`, `SYNTHESIS_METACOGNITION.md`
link #2, reported 0.81-0.82 AUROC, de-confounded from grad-norm). Labels
(`good`/`garbage`/`ambiguous`) in `runs/b2_vanilla/labels.csv` are derived by
thresholding `nn_dist` (`thresholds.json`). If `p_bad` from the shape probe
is itself mostly a function of `nn_dist`, the "descent dynamics predict
failure" claim is confounded the same way the energy claim was.

## Method
Same held-out protocol as `energy_ood_head.py`: 70/30 split, fit logistic
probe on `shape_feats` (magnitude-normalized descent dynamics) on train,
score held-out only (`shape_probe_nndist_check.py`). Then:
1. raw shape-probe AUROC (held-out)
2. Pearson / Spearman corr(p_bad, nn_dist)
3. AUROC of p_bad residual after regressing out nn_dist
4. within-nn_dist-decile AUROC (matched-distance control)

## Result (5 seeds, `runs/b2_vanilla`)

| seed | raw AUROC | residual-after-nn_dist AUROC |
|---|---|---|
| 0 | 0.826 | 0.570 |
| 1 | 0.815 | 0.567 |
| 2 | 0.799 | 0.556 |
| 3 | 0.819 | 0.556 |
| 4 | 0.827 | 0.563 |
| **mean** | **0.817** | **0.562** |

Pearson corr(p_bad, nn_dist) ≈ 0.55, Spearman ≈ 0.54 (seed 0; stable order of
magnitude across seeds). Within-nn_dist-decile AUROC (seed 0): 0.659, but on
**only 1/10 usable bins** — nn_dist alone determines the label almost
completely in 9 of 10 deciles, so the matched-distance control barely has
anywhere to run (same failure mode `dynamics_probe.py`'s `MIN_USABLE_BINS`
guard was built to catch — it just wasn't applied against `nn_dist`, only
against the norm-magnitude confound).

## CORRECTION (user, 2026-07-02): residual test over-controls

Labels are `y = 1[nn_dist > τ]` (thresholded, per `thresholds.json`). Regressing
`nn_dist` out of `p_bad` and then scoring against a label *defined by* `nn_dist`
removes the label-generating variable itself — a good predictor is EXPECTED to
collapse under this test, near-tautologically. This is not evidence the probe
lacks signal; it's evidence the test asks an over-constrained question ("does
the probe predict this label using zero information correlated with the
variable that made the label"). Downgrade the verdict below from "bug
invalidating prior result" to "mechanistic caveat on interpretation":

- **Not supported:** "shape reveals semantic OOD independent of distance."
- **Still standing:** "early trajectory shape predicts eventual nn_dist-defined
  failure" (claim #2, restated correctly) and "acting on it improves FID"
  (claim #3, an intervention result, untouched by this whole discussion).
- The 1/10-usable-bin decile result also doesn't show the probe is useless —
  it shows the label is too tightly coupled to `nn_dist` for that particular
  control to be well-powered, by construction.

**Corrected framing:** *"The shape probe's current validation target is
nn_dist-defined failure. Residualizing against nn_dist removes the
label-defining signal, so collapse is expected and uninformative on its own.
This does not invalidate the selector or the early-trajectory-predicts-
eventual-failure result; it only means we cannot yet claim semantic OOD
independent of distance without a non-nn_dist label."*

**Better next test (not yet run):** evaluate the shape probe against a
non-`nn_dist` quality target — `max_softmax` (already cached in `labels.csv`,
zero new compute), classifier confidence, Inception-feature quality bucket,
or per-sample FID contribution. If shape probe AUROC holds against a target
that isn't distance-derived, the distance-independent claim is supported
properly instead of via residualization.

## Original verdict text (superseded by correction above, kept for record)

The residual AUROC (~0.56, stable across 5 seeds) falls to essentially the
same floor the endpoint `E_ψ` residual hit (0.524). **The shape probe's
0.81-0.82 headline number is substantially explained by `nn_dist`, the same
signal used to define the labels.** This is not a partial-drop case — it's
the same collapse pattern as the energy head, just starting from a higher
raw number.

## What this means for the project thesis

`SYNTHESIS_METACOGNITION.md` claim #2 ("descent dynamics predict failure,
de-confounded from grad-norm, AUROC 0.82") was de-confounded against
**norm magnitude only**, never against `nn_dist`. That de-confounding was
real and correctly done for what it tested — but it leaves open exactly the
confound this check finds: the shape probe may be predicting *distance to
the nearest real-image neighbor* (which the labels are built from) rather
than a distance-independent notion of "will this descent produce a bad
sample."

**Required downgrade:** until shown otherwise, describe the result as
*"early/full trajectory shape predicts nn_dist-defined failure — largely
explained by the trajectory shape correlating with nn_dist itself"*, not
*"trajectory dynamics reveal semantic OOD / sample quality independent of
distance."* This affects claim #2 and everything downstream that cites it
(#3 restart-improves-FID is a separate, action-based claim and is NOT
directly invalidated — restart still improves FID regardless of *why* the
probe works — but the *mechanistic* story "it's the dynamics, not distance"
needs re-examination).

## Immediate caveat for all in-flight framing
- Do not describe the shape probe as "de-confounded from grad-norm" without
  also noting it has **not** been de-confounded from `nn_dist`, and that
  when it is, residual AUROC collapses to ~0.56.
- `SELECTOR_LOCKDOWN_RESULTS.md`, `SYNTHESIS_METACOGNITION.md`, and any
  paper-shape claim depending on "the signal is dynamics not magnitude"
  need an explicit nn_dist ablation before the workshop draft.

## Next steps (not yet run, per explicit instruction)
- Cross-seed / cross-shard held-out test (same requirement as for `E_ψ`).
- Check whether the *action* result (probe-restart improves FID) survives
  even if the *mechanism* explanation is "distance to nearest real image"
  rather than "descent instability" — these could be functionally the same
  intervention with a different causal story.
- Re-run the residual check with more nn_dist bins / finer resolution to
  see if the 1/10-usable-bin collapse is a labeling artifact (labels are
  literally thresholds on nn_dist, so most of the joint distribution is
  deterministic) vs a genuine information-theoretic ceiling.

## PI-update trigger
Per AGENTS.md PI-update protocol: this is a "bug invalidating prior reported
result" trigger for claim #2 in `SYNTHESIS_METACOGNITION.md`. Draft owed in
`pi-updates.md`; `pipeline.json:needs_user_input.value` should be set true
pointing here. Not sent — user reviews first.
