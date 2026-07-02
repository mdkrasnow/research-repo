# Shape probe / E_psi vs non-nn_dist target (max_softmax) — 2026-07-02

## Why
Per user correction to `shape-probe-nndist-skepticism-2026-07-02.md`: residualizing
against `nn_dist` over-controls because labels are `nn_dist`-thresholded. Correct
test for the distance-independent claim is a genuinely different quality axis.
`max_softmax` (already cached in `labels.csv`) is only moderately anti-correlated
with `nn_dist` (Pearson -0.427, all 3000 samples) — a distinct signal, not a
relabeling of the same one.

## Method
Quartile-extreme split on `max_softmax` (bottom 25% = low-confidence/bad, top 25%
= high-confidence/good, n=750/750, symmetric with the original nn_dist-based
construction). 70/30 held-out split. Three numbers:
(a) shape probe freshly fit on this new label, held-out AUROC
(b) the OLD nn_dist-trained shape probe scored against this new label (transfer)
(c) E_ψ freshly fit on this new label, held-out AUROC (5 seeds)

## Result (`runs/b2_vanilla`, seed 0)

| check | AUROC |
|---|---|
| (a) SHAPE probe, fresh-fit on max_softmax label | 0.661 |
| (b) OLD nn_dist-trained shape probe, transfer to max_softmax label | 0.659 |
| (c) E_psi, fresh-fit on max_softmax label (5-seed mean) | 0.615 ± 0.009 |

For reference: nn_dist-label numbers were shape probe 0.817 (5-seed mean), E_ψ 0.753.

## VERDICT: WEAK SUPPORT — real, smaller than headline

Trajectory shape (and to a lesser extent the endpoint) **does** carry signal about
a genuinely distance-independent quality axis (max_softmax) — 0.66 clears chance
and clears the dead endpoint-energy floor (0.50-0.57) by a real margin, and (a)
vs (b) being nearly identical (0.661 vs 0.659) means the ORIGINAL nn_dist-trained
probe transfers almost losslessly to this new axis without retraining — that's
a meaningful transfer result on its own.

But the effect size is much smaller than the 0.81-0.82 headline: roughly half the
distance-above-chance. Honest reading: **some of the shape probe's 0.81-0.82
nn_dist-label AUROC is genuinely distance-independent signal (~0.66 worth), and
some is nn_dist-specific** (the two labeling schemes aren't fully redundant —
Pearson -0.43 — so the gap between 0.82 and 0.66 is a real decomposition, not
noise).

## Updated claim status
- "Trajectory shape reveals SOME distance-independent quality signal" — **now
  supported**, ~0.66 AUROC, transfers without retraining.
- "Trajectory shape reveals semantic OOD as strongly as the 0.81-0.82 headline
  suggests" — **not supported**; that number is inflated by nn_dist-specific
  content baked into the label construction.
- Correct AUROC to cite for a distance-independent framing: **~0.66**, not 0.81.
- Restart-improves-FID intervention (claim #3) still untouched by any of this —
  it's an action result, orthogonal to which quality axis explains the mechanism.

## Next steps (not run yet)
- Cross-seed / cross-shard replication of this 0.66 number specifically (still
  the standing ask from earlier).
- Try a probe fit on BOTH nn_dist and max_softmax jointly as parallel targets —
  see if a single shape representation predicts each axis about as well as a
  probe dedicated to that axis alone (would support "one shared trajectory-shape
  signal, multiple correlated symptoms" over "several unrelated coincidences").
