# Shape probe / E_psi vs a NON-nn_dist target (max_softmax)

pearson(nn_dist, max_softmax) = -0.427 (moderate anti-corr, distinct axis, not a relabel of nn_dist)
max_softmax-quartile split: low-conf(bad)=750 high-conf(good)=750, held-out n=450

(a) SHAPE probe freshly fit on max_softmax label, held-out AUROC: 0.6612
(b) OLD nn_dist-trained shape probe, scored against max_softmax label (transfer): 0.6586
(c) E_psi freshly fit on max_softmax label, held-out AUROC (5 seeds): 0.6147 +/- 0.0087

## VERDICT: WEAK SUPPORT: 0.661 on max_softmax label -- some transfer beyond nn_dist, but much weaker than the 0.81-0.82 nn_dist-label number.
