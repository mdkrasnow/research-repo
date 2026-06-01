"""Experiment 3: Fidelity-Diversity & Mode Coverage (vanilla EqM vs ANM EqM, IN-1K).

Tests whether ANM's known FID gain (IN-1K-256 EqM-B/2 80ep: vanilla 31.41 ->
ANM lambda=0.3 27.09) is accompanied by preserved/improved diversity, class
coverage and mode coverage -- or whether ANM merely sharpens samples while
dropping modes.

FID alone is NOT sufficient here. The verdict must state the diversity/coverage
conclusion, not just the FID delta.

Pipeline (all on the SAME pytorch_fid InceptionV3 pool3 2048-d features used by
the trusted FID run that produced 27.09/29.01/31.41):
  schedule.py            shared balanced label + per-sample seed schedule
  sample_scheduled.py    DDP generation with per-index seed+label (fork of sample_gd.py)
  features.py            pytorch_fid Inception features + resnet50 classifier preds
  prdc_vendored.py       pure-numpy precision/recall/density/coverage
  metrics.py             FID, KID, PRDC, bootstrap CIs, per-class, classifier histogram
  eval_fidelity_diversity.py   orchestrator + delta + verdict
  plots.py               required plots
"""
