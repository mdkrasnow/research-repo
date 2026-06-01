"""Experiment 4: seed stability, train-val generalization gap, memorization audit.

Compares vanilla EqM vs ANM-EqM (v10 PGD hard-example mining) under:
  - step-matched and compute-matched regimes,
  - validation-reference AND training-reference FID/KID (train-val gap),
  - nearest-neighbor memorization + duplicate audit.

This package only CONSUMES generated samples + precomputed reference banks; it
never trains and never re-implements EqM sampling. Generate samples with the
repo sampler (eqm-upstream/sample_gd.py) and build references with
build_references.py before running eval_stability_memorization.py.

See documentation/experiment-4-agent-prompt.md for the full design + the two
load-bearing caveats (B1: trusted FIDs are train-reference; B2: single-seed
checkpoints today -> variability audit, not a seed audit).
"""
