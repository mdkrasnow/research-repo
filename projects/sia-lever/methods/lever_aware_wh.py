"""Lever-aware W+H policy (our contribution / positioning).

SIA showed H+W beats H-only. The missing layer is ATTRIBUTION: choose H vs W vs H_THEN_W per
failure. This policy first verifies whether the deployed harness is valid (oracle sandwich), then:
  - harness broken            -> H        (never train weights against a broken evaluator)
  - harness ok but cheater    -> H_THEN_W (fix the verifier, then retrain)
  - harness ok, model weak     -> W

vs paper-style plateau_then_w: that scheduler escalates H->W by trial-and-error re-measurement and
cannot tell 'harness broken' from 'model weak' in one shot, so it over-pulls W (coupled-Goodhart
risk: training on bad feedback). The lever-aware policy makes the attribution explicit.

The deterministic form IS the oracle-sandwich rule. The LEARNED form is gpt-oss-120b(+LoRA) trained
to reproduce/generalize it from traces (see methods/learned_selector.py).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_sandwich_selector import select as _rule_select  # noqa: E402


def select(observable_trace):
    return _rule_select(observable_trace)
