"""Oracle-sandwich lever selector (deterministic rule baseline / upper-bound diagnostic).

Label-free: decides from the observable trace only (no hidden score). This is the hand-coded
diagnostic the learned gpt-oss selector is meant to approximate. It is an UPPER-BOUND baseline,
not a learned policy.

Rule (in order):
  1. harness does NOT accept a known-good reference model  -> H        (evaluator broken)
  2. shortcut-cheat signature present                      -> H_THEN_W (weak harness passed a cheater)
  3. model cannot predict clean examples                   -> W        (valid harness, weak model)
  4. otherwise                                             -> PROMOTE  (nothing wrong)
"""


def select(observable_trace):
    obs = observable_trace
    if not obs.get("harness_accepts_known_good_model", True):
        return "H"
    if obs.get("shortcut_cheat_signature", False):
        return "H_THEN_W"
    if not obs.get("predicts_clean", False):
        return "W"
    return "PROMOTE"


def select_from_episode(ep):
    return select(ep["observable_trace"])
