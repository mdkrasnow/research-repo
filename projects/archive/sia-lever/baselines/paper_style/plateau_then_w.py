"""Paper-style 'plateau-then-W' scheduling baseline.

This models the SIA paper's W+H scheduling STRUCTURE: iterate on the harness (H), and when
harness-side progress plateaus, switch to a weight update (W). Crucially it does NOT use our
oracle-sandwich per-failure-mode diagnosis; it decides by re-measuring after an H step.

On the single-decision SIA-Lever episodes we model it operationally:
  - Apply H, then RE-MEASURE the resulting model's outcome (a real measured value from the cache).
  - If H alone already 'solves' it (outcome >= solve_thresh), stop at H.
  - Otherwise the harness-only path has plateaued -> escalate to H_THEN_W.

This is a *paper-style* baseline, NOT an exact SIA reproduction. It is a credible scheduler: it
gets shortcut_leak and bad_verifier right, but cannot distinguish 'model just needs W' (model_prior
_gap) from 'harness was the only problem', so it tends to over-escalate to W and pay its cost.
"""

DEFAULT_SOLVE_THRESH = 0.5


def select(reward_by_action, solve_thresh=DEFAULT_SOLVE_THRESH):
    """reward_by_action must contain measured 'H' and 'H_THEN_W' outcomes (real reruns)."""
    if reward_by_action["H"] >= solve_thresh:
        return "H"            # harness fix alone reached target -> no weight update needed
    return "H_THEN_W"          # harness-only plateaued -> escalate to a weight update


def select_from_episode(ep, solve_thresh=DEFAULT_SOLVE_THRESH):
    return select(ep["reward_by_action"], solve_thresh)
