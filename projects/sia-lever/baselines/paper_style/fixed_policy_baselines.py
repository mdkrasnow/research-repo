"""Fixed lever policies (naive baselines). Decisions ignore the trace content."""


def h_only(_ep, idx=0):
    return "H"


def w_only(_ep, idx=0):
    return "W"


def alternating(_ep, idx=0):
    return "H" if idx % 2 == 0 else "W"


POLICIES = {"H_only": h_only, "W_only": w_only, "alternating": alternating}
