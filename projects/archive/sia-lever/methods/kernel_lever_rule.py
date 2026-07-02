"""SIA-Lever deterministic rule for the TriMul GPU-kernel task (label-free).

Decides the lever from the NUMERIC observable trace only (no hidden score). This is the hand-coded
diagnostic the learned gpt-oss selector is meant to approximate on the kernel task — the analog of
oracle_sandwich_selector, but reading kernel signals (pass-rates / accept-rate / latency gap)
instead of the rotation-toy booleans.

Trace fields (see experiments/trimul_task.py / sia_task_trimul/data/public/task.md):
  deployed_weak_pass_rate, deployed_heldout_pass_rate, weak_minus_heldout_gap,
  deployed_latency_us, known_good_kernel_accept_rate

Rule (in order):
  1. verifier rejects a known-good kernel (accept_rate < 0.5)  -> verifier is broken.
       - emitted kernel itself correct (heldout high) -> H        (just fix the verifier)
       - emitted kernel also wrong                    -> H_THEN_W  (fix verifier AND re-select)
  2. large weak-minus-heldout gap (a cheat passed a weak verifier) -> H_THEN_W
  3. emitted kernel fails on held-out inputs (heldout low)         -> H_THEN_W (re-select a correct kernel)
  4. otherwise (emitted correct, verifier sound)                   -> H        (nothing more to gain)
"""

GAP_CHEAT = 0.30
HELDOUT_OK = 0.99
ACCEPT_OK = 0.50


def select(observable_trace):
    t = observable_trace
    heldout = t.get("deployed_heldout_pass_rate", 0.0)
    gap = t.get("weak_minus_heldout_gap", 0.0)
    accept = t.get("known_good_kernel_accept_rate", 1.0)

    if accept < ACCEPT_OK:                       # verifier broken (rejects known-good kernel)
        return "H" if heldout >= HELDOUT_OK else "H_THEN_W"
    if gap >= GAP_CHEAT:                          # cheat slipped past a weak verifier
        return "H_THEN_W"
    if heldout < HELDOUT_OK:                      # emitted kernel not actually correct
        return "H_THEN_W"
    return "H"                                    # emitted correct + verifier sound


def select_from_episode(ep):
    return select(ep["observable_trace"])
