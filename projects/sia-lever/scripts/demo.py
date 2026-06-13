#!/usr/bin/env python3
"""
SIA-LEVER LIVE DEMO  --  "name the right fix, don't waste a GPU retrain"
========================================================================

WHAT IS REAL vs REPLAYED (read this before you trust the numbers):
  REAL  : every episode in gpt_oss/data/out/kernel_cache_gpu.jsonl is a MEASURED
          rerun. For each failed agent run we actually executed each candidate
          fix-lever (H = hot-patch the verifier, W = weight-retrain, H_THEN_W =
          patch-then-retrain) on a real GPU-kernel verifier and recorded the
          resulting reward (`reward_by_action`) and CUDA latency. The cheat
          signatures (a kernel that PASSES the weak verifier but FAILS held-out
          inputs) and the "oracle sandwich" are measured, not authored.
  REAL  : the scoring (cost_adjusted_best / regret / retrain accounting) is the
          exact production code in gpt_oss/lever_io.py -- this demo imports it,
          it does NOT reinvent it.
  REPLAY: this CLI does NOT call a GPU or an LLM. It REPLAYS the cached measured
          episodes locally (no GPU / no VM / no network, < 10s). The "SIA-Lever"
          pick is the REAL label-free decision rule (methods/kernel_lever_rule.py)
          run live on each trace -- so its accuracy is our HONEST number (~0.75),
          NOT the oracle. The oracle (cost_adjusted_best, the upper bound) is
          shown as a separate ceiling row so the gap is visible.

Story: a self-improving agent emits a GPU kernel that fails. A naive operator
always retrains the weights (always-W) -- expensive, and for a CHEATING kernel
the retrain doesn't even fix it (the verifier itself is the problem). SIA-Lever
reads the failure trace and names the RIGHT lever, saving wasted retrains.
"""

import argparse
import json
import os
import sys
import time

# import the REAL production scoring -- do not reinvent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_oss.lever_io import (  # noqa: E402
    cost_adjusted_best,
    regret_of_action,
    outcome_for_action,
    W_RETRAINS,
)
# the REAL label-free SIA-Lever decision rule (reads only the observable trace,
# never the hidden reward) -- this is "ours", NOT the oracle.
from methods.kernel_lever_rule import select as sia_select  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "gpt_oss", "data", "out", "kernel_cache_gpu.jsonl")

# --- cost assumption (explicit + tunable) ---------------------------------
# H200 on-demand is ~$4.63/hr (Nebius/Lambda-class pricing, 2026). A single
# weight-retrain of the agent for one of these kernel episodes is estimated at
# ~0.75 GPU-hours. Tune these two constants to re-price the whole story.
H200_USD_PER_HR = 4.63
HOURS_PER_RETRAIN = 0.5
USD_PER_RETRAIN = H200_USD_PER_HR * HOURS_PER_RETRAIN  # ~$3.47

# --- ANSI ------------------------------------------------------------------
class C:
    R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    RED = "\033[91m"; GRN = "\033[92m"; YEL = "\033[93m"
    BLU = "\033[94m"; MAG = "\033[95m"; CYN = "\033[96m"; GRY = "\033[90m"

LEVER_NAME = {
    "H": "H  (hot-patch verifier, NO retrain)",
    "W": "W  (full weight retrain)",
    "H_THEN_W": "H_THEN_W (patch verifier, THEN retrain)",
    "NOOP": "NOOP", "KILL": "KILL", "PROMOTE": "PROMOTE",
}

SLOW = False


def pause(t):
    if SLOW:
        sys.stdout.flush()
        time.sleep(t)


def box(title, color=C.CYN, width=74):
    print(f"{color}{C.B}+{'-'*(width-2)}+{C.R}")
    print(f"{color}{C.B}| {title.ljust(width-4)} |{C.R}")
    print(f"{color}{C.B}+{'-'*(width-2)}+{C.R}")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pick_episodes(rows):
    """Hand-pick a few vivid, DISTINCT kernel episodes."""
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], r)

    def find(pred):
        for r in rows:
            if pred(r):
                return r
        return None

    picks = []
    # 1) THE CHEATING KERNEL: passes weak verifier (1.0) but fails held-out (0.0)
    cheat = find(lambda r: r["observable_trace"]["weak_minus_heldout_gap"] >= 0.99
                 and "approx" in r["mode"])
    # 2) memorize cheat (different flavour of broken kernel)
    memo = by_mode.get("v50|k_memorize")
    # 3) ALREADY-CORRECT kernel where W would be pure waste (correct_action == H)
    good = find(lambda r: r["correct_action"] == "H")
    # 4) a loop-kernel cheat for variety
    loop = by_mode.get("v50|k_loop")
    for r in (cheat, memo, good, loop):
        if r is not None and r not in picks:
            picks.append(r)
    return picks


def naive_pick(r):
    """The naive baseline: a self-improving agent's reflex is 'just retrain the
    weights'. So always-W. (For the already-correct kernel this is the most
    obviously wasteful: nothing was wrong with the weights.)"""
    return "W"


def render_trace(trace_text):
    for line in trace_text.strip().splitlines():
        ls = line.strip()
        col = C.GRY
        if "WEAK" in ls and "0.0" not in ls.split(":")[-1]:
            col = C.YEL
        if "HELD-OUT" in ls:
            col = C.RED if ls.rstrip().endswith("0.0") else C.GRN
        if "cheat signature" in ls:
            col = C.MAG if not ls.rstrip().endswith("0.0") else C.GRY
        if "ORACLE SANDWICH" in ls:
            col = C.BLU
        print(f"    {col}{ls}{C.R}")
        pause(0.12)


def lever_line(label, action, r, color):
    out = outcome_for_action(action, r["reward_by_action"])
    reg = regret_of_action(action, r["reward_by_action"])
    retr = W_RETRAINS[action]
    tag = "RETRAIN" if retr else "no-retrain"
    tagc = C.RED if retr else C.GRN
    print(f"    {color}{C.B}{label:<14}{C.R} -> {color}{LEVER_NAME[action]:<40}{C.R}")
    print(f"        outcome={out:+.3f}  regret={reg:.3f}  "
          f"cost={tagc}{tag}{C.R}")
    pause(0.25)
    return out, reg, retr


def episode_panel(idx, r):
    ot = r["observable_trace"]
    is_cheat = ot["weak_minus_heldout_gap"] >= 0.5
    is_good = r["correct_action"] == "H"
    if is_good:
        sub = "ALREADY-CORRECT KERNEL (a retrain here is pure waste)"
    elif is_cheat:
        sub = "CHEATING KERNEL: passes the weak verifier, fails held-out!"
    elif ot["deployed_weak_pass_rate"] == 0.0:
        sub = "BROKEN KERNEL: fails even the weak verifier"
    else:
        sub = "MISBEHAVING KERNEL: needs verifier patch THEN retrain"
    box(f"EPISODE {idx}: {r['mode']}   [{sub}]", C.CYN)
    print(f"  {C.B}1) THE FAILURE (replayed measured trace){C.R}")
    render_trace(r["trace_text"])
    pause(0.3)

    gold = cost_adjusted_best(r["reward_by_action"])
    print(f"\n  {C.B}2) NAIVE BASELINE (always retrain the weights){C.R}")
    n_act = naive_pick(r)
    _, n_reg, n_retr = lever_line("NAIVE always-W", n_act, r, C.RED)
    if outcome_for_action(n_act, r["reward_by_action"]) <= 0:
        print(f"        {C.RED}=> burned a GPU retrain and the failure PERSISTS.{C.R}")
    pause(0.3)

    print(f"\n  {C.B}3) SIA-LEVER SELECTOR (reads trace, names the right lever){C.R}")
    s_act = sia_select(r["observable_trace"])
    hit = (s_act == gold)
    scol = C.GRN if hit else C.YEL
    _, s_reg, s_retr = lever_line("SIA-Lever", s_act, r, scol)
    why = ("patch the broken verifier first, THEN retrain"
           if s_act == "H_THEN_W" else
           "verifier is fine + kernel already correct: HOT-PATCH only, skip retrain"
           if s_act == "H" else "retrain")
    mark = f"{C.GRN}correct (matches oracle){C.R}" if hit else f"{C.YEL}miss (oracle={gold}){C.R}"
    print(f"        {scol}=> {why}.{C.R}  [{mark}]")
    if s_retr < n_retr and hit:
        print(f"        {C.GRN}{C.B}=> SAVED {n_retr - s_retr} needless retrain "
              f"(${(n_retr-s_retr)*USD_PER_RETRAIN:.2f}), correct fix.{C.R}")
    elif s_retr < n_retr and not hit:
        print(f"        {C.YEL}=> used fewer retrains but UNDER-fixed "
              f"(regret {s_reg:.2f}); see oracle.{C.R}")
    pause(0.4)
    print()
    return {"naive_retr": n_retr, "naive_reg": n_reg,
            "sia_retr": s_retr, "sia_reg": s_reg,
            "correct": s_act == gold}


def summary(rows):
    """Full-cache summary over all measured episodes -- the headline numbers."""
    def run(policy):
        acc = retr = 0
        reg = 0.0
        for r in rows:
            a = policy(r)
            gold = cost_adjusted_best(r["reward_by_action"])
            acc += (a == gold)
            retr += W_RETRAINS[a]
            reg += regret_of_action(a, r["reward_by_action"])
        n = len(rows)
        return acc / n, retr, reg

    sia = run(lambda r: sia_select(r["observable_trace"]))
    alw = run(lambda r: "W")
    ora = run(lambda r: cost_adjusted_best(r["reward_by_action"]))
    n = len(rows)

    box(f"SUMMARY over {n} measured kernel episodes", C.MAG, 74)
    hdr = f"  {'policy':<16}{'accuracy':>10}{'retrains':>11}{'total-regret':>15}"
    print(C.B + hdr + C.R)
    print(C.GRY + "  " + "-" * 70 + C.R)
    for name, (a, rt, rg), col in [
        ("SIA-Lever", sia, C.GRN),
        ("always-W (naive)", alw, C.RED),
        ("oracle (ceiling)", ora, C.BLU),
    ]:
        print(f"  {col}{name:<16}{a:>9.3f} {rt:>10d} {rg:>14.3f}{C.R}")

    saved = alw[1] - sia[1]
    dollars = saved * USD_PER_RETRAIN
    print()
    print(f"  {C.B}Cost assumption:{C.R} H200 @ ${H200_USD_PER_HR}/hr x "
          f"{HOURS_PER_RETRAIN} hr/retrain = {C.B}${USD_PER_RETRAIN:.2f}/retrain{C.R}")
    print(f"  {C.GRN}{C.B}RETRAINS SAVED by SIA-Lever vs naive always-W: "
          f"{saved}  =>  ${dollars:,.2f} saved{C.R}")
    print(f"  {C.GRN}Regret crushed: {alw[2]:.2f} (naive) -> "
          f"{sia[2]:.2f} (SIA-Lever)  [oracle {ora[2]:.2f}]{C.R}")
    print(f"  {C.DIM}Cross-check (SIA-Lever-HARD rung, ladder_results.tsv): trained "
          f"selector acc 0.542 vs base 0.333 vs majority 0.458; regret 0.161->0.043.{C.R}")
    print(f"{C.MAG}{C.B}+{'-'*72}+{C.R}")


def main():
    global SLOW
    ap = argparse.ArgumentParser(description="SIA-Lever live demo (replays measured episodes)")
    ap.add_argument("--slow", action="store_true", help="animate for live presentation")
    ap.add_argument("--episodes", type=int, default=4, help="how many vivid episodes to show")
    args = ap.parse_args()
    SLOW = args.slow

    rows = load(CACHE)
    print()
    box("SIA-LEVER  --  name the right fix, don't waste a GPU retrain", C.B + C.CYN, 74)
    print(f"  {C.DIM}Replaying {len(rows)} MEASURED GPU-kernel episodes "
          f"(no GPU/VM/network).{C.R}\n")
    pause(0.5)

    picks = pick_episodes(rows)[:args.episodes]
    for i, r in enumerate(picks, 1):
        episode_panel(i, r)
        pause(0.4)

    summary(rows)
    print()


if __name__ == "__main__":
    main()
