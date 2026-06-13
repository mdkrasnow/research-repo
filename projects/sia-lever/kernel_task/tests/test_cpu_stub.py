#!/usr/bin/env python3
"""CPU end-to-end regression for the kernel-task lever loop (no model/GPU). Asserts the phenomenon:
W-only entrenches a fast-but-wrong kernel; H_THEN_W and the selector repair to a correct one; the
selector picks H_THEN_W from the shortcut signature. Run: python kernel_task/tests/test_cpu_stub.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import harness as Hn  # noqa: E402
import lever_loop as L  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    # 1) seed harness sanity: the cheat passes weak, fails strong; bmm correct both
    cheat = Hn.evaluate(Hn.load_seed("memorize_cheat"), "weak")
    cheat_s = Hn.evaluate(Hn.load_seed("memorize_cheat"), "strong")
    bmm = Hn.evaluate(Hn.load_seed("torch_bmm"), "strong")
    check("cheat_passes_weak", cheat["observable"]["passes_deployed_verifier"])
    check("cheat_fails_strong", not cheat_s["observable"]["passes_deployed_verifier"])
    check("cheat_not_heldout_correct", not cheat["hidden"]["heldout_correct"])
    check("bmm_correct_strong", bmm["observable"]["passes_deployed_verifier"]
          and bmm["hidden"]["heldout_correct"])

    # 2) lever loop phenomenon
    res = {p: L.run_episode("stub", p) for p in ["W_only", "H_only", "H_THEN_W", "selector"]}
    check("W_only_entrenches_wrong", res["W_only"]["final_heldout_correct"] is False,
          f"speedup={res['W_only']['final_speedup_vs_baseline']}")
    check("H_only_cannot_repair", res["H_only"]["final_heldout_correct"] is False)
    check("H_THEN_W_repairs", res["H_THEN_W"]["final_heldout_correct"] is True)
    check("selector_repairs", res["selector"]["final_heldout_correct"] is True)
    check("selector_picks_H_THEN_W_first",
          res["selector"]["trajectory"][0]["lever"] == "H_THEN_W")

    # 3) isolation: a hanging/garbage kernel must not crash the harness
    bad = Hn.evaluate("def kernel(a, b):\n    return a @ b  # wrong shape on purpose", "strong")
    check("garbage_kernel_handled", bad["observable"]["compiles"] in (True, False)
          and bad["hidden"]["heldout_correct"] is False)

    print("=" * 50)
    if FAILS:
        print(f"FAILED: {FAILS}")
        sys.exit(1)
    print("ALL KERNEL-TASK CPU CHECKS PASS")


if __name__ == "__main__":
    main()
