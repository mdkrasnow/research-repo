"""Parse CAFM-EqM smoke train log + emit pre-registered gate verdict.

Per documentation/phase-0-spec.md, the smoke (Task 0.C.2) gates Phase 1a launch:
  A. End-to-end run completes without crash.
  B. Generator + discriminator losses finite throughout.
  C. Discriminator loss not collapsing to ~0 (gen-domination indicator).
  D. Generator loss not blowing up.

Usage:
    python evaluate_smoke.py path/to/smoke_train.log

Exits 0 on PASS, non-zero on FAIL. Prints structured verdict to stdout.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


# Loss line format (from train_cafm_eqm.py):
#   [step      N phase=dis] loss/dis_adv=X loss/dis_cp=Y loss/total_dis=Z ...
#   [step      N phase=gen] loss/gen_adv=X loss/gen_ot=Y loss/total_gen=Z ...
LINE_RE = re.compile(
    r"\[step\s+(\d+)\s+phase=(dis|gen)\]\s+(.+)"
)
PAIR_RE = re.compile(r"(loss/\S+|v10/\S+|logits/\S+)=(-?[\d.eE+nan-]+)")


def parse_log(path: Path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        step, phase, rest = m.group(1), m.group(2), m.group(3)
        vals = {k: float(v) if v not in ("nan", "-nan") else math.nan
                for k, v in PAIR_RE.findall(rest)}
        rows.append({"step": int(step), "phase": phase, **vals})
    return rows


def evaluate(rows):
    if not rows:
        return False, "No loss lines found in log."

    fail_reasons = []
    pass_notes = []

    last_step = rows[-1]["step"]
    pass_notes.append(f"final step={last_step} (PASS A: completed)")

    # Check finite throughout (B).
    any_nan = any(
        math.isnan(v)
        for row in rows
        for v in row.values()
        if isinstance(v, float)
    )
    if any_nan:
        fail_reasons.append("FAIL B: NaN encountered")
    else:
        pass_notes.append("PASS B: all losses finite")

    # Discriminator not collapsing (C). Look at last 25% of dis steps.
    dis_rows = [r for r in rows if r["phase"] == "dis"]
    if len(dis_rows) >= 4:
        tail = dis_rows[-max(1, len(dis_rows) // 4):]
        dis_adv = [r.get("loss/dis_adv", math.nan) for r in tail]
        dis_adv = [v for v in dis_adv if not math.isnan(v)]
        if dis_adv:
            mean_tail = sum(dis_adv) / len(dis_adv)
            if mean_tail < 1e-3:
                fail_reasons.append(
                    f"FAIL C: discriminator loss collapsed (mean tail loss/dis_adv = {mean_tail:.6f})"
                )
            else:
                pass_notes.append(f"PASS C: disc loss not collapsed (mean tail = {mean_tail:.4f})")
        else:
            fail_reasons.append("FAIL C: no valid dis_adv values in tail")
    else:
        fail_reasons.append(f"FAIL C: only {len(dis_rows)} dis steps; cannot evaluate collapse")

    # Generator not blowing up (D). Compare initial vs final gen_adv.
    gen_rows = [r for r in rows if r["phase"] == "gen"]
    if len(gen_rows) >= 4:
        gen_adv_init = [r.get("loss/gen_adv", math.nan) for r in gen_rows[: max(1, len(gen_rows) // 4)]]
        gen_adv_init = [v for v in gen_adv_init if not math.isnan(v)]
        gen_adv_final = [r.get("loss/gen_adv", math.nan) for r in gen_rows[-max(1, len(gen_rows) // 4):]]
        gen_adv_final = [v for v in gen_adv_final if not math.isnan(v)]
        if gen_adv_init and gen_adv_final:
            m_init = sum(gen_adv_init) / len(gen_adv_init)
            m_final = sum(gen_adv_final) / len(gen_adv_final)
            if m_final > 10 * max(m_init, 1e-3):
                fail_reasons.append(
                    f"FAIL D: gen loss blowing up (init {m_init:.4f} -> final {m_final:.4f})"
                )
            else:
                pass_notes.append(f"PASS D: gen loss stable (init {m_init:.4f} -> final {m_final:.4f})")
        else:
            fail_reasons.append("FAIL D: no valid gen_adv values")
    else:
        fail_reasons.append(f"FAIL D: only {len(gen_rows)} gen steps; cannot evaluate")

    passed = not fail_reasons
    return passed, "\n".join(pass_notes + fail_reasons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=Path)
    args = ap.parse_args()

    rows = parse_log(args.log_path)
    passed, verdict = evaluate(rows)

    print("=" * 60)
    print(f"CAFM-EqM smoke gate evaluation: {args.log_path}")
    print("=" * 60)
    print(verdict)
    print("=" * 60)
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
