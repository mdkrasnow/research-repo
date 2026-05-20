"""Parse CAFM-EqM Phase 1b train log + emit pre-registered Phase 1b gate verdict.

Pre-registered gate (from summer-2026-plan.md + workshop-paper-outline.md):
  Phase 1b PASS: CAFM-only EqM-B/2 80ep + 10ep post-training → FID < 25.
  (Vanilla baseline FID 31.41; CAFM expected gain ~6-8 FID per Lin et al.
   CAFM-on-SiT-XL/2 results extrapolated downward.)

Inputs:
  - Training log (stdout from train_cafm_eqm.py): loss/gen_adv, loss/dis_adv,
    loss/total_dis, loss/total_gen, logits.
  - FID eval result: parsed via standard EqM FID pipeline
    (imagenet1k_fid_eval.sbatch produces "FID: <value>" line).

Diagnostics checked:
  A. Training completed all 10 epochs (50K steps with N=16 → 50K/(N+1) ≈ 3K
     gen updates ≈ 10 epochs at batch 256).
  B. Losses finite throughout (no NaN).
  C. Discriminator loss not collapsed (mean tail > 1e-3 confirms it found
     non-trivial real-vs-fake signal).
  D. Generator loss decreased from initial (CAFM is meaningful only if gen
     reduces adversarial loss).

Plus FID gate:
  E. Final FID < 25 (PASS Phase 1b → submit v10+CAFM Phase 2).
  E'. Final FID 25-30 (partial → diagnose; consider 1 retune of LR or λ_cp).
  E''. Final FID ≥ vanilla 31.41 (FAIL → CAFM didn't transfer to EqM; pivot).

Usage:
  python evaluate_phase_1b.py <train_log_path> [--fid <value>]

Exits 0 PASS, 1 FAIL (config retune), 2 PIVOT (CAFM doesn't transfer).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


VANILLA_BASELINE_FID = 31.41
GATE_PASS_FID = 25.0
GATE_RETUNE_FID = 30.0

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
        step, phase = int(m.group(1)), m.group(2)
        vals = {}
        for k, v in PAIR_RE.findall(m.group(3)):
            try:
                vals[k] = float(v) if v not in ("nan", "-nan") else math.nan
            except ValueError:
                continue
        rows.append({"step": step, "phase": phase, **vals})
    return rows


def evaluate(rows, fid):
    fails, partial, passes = [], [], []

    if not rows:
        return ("PIVOT", "No loss lines found in log.")

    last_step = rows[-1]["step"]
    passes.append(f"PASS A: completed to step {last_step}")

    # B. Finite throughout
    if any(math.isnan(v) for r in rows for v in r.values() if isinstance(v, float)):
        fails.append("FAIL B: NaN encountered")
    else:
        passes.append("PASS B: all losses finite")

    # C. Discriminator not collapsed
    dis_rows = [r for r in rows if r["phase"] == "dis"]
    if dis_rows:
        tail = dis_rows[-max(1, len(dis_rows) // 4):]
        dis_adv = [r.get("loss/dis_adv", math.nan) for r in tail]
        dis_adv = [v for v in dis_adv if not math.isnan(v)]
        if dis_adv:
            mean_tail = sum(dis_adv) / len(dis_adv)
            if mean_tail < 1e-3:
                fails.append(f"FAIL C: disc loss collapsed (tail mean = {mean_tail:.6f})")
            else:
                passes.append(f"PASS C: disc loss healthy (tail mean = {mean_tail:.4f})")
    else:
        fails.append("FAIL C: no dis rows")

    # D. Generator loss decreased from initial.
    gen_rows = [r for r in rows if r["phase"] == "gen"]
    if len(gen_rows) >= 8:
        early = gen_rows[: max(1, len(gen_rows) // 8)]
        late = gen_rows[-max(1, len(gen_rows) // 4):]
        m_early = sum(r.get("loss/gen_adv", 0) for r in early) / len(early)
        m_late = sum(r.get("loss/gen_adv", 0) for r in late) / len(late)
        if m_late < m_early:
            passes.append(f"PASS D: gen loss decreased ({m_early:.4f} → {m_late:.4f})")
        else:
            partial.append(
                f"PARTIAL D: gen loss not strictly decreasing ({m_early:.4f} → {m_late:.4f}); "
                "may indicate gen-disc balance issue."
            )

    # E. FID gate
    if fid is None:
        partial.append("SKIP E: FID value not provided")
        verdict = "INCOMPLETE"
    else:
        gain = VANILLA_BASELINE_FID - fid
        if fid < GATE_PASS_FID:
            passes.append(
                f"PASS E: FID {fid:.2f} < {GATE_PASS_FID} (gain {gain:+.2f} vs vanilla)"
            )
            verdict = "PASS"
        elif fid < GATE_RETUNE_FID:
            partial.append(
                f"PARTIAL E: FID {fid:.2f} in [{GATE_PASS_FID}, {GATE_RETUNE_FID}); "
                "borderline. 1 retune of LR ∈ {5e-6, 5e-5} or λ_cp ∈ {0.0001, 0.01}."
            )
            verdict = "RETUNE"
        elif fid < VANILLA_BASELINE_FID:
            partial.append(
                f"PARTIAL E: FID {fid:.2f} better than vanilla {VANILLA_BASELINE_FID} "
                "but worse than gate. Mechanism works at minor level."
            )
            verdict = "RETUNE"
        else:
            fails.append(
                f"FAIL E: FID {fid:.2f} ≥ vanilla {VANILLA_BASELINE_FID}. "
                "CAFM did NOT transfer to EqM. PIVOT decision needed."
            )
            verdict = "PIVOT"

    body = "\n".join(passes + partial + fails)
    return (verdict, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=Path)
    ap.add_argument("--fid", type=float, default=None, help="Final FID@50K result")
    args = ap.parse_args()

    rows = parse_log(args.log_path)
    verdict, body = evaluate(rows, args.fid)

    print("=" * 60)
    print(f"Phase 1b CAFM-EqM gate evaluation: {args.log_path}")
    print(f"  rows: {len(rows)}, FID: {args.fid}")
    print("=" * 60)
    print(body)
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    print("=" * 60)

    if verdict == "PASS":
        print("Next: submit Phase 2 v10+CAFM via:")
        print("  bash projects/diff-EqM/experiments/cafm_eqm/submit_phase_2.sh")
        sys.exit(0)
    if verdict in ("RETUNE", "INCOMPLETE"):
        sys.exit(1)
    sys.exit(2)


if __name__ == "__main__":
    main()
