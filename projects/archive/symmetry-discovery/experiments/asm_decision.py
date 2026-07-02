"""asm_decision — reads ASM CPU-ladder stage JSONs and emits the NEXT action per the gated decision tree.

No manual interpretation: the gates in asm_cpu_ladder.py write gate_*_pass booleans; this script chains
them into "run next stage X" or "launch GPU arms {..}" or "STOP, gap is flagship". Mirrors the program's
decision tree exactly.

Run: python projects/symmetry-discovery/experiments/asm_decision.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASMDIR = ROOT / "projects" / "symmetry-discovery" / "results" / "asm"


def load(stage, seed=0):
    p = ASMDIR / f"stage_{stage}_seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def decide(seed=0):
    A = load("A", seed); B = load("B", seed); C = load("C", seed); D = load("D", seed)
    out = []
    if A is None:
        return ["RUN CPU-A (unit/validity smoke) first."]
    if not A.get("gate_A_pass"):
        return [f"CPU-A FAIL {A.get('gate_reasons')} -> fix asm_miner validity/firewall before any science."]
    out.append("CPU-A PASS (miner valid).")

    if B is None:
        out.append("NEXT: run CPU-B (positive-control desat gap).")
        return out
    if not B.get("gate_B_pass"):
        r = B.get("gate_reasons", {})
        out.append(f"CPU-B FAIL {r} -> repair scorer/hardness; NO GPU. "
                   f"(summary: {B.get('summary')})")
        return out
    out.append(f"CPU-B PASS: {B['summary']['best_asm_arm']} selected "
               f"{B['arms'][B['summary']['best_asm_arm']]['selected']}, "
               f"beats random by {B['summary']['margin_asm_vs_random']}. ASM can mine the missing factor.")

    if C is None:
        out.append("NEXT: run CPU-C (full-CIFAR TinyEqM/v10-lite ladder).")
        return out
    solo = C.get("gate_solo_pass"); hybrid = C.get("gate_hybrid_pass")
    if not solo and not hybrid:
        out.append("CPU-C BOTH FAIL -> no full-CIFAR GPU ASM. Report gap15 as flagship. NO HP tuning.")
        return out
    gpu_arms = ["v00", "v10"]
    if solo:
        out.append("CPU-C SOLO PASS -> GPU ladder includes ASM alone.")
        gpu_arms += ["random_valid", "static_v17", "ASM_best"]
    if hybrid:
        out.append("CPU-C HYBRID PASS -> GPU ladder includes v10+ASM.")
        gpu_arms += ["v10+ASM_best"]
        if D and D.get("gate_scheduled_better"):
            gpu_arms += ["scheduled_v10+ASM"]
            out.append("CPU-D: scheduled hybrid wins -> include scheduled arm.")
    out.append(f"LAUNCH GPU-G0 smoke then G1 seed0 with arms: {sorted(set(gpu_arms))}")
    out.append("G1 decision: promote a comparison to seeds 1,2 only if margin >= 0.3 FID; "
               "else inconclusive (NO tuning); if worse, diagnose+stop.")
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for line in decide(seed):
        print("•", line)
