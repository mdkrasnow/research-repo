"""asm_diagnostic — interpretable readout of an ASM stage result + failure-cause analysis.

Run: python projects/symmetry-discovery/experiments/asm_diagnostic.py B [seed]
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASMDIR = ROOT / "projects" / "symmetry-discovery" / "results" / "asm"
SPATIAL = {"translate_x", "translate_y", "rotate", "scale", "hflip", "pad_crop"}
PHOTO = {"hue", "bright", "contrast", "saturate"}
DECOY = {"crop_erase", "big_shear", "color_collapse"}


def readout_A(d):
    c = d["checks"]
    print(f"families ok: {len(c['families_ok'])}/{len(c['families_ok'])+len(c['families_err'])}  "
          f"err={c['families_err']}")
    print(f"backward grad_norm={c['backward_grad_norm']}  n_valid={c['n_valid']}  "
          f"decoy_in_top={c['decoy_in_top']}")
    hbf = c["hardness_by_family"]
    hard = sorted(hbf.items(), key=lambda kv: -kv[1])
    print("hardest valid families:", [(k, v) for k, v in hard if k not in DECOY][:5])
    print("decoy hardness (should be rejected regardless):", {k: hbf[k] for k in DECOY})


def readout_B(d):
    print("=== CPU-B arms (eqm_full lower=better) ===")
    for a, r in d["arms"].items():
        print(f"  {a:16s} eqm_full={r['eqm_full']}  selected={r['selected']}  decoy={r.get('decoy_usage')}")
    s = d["summary"]
    print(f"\nbase {s['base']} | random {s['random']} | best_asm {s['best_asm']} ({s['best_asm_arm']}) "
          f"| static {s['static']}")
    print(f"ASM vs random margin: {s['margin_asm_vs_random']}  GATE_B={d.get('gate_B_pass')}")
    if not d.get("gate_B_pass"):
        r = d.get("gate_reasons", {})
        print("FAILURE CAUSE:")
        if not r.get("saturate_selected"):
            print("  - saturate NOT selected -> hardness/validity not steering to the missing factor.")
        if not r.get("beats_random"):
            print("  - does not beat random -> ASM objective not buying anything over sampling valid augs.")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "A"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    p = ASMDIR / f"stage_{stage}_seed{seed}.json"
    if not p.exists():
        print(f"no {p}"); return
    d = json.loads(p.read_text())
    {"A": readout_A, "B": readout_B}.get(stage, readout_A)(d)


if __name__ == "__main__":
    main()
