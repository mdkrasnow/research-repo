#!/usr/bin/env python3
"""Capability-eval pack for the v13 SE2 bridge — runs ONLY if v13 wins on FID.

Beyond FID: does v13 improve the capabilities its mechanism predicts (translation robustness, coverage,
operator-aligned gains)? This is dry-run capable and robust to missing checkpoints so it can be prepared
NOW without touching the training jobs.

Eval plan (per arm checkpoint):
  1. transform_robustness   — feature MSE/cos of model field under small translations/crops of inputs
  2. coverage_diversity     — precision/recall or NN-diversity of generated samples in feature space
  3. translation_gap_score  — proxy-style held-out-translation feature gap, but on GENERATED samples
  4. operator_aligned       — does v13 improve specifically along the discovered tx/ty (from operator_diag)?
  5. sample_grid            — fixed-seed sample grids per arm (+ translated views), cheap visual check

Usage:
    python projects/diff-EqM/experiments/bridge_capability_eval.py --dry-run        # inventory + plan
    python projects/diff-EqM/experiments/bridge_capability_eval.py --run --eval operator_aligned
  (sampling-based evals require the EqM sampler from dganm_variants._common; wired behind --run and
   guarded so nothing executes unless the checkpoint exists.)
"""
import argparse, glob, json, os

RR = "/n/home03/mkrasnow/research-repo"
RESULTS = f"{RR}/projects/diff-EqM/results"

# arm -> (variant_name, job_id). Checkpoint dir = variant_<variant>_<job>_seed0.
ARMS = {
    "v00_base":           ("v00_vanilla",        "19242268"),
    "v10_hardneg":        ("v10_hard_example",   "19405301"),
    "vK_translate_crop":  ("vK_known_aug",       "19404840"),
    "v13_random_se2":     ("v13_stable_se2_aug", "19404851"),
    "v13_discovered_se2": ("v13_stable_se2_aug", "19404844"),
}
EVALS = ["transform_robustness", "coverage_diversity", "translation_gap_score",
         "operator_aligned", "sample_grid"]


def ckpt_path(variant, job):
    base = f"{RESULTS}/variant_{variant}_{job}_seed0"
    for name in ("final.pt", "checkpoint.pt"):
        p = f"{base}/{name}"
        if os.path.isfile(p):
            return p
    hits = sorted(glob.glob(f"{base}/**/*.pt", recursive=True))
    return hits[-1] if hits else None


def operator_diag(variant, job):
    p = f"{RESULTS}/variant_{variant}_{job}_seed0/operator_diag.json"
    return json.load(open(p)) if os.path.isfile(p) else None


def inventory():
    print("## capability-eval inventory\n")
    print("| arm | ckpt ready? | path | operator tx/ty/det |")
    print("|---|---|---|---|")
    ready = {}
    for arm, (variant, job) in ARMS.items():
        cp = ckpt_path(variant, job); ready[arm] = cp
        diag = operator_diag(variant, job)
        op = "-"
        if diag and isinstance(diag.get("tx_px"), (int, float)):
            op = f"tx={diag['tx_px']:.2f} ty={diag.get('ty_px',0):.2f} det={diag.get('det',0):.3f}"
        print(f"| {arm} | {'YES' if cp else 'no'} | {cp or '(pending)'} | {op} |")
    return ready


def eval_operator_aligned():
    """Cheap, no sampling: summarize discovered vs random operator quality from operator_diag."""
    print("\n### operator_aligned (from operator_diag.json)")
    for arm in ("v13_discovered_se2", "v13_random_se2"):
        variant, job = ARMS[arm]; diag = operator_diag(variant, job)
        if not diag:
            print(f"- {arm}: operator_diag not present yet"); continue
        b, a = diag.get("anchor_baseline_real_real"), diag.get("anchor_final_T_real")
        print(f"- {arm}: tx={diag.get('tx_px')} ty={diag.get('ty_px')} det={diag.get('det')} "
              f"cond={diag.get('cond')} lin_off={diag.get('lin_off_identity')} "
              f"anchor {b}→{a} shift_consistency={diag.get('feature_shift_consistency')}")
    print("  (discovered should be near-isometric translation w/ anchor-improvement; random should not)")


def _needs_sampling(name):
    print(f"\n### {name}")
    print(f"  [NOT RUN] requires EqM sampling from a checkpoint (dganm_variants._common Euler sampler "
          f"+ frozen-conv/Inception features). Wire here when a winning ckpt exists; guarded so it never "
          f"runs without the checkpoint. Cheap version: reuse feature_gap_proxy_cifar_se2 metric on "
          f"GENERATED samples instead of rotated training images.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--eval", choices=EVALS + ["all"], default="all")
    args = ap.parse_args()

    ready = inventory()
    if args.dry_run or not args.run:
        print("\nplan:", EVALS)
        print("dry-run only — no sampling, no checkpoints touched. Use --run --eval <name> when v13 wins.")
        return

    sel = EVALS if args.eval == "all" else [args.eval]
    for name in sel:
        if name == "operator_aligned":
            eval_operator_aligned()
        else:
            if not any(ready.values()):
                print(f"\n### {name}\n  [skip] no checkpoints present yet")
            else:
                _needs_sampling(name)


if __name__ == "__main__":
    main()
