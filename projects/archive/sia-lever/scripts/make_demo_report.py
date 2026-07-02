#!/usr/bin/env python3
"""Assemble all SIA-Lever-120B figures + key numbers into one results/DEMO_REPORT.md.

Auto-detects real gpt-oss results vs the synthetic PREVIEW, and marks GPU-pending sections. Safe to
run anytime on CPU; re-run after the GPU comparison to refresh with real numbers.
"""

import glob
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(p):
    return os.path.relpath(p, PROJ)


def newest(pat):
    m = sorted(glob.glob(os.path.join(PROJ, pat)))
    return m[-1] if m else None


def read(pat):
    p = newest(pat)
    return (open(p).read() if p else None), p


def fig(path, caption):
    if path and os.path.exists(path):
        return [f"![{caption}]({rel(path)})", f"*{caption}* — `{rel(path)}`", ""]
    return [f"_(pending) {caption}_", ""]


def main():
    L = ["# SIA-Lever-120B — DEMO REPORT", "",
         "One place for the demo: the phenomenon, the lever-attribution result, and the "
         "gpt-oss-120b base-vs-LoRA comparison. Figures are real CPU measurements unless marked "
         "PREVIEW (synthetic, replaced by the GPU run) or pending.", "",
         "Refresh: `python3 scripts/make_demo_report.py` (after `run_gpu_comparison.sh` for real "
         "gpt-oss numbers).", "", "---", ""]

    # 1. phenomenon (CPU, real)
    L += ["## 1. The phenomenon (CPU, measured, 15 seeds)",
          "W-only PRESERVES shortcut-cheating; H→W REPAIRS it. Welch t: shortcut_sens d=16.3 "
          "(p=1.6e-16), composition d=2.58. Gate 15/15.", ""]
    L += fig(os.path.join(PROJ, "results", "episode_plot.png"),
             "Phase 1: W-only preserves the shortcut; H→W learns the real group")
    L += fig(os.path.join(PROJ, "results", "phase3_plot.png"),
             "Phase 3: lever-selector lowest regret across the whole W_COST sweep")

    # 2. policy comparison (CPU, real)
    L += ["## 2. Lever-attribution policy comparison (measured regret)", ""]
    cmp_md, _ = read("results/final_comparison.md")
    if cmp_md:
        L += [cmp_md.split("Notes:")[0].strip(), ""]
    L += fig(os.path.join(PROJ, "plots", "final_comparison.png"),
             "Regret / lever-accuracy / W-call-rate by policy (green=oracle/rule, gray=baseline)")

    # 3. gpt-oss base vs LoRA (real if present, else preview)
    real_adapter_md = newest("results/gpt_oss/adapter_eval_*.md")
    preview_dir = os.path.join(PROJ, "results", "gpt_oss", "preview")
    is_preview = real_adapter_md is None
    L += [f"## 3. gpt-oss-120b base vs LoRA selector {'(PREVIEW — synthetic)' if is_preview else '(real)'}",
          ""]
    if is_preview:
        L += ["> **PREVIEW**: synthetic base+LoRA rollouts illustrate the figures. The GPU run "
              "(`run_gpu_comparison.sh`) overwrites these with real gpt-oss-120b measurements.", ""]
        amd, _ = read("results/gpt_oss/preview/adapter_eval_*.md")
        apng = newest("results/gpt_oss/preview/adapter_eval_*.png")
        per = newest("results/gpt_oss/preview/adapter_per_mode_*.png")
    else:
        amd = open(real_adapter_md).read()
        apng = newest("results/gpt_oss/adapter_eval_*.png")
        per = newest("results/gpt_oss/adapter_per_mode_*.png")
    if amd:
        L += [amd.split("## What LoRA changed")[0].strip(), ""]
        if "## What LoRA changed" in amd:
            L += ["### What LoRA changed vs base", "",
                  "## What LoRA changed".join(amd.split("## What LoRA changed")[1:]).strip(), ""]
    L += fig(apng, "Base vs +LoRA: lever accuracy ↑, mean regret ↓, invalid-JSON ↓")
    L += fig(per, "Per-mode lever accuracy: base vs LoRA")

    # 4. training curve (pending until GPU)
    tc = newest("adapters/gpt_oss_120b/*/training_curve.png")
    L += ["## 4. LoRA training curve", ""]
    L += fig(tc, "SFT/DPO loss (and GRPO reward) vs step")

    # 5. official SIA-H + LawBench (pending until run)
    L += ["## 5. Official SIA-H + LawBench (stretch)", ""]
    L += fig(os.path.join(PROJ, "results", "official_sia_generation_curve.png"),
             "Official SIA-H: lever accuracy / regret vs generation")
    L += fig(os.path.join(PROJ, "plots", "lawbench_compare.png"),
             "LawBench: ours vs paper (13.5 / 50.0 / 70.1; * = reduced split)")

    L += ["---", "",
          "## Diagnostics (for debugging a run)",
          "- per-episode mistakes + raw model output: `results/gpt_oss/<tag>_diagnostics.md`",
          "- base→LoRA fixed/regressed episode diff: in the adapter_eval `.md`",
          "- action distribution + per-mode accuracy: `<tag>_action_dist.png`, `<tag>_per_mode.png`",
          "- adapter provenance (hashes/gpu/config): `adapters/gpt_oss_120b/<run>/`",
          "- env/endpoint readiness: `python3 gpt_oss/check_env.py`",
          ""]

    out = os.path.join(PROJ, "results", "DEMO_REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {rel(out)}  ({'PREVIEW gpt-oss figures' if is_preview else 'real gpt-oss figures'})")


if __name__ == "__main__":
    main()
