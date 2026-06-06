# SIA-Lever / SIA-Lever-120B

**When should a self-improving agent update its harness (H) vs its weights (W)?**

Hackathon research project. Phenomenon-first.

---

## GPT-OSS-120B / H200 Pivot

**SIA-Lever-120B: Learning When to Update Harness vs Weights in Self-Improving Agents.**

SIA proved H+W beats H-only; the open problem is **attribution** — which lever to pull. We turn the
Feedback-Agent's frozen H/W choice into a **learned, evaluated policy**: gpt-oss-120b (± LoRA rank 32)
maps a failed-run trace → `H | W | H_THEN_W | PROMOTE | KILL`, trained on **real measured outcomes**
(no synthetic transition table). Aligns with the SIA paper's future-work direction (learn the
action-selection policy from trajectory/action/outcome triples).

### Layout (pivot)
- `sia_task/` — SIA-Lever as an official SIA custom task (public traces, private measured outcomes,
  `evaluate.py` exposing `evaluate(submission_path)->dict`).
- `gpt_oss/` — gpt-oss-120b lever selector: data builders, base/adapter rollouts, LoRA SFT/DPO/GRPO,
  eval, serve. Shared scoring in `gpt_oss/lever_io.py`.
- `methods/` — oracle-sandwich rule, learned selector, lever-aware W+H policy.
- `baselines/` — vendored official SIA (`vendor/sia`, pinned), paper-style `plateau_then_w`, runners.
- `paper_benchmarks/lawbench/` — LawBench W lane (stretch); `trimul`/`denoising` feasibility notes.
- `scripts/` — CPU regression, smoke, LoRA launchers, `run_gpu_comparison.sh` (one-command).
- `documentation/` — `gpt_oss_pivot.md`, `positioning_next_layer.md`, `paper_repo_notes.md`,
  `reproduction_limits.md`, `gpu_runbook.md`.

### CPU fallback (no GPU/endpoint) — runs now
```bash
bash scripts/run_cpu_regression.sh                 # Phases 0-3 + data pipeline + policy table
python3 gpt_oss/eval/compare_policies.py           # measured rule/fixed/paper-style policy comparison
python3 sia_task/data/public/evaluate.py --gen-dir <dir with submission.json>
python3 scripts/make_demo_preview.py               # render ALL gpt-oss figures now (synthetic, labeled PREVIEW)
python3 scripts/make_demo_report.py                # assemble results/DEMO_REPORT.md (one place for the demo)
```

### Demo figures + diagnostics
- **`results/DEMO_REPORT.md`** — single demo deck: phenomenon → policy comparison → base-vs-LoRA →
  training curve → LawBench. Marks real vs PREVIEW vs pending.
- Each eval emits a figure + human `.md` + per-episode `*_diagnostics.md` (mistakes w/ raw model
  output) + `*_per_mode.png` + `*_action_dist.png`. Adapter eval adds a base-vs-LoRA headline figure
  and a fixed/regressed episode diff. Trainers save `training_curve.png`. SIA-H/LawBench have their
  own curves. So any run is debuggable from the saved artifacts.

### GPU run (when H200 + endpoint available)
```bash
python3 gpt_oss/check_env.py && python3 gpt_oss/smoke_infer.py --model "$GPT_OSS_MODEL"
bash scripts/run_gpu_comparison.sh                 # env->cpu->data->base eval->SFT LoRA->adapter eval->compare
```
Full command list: `documentation/gpu_runbook.md`.

### Expected result shape
Naive policies (H_only/W_only/alternating) high regret; `plateau_then_w` low regret but mis-attributes
(~0.67 lever-acc); oracle-sandwich rule 1.00 acc / 0 regret; **gpt-oss-120b + LoRA improves over base
gpt-oss-120b** on lever accuracy + regret + invalid-JSON. Artifacts: `results/final_comparison.{csv,md}`,
`plots/final_comparison.png`, `results/gpt_oss/*`.

### Honesty
Public SIA = **harness loop only** (0 weight-update files at the pinned commit); the W lever and the
learned selector are **ours**. We do not claim to beat the paper's benchmarks unless run on matched
splits/budget. See `documentation/reproduction_limits.md`.

---

## Original CPU project (Phases 0–3)

## The claim (proven once, measured)
On a 2D-rotation + shortcut-channel symmetry toy (an **adversarial shortcut trap**, not a
naturalistic benchmark):
- A model can hit low prediction error by exploiting a leaked shortcut (fake symmetry win).
- **Weight-only (W) updates PRESERVE the cheating** — structural errors stay high.
- **Harness-first then weight (H→W)** — fix the verifier to detect the shortcut, *then* retrain —
  produces real structural improvement (group-axiom errors collapse to ~0).

Money line, measured not asserted: *"W-only preserved the shortcut failure; H-then-W repaired it."*

## Why symmetry
Fake progress is detectable: a cheater predicts well but fails group-structure tests
(composition, equivariance) and succeeds on a broken-symmetry negative control where an
honest learner must fail. That gives ground truth for which lever was actually needed.

## Structure
- `experiments/` — toy + measurement (`data.py`, `model.py`, `verifier.py`, `train.py`, `run_episode.py`)
- `documentation/plan.md` — phased plan (Phase 0 → 4) + exit gates
- `results/summary.md` — measured numbers as they land
- `figures/` — diffs, plots
- `runs/` — per-run artifacts
- `.state/pipeline.json` — phase + gate tracking

## Execution
Local CPU only. Tiny MLP, seconds per train. No SLURM, no GPU.

## Quick start (Phase 0, once implemented)
```bash
python experiments/run_episode.py --stage 1   # prediction-only baseline
```

## Status
Phase 0 (toy + measurement). See `documentation/plan.md` and `.state/pipeline.json`.
