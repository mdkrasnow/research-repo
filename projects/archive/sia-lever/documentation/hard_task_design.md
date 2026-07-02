# SIA-Lever-HARD — making the lever task worth a GPU

## Why the original task can't justify Phase 4
`experiments/trace_difficulty_probe.py` on the 30-episode cache (train seeds 0–6, eval 7–9):

| trace | tiny-model (2-feature rule) held-out acc |
|---|---|
| current (with giveaway booleans) | **1.00** |
| hardened (raw numbers only, booleans stripped) | **1.00** |

The triviality is **structural, not cosmetic**: the three failure modes are constructed pure, so
`(neg_control_mse, composition_error)` already separates them perfectly. Stripping the trace flags
does nothing. A gpt-oss-120b LoRA cannot demonstrate "learned attribution" on a task a 2-threshold
rule solves on held-out data. **Conclusion: don't spend the H200 on the original task.**

## The fix is task-side, not trace-side
Keep the integrity bar (real reruns, no transition table, label-free trace) but break the clean
separability of the *latent fault*:

1. **Continuous severity.** Start models from a grid of `(init objective, train steps, leak
   strength)` instead of three pure archetypes → a spectrum of capability × cheating, including
   mediocre boundary models where the best lever flips on small differences.
2. **Compound faults.** A buggy/weak harness *and* a weak/cheating model at once → the right lever
   depends on the interaction (which fault dominates), not one axis.
3. **Measurement noise.** Several seed folds → thresholds fit on train don't transfer cleanly, so a
   tiny model can't memorize the boundary; only a model that actually reasons over several signals
   generalizes.

Ground truth stays honest: each episode APPLIES every lever for real (retrain/keep), MEASURES the
resulting hidden true-score (capability × honesty), and takes the cost-adjusted argmax as gold. No
hand-authored labels.

`experiments/hard_task.py` emits `gpt_oss/data/out/hard_cache.jsonl` (same schema as the measured
cache, so the SFT/DPO/GRPO builders and the evaluator consume it unchanged).

## The gate
A harder task is only worth it if a tiny model now FAILS on held-out while headroom remains for a
capable model. The acceptance test is the difficulty probe itself:

```bash
python experiments/hard_task.py --reps 4 --steps 500
python experiments/trace_difficulty_probe.py --cache gpt_oss/data/out/hard_cache.jsonl --eval-seeds 1
```

- tiny-model held-out **≈ 1.0**  → still trivial, iterate the grid (more boundary/compound episodes).
- tiny-model held-out **≪ 1.0** (and gold classes balanced) → real headroom → Phase 4 informative.

## Results (build: 72 episodes, reps=3, steps=300, real reruns)

`python experiments/trace_difficulty_probe.py --cache gpt_oss/data/out/hard_cache.jsonl --eval-seeds 1`

| measure | original 30-ep task | **SIA-Lever-HARD** |
|---|---|---|
| tiny 2-feature model, held-out | **1.00** | **0.46** |
| always-majority floor (eval) | 0.33 | 0.46 |
| hand mechanism-rule (oracle_sandwich) | 1.00 | **0.29** (below random — compound faults break it) |
| latent-config ceiling (achievable) | 1.00 | **0.81** |
| gold balance | clean 3-way | H 34 / H_THEN_W 23 / W 15 |

**Verdict: headroom is real.** Tiny threshold model and the hand rule are stuck at/below the
majority floor (~0.46), while the latent fault config determines the answer up to **0.81** (the
remaining ~19% is irreducible seed noise — gold flips across replicates for 14/24 configs). The
band **[0.47 → ~0.81]** is exactly what a model that *reasons over multiple trace signals* can climb
and a 2-feature rule cannot. The depth-2 pair even **overfits** (train 0.69 → eval 0.46), confirming
no tiny model generalizes.

Why the hand rule collapses to 0.29: on **compound** faults (e.g. buggy harness AND weak model) the
single-signal rule says "harness broken → H" but the measured best is H_THEN_W (must also retrain).
Single-axis thresholds fail precisely where faults interact — the intended difficulty.

### What this unlocks
- Phase 4 (gpt-oss-120b ± LoRA) on `hard_cache.jsonl` now has something to learn: beat ~0.47 toward
  ~0.81. A base-vs-LoRA gap here would be a *real* attribution result, not a JSON-formatting artifact.
- The SFT/DPO/GRPO builders + evaluator consume `hard_cache.jsonl` unchanged (same schema), so the
  whole gpt-oss lane retargets by pointing `--cache` at it.

### Caveats (honest)
- Ceiling ≈ 0.81, not 1.0: ~19% of episodes are seed-ambiguous. Report gains as movement within the
  band, never as "perfect attribution."
- Still a toy (rotation+shortcut); harder ≠ naturalistic. It tests "can the selector combine signals
  under noise + compound faults," not real-world SIA.
- The hand rule's 0.29 means the *original* oracle_sandwich rule is NOT a valid baseline on the hard
  task — a new rule (or the learned model) must be built and scored honestly against the 0.47 floor.

## Tomorrow (GPU) — turnkey
The whole gpt-oss lane retargets at the hard task with ONE env var (every script now takes `--cache`;
the one-command script threads it):
```bash
CACHE=gpt_oss/data/out/hard_cache.jsonl bash scripts/run_gpu_comparison.sh \
  --model "$GPT_OSS_MODEL" --base-url "$GPT_OSS_BASE_URL"
```
Step [3b] runs the headroom gate automatically and prints INFORMATIVE/NO-HEADROOM before any GPU
spend. Verified end-to-end on CPU (dry-run + offline builders + rollout) — only the model calls need
the endpoint/H200.

## Metric to report on the hard task: LEVER ACCURACY, not regret
On the hard cache, `plateau_then_w` still gets ~0.016 mean regret because it PEEKS at the measured
`reward_by_action` (it re-measures after H). Regret is therefore gamed by a measurement-peeking
baseline and is NOT the discriminating metric here. The trace-only learnable signal is **lever
accuracy**: tiny rule + original oracle_sandwich rule sit at 0.29–0.46, ceiling 0.81. Report
base-vs-LoRA on **lever accuracy** (and per-mode accuracy), state the 0.47 floor and 0.81 ceiling,
and treat regret as secondary.

## The reusable gate
`trace_difficulty_probe.py` prints `NO headroom` (don't spend GPU) vs `INFORMATIVE` (headroom band).
Run it on any candidate cache before elevating to the GPU rung. It is the cheap CPU pre-flight that
makes the expensive rung conditional on there being something to learn.
