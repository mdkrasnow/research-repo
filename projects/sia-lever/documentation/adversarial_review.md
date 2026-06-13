# Adversarial Review — SIA-Lever / SIA-Lever-120B

Date: 2026-06-06. Reviewer: adversarial audit pass (assume every claim wrong until proven).
Method: read all core code + docs, reran the full CPU regression, broke the evaluator with
malformed input, verified provenance byte-for-byte, traced the gold/label definitions across every
scorer, and added a standing integrity regression. Patches applied and reverified.

## TL;DR verdict
- **The CPU core phenomenon is REAL and reproduces exactly.** Phase 0/1/2/3 numbers match the
  committed summary; the Phase-2 agentic-H diff is byte-for-byte reproducible from the weak harness.
- **The headline framing was over-strong in two places** (now corrected in README + summary):
  1. The lever-attribution task is **near-deterministic from the trace** — a 3-line boolean rule
     scores **9/9** on the private eval. High selector/LLM accuracy is expected, not a discovery.
  2. Against the **strongest** baseline (`plateau_then_w`) the selector does **not** lower regret on
     the held-out eval (both 0 regret, both 6/9 W-calls); its only edge is *nominal, outcome-neutral*
     lever-accuracy.
- **The gpt-oss-120b / LoRA result does not exist yet** (GPU/endpoint blocked). No base-vs-LoRA
  number may be quoted as a result. Scaffold is runnable to the GPU boundary.
- **No data leakage**: private measured outcomes are read only by the evaluator + task builder; the
  reference agent reads public only; train/eval split is disjoint by seed and matches the published
  private eval seeds (7,8,9).
- Several robustness bugs found and **fixed** (evaluator crash on malformed JSON, silent duplicate
  dedupe, missing provenance hashes, check_env display bug, missing dry-run on rollout_adapter).

## Commands run (this audit) — all PASS
| command | result |
|---|---|
| `python experiments/run_seeds.py --seeds 15 --steps 2000` | gate 15/15; matches summary (shortcut_sens 1.009→0.002, comp 4.66→0.001, Welch d=16.3) |
| `python harness/verifier.py` | detection PASS (cheater 5/5, honest 0/5) |
| `python experiments/phase3.py --seeds 10 --steps 800` | selector 0.97 acc / 0.014 regret; Wilcoxon p=0.013; lowest regret at every W_COST |
| `python gpt_oss/eval/compare_policies.py` | rule 1.00/0.0; **plateau_then_w 0.67/0.0 (ties on regret, 6/9 W)** |
| `bash scripts/run_cpu_regression.sh` | full pipeline PASS incl. new robustness harness |
| `python scripts/robustness_tests.py` | ALL ROBUSTNESS CHECKS PASS |
| evaluator break tests (valid/missing/invalid/dup/malformed/empty/extra) | handled; malformed now fail-soft (was a crash) |
| `patch weak_v0 < harness_update.diff` vs `verifier.py` | IDENTICAL (provenance verified) |
| `check_env / build_{trace,sft,dpo,grpo} / rollout_{base,adapter} --dry-run` | run to GPU/endpoint boundary cleanly |
| trace-dataset spine `--seeds 1` | real reruns produce cache + hash; reproduces the degenerate bad_verifier mismatch |

## Bugs found & patches made
| # | severity | file | issue | fix |
|---|---|---|---|---|
| 1 | med | `sia_task/data/public/evaluate.py` | malformed/non-object submission JSON **crashed** the evaluator (uncaught `JSONDecodeError`) — a target agent could DoS the scorer | fail-soft: return worst-case results dict with an `error` field |
| 2 | low-med | `sia_task/data/public/evaluate.py` | duplicate `episode_id`s silently deduped to **last** occurrence (could hide conflicting predictions) | keep FIRST deterministically + report `duplicate_episode_ids` |
| 3 | med | `gpt_oss/train/provenance.py` | provenance omitted **harness hash, evaluator hash, adapter-weights hash** (goal-required) | added all three + consolidated `provenance.json` record |
| 4 | low | `gpt_oss/check_env.py` | operator-precedence bug: `HF_TOKEN` always printed "SET" even when unset | fixed `is_secret` logic |
| 5 | low | `gpt_oss/rollout/rollout_adapter.py` | no `--dry-run` (rollout_base had one); `--adapter` required even for plumbing tests | added `--dry-run`; `--adapter` optional under dry-run |
| 6 | doc | `README.md`, `results/summary.md`, `experiments/run_seeds.py` | overclaims (see below) | downgraded to defensible wording |

## Findings that change interpretation (not crashes — framing)
1. **Task is near-deterministic from the observable trace.** `observable_trace` prints
   `shortcut_cheat_signature`, `harness_accepts_known_good_model`, `predicts_clean`. The gold action
   is a pure function of those three booleans:
   `(harness_ok, cheat, predicts) → {(T,T,T):H_THEN_W, (T,F,F):W, (F,F,T):H}`. A 3-line rule scores
   **9/9** on the private eval. So `oracle_sandwich_rule` getting 1.00 is tautological, and any
   future "gpt-oss reads the trace" result measures JSON-formatting robustness, not attribution
   discovery. This is now stated in README + summary. It does not invalidate the prototype; it bounds
   the claim.
2. **Strongest baseline ties on regret.** `plateau_then_w` reaches **0 regret with 6/9 W-calls** on
   the held-out eval — identical outcome to the rule. The selector's measured advantage is *nominal*
   lever-accuracy (1.00 vs 0.67), and it is **outcome-neutral** (names `W` where plateau names the
   equal-reward `H_THEN_W`). `reproduction_limits.md` already conceded this; README/summary now do
   too. (Caveat: `plateau_then_w` is also given measured H/H_THEN_W outcomes to decide, so it has
   more info than a trace-only selector — it is a strong, not a weak, baseline.)
3. **Phase-1 "W-only preserves / H→W repairs" is partly definitional.** S2 (W-only) continues the
   SAME prediction-only objective the shortcut already satisfies, so "more W can't fix structure" is
   true by construction under a weak harness; S4 differs from S2 ONLY by switching to the structural
   objective (= the harness change). The phenomenon is a valid *illustration* of "the harness defines
   what a W-update optimizes," not a surprising empirical discovery. Already framed as an "adversarial
   shortcut trap"; flagged here for honesty.
4. **Two gold definitions coexist.** `action_outcome_cache.jsonl` stores `correct_action =`
   pre-registered `ep['correct']`; `sia_task` private data + `compare_policies` use
   `cost_adjusted_best` (measured argmax). They disagree on **2/30** episodes
   (`bad_verifier_seed_000`, `shortcut_leak_seed_002`), **both in train seeds** — the held-out eval
   (7,8,9) is consistent, so the headline is unaffected. `shortcut_leak_seed_002` is **degenerate**:
   every lever scores ~0 (H_THEN_W retrain collapsed to 0.0), so its measured-best label flips to H,
   contradicting the cheat-signature trace → one noisy SFT training row. New
   `scripts/robustness_tests.py::test_gold_definition` asserts eval-seed consistency and surfaces the
   train mismatches every run.
5. **identity/inverse don't "collapse to ~0".** After H→W they drop ~5× (≈1.7→0.32, 2.7→0.31) but
   stay ~0.3. summary.md's table was already honest; the inline interpretation string was loosened.

## Provenance (Phase 2) — VERIFIED
`harness/verifier_weak_v0_backup.py` + `figures/harness_update.diff` == `harness/verifier.py`
byte-for-byte (reconstructed sha == current sha `692e12a487789415`). The patched harness is NOT a
copy of the reference structural verifier (`experiments/verifier.py`): different metric battery
(is_cheating signature vs identity/inverse axioms). Recorded with checksums in
`harness/PROVENANCE.md` + reproduce command. Detection reproduced: cheater 5/5, honest 0/5.

## Baseline fairness — HONEST
- Official SIA vendored at pinned commit `99db0e87cbe3a67f6fc251d33b72c88ee1edfac5` (real git
  checkout, matches `sia_commit.txt`). `grep -rilE "lora|peft|grpo|trl|bitsandbytes"` over `sia/` →
  **0 files**: public SIA is the harness (H) loop only. The W lever + learned selector are ours.
- `plateau_then_w` is labeled **paper-STYLE, not an exact reproduction**, and structurally cannot
  return pure `W` (caps its accuracy by design). Labels are honest.
- No doc claims paper-benchmark reproduction; `reproduction_limits.md` states LawBench/TriMul/
  denoising are NOT reproduced and require matched split/label-set/model/budget.

## Robustness additions (new this audit)
- `scripts/robustness_tests.py` (wired into `run_cpu_regression.sh`): evaluator robustness
  (valid/missing/invalid/duplicate/**malformed**/empty), no-private-data-access scan, cache schema +
  reward-bounds + **meta-hash-matches-bytes**, SFT/DPO train↔eval seed disjointness + private-eval-
  matches-heldout, and gold-definition consistency. All PASS.
- W_COST sweep: already in `phase3.py` (selector lowest at every cost incl. 0.0). leak_alpha sweep:
  `experiments/leak_sweep.py`. Seed sensitivity: 15-seed Welch (Phase 1) + 10-seed Wilcoxon (Phase 3).
- Provenance hashing extended to harness + evaluator + adapter (`gpt_oss/train/provenance.py`).

## Remaining blockers (cannot resolve on CPU)
- gpt-oss-120b **base eval** needs `GPT_OSS_BASE_URL` + key; **LoRA** needs H200 + `requirements-gpu`.
  Until then the entire "learned policy improves over base" thesis is **unvalidated** — and even when
  run, finding 1 bounds what a positive result would mean.
- Official LawBench/TriMul/denoising reproduction needs GPU + matched data (out of scope here).

## Honest demo claims that SURVIVE the audit
1. On an adversarial rotation+shortcut toy, a model hits ~0 clean MSE by reading a leaked shortcut;
   the weak prediction-only harness **prefers** the cheater (0.000152 < 0.002478). [measured]
2. Continuing weight updates under that weak harness **preserves** the shortcut (15 seeds; structural
   errors unchanged); switching the harness to a structural objective and retraining (**H→W**)
   **repairs** it (shortcut_sens & composition → ~0; Welch d=16.3, p=1.6e-16). [measured, reproduced]
3. A coding subagent, given only the weak harness + failed trace, authored a structural verifier that
   flags the cheater (5/5) and clears the honest model (0/5); the diff is byte-reproducible. [verified]
4. A label-free oracle-sandwich rule names the minimal correct lever on a 3-mode prototype with 0
   measured regret. [measured] — with the caveat that the task is near-deterministic from the trace
   and a competent scheduler ties it on regret.

## Claims to NOT make
- ❌ "gpt-oss-120b + LoRA improves over base" — not run.
- ❌ "selector achieves lower regret than the paper-style scheduler" — it ties (0 vs 0) on eval.
- ❌ "we reproduce SIA paper benchmarks" — not run on matched setup.
- ❌ "structural errors collapse to zero" — identity/inverse stay ~0.3.
