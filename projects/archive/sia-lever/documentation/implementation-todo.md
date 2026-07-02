# Implementation TODO — SIA-Lever

## Phase 0 — Toy + measurement (HOP) — DONE
- [x] (T1) `data.py` — rotation task + neg-control + full-vector shortcut leak
- [x] (T2) `model.py` — enc → A(Δ) → dec
- [x] (T3) `verifier.py` — v0 (pred-only) + v1 (structural battery)
- [x] (T4) `train.py` — prediction_only + structural objectives
- [x] (T5) `run_episode.py` — 4-stage driver, gate check
- [x] (T6) **GATE PASS**: pred-only clean_mse~0 AND neg_control low + composition_err~5.6 (shortcut win real)

## Phase 1 — Four-stage episode (THE WIN) — DONE
- [x] (T7) Stage 1 pred-only
- [x] (T8) Stage 2 W-only (cheat persists)
- [x] (T9) Stage 3 v0→v1 re-score (shortcut exposed)
- [x] (T10) Stage 4 H→W structural retrain (clean_win)
- [x] (T11) **GATE PASS 5/5**: stage-4 neg_control > stage-2. `run_seeds.py` → table + plot.

## Phase 2 — Agentic H (WALK) — DONE
- [x] (T12) Subagent patched harness/verifier.py from weak harness + trace (no human-written tests)
- [x] (T13) figures/harness_update.diff (178 lines); cheater flagged 5/5, honest clean 0/5

## Phase 3 — Lever selector (RUN) — DONE
- [x] (T14) model_prior_gap (valid harness, honest-but-weak model → W)
- [x] (T15) bad_verifier (buggy harness rejects good model → H)
- [x] (T16) oracle-sandwich selector reads real trace → action; regret by REAL rerun (no lookup table)
- [x] (T17) **GATE PASS** (10 seeds, 30 ep): selector regret 0.014 << alternating 0.148 << H_only 0.318 << W_only 0.421; selector lowest at every W_COST incl 0.00; Wilcoxon p=0.013

## Rigor pass (paper-level) — DONE
- [x] (R1) Replace ill-posed equivariance proxy with group-axiom metrics (identity, inverse)
- [x] (R2) Phase 1: 5→15 seeds + Welch t + 95% CI + Cohen's d on W-only vs H→W
- [x] (R3) Wording: "preserves/repairs" not "made worse/reinforced"
- [x] (R4) Phase 2 provenance doc (exact prompt + before/after + detection output)
- [x] (R5) Phase 3: separate measurement from scoring; W_COST sweep {0,0.01,0.03,0.05,0.10}; 10 seeds; raw+adjusted; Wilcoxon
- [x] (R6) Leak-strength robustness sweep (data.py leak_alpha); honest two-finding result
- [x] (R7) Paper-style writeup documentation/writeup.md

## Phase 4 — Gemma garnish (FLY) — BLOCKED (GPU / user decision)
- [ ] (T18) Gemma LLM selector — needs ollama model pull (paused; user rejected pull)
- [ ] (T19) LoRA on trace→action — needs **GPU** (hackathon blocker)

## Blocked
(none)

## Completed
(none yet)
