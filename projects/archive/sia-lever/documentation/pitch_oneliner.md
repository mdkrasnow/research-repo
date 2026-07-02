# SIA-Lever — pitches at three lengths

## One sentence

When a self-improving agent stalls, SIA-Lever reads the label-free failure trace and
names the minimal correct repair — fix the evaluator, retrain the weights, or both — so
you stop burning full fine-tunes on bugs that were really in the evaluator.

## 30 seconds

Self-improving agents that plateau usually default to "retrain the weights" — the most
expensive fix — even when the real bug was the evaluator. SIA-Lever is a LoRA fine-tune of
gpt-oss-120b that reads the failure trace alone and picks the right lever: H (fix harness),
W (retrain), or H_THEN_W (both). The training signal isn't hand-labels or distilled LLM
targets — it's ground truth from *actually applying every lever and re-running*. On a hard
compound-fault testbed it crushes regret 0.161 → 0.043 over the base model without
collapsing, and on a real AlphaFold-3 GPU kernel task it reaches the right lever with 48
weight-retrains instead of a re-measuring scheduler's 72 — a third fewer expensive calls.

## 2 minutes

A self-improving agent has three levers when it stops improving: fix the harness/evaluator
(cheap, no training), retrain the weights (expensive), or do both. Pick wrong and you pay
twice — you retrain when the evaluator was broken, or you patch the harness when the model
genuinely needed weights. The default guess is always "retrain," so the common failure is a
wasted full fine-tune.

SIA-Lever diagnoses instead of guessing. It's a LoRA fine-tune of gpt-oss-120b trained to do
attribution: read a label-free failure trace, infer which subsystem is at fault, name the
minimal correct lever. The honest core is how we get ground truth — for every failing episode
we *actually apply all three levers and re-run*, and the correct lever is the one that
measurably fixed it. No human labels, no LLM-distilled targets.

Results, leading with cost. On our HARD compound-fault testbed (24 held-out episodes, served
on a real H200 with vLLM MXFP4), a balanced LoRA beats the base model's accuracy (0.333 →
0.542), beats the best constant guess (0.458), and crushes mean regret 0.161 → 0.043 while
keeping a non-collapsed policy that uses all three levers. We earned that result honestly: an
earlier run hit the same 0.458 but had collapsed to always-predict-H — our anti-collapse gate
caught that false win, and only the balanced run passes all four gate checks. On a real
AlphaFold-3 Triangle Multiplicative Update GPU kernel, our trace-only selector reaches the
right lever with 48 weight-retrains versus a paper-style re-measuring scheduler's 72.

Honest limits: the kernel selector is currently a hand-written rule with a trained LoRA in
progress, the W+H baseline is our reconstruction (and privileged — it re-measures), so we
claim "minimal correct lever, more cheaply," not "we beat the published paper." Stack: Nebius
Token Factory managed LoRA, self-hosted vLLM MXFP4 on H200, hot-swap adapters. Ask: support to
finish the trained kernel-LoRA and multi-seed validation.
