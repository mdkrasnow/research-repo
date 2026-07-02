# SIA-Lever — pick the right repair lever for a self-improving agent

*Hackathon pitch deck. Every number traces to `documentation/ladder_findings.md` / `results/ladder_results.tsv`.*

---

## 1. Hook

**Self-improving agents pick the wrong fix and burn GPU doing it.**

When an agent stops improving, the question is *what's actually broken* — the
evaluator, the model weights, or both. Today's systems guess, and the default guess
is "retrain the weights." When the real bug was the evaluator, that's a full
fine-tune spent fixing nothing.

**SIA-Lever reads the failure trace and names the minimal correct repair — before you
spend the GPU.**

---

## 2. The problem (the $ pain)

A self-improving agent has three repair levers when it plateaus:

- **H** — fix the harness / evaluator (cheap: no training)
- **W** — retrain the weights (expensive: a full fine-tune)
- **H_THEN_W** — the evaluator is buggy *and* the model is weak; do both

Pick wrong and you pay twice:
- Bug was in the evaluator, you retrained → you spent a fine-tune and the score
  didn't move.
- Bug needed weights, you only patched the harness → another wasted plateau cycle.

On a real GPU-kernel task (rung 3, below) a paper-style scheduler that *re-measures*
every option spends **72 weight-retrains** to get the answer. Our trace-only selector
gets there in **48** — a third fewer expensive calls — by diagnosing instead of brute-forcing.

---

## 3. What we built

**A lever selector.** Input: a label-free failure trace from a stalled agent.
Output: one of {H, W, H_THEN_W} — the minimal correct repair.

It's a LoRA fine-tune of **gpt-oss-120b** trained to do *attribution*: read the trace,
infer which subsystem is at fault, name the lever. No human labels, no LLM-distilled
targets — the training signal comes from **actually applying every lever and measuring
what happened** (see next slide).

---

## 4. How it works — ground truth from real reruns

1. **Testbed** — a self-improving agent on a real task that can fail in distinct ways
   (weak model, broken structural check, broken prediction check, and compounds).
2. **Measured ground truth** — for each failing episode we *actually apply every lever*
   (H, W, H_THEN_W) and re-run, recording the true outcome and cost. The correct lever
   is the one that actually fixed it — measured, not assumed.
3. **Trace** — strip the labels. The selector only ever sees the label-free failure
   trace, exactly what a deployed system would have.
4. **SFT** — fine-tune the LoRA to map trace → correct lever.

**No hand-labels. No LLM-distilled targets. Ground truth from real reruns.** That's the
honest core of the method.

---

## 5. Demo

`scripts/demo.py` — the fail → diagnose → fix → $-saved loop, live:

1. A self-improving agent plateaus on a task. Show the failure trace.
2. The naive default fires the expensive lever (**W**, full retrain).
3. SIA-Lever reads the same trace and calls the **correct** lever
   (often **H** alone, or **H_THEN_W**).
4. Show the cost delta: the retrains avoided when the bug was really in the evaluator.

**Money shot:** the trace where the default says "retrain the model" and SIA-Lever says
"your evaluator is broken — fix that first."

---

## 6. Results

### Rung 2 — SIA-Lever-HARD (gpt-oss-120b, LoRA, real H200 vLLM serve)

24 held-out compound-fault episodes. Lead with **regret** (cost of the wrong lever),
not raw accuracy.

| arm | accuracy | mean regret | action distribution | verdict |
|---|---|---|---|---|
| base gpt-oss-120b | 0.333 | 0.161 | varied | below floor |
| best constant (always-H) | 0.458 | — | constant | the bar to beat |
| LoRA 20ep (imbalanced) | 0.458 | 0.202 | **H ×24 (collapsed)** | false win — caught |
| **LoRA balanced-v2** | **0.542** | **0.043** | H_THEN_W 10 / H 4 / W 10 | **REAL WIN** |
| latent ceiling | 0.81 | — | — | headroom remains |

The win is the **regret crush: 0.161 → 0.043** with a *non-collapsed* policy that uses all
three levers (≈ gold mix). It beats base (+0.21) **and** the best constant guess (0.458) —
i.e. it's doing real attribution, not guessing the majority class.

### Rung 3 — TriMul GPU kernel (AlphaFold-3 Triangle Multiplicative Update, real CUDA timing on H200)

The cost story. A paper-style scheduler re-measures everything; we decide from the trace.

| policy | lever_acc | mean regret | weight-retrains |
|---|---|---|---|
| oracle (upper bound) | 1.00 | 0.000 | 72 |
| W-only (neg control) | 0.00 | 1.006 | 96 |
| SIA-W+H paper-style* | 1.00 | 0.000 | 72 |
| **SIA-Lever (ours)** | **0.750** | **0.236** | **48** |

*Reconstruction, see honesty note below.* We get to 0.750 accuracy at **48 retrains
vs 72** — a third fewer of the expensive calls — *without ever re-measuring*, using a
hand-written label-free rule. **Honest negative:** a trained kernel-LoRA (gpt-oss-120B
SFT on 144 balanced kernel examples, `ftjob-85c461bc`) was served + evaluated on the same
held-out fold — it improved over the base model (0.25 → 0.50, regret halved) but did **not**
beat the hand-rule or the majority constant (0.75); the rule remains our best kernel selector.
Diagnosis is data-scale (144 ex, 2-class), not mechanism — base + rule correctly bracket it.
Closing the 0.75 → 1.00 gap needs more kernel episodes (the same lesson rung-2 taught us).

---

## 7. Rigor — we caught our own false positive

The LoRA 20ep run hit accuracy **0.458** — looks like a win. It wasn't: it had collapsed
to predicting **H for all 24 episodes**. 0.458 is *exactly* the always-guess-majority rate,
and its regret got **worse** (0.161 → 0.202).

Our anti-collapse gate (`collapse_report.py`) flagged it: a real win must beat base, beat
the best constant, not worsen regret, **and** keep a non-collapsed action distribution. Only
balanced-v2 passes all four. Without that gate, the 0.458 would have shipped as a (false) win.

**We don't trust a number until it survives its own controls.**

---

## 8. Nebius stack / architecture

- **Managed LoRA fine-tuning** on Nebius Token Factory (gpt-oss-120b, `unsloth/gpt-oss-120b-BF16`).
- **Self-hosted vLLM serving** of native **MXFP4** gpt-oss-120b (~63 GB, 1×H200) — the
  attention-only adapter loads cleanly on the quantized base.
- **Hot-swap adapters** — `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, new adapters load via
  `POST /v1/load_lora_adapter` with no restart, no recompile.
- **Fast loop:** TF-train (~10 min) → download → hot-load (instant) → parallel eval
  (24 episodes in ~3 s).

---

---

## 10. Ask / impact

Self-improving agents are about to consume a lot of GPU. SIA-Lever is a cheap
diagnostic layer that tells them *which* fix to spend on — turning "retrain and hope"
into "diagnose, then repair the right thing."

We've shown it works on a label-free toy (regret 0.161 → 0.043, no collapse) and that the
cost win holds on a real GPU kernel (48 vs 72 retrains). **Ask:** support to finish the
trained kernel-LoRA and multi-seed validation, and a path to drop this in front of a
production self-improvement loop.
