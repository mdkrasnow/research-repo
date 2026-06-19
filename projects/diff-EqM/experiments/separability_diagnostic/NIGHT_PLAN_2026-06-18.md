# Overnight autonomous research plan — 2026-06-18 → 19

**Goal (user, AFK):** push the SCOPE of trajectory-metacognition. Three threads, each
mechanism-grounded, with positive + negative controls, equal-NFE, and exact labels where
available. Deliver decision-grade verdicts + honest scope. SSH up all night for GPU jobs.

Unifying prior finding: the probe keys on descent **instability** → rescues
collapse/broken-structure failures (image gen, maze planning), blind to confident-wrong
(MNIST inpaint). Each thread tests an edge of that boundary.

## Thread 1 — Does it aid OOD GENERALIZATION? (cheap, reuse maze)
`maze_ood_scaling.py`. One c7-trained EqM, eval on tiers c5…c13 (increasing OOD), FIXED
budget (equal NFE everywhere). Probe trained on c7 in-dist, FROZEN, applied to all tiers.
- H1 vanilla degrades with OOD. H2 probe−vanilla advantage GROWS with OOD. H3 frozen
  in-dist probe still flags OOD failures (AUROC>chance OOD).
- Positive control: oracle. Negative: random-restart + vanilla. KILL if H2 corr ≤ 0.
- Why it matters: "metacognition is a generalization aid, advantage widens under shift" is
  a strong, novel claim distinct from image-FID.

## Thread 2 — A THIRD task type: constraint reasoning (Sudoku-EqM) (build, GPU)
`sudoku_eqm/`. Conditional EqM: clues → solved grid. Invalid = constraint violation (exact
checker = oracle). Distinct from planning (no path; pure CSP). Metacog: probe>random at
equal NFE?
- Mechanism: constraint-violating fills = inconsistent local minima = disturbed descent →
  probe should grip (instability-type). If it works → mechanism spans gen/planning/CSP.
- Positive: oracle. Negative: random + vanilla. KILL if AUROC≈chance AND gap≈0 (then it's
  a confident-wrong task like inpaint — also informative).

## Thread 3 — Can INPAINTING be made positive? (cheap, reuse MNIST)
`mnist_eqm/mnist_difficulty_sweep.py`. Hypothesis: MNIST inpaint was null because failures
were confident-wrong (clean descent). Push mask fraction 0.3→0.95: at extreme masks the
model must hallucinate structure → some failures become collapse/incoherent (instability),
which the probe CAN grip. Sweep AUROC + probe−random vs mask size. Also add a
structural-coherence oracle (broken-blob vs coherent-digit) alongside the classifier oracle.
- Find the regime (if any) where AUROC rises above chance and gap>0.
- If a positive regime exists → confirms the scope-boundary mechanism (instability-failures
  are detectable even in inpainting). If none → boundary is hard; strengthens the limit.
- Negative control built-in: the low-mask regime (known null). Positive: oracle ceiling.

## Discipline
- Smoke each script locally (tiny n, CPU) before any cluster submit.
- Equal-NFE in every action comparison. De-confound AUROC within norm bins.
- Commit + push after each thread produces a verdict. Honest nulls are results.
- Job tracking: every submit → pipeline.json active_runs + commit. Poll via background watchers.
- No overclaim: small EqMs, single-checkpoint caveats stated.

## Priority / order
1. Thread 1 (fastest to signal, full reuse) — submit GPU first.
2. Thread 3 (cheap CPU, reuse) — run locally in parallel.
3. Thread 2 (most build) — build data+train+metacog, submit GPU.
Monitor all via background watchers; reconcile + write results as each lands.
