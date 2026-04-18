# Publishability Plan — DG-ANM for EqM

## North Star

**Goal: publish at NeurIPS, ICML, or ICLR.**

Every experimental decision must be evaluated against: *does this move us closer to a credible top-venue submission?* If no, de-prioritize.

## What the target looks like

A top-venue paper requires **at least one** of the following credible claims, backed by rigorous evidence:

1. **SOTA or near-SOTA on a standard benchmark** — e.g., match or beat EqM-XL/2's FID 1.90 on ImageNet-256 class-conditional, or equivalent on CIFAR-10 (EqM: 3.36).
2. **Consistent relative improvement at matched compute across scales** — DG-ANM beats vanilla EqM at S/2, B/2, L/2, XL/2 with the same training budget. Scaling curves, not single points.
3. **A compelling mechanistic story** — DG-ANM solves a specific failure mode of EqM (e.g., off-manifold energy landscape pathology) with theory + targeted empirical evidence. Improvement can be smaller if the story is deep.

## Honest assessment of current state

- **Autoresearch scale (2-epoch IN-100, bs=16, FID ~253-278)**: a proxy, not a result. FID at this scale is ~100× worse than the EqM paper's reporting regime (~1.9–50). A 6% relative gain here is **not publishable** and may not survive at full scale. Reviewers will ask: does it hold at EqM-B/2, 80 epochs? Does it hold on CIFAR-10? On 2+ seeds?
- **Prior IN-100 80-epoch result**: FID 112.58 with gamma=4.0 (old best). This is a real data point but still ~30× worse than published ablation-scale FIDs — suggests we are training a *much* smaller/shorter model than EqM-B/2 @ 80ep.
- **Strengths**: working autoresearch loop, clear method (DG-ANM), results.tsv discipline, reproducible SLURM infra.
- **Gaps**: no full-scale confirmation, single seed, single dataset, no mechanism-level analysis, no comparison to baselines at matched compute.

## Stage plan

### Stage A — Finish the pilot sweep (current)
**Objective**: identify the best DG-ANM config on the cheap proxy.
- Complete round 4 (9 candidates across 4 dimensions).
- Do a **combination round** using the top-1 from each productive dimension.
- Do a **repeatability check**: run the best config 3× with different seeds. If the gain is within FID noise (±2–3 on 2K samples), the result is not real.
- **Exit criterion**: we have a single best config whose advantage over vanilla-EqM baseline is ≥1 FID across 3 seeds on the proxy.
- **If gain does not survive 3-seed check**: pause autoresearch, re-evaluate method, do not scale up.

### Stage A.5 — Reproduce vanilla EqM on CIFAR-10 (reproducibility gate)
**Objective**: confirm our training/eval stack is sound by matching the EqM paper's own number.
- Our current CIFAR-10 80ep run gets FID ~497, vs. the paper's 3.36 (Appendix B.1, Table 8). This is a bug, not a scale issue.
- Train **vanilla EqM** on CIFAR-10 with the paper's recipe (use their Appendix B.1 config; likely a non-transformer backbone per the paper's text).
- **Exit criterion**: vanilla EqM on CIFAR-10, 50K-sample FID within ~0.3 of 3.36 (paper's own number).
- **If we cannot reproduce the paper's baseline**, nothing downstream is credible. Fix this before Stage B.
- This stage is a gate, not a result. CIFAR is NOT the paper's main benchmark (EqM is *worse* than standard Flow Matching there, 3.36 vs. 2.09). We reproduce it only to certify our stack.

### Stage B — Confirm at EqM-B/2, 80 epochs, ImageNet-256 class-conditional (first real result)
**Objective**: produce the paper's headline result — DG-ANM vs. vanilla EqM on the benchmark where EqM itself is strong.
- ImageNet-256 class-conditional is **the paper's primary benchmark** (EqM-XL/2 = 1.90 FID, B/2 ablations ~33–50 FID).
- Train DG-ANM vs. vanilla EqM at **EqM-B/2, 80 epochs, ImageNet-256**, matching the paper's training recipe.
- **3 seeds each**, 50K-sample FID.
- Compare: DG-ANM mean FID ± std vs. vanilla EqM mean FID ± std.
- **Exit criterion**: DG-ANM beats vanilla EqM by a margin > 2× the seed std.
- Compute: this is the single biggest ask. Plan for ≥1–2 weeks wall-clock. Start vanilla baseline in parallel with Stage A as soon as A.5 passes.
- **If compute is infeasible** for 3 seeds at B/2: consider smaller model (S/2) OR reframe paper as a "sample efficiency" contribution at lower epoch counts.

### Stage C — Scaling + second-dataset CIFAR (supporting evidence)
**Objective**: show the gain is not an artifact of one scale or one dataset.
- **Scaling plot**: DG-ANM vs. EqM at S/2, B/2 (and L/2 if compute allows) on ImageNet-256. Shows the gain is not scale-dependent noise.
- **CIFAR-10 secondary result**: DG-ANM vs. vanilla EqM on CIFAR-10 at paper-matching config, 3 seeds. Mirrors the paper's own Appendix B.1 placement (supporting, not headline). Our target: beat vanilla EqM's 3.36 at matched compute; aspirational stretch: approach or beat Flow Matching's 2.09.

### Stage D — Mechanism + ablations (paper polish)
**Objective**: produce the paper's analysis tables and figures.
- **Mechanism**: why does DG-ANM work? Candidate analyses: energy landscape visualizations around data, per-timestep loss curves, negative-sample diversity vs. training step, gradient-alignment between mined negatives and natural failure modes. One compelling figure beats three weak ones.
- **Ablations**: gamma, epsilon, mine_every, mining_steps — already most of what autoresearch has been doing; formalize as a table.
- **Baselines to compare**: vanilla EqM, DiT-XL/2, SiT-XL/2, and at least one EBM-style negative-mining baseline (e.g., CD, PCD, or a contemporary hard-negative method) at matched compute.

### Stage E — Writing and submission
- Target venue selection: NeurIPS (May deadline), ICML (Jan deadline), ICLR (Sep deadline). Pick based on when Stage C finishes.
- Threats to validity section: proxy-to-full-scale translation, seed variance, compute constraints.
- Supplementary: full autoresearch log (results.tsv), commit-level reproducibility.

## Decision rules

- **Do not scale up until proxy gains survive a 3-seed check.** Scaling a non-reproducible gain is wasted compute.
- **Every claim in the paper must be backed by ≥3 seeds at the claimed scale.** Single-seed results are for autoresearch, not for the paper.
- **If Stage A.5 fails (we cannot reproduce vanilla EqM on CIFAR)**, nothing downstream is credible. Fix the stack before any further experiments.
- **If Stage B fails (gain does not transfer to B/2 + 80ep + ImageNet-256)**, the method as currently formulated is not publishable. Options: rethink the mechanism, reformulate as a theoretical/analysis paper, or pivot.
- **Do not chase small FID gains at proxy scale.** Beyond Stage A convergence, additional proxy sweeps are diminishing returns — the compute should go to Stage B/C.

## Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Cannot reproduce vanilla EqM's 3.36 on CIFAR-10 (stack bug) | HIGH | Stage A.5 gate; fix before anything else |
| Proxy-to-full-scale gap: gain at 2ep IN-100 does not transfer to 80ep B/2 | HIGH | Stage B gate before any larger investment |
| Seed variance swamps gain | HIGH | 3-seed check at end of Stage A |
| Cluster compute for Stage B ImageNet-256 B/2 80ep 3-seed is infeasible | HIGH | Start vanilla baseline early; plan fallbacks (S/2, fewer epochs, reframed contribution) |
| No mechanistic story emerges | MED | Stage D analyses; one strong figure is enough |
| Scooped by concurrent work | LOW-MED | EqM is recent (Oct 2025); negative-mining for generative models is crowded but specific DG-ANM framing is defensible |

## Immediate next actions (queue after round 4)

0. **Stage A.5 (NEW, start in parallel with round 4)**: reproduce vanilla EqM on CIFAR-10 at paper-matching FID 3.36. Audit our CIFAR config against the EqM repo's Appendix B.1 setup; identify why our vanilla got FID 497.
1. Tournament for round 4's 9 candidates; identify winners per dimension.
2. Round 5: combination of top dimensions (tests additivity).
3. **Seed-variance check**: best config × 3 seeds on the proxy. This is the Stage A exit gate.
4. Design Stage B experiment: EqM-B/2 + CIFAR-10 + 80ep + 3 seeds. Estimate wall-clock & submit.
