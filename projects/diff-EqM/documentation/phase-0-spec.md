# Phase 0 Spec — diff-EqM Summer 2026 (Branch B-Both)

**Window**: 2026-05-19 → 2026-06-09 (3 weeks)
**Goal**: Lock literature positioning + CAFM-on-EqM port design + reproduce Lin's CAFM smoke before any IN-1K spend.
**Exit gate**: SYNTHESIS.md written + CAFM smoke on SiT-B/2 passes + CAFM-to-EqM port design doc reviewed.

Compute budget: **≤ 50 GPU-h**.

This is an **agentic phase**. Pre-registered gates control transitions.

---

## Task index

| ID | Title | Compute | Status | Output |
|---|---|---|---|---|
| **0.A.1** | Build literature read slate | 0 | TODO | `literature/read-slate.md` |
| **0.A.2** | Read + per-paper notes (~18 papers, parallel) | 0 | TODO | `literature/<paper>.md` × N |
| **0.A.3** | Synthesis memo | 0 | TODO | `literature/SYNTHESIS.md` |
| **0.B** | Apply synthesis edits to CLAUDE.md, summer-plan, this spec | 0 | TODO | committed edits |
| **0.C.1** | Clone Lin AFM/CAFM repo; inspect | 0 | TODO | `documentation/cafm-repo-inspection.md` |
| **0.C.2** | Reproduce CAFM-on-SiT smoke (small scale) | ~40 GPU-h | TODO | reproduction note + `active_runs` entry |
| **0.1** | v02 IN-1K cancellation postmortem | 0 | TODO | `debugging.md` entry |
| **0.2** | CAFM-to-EqM port design doc | 0 | TODO | `documentation/cafm-eqm-port-design.md` |
| **0.3** | v10 + CAFM combined loss design doc | 0 | TODO | `documentation/v10-cafm-combination-design.md` |
| **0.4** | v11 / v12 fallback sketches | 0 | TODO | `documentation/v11_fallback_sketch.md` |
| **0.5** | Arxiv weekly sweep log started | 0 | TODO | `literature/arxiv-weekly-sweep.md` |

**Execution order**:
```
Week 1: 0.A.1 → 0.A.2 (parallel reads, ~18 papers) → 0.A.3 → 0.B
Week 2: 0.C.1 + 0.1 + 0.5 (parallel) → 0.C.2 (smoke) → 0.2 (port design)
Week 3: 0.3 (combination design) → 0.4 (fallbacks) → exit gate review
```

---

## Task 0.A.1 — Build literature read slate

### Output: `documentation/literature/read-slate.md`

Six buckets (added Bucket 6 for Lin et al. + ACT after pivot). ~18–22 papers.

#### Bucket 1 — Direct competitors / scoop check (HIGH, full-read)
1. Lin et al. 2026 AFM (arxiv 2511.22475) — ICML 2026.
2. Lin et al. 2026 CAFM (arxiv 2604.11521).
3. Wu et al. 2025 DAT (arxiv 2510.13872).
4. Geng et al. 2024 AEBM-Diff (arxiv 2403.01666).
5. Wang et al. 2025 EqM (arxiv 2510.02300).
6. Kong et al. 2024 ACT (arxiv 2311.14097).

#### Bucket 2 — Adversarial training for generative models (MEDIUM, abstract+method)
7. Madry 2017 PGD (arxiv 1706.06083).
8. "What is Adversarial Training for Diffusion Models?" (arxiv 2505.21742).
9. VeCoR (arxiv 2511.18942).
10. Luo 2023 DCD (arxiv 2307.01668).
11. Rob-GAN (arxiv 1807.10454) — old PGD+GAN combination.

#### Bucket 3 — EBM family + flow/regression connections (MEDIUM)
12. Du & Mordatch 2019 EBM-CD (arxiv 1903.08689).
13. JEM (Grathwohl 2020) (arxiv 1912.03263).
14. Lipman 2023 Flow Matching (arxiv 2210.02747).
15. Song 2023 Consistency Models (arxiv 2303.01469).

#### Bucket 4 — SiT / DiT background (essential since Phase 5 uses SiT)
16. SiT paper (find arxiv) — full-read for Phase 5.
17. DiT (Peebles 2022, arxiv 2212.09748) — abstract+method.

#### Bucket 5 — Hard-example / mining priors (LOW, skim)
18. OHEM (arxiv 1604.03540).
19. Focal Loss (arxiv 1708.02002).

#### Bucket 6 — Already-on-the-radar Lin follow-ups + watch list
- AAPT (arxiv 2506.09350) — autoregressive adversarial post-training for video.
- FlowEqProp (arxiv 2604.08150) — confirmed unrelated to EqM-Wang.
- FAIL (arxiv 2602.12155) — flow matching adversarial imitation.
- FMVP (arxiv 2601.02228) — adversarial purification with flow matching (not generation).

### Definition of done
- Slate written with arxiv ID, bucket, depth tag, threat-level pre-assessment per entry.

---

## Task 0.A.2 — Read + take per-paper notes

Use template in `literature/README.md`. Parallel reads via WebFetch on different URLs.

### Quality bar
- ≥ 200 words / ≤ 800 words per note.
- "Differentiation from us" = single specific sentence.
- For Bucket 1 entries (HIGH): also extract loss equations into the note.

### Definition of done
- One file per paper committed.

---

## Task 0.A.3 — Synthesis memo

### Output: `literature/SYNTHESIS.md`

Required sections:
1. Landscape map (1 paragraph).
2. Updated positioning ≤ 50 words.
3. Mechanism design adjustments (concrete HP / loss-formulation changes).
4. New / sharpened risks (especially follow-up scoop potential from Lin lab).
5. Recommended edits: CLAUDE.md, summer-plan, this spec, related-work memo (path + section + before/after).
6. Citation map (every paper → section in eventual paper).

### Definition of done
- ≤ 4 pages.
- §5 has concrete edits, not vibes.

---

## Task 0.B — Apply lit-review findings

Execute every edit in `SYNTHESIS.md §5`. One commit per file.

### Definition of done
- All edits applied; commit messages: `Apply lit-review finding: <summary>`.
- `pipeline.json:events` entry: `LITERATURE_REVIEW_APPLIED`.

---

## Task 0.C.1 — Clone Lin repo + inspect

```bash
cd projects/diff-EqM/
mkdir -p external
git clone https://github.com/ByteDance-Seed/Adversarial-Flow-Models.git external/Adversarial-Flow-Models
cd external/Adversarial-Flow-Models
ls -la
```

### Output: `documentation/cafm-repo-inspection.md`

Document:
- Repo structure (top-level dirs/files).
- Training scripts inventory: AFM vs CAFM, SiT vs JiT vs Z-Image.
- Discriminator architecture file location.
- Dependencies (requirements.txt — Python version, PyTorch version, CUDA).
- Pretrained checkpoint URLs (HuggingFace).
- Smoke-test entrypoint (the smallest reproducible training run).
- Estimated GPU + time for SiT-B/2 CAFM smoke.

### Definition of done
- Repo cloned to `projects/diff-EqM/external/Adversarial-Flow-Models/`.
- Inspection doc committed.
- `.gitignore` updated to exclude `external/` from our repo (use as read-only ref).

---

## Task 0.C.2 — Reproduce CAFM-on-SiT smoke

### Why
Before porting CAFM to EqM, **must verify we can run CAFM ourselves** on their stock setup. This catches: dependency hell, pretrained-ckpt mismatch, dataset wiring, basic sanity.

### Smoke run
- SiT-B/2 (smallest available) IN-256 latent space.
- CAFM 1 epoch post-training (vs the 10 ep recipe).
- Use Lin's CAFM training script from cloned repo.
- Single GPU OK (no DDP).
- Goal: training loop runs, discriminator loss visible, FID computed at end (rough is fine).

### `active_runs` entry (submission commit)
```json
{
  "run_id": "phase0_cafm_sit_b2_smoke",
  "job_id": "<TBD>",
  "partition": "gpu",
  "status": "pending",
  "description": "Phase 0.C.2: reproduce CAFM-on-SiT-B/2 IN-256 1-epoch smoke (verify Lin recipe before EqM port).",
  "submitted_at": "<ISO>",
  "git_sha": "<SHA>",
  "sbatch_path": "projects/diff-EqM/slurm/jobs/phase0_cafm_smoke.sbatch",
  "expected_runtime": "~6h",
  "phase": "0",
  "gate": "CAFM smoke pass = training loop runs, no NaN, FID reasonable"
}
```

### Gate (PASS = ALL)
- A. End-to-end run completes 1 epoch without crash.
- B. Generator loss + discriminator loss both finite throughout.
- C. Discriminator loss does NOT collapse to ~0 (= sign of generator dominance).
- D. Final FID (rough single-GPU estimate) within 2× of Lin's published 1-epoch number (extrapolated; their reporting is at 10 ep).

### Fail handling
- Crash or NaN → diagnose deps/data; fix; retry. Hard cap 3 attempts.
- Loss collapse → reduce learning rate (1e-5 → 5e-6) per Lin's ablation.
- FID way off → likely dataset preprocessing mismatch; document and seek next round of debug.

### Definition of done
- Smoke run completes.
- Reproduction notes recorded in `documentation/cafm-repo-inspection.md`.
- Lessons-learned section about Lin's recipe quirks.

---

## Task 0.1 — v02 IN-1K cancellation postmortem

Fetch logs 10198798 + 10387316. Categorize cause (OOM | timeout | user | diagnostic). Document in `debugging.md`. Apply partition + cost mitigation to Phase 2 sbatch settings.

### Definition of done
Cancellation cause documented.

---

## Task 0.2 — CAFM-to-EqM port design doc

### Output: `documentation/cafm-eqm-port-design.md`

Required sections:
1. **What CAFM does on SiT**: discriminator D(x_t, t) on JVP of velocity field, least-squares GAN loss, N=16 discriminator updates per generator update, 10 ep post-training of pretrained FM model.
2. **EqM differences**: EqM has c(γ) decay, hardcoded interp=0.8 λ=4 (transport.py:122-126). Time conditioning may differ from SiT (verify in EqM code).
3. **Compatibility analysis**: each CAFM component → does it transfer to EqM unchanged, need adapt, or break entirely?
   - JVP discriminator on velocity → likely transfers (EqM also has velocity prediction).
   - Time conditioning (t ∈ [0,1] vs γ ∈ [eps, 1-eps] truncated) → adapt.
   - OT regularization term (λ_ot · ||G(x_t,t)||²) → does it conflict with EqM's c(γ)-weighted target?
   - Post-training schedule (10 ep on top of pretrained) → use vanilla EqM 80ep (FID 31.41) as start.
4. **Required code changes**:
   - New file: `experiments/cafm_eqm/discriminator.py` (DiT-style with [CLS], JVP head).
   - New file: `experiments/cafm_eqm/cafm_train.py` (post-training loop).
   - New file: `experiments/cafm_eqm/losses.py` (least-squares GAN + OT + centering).
   - New file: `configs/cafm/eqm_b2_in1k_cafm.json`.
   - New file: `slurm/jobs/cafm_eqm_b2_in1k.sbatch`.
5. **Diagnostics to log**: gen loss, disc loss, OT loss, disc gradient norm, generator gradient norm, FID at every 1 ep.
6. **Open questions**: 3-5 things we don't know yet.

### Definition of done
- Doc written.
- All design questions answered or marked OPEN.

---

## Task 0.3 — v10 + CAFM combined loss design

### Output: `documentation/v10-cafm-combination-design.md`

Required sections:
1. **Combined loss**:
   ```
   L_G_total = L_CAFM_gen(G, D) + λ_v10 · L_base(x_t + δ*)
   δ* = argmax_{||δ||≤ε} L_base(x_t + δ)
   ```
2. **Where v10 fires in the CAFM loop**: every Nth generator step (decoupled from discriminator update schedule).
3. **Hyperparameters**:
   - λ_v10: start at 0.1 (per v10 proposal); allow sweep.
   - K: 3 PGA steps.
   - ε_radius: 0.3.
   - mine_every: every gen update OR every 4 gen updates (compute trade-off).
4. **Variant proposal template** (per CLAUDE.md research rules) filled.
5. **Mechanism hypothesis**: discriminator catches global mismatch; v10 catches local regression failure. Compose because attacks different failure modes.
6. **Diagnostics**: L_base, L_hard, L_disc, ||δ||, ratio L_hard/L_base.
7. **Expected pass signature**: combined FID < CAFM-alone FID by ≥0.3 at seed 0.
8. **Kill conditions**: λ_v10 sweep at {0.03, 0.1, 0.3, 1.0} all worse than CAFM-alone → kill combination.

### Definition of done
- Doc written.
- Variant template filled.

---

## Task 0.4 — v11 / v12 fallback sketches

### Output: `documentation/v11_fallback_sketch.md`

If v10+CAFM compounding fails (Phase 2 gate), what next?

- **v11** = gamma-weighted hard mining (auxiliary leg sampled from informative γ region).
- **v12** = stale-EMA-PGA (mine against EMA model, train current model — two-time-scale AT).
- **v13** = velocity-correlation hard mining (direction-sensitive regression objective).

Each sketch: mechanism check per CLAUDE.md template.

### Definition of done
File committed.

---

## Task 0.5 — Arxiv weekly sweep log

### Output: `literature/arxiv-weekly-sweep.md`

Weekly Monday entries through Oct 1. Keywords + Lin-lab author watch.

### Definition of done
File created with week-1 entry.

---

## Phase 0 exit checklist

- [ ] **0.A.1** Read slate written.
- [ ] **0.A.2** ~18 per-paper notes committed.
- [ ] **0.A.3** SYNTHESIS.md written.
- [ ] **0.B** Synthesis edits applied to all artifacts.
- [ ] **0.C.1** Lin repo cloned + inspected.
- [ ] **0.C.2** CAFM-on-SiT smoke PASS.
- [ ] **0.1** v02 postmortem in `debugging.md`.
- [ ] **0.2** CAFM-to-EqM port design doc.
- [ ] **0.3** v10+CAFM combination design doc.
- [ ] **0.4** Fallbacks sketched.
- [ ] **0.5** Arxiv sweep log started.
- [ ] `pipeline.json:phase` → `PHASE-1A-CAFM-PORT` (if 0.C.2 PASS) or `PHASE-0-DEBUG` (if FAIL).
- [ ] PI update draft in `pi-updates.md` for Phase 0 exit.

---

## Outputs traceability

| Task | Artifact | Required by |
|---|---|---|
| 0.A.* | `literature/*.md`, `SYNTHESIS.md` | 0.B + every later phase |
| 0.B | edits across artifacts | 0.2, 0.3, 0.4 |
| 0.C.1 | `cafm-repo-inspection.md` | 0.C.2, 0.2 |
| 0.C.2 | smoke FID + reproduction notes | 0.2 (confirms Lin recipe works) |
| 0.1 | `debugging.md` v02-cancel section | Phase 1a partition + budget |
| 0.2 | `cafm-eqm-port-design.md` | Phase 1a implementation |
| 0.3 | `v10-cafm-combination-design.md` | Phase 2 implementation |
| 0.4 | `v11_fallback_sketch.md` | Phase 2 contingency |
| 0.5 | `arxiv-weekly-sweep.md` | weekly through Oct 1 |

---

## Phase 0 closeout (fill on completion)

```
- Started: 2026-05-19
- Lit-review completed: <date>
- Synthesis-driven changes to plan: <summary>
- CAFM-on-SiT smoke outcome: <PASS | FAIL>
- CAFM-to-EqM port design ready: <yes/no>
- v10+CAFM combination design ready: <yes/no>
- v02 cancellation cause: <category>
- Decision: <Phase 1a launch | port-design redo | B-SiT fallback>
- Commits: <hashes>
- PI update drafted: <yes/no>
```
