---
name: diff_eqm_exp2_field_robustness
description: Exp 2 off-trajectory field robustness result — ANM field-robustness mechanism confirmed at IN-1K B/2 (small but significant)
metadata: 
  node_type: memory
  type: project
  originSessionId: d76ef35d-d4fb-4e29-b908-cd58b2feae90
---

Exp 2 (off-trajectory field robustness diagnostic, 2026-06-01): does v10/ANM make the
EqM field more accurate off the vanilla trajectory, not just better FID? Run on 5120
IN-1K val latents, frozen vanilla (FID 31.41) vs v10 final.pt (FID 29.01) EqM-B/2,
paired bootstrap CI over sample_id. Jobs 17788287 (random) + 17788329 (sampler), both exit 0.

**Result: mechanism CONFIRMED but small.** ANM lower MSE / higher cosine at every
radius, all SIG. Gap widens off-trajectory (random dMSE −0.0164 r0 → −0.0277 r0.1).
**Largest gap at the real v10-mined perturbation: dMSE −0.0368 [CI −0.0396,−0.0340],
dCos +0.00109** — ANM most accurate exactly where its mining trains. BUT effect tiny:
dMSE ~−0.02 to −0.037 on vanMSE ~10.3 (0.2–0.4% rel). Consistent with the B/2 NULL
capability eval [[diff_eqm_phase_0_v10_pass]] — v10 benefit at B/2 real but modest;
may amplify at XL/2 (untested). Not pure r0 parity (small −0.0164 on-path gap too).

**DOSE-RESPONSE (2026-06-01, jobs 17829106+17829107):** re-ran Exp 2 on lambda=0.3 ckpt
(FID 27.09 vs lambda=0.1 FID 29.01). lambda=0.3 field gap ~2x larger at EVERY radius +
perturb type: random r0.1 −0.0607 (vs −0.0277), mined −0.0565 (vs −0.0368), drift −0.0397
(vs −0.0194). Clean monotonic dose-response: more mining -> better FID AND more field
robustness. At lambda=0.3 random-r0.1 gap exceeds mined gap (robustness spreads beyond mined
region as mining strengthens — not just memorizing mined δ). Strong paper story.

Diagnostic: `projects/diff-EqM/experiments/diagnostics/offtraj_field_robustness.py`.
Results doc: `documentation/exp2-offtraj-field-robustness-results.md`.

**Reusable cluster gotchas learned:**
- SLURM `--export=KEY=a,b,c` SPLITS on commas → comma-list args (T_VALUES, RADII)
  silently truncate. Pass them as shell env BEFORE sbatch with `--export=ALL`.
- diagnostic sbatch `--output=slurm/logs/...` is relative to submit cwd = repo root,
  so logs land in `/n/home03/mkrasnow/research-repo/slurm/logs/`, NOT projects/diff-EqM/slurm/logs.
- EqM.forward calls `x0.requires_grad_(True)` → use torch.no_grad() NOT inference_mode().
- Authoritative EqM target = transport.py `(x1−x0)·get_ct(t)` (x1=data,x0=noise),
  matches train_imagenet `_v10_pgd_hard_example_step`. The CAFM `eqm_target.py` uses
  OPPOSITE label convention (retired branch) — never mix.
