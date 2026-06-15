# Overnight summary — 2026-06-15 (AFK run)

## Headline
Cluster was DOWN all night (MGHPCC annual power downtime Jun 15–18, **not** VPN — login
nodes physically off, back ~Jun 18 5 PM). So maze-GPU + online-50k **could not run** — they
are queued to auto-fire on return. All overnight compute pivoted to local CPU, which
produced two real results.

## What ran (local CPU) — results
1. **Maze planning metacognition — CONFIRMED (multi-seed).** 5 seeds × 2 OOD tiers:
   probe-restart > random at equal NFE, pooled **+0.174±0.034 valid-rate, 10/10 positive**,
   AUROC 0.67–0.75. (`maze_eqm/STEP3_RESULTS.md`, `runs/maze_sweep/`.)
2. **MNIST inpainting metacognition — WEAK/NULL (honest, with an insight).** Real MNIST +
   RePaint masks + classifier oracle. Probe ~chance (AUROC 0.60), gap **+0.015±0.011** —
   far below maze. **Finding:** the method keys on descent *instability*, so it rescues
   collapse/broken-structure failures (generation, planning) but NOT *confidently-wrong*
   failures (inpainting descends cleanly to a plausible-wrong digit). A scope boundary that
   sharpens the paper. (`mnist_eqm/MNIST_RESULTS.md`.)

## What I built (no compute / design / writing)
- **Cross-task synthesis** (`separability_diagnostic/SYNTHESIS_METACOGNITION.md`) — the
  6-link evidence chain (energy dead → shape probe → best-of-R 50k + online → maze → MNIST)
  + the scope boundary + paper shape.
- **Scoop check** — found + differentiated concurrent **CFG-Rejection** (2505.23343): their
  signal needs CFG (undefined for our cfg=1.0 EqM), post-hoc/images-only/no de-confound.
- **RePaint-on-IN-1K-EqM spec** (`REPAINT_EQM_SPEC.md`) — real image inpainting, latent-space
  design, queued for June 18. (Predicted weak, same reason as MNIST.)

## Queued for June 18 (cluster return)
Run `scripts/cluster/fire_overnight.sh` (idempotent) once `ssh` works. Submits:
- maze GPU scale-up × 3 seeds (wider EqM, native c7 training) — tightens the maze AUROC.
- online 50k adaptive × 3 seeds — promotes the online claim to 50k/3-seed.
Then optionally build + run RePaint-on-IN-1K-EqM per the spec.

## State
All pushed to main (latest: MNIST result + scope boundary). `pipeline.json` reconciled;
cluster jobs marked blocked-until-Jun-18. Nothing is running now — clean stop.

## One-line status
Image FID (50k 3-seed) + online (15k) + maze planning (5-seed) = decision-grade. MNIST
inpainting = honest null that defines the method's scope. Image-scale inpainting + maze
GPU scale-up wait on the cluster (Jun 18).
