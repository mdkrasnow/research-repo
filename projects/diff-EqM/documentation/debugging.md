# Debugging — DG-ANM for EqM

## Active Issues

### R2 v03/v06 OOM on gpu_test 20G card (RESOLVED 2026-04-27)
- **Symptom**: jobs 8898137 (v03) and 8898147 (v06) FAILED at 2-3 min wall-time on `gpu_test` partition. `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of 19.62 GiB of which 32.88 MiB is free`.
- **Root cause**: mining variants need 3-4x the activation memory of vanilla (PGA does multiple forward+backward per step). bs=128 fits in 40G A100 but not in the 20G test card. v00_vanilla ran fine on `gpu` (40G) at the same bs because it has no mining.
- **Fix**: resubmit mining variants only to `gpu` partition. v02 (8898150) and v04 (8898152) already there and unaffected.
- **Resubmits**: v03→8941251, v06→8941252 on `gpu`.
- **Future rule**: never submit mining variants (v01-v04, v06) to `gpu_test`. Add to variant_pilot.sbatch header comment, or split sbatches per-partition. Non-mining variants (v00, v05) can still use gpu_test for queue speed.



### CIFAR-10 vanilla FID 497 vs paper's 3.36 (ROOT CAUSE IDENTIFIED, 2026-04-18)
- **Symptom**: vanilla EqM CIFAR 80ep → FID 497.55 (job 3053350); DG-ANM → FID 497.04 (job 3104472). Both stuck at pathological floor, no method separation.
- **Root cause**: wrong architecture. Paper Appendix B.1 uses U-Net on CIFAR from the Flow Matching (Lipman et al., 2024) codebase. We used EqM-S/2 transformer (patch=4) because our vendored `eqm-upstream/` only contains the SiT-style transformers from the ImageNet branch — no U-Net.
- **Full audit + fix plan**: `documentation/stage-a5-audit.md`
- **Action**: do not run any more CIFAR experiments on the current stack. Decide fix path (port FM UNet / DDPM UNet smoke / defer CIFAR) before more compute.
- **Good news**: Stage B (ImageNet-256) is NOT affected — our upstream's transformer models are what the paper used on ImageNet. Stage B can proceed in parallel with the CIFAR fix.

## Resolved
(none yet)

---
## Common Failure Modes (Preemptive Checklist)
- [ ] Import errors (timm, diffusers not installed on cluster)
- [ ] OOM with geometry estimation (P_T/P_N are BxDxD — may need batching for large D)
- [ ] Mining gradient explosion (adversarial search can produce NaN)
- [ ] EqM field returns NaN at large perturbations
- [ ] Checkpoint path mismatch between train and eval configs
- [ ] CIFAR-10 download fails on cluster (use pre-cached data_dir)

---

## v02 IN-1K 80ep cancellations (2026-05-05 + 2026-05-06) — POSTMORTEM PENDING

**Jobs**: 10198798 (elapsed 04:54, cancelled 2026-05-05), 10387316 (elapsed 09:57, cancelled 2026-05-06).
**Status as of 2026-05-19**: SSH to cluster returning "Permission denied (keyboard-interactive)" — cannot fetch sacct details or logs.
**Required**: re-authenticate `scripts/cluster/ssh.sh` (likely needs MFA or session refresh). Then:
```bash
bash scripts/cluster/ssh.sh "sacct -j 10198798,10387316 --format=JobID,JobName,State,ExitCode,ReqMem,MaxRSS,Elapsed,DerivedExitCode,Reason -P"
bash scripts/cluster/remote_fetch.sh diff-EqM
ls projects/diff-EqM/slurm/logs/stageb-v02-in1k-80ep_*.{out,err}
```
**Implication for Phase 1a**: blocking only if cancellation cause was OOM (would constrain CAFM-EqM port memory budget). Given CAFM uses 80G A100s and our vanilla baseline trained fine on `seas_gpu`, working assumption is OOM was NOT the cause and Phase 1a can proceed on `seas_gpu`. Update postmortem when SSH restored.

---

## POSTMORTEM UPDATE (2026-05-19): v02 IN-1K cancellations RESOLVED

Replaces the "POSTMORTEM PENDING" entry below.

**Root cause**: **mechanism saturation, USER-CANCELLED** (not OOM, not timeout, not crash).

**Evidence from logs** (`slurm/logs/stageb-v02-in1k-80ep_{10198798,10387316}.out`):
- Diagnostics across epochs 9-11: `v02(pos=0.005, neg=0.999, |v_neg|=220)` — completely flat.
- **`neg=0.999`** = PGA-found cosine similarity ≈ 1.0 (max anti-alignment achieved); objective saturated.
- **`pos=0.005`** = clean cosine ≈ 0; model perfectly aligns with target.
- **`|v_neg|=220`** constant — EqM-B/2 output magnitude huge; confirms v10-proposal-line-14: "v02 cosine saturated on EqM-B/2 because |v|≈220 >> |J·δ|, cos≈1 for any small δ, PGA gradient vanishes."
- `Aux: 0.20` constant. v02 auxiliary loss doing NOTHING by epoch 9.

**SACCT confirms**:
- ExitCode 0:0 (clean), CANCELLED by user UID.
- MaxRSS 19-20 GB / 256 GB ReqMem → no OOM.
- Throughput: 3.25 sps fast / 1.62 sps slow. 80ep ETA: 27h fast, 54h slow. Attempt 2 would have exceeded 48h walltime.

**Implications**:
1. CAFM-EqM throughput: with N=16 disc/gen, expect ~24-36h for 10ep post-training. Set sbatch time=48h.
2. OOM not a concern for CAFM port (same memory profile).
3. v10's **L2-regression objective avoids v02's saturation**: regression has unbounded gradient even at perfect alignment.
4. Confirms Branch B framing: v02 saturation is mechanism-level; v10 + CAFM mechanistically immune.

**No remediation needed**. v02 path dead. v10 + CAFM is correct continuation.

---

## 2026-05-23: CAFM-EqM Phase 1b CATASTROPHIC FID 341.25

Full postmortem: `postmortem-cafm-eqm-2026-05-23.md`.

**Headline**: CAFM port to EqM, after clean training (gen loss 4.0→1.7, dis loss 0.9-2.1 oscillating), produced 50K-sample FID 341.25 vs vanilla 31.41 (10× worse). Ckpt_5000 diagnostic FID 369.64 confirmed instant (not cumulative) collapse.

**Root cause**: Vanilla EqM was trained by pure regression; freshly-initialized discriminator trivially discriminates "vanilla-EqM output ≠ training-data velocity" and pushes generator off the EqM target manifold. `c(γ)→0` near data manifold amplifies asymmetry — any adversarial gradient at high γ preferred over vanishing regression target.

**Process failures**:
1. Smoke validated loss-finiteness + exit 0, NOT sample quality.
2. Misread one-sided monotonic dis-loss decrease as healthy convergence.
3. Design doc reasoned about JVP geometry, not about discriminator-shortcut against non-adversarially-trained generators.

**Fixes landed in CLAUDE.md**:
- Mandatory smoke-time sample-quality probe for new losses on gen models.
- Discriminator-loss oscillation check (one-sided decrease = STOP).
- Branch B-Both retired; v10-only pivot confirmed.

---

## 2026-05-24/25: Home-quota deadlock killed v10 train 15290932

**Headline**: Main job 15290932 wedged at step 74,200 of 380K. Symptom: SLURM .out write blocked → Python stdout buffer filled → training process froze (no exit, no crash, no error).

**Root cause**: Home quota 100GB hard limit hit. ckpt-every=5000 + 30min rsync sync from /tmp persisted ~70 ckpts × 2GB = 140GB total over ~13h. Combined with prior CAFM dirs (66GB) + ref-stats files (114MB) = >100GB triggered quota wall. Process froze on first write attempt that exceeded quota.

**Recovery**: cancelled 15290932; deleted ~66GB of CAFM dead-tree dirs (with user approval); resumed v10 from ckpt_65000.pt as 15638767. Completed clean in 1d11h32m.

**Mitigation deployed**: `slurm/jobs/prune_v10_ckpts.sbatch` — periodic shared-partition pruner watching MAIN_JOB_ID, keeps anchors {5000, 65000} + latest 2, prunes every 10min, exits on main-job completion. Used during 15638767 successfully (peak quota 50G/95G, never wedged).

**Outstanding bug**: main job's sync_checkpoints rsync re-uploads ckpts from /tmp source every 30min, resurrecting pruner-deleted files. Requires 1-2 manual prunes per train. Fix: patch sync_checkpoints to thin /tmp side BEFORE rsync. Deferred to next sbatch revision (applies only to future runs, not in-flight).

---

## 2026-05-27: FID eval failed on .JPEG case-sensitivity (16327377)

**Headline**: Phase 1 gate FID resubmission 16327377 (gpu_requeue) failed at 2m27s. Error: "Flat reference: 0 images" → "ERROR: No reference images found in /n/holylabs/.../imagenet/train".

**Root cause**: `imagenet_fid.sbatch` line 91 used `find -name "*.jpg" -o -name "*.jpeg" -o -name "*.png"` (case-sensitive). IN-1K data uses `.JPEG` (uppercase). Find matched 0 files → 0 symlinks → 0 ref → exit.

**Why missed earlier**: `imagenet_fid.sbatch` was generic (IN-100 was lowercase .jpg). Trusted IN-1K baseline FID 31.41 (12590806) used a different sbatch: `imagenet1k_fid_eval.sbatch` which has both `.JPEG` and `.jpg` patterns + 50K shuf subsample (handles 1.3M file count). Pre-staged helper `submit_v10_phase1_fid.sh` pointed to the wrong sbatch.

**Fix**: commit `fef80d7` — switched helper to `imagenet1k_fid_eval.sbatch`. Resubmitted as 16328965 → COMPLETED FID 29.01 in 2h20m.

**Lesson**: Pre-staged helpers must match the trusted-baseline path verbatim. Audit any sbatch interaction with NFS-mounted datasets for case-sensitivity + extension assumptions.

---

## 2026-05-27/28: gpu_requeue MIG roulette + preempt

Two events on gpu_requeue:

**Event 1 — 16371645 K=3 B/2 smoke FAILED 5min**: Node `holygpu7c0705` was MIG-sliced. NCCL `Duplicate GPU detected: rank 1 and rank 0 both on CUDA device 6000`. Same class as gpu_test MIG bug. Resubmit to seas_gpu as 16374706 → PASS (timeout but loss curve healthy).

**Event 2 — 16376978 vanilla L/2 PREEMPTED 3min**: Got non-MIG node (DDP init OK) but preempted before 5K-step ckpt. No state saved → had to start from scratch. Resubmit to seas_gpu as 16406011 → RUNNING successfully.

**Decision**: gpu_requeue is too unreliable for long multi-GPU DDP trains. Use only for: (a) single-GPU jobs (eval, FID); (b) jobs that tolerate restart from scratch within minutes; (c) when seas_gpu/gpu queues catastrophic.

CLAUDE.md "gpu_requeue MIG roulette" + "Auto-pruner standing infrastructure" sections added.

---

## 2026-05-28: Smoke ckpt accumulation + rsync temp file leak (quota hit 95G)

**Headline**: Quota hit 95G/100G with 6 jobs running. Investigation found 21G in dead XL/2 smoke dir = all hidden rsync temp files `.0015000.pt.yJJYZ6` etc, not actual `*.pt` ckpts.

**Root cause**: When SLURM signals smoke jobs at wall-time (TIMEOUT), in-flight rsync from /tmp to home leaves partial-transfer temp files. Pruner glob `*.pt` doesn't match hidden temp pattern. Smoke dir keeps growing across multiple aborted rsync calls.

**Triple fix**:
1. Manual mass-prune: `find $RESULTS -name '.*.pt.*' -delete` (freed 27G immediately).
2. Pruner patched (commit `a75fd5e`) to include same find-delete every 5min cycle.
3. CLAUDE.md "Rsync temp-file failure mode" section added for durable cluster discipline.

**Other contributor**: 4 smokes ran with `--ckpt-every 5000` (default in scaling sbatch) → each saved 3-15 ckpts × 1-7GB = up to 21G per smoke. Smokes have no use for ckpts. Pruner's smoke-dir loop deletes all `*.pt` in smoke dirs (and now temps too).

## Exp1 sampler-robustness: torchrun master-port collision (2026-06-01)
- **Symptom**: intermittent cell failures in the full sweep (job 17828606), ~3% (1/33 early). job.log: `RuntimeError: ... EADDRINUSE ... port: 29629` from torch.distributed.elastic static_tcp_rendezvous. Failed cell recorded with `generate_rc=1`, fid empty, n=0; driver continues (no crash).
- **Root cause**: `generate()` set `master_port = 29500 + hash((sampler,nfe,sm))%1000`. Across 80 sequential torchrun launches, ports collide (birthday) and/or reuse a port still in TIME_WAIT from the previous cell.
- **Fix (commit 8aa5308)**: OS-assigned free port via `socket.bind(("",0))` + 3x retry with a fresh port on nonzero rc. Applies to future runs (50k resume / reruns). The in-flight 17828606 keeps old code -> expect ~2-4 holes/80; analysis dropna's them, result still interpretable.
