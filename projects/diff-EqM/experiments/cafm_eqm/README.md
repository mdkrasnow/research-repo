# CAFM → EqM Port (Phase 1a)

Adapts Lin et al. CAFM (`external/Adversarial-Flow-Models/`, MIT) to post-train
our trusted vanilla EqM-B/2 80ep checkpoint (FID 31.41).

## Design doc
`projects/diff-EqM/documentation/cafm-eqm-port-design.md`

## Modules

| File | Purpose |
|---|---|
| `eqm_target.py` | c(γ) truncated decay + target computation matching `transport.py`. Single source of truth. |
| `generator_wrapper.py` | Wraps EqM model with CAFM-compatible (x, y, t) interface. Internally fixes t=0. |
| `v10_mining.py` | PGD hard-example mining on the EqM base loss. Used standalone + in v10+CAFM. |

## TODO Phase 1a (next sessions)

1. **`discriminator_gamma.py`** — port Lin's `models/cafm/sit/discriminator.py` with γ-conditioning instead of t-conditioning. Reuse `models.cafm.jvp.discriminator.DiscriminatorJVP` from the cloned external repo via path-based import.
2. **`train_cafm_eqm.py`** — adapt Lin's `train_continuous_adversarial_flow_imagenet.py` with EqM target substitution (multiply CAFM's `velocity_real = ε − x` by c(γ) so both sides match).
3. **`train_v10_cafm_eqm.py`** — extends `train_cafm_eqm.py` with the v10 mining step (Phase 2).
4. **Config**: `projects/diff-EqM/configs/cafm/eqm_b2_in256_cafm.yaml` based on Lin's `train_cafm_sit.yaml` template.
5. **Sbatch**: `projects/diff-EqM/slurm/jobs/cafm_eqm_b2_in256.sbatch`.

## Smoke-test plan

Once `train_cafm_eqm.py` exists:
1. Local 100-step CPU smoke — shape check, loss flow.
2. Cluster 1-epoch CAFM-EqM smoke on B/2 ckpt.
3. Cluster 10-epoch full CAFM-EqM post-training, seed 0.

## Key port adaptations vs CAFM-on-SiT

- Generator takes (x, y, γ) where γ replaces t but is NOT passed to the EqM model (model receives t=0).
- Target = (ε − x) · c(γ) per `eqm_target.eqm_target()`. CAFM's `velocity_real` is replaced by this.
- Discriminator becomes γ-conditional via existing t-embedding path.
- N=16, λ_cp=0.001, λ_ot=0 (Lin's CAFM defaults; keep).
- Optimizer: Adam β=(0, 0.95), LR=1e-5 (Lin's CAFM default; keep).
- EMA decay 0.99.
- 10 epochs post-training on top of our 31.41 baseline.
