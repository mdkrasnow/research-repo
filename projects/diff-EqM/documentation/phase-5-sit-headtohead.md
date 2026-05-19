# Phase 5: SiT Head-to-Head (Aug 29 → Oct 1)

Goal: confirm v10+CAFM combination transfers from EqM to standard flow matching (SiT). Direct comparison to Lin et al.'s published CAFM numbers strengthens ICLR claim by removing "EqM-only" caveat.

## Why Phase 5 strengthens the paper
- Workshop paper headlines: v10+CAFM on EqM.
- Reviewer Q: "Why does this generalize beyond EqM?" → Phase 5 SiT result is the answer.
- Lin CAFM published SiT-XL/2 numbers: FM 8.26 → CAFM 3.63 (guidance-free), 2.06 → 1.53 (CFG). We target same setup at smaller scale.

## Setup

### Pretrained backbone
- Download Lin's pretrained SiT-XL/2 ckpt (mentioned in their README):
  `https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt`
- Place in `projects/diff-EqM/external/Adversarial-Flow-Models/checkpoints/`.

### Compute reality
- SiT-XL/2 is **5× more params than EqM-B/2** (675M vs 130M).
- CAFM 10ep post-training of SiT-XL/2 on 4× A100s ≈ 30-40h.
- 3 seeds × 2 conditions (CAFM-only vs v10+CAFM) = 6 runs × 35h = 210 GPU-h.
- Manageable within Phase 5 buffer.

### Alternative: smaller SiT
- If Lin doesn't publish SiT-B/2 ckpt, use SiT-B/2 trained from scratch (Lin's repo supports). 80ep ≈ 24h per seed.
- 3 seeds × 2 conditions × 24h (training) + 35h × 2 conditions × 3 seeds (CAFM post-training) ≈ 350 GPU-h. Tight.

**Recommended**: SiT-XL/2 with Lin's pretrained ckpt. Avoids retraining.

## Code plan

Adapt Lin's `train_continuous_adversarial_flow_imagenet.py` to add v10 mining hook. Since their code is OOP-heavy (Entrypoint/PersistenceMixin), the cleanest approach is:

1. Add a new file `projects/diff-EqM/experiments/cafm_eqm/v10_sit_trainer.py` that subclasses Lin's `ContinuousAdversarialFlowTrainer`.
2. Override `training_step` to inject v10 mining step (re-use `cafm_eqm/v10_mining.py`).
3. New config `configs/cafm/sit_b2_in256_v10_cafm.yaml` based on `train_cafm_sit.yaml`.

### Pseudo:
```python
class V10ContinuousAdversarialFlowTrainer(ContinuousAdversarialFlowTrainer):
    def training_step(self, step, *, latents, labels, noises, timesteps):
        out = super().training_step(step, latents=latents, labels=labels, noises=noises, timesteps=timesteps)
        if not self.is_dis_step(step) and self.config.v10.lambda_v10 > 0:
            # CAFM gen step already computed; add v10 PGD aux loss.
            from cafm_eqm.v10_mining import v10_aux_loss
            latents_noised = self.interpolate(latents, noises, timesteps)
            velocity_real = self.velocity(latents, noises, timesteps)  # flow matching target
            def fwd(x):
                return self.gen(x, labels, timesteps)
            l_hard, diag = v10_aux_loss(fwd, latents_noised, velocity_real,
                                         K=self.config.v10.K,
                                         eps_radius=self.config.v10.eps_radius,
                                         lr=self.config.v10.lr)
            out[0]["loss/v10_hard"] = l_hard
            out[0]["loss/total"] = out[0]["loss/total"] + self.config.v10.lambda_v10 * l_hard
            for k, v in diag.items():
                out[0][k] = v
        return out
```

Important: **for SiT (flow matching), v10's target is `velocity_real = noises - latents` (no c(γ) scaling)**, unlike EqM. The mechanism is identical otherwise.

## Comparison metric

Report on standard SiT-XL/2 IN-256 setup (guidance-free + CFG):

| Condition | Guidance-Free FID | CFG FID |
|---|---|---|
| Vanilla SiT-XL/2 1400ep (Lin's reference) | 8.26 | 2.06 |
| + CAFM 10ep (Lin's published) | 3.63 | 1.53 |
| + v10 only 10ep (ablation, our addition) | ? | ? |
| + v10 + CAFM 10ep (our contribution) | ? | ? |

Goal: v10+CAFM beats CAFM-only by ≥0.3 FID on the same setup.

## Gates

### Phase 5 PASS (ICLR claim hold):
- v10+CAFM 3-seed mean beats CAFM-only 3-seed mean by ≥0.3 FID (guidance-free).
- Same direction on CFG.
- Diagnostics show v10 mining active (L_hard > L_clean across training).

### Phase 5 FAIL:
- v10+CAFM ties or loses on SiT → restrict claim to EqM family only.
- Workshop result stands; ICLR paper still publishable but narrower.

## Timeline

| Week | Task | GPU-h |
|---|---|---|
| 13 (Aug 25-31) | Workshop submission. Begin Phase 5 setup. | ~0 |
| 14 (Sep 1-7) | Lin code adapter; SiT-XL/2 pretrained ckpt downloaded; CAFM-only reproduction (seed 0). | ~50 |
| 15 (Sep 8-14) | CAFM-only + v10+CAFM seed 0. Compare to Lin's published numbers. | ~80 |
| 16 (Sep 15-21) | Multi-seed runs (seeds 1, 2 for both conditions). | ~140 |
| 17 (Sep 22-28) | FID eval (all 6 ckpts × 50K samples). ICLR draft. | ~50 |
| 18 (Sep 29-Oct 1) | ICLR write-up + submit. | 0 |

Total Phase 5: ~320 GPU-h. Within budget.
