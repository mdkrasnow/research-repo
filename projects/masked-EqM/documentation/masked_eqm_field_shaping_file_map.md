# Masked EqM field-shaping file and function map

Date: 2026-08-03

This map records the repository locations that govern the predeclared pixel-masked continuation experiment. The active model is EqM-B/2 at 256 px with `ebm=none`, class conditioning enabled in the architecture, and unconditional sampling through the null class token where specified.

| Concern | Repository location | Relevant symbol or behavior |
|---|---|---|
| Main training entry point | `train.py` | `main`; DDP initialization, ImageNet `ImageFolder`, frozen VAE encoding, optimizer, EMA, checkpointing |
| EqM loss | `transport/transport.py` | `Transport.training_losses`; compares model output to the path velocity after the existing `get_ct` multiplier |
| Corruption endpoint selection | `transport/transport.py` | `Transport.sample`; ordinary endpoint is `torch.randn_like(x1)` |
| Interpolation | `transport/path.py` | `ICPlan.compute_mu_t`; `x_t = t*x_clean + (1-t)*x_corruption` |
| Base path velocity | `transport/path.py` | `ICPlan.compute_ut`; `u_t = x_clean - x_corruption` for the linear path |
| c(gamma) | `transport/transport.py` | `Transport.get_ct`; existing piecewise multiplier, applied in `training_losses` without modification |
| Target sign | `transport/path.py`, `transport/transport.py` | target points from corruption toward clean: `(x_clean-x_corruption)*c(t)` |
| Sampling sign | `sample_gd.py`, `eval_fid.py` | GD update is `x <- x + eta * model(x)`; EqM-B/2 default `eta=0.003` from `sampling_defaults.py` |
| Dataset and preprocessing | `train.py` | `ImageFolder`; ADM center crop, random horizontal flip, tensor conversion, normalization to `[-1,1]` |
| Distributed sampler | `train.py` | `DistributedSampler(..., shuffle=True, seed=global_seed)` and `sampler.set_epoch(epoch)` |
| VAE encoding | `train.py` | frozen `AutoencoderKL` `stabilityai/sd-vae-ft-ema`; posterior sample scaled by `0.18215` |
| Checkpoint loading | `download.py`, `train.py` | `find_model`; restores model, EMA, and AdamW when present |
| EMA update | `train.py` | `update_ema`, decay `0.9999`; evaluation utilities load the saved EMA weights |
| Model architecture | `models.py` | `EqM_models["EqM-B/2"]`; 4-channel 32x32 latent input at 256 px |
| Standard generation | `sample_gd.py` | paper-style GD sampler and PNG/NPZ generation |
| FID generation/calculation | `eval_fid.py` | EMA sampling plus `pytorch_fid.fid_score.calculate_fid_given_paths` |
| Existing masked recovery | `eval_masked_recovery.py` | latent-mask recovery utility; not reused for the primary pixel-space block-mask protocol |
| LPIPS | `eval_masked_recovery.py`, `eval_fourier_recovery.py` | `lpips.LPIPS(net="alex")` convention |
| Scheduler conventions | `slurm/jobs/*.sbatch`, repository `scripts/cluster/remote_submit.sh` | immutable Git SHA clone, remote-only SLURM, job-scoped results directories |
| Experiment state | `.state/pipeline.json` | authoritative `active_runs` and `completed_runs` ledger |
| Historical aggregation | existing experiment-specific analysis modules | paired/hierarchical analysis patterns; the new primary uses image-cluster bootstrap |

## Exact base checkpoints

The three files are the `ebm=none` checkpoints from one continuation lineage and were verified on Netscratch on 2026-08-03.

- Epoch 15, step 600,540: `/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer15_none_seed0_job35524599/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch15.pt`
- Epoch 40, step 1,601,440: `/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer40_none_seed0_ckpt50k_job36359207/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch40.pt`
- Epoch 80, step 3,202,880: `/n/netscratch/ydu_lab/Lab/mkrasnow_eqm/direct_energy_longer_retry/longer80_none_seed0_ckpt50k_job36632776/000-EqM-B-2-Linear-velocity-None-ebm-none/checkpoints/epoch80.pt`

Each checkpoint has keys `model`, `ema`, `opt`, `args`, `epoch`, and `step`. All use AdamW at learning rate `1e-4`, weight decay `0`, global batch size 32, global seed 0, linear velocity prediction, the EMA VAE, and the ordinary Gaussian endpoint. No scheduler object exists because the repository uses a constant learning rate. One normal epoch is exactly 40,036 optimizer updates; ten epochs are therefore locked to 400,360 updates.

## Sign sanity check

For a scalar linear-path example with corruption endpoint 0, clean endpoint 1, and `t=0.5`, `ICPlan.plan` returns `x_t=0.5` and base velocity `u_t=+1`. `Transport.training_losses` multiplies that velocity by the positive existing `get_ct(0.5)` factor. The sampler adds the predicted field (`x <- x + eta*f(x)`), so the repository convention is unequivocally corruption-to-clean. Pixel-masked continuation must replace only the endpoint and must not negate or independently rederive this target.
