"""CAFM-EqM training step — the per-batch loss computation, framework-agnostic.

This module is intentionally backbone-agnostic: it does NOT depend on Lin's
`Entrypoint`/`PersistenceMixin` infrastructure. Phase 1a wraps it with a minimal
DDP training loop in `train_cafm_eqm.py`; Phase 2 wraps the v10 variant in
`train_v10_cafm_eqm.py`.

Drop-in mapping vs Lin's `training_step` (in
`external/Adversarial-Flow-Models/train_continuous_adversarial_flow_imagenet.py`,
the `ContinuousAdversarialFlowTrainer.training_step` method):
- Replace `velocity_real = noises - latents` → `target = (noises - latents) * c(γ)`.
- Replace `velocity_pred = self.gen(latents_noised, labels, timesteps)` with the
  EqM gradient field output (model_kwargs={'y': labels}, t=0 internally).
- Replace `timesteps` → `gamma` everywhere (same distribution U(0,1); semantically
  EqM's mixing coefficient instead of flow-matching time).
- Keep CAFM's JVP discriminator unchanged; pass γ as t.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .eqm_target import eqm_base_loss, eqm_target, interpolate
from .v10_mining import v10_aux_loss


# ---------------------------------------------------------------------------
# CAFM losses (least-squares GAN + centering penalty + optional OT)
# ---------------------------------------------------------------------------


def lsgan_dis_loss(
    logits_real: Tensor, logits_fake: Tensor, logits_centering: Tensor,
    *, cp_scale: float = 0.001,
) -> dict[str, Tensor]:
    """Lin CAFM discriminator loss. Matches their `training_step` dis branch."""
    dis_adv_real = logits_real.sub(1.0).square().mean()
    dis_adv_fake = logits_fake.add(1.0).square().mean()
    dis_adv = dis_adv_real + dis_adv_fake
    dis_cp = logits_centering.square().mean().mul(cp_scale)
    return {
        "loss/dis_adv": dis_adv,
        "loss/dis_cp": dis_cp,
        "loss/total_dis": dis_adv + dis_cp,
        "logits/real": logits_real.detach().mean(),
        "logits/fake": logits_fake.detach().mean(),
        "logits/centering": logits_centering.detach().mean(),
    }


def lsgan_gen_adv_loss(logits_fake: Tensor) -> Tensor:
    """Lin CAFM generator adversarial loss (LSGAN, push logits_fake toward 1)."""
    return logits_fake.sub(1.0).square().mean()


# ---------------------------------------------------------------------------
# EqM-specific input prep
# ---------------------------------------------------------------------------


def prepare_eqm_inputs(latents: Tensor, labels: Tensor) -> dict[str, Tensor]:
    """Sample γ and noise, build x_γ and target. Returns dict for training step."""
    device = latents.device
    b = latents.size(0)
    gamma = torch.rand([b], device=device)
    noises = torch.randn_like(latents)
    x_gamma = interpolate(latents, noises, gamma)
    target = eqm_target(latents, noises, gamma)
    return {
        "latents": latents,
        "labels": labels,
        "noises": noises,
        "gamma": gamma,
        "x_gamma": x_gamma,
        "target": target,
    }


# ---------------------------------------------------------------------------
# Discriminator step (drop-in for CAFM dis branch)
# ---------------------------------------------------------------------------


def cafm_dis_step(
    dis_jvp: Any,   # DiscriminatorJVP from external repo
    gen: Any,       # EqMGeneratorWrapper
    inputs: dict[str, Tensor],
    *, cp_scale: float = 0.001,
) -> dict[str, Tensor]:
    """Compute discriminator loss. Matches Lin's CAFM dis branch with EqM target."""
    x_gamma = inputs["x_gamma"]
    labels = inputs["labels"]
    gamma = inputs["gamma"]
    target = inputs["target"]

    # Generator forward (no grad — dis step doesn't update gen).
    with torch.no_grad():
        pred = gen(x_gamma, labels, gamma)

    # JVP tangents: stack [real_target, gen_prediction] for batched dis_jvp.
    tangent_x = torch.stack([target, pred])
    tangent_t = torch.stack([torch.ones_like(gamma), torch.ones_like(gamma)])

    out, out_jvp = dis_jvp(
        x=x_gamma, y=labels, t=gamma, dx=tangent_x, dt=tangent_t,
    )
    logits_real, logits_fake = out_jvp.chunk(2, 0)

    return lsgan_dis_loss(
        logits_real=logits_real,
        logits_fake=logits_fake,
        logits_centering=out,
        cp_scale=cp_scale,
    )


# ---------------------------------------------------------------------------
# Generator step (drop-in for CAFM gen branch)
# ---------------------------------------------------------------------------


def cafm_gen_step(
    dis_jvp: Any, gen: Any, inputs: dict[str, Tensor],
    *, ot_scale: float = 0.0,
) -> dict[str, Tensor]:
    """Compute generator loss for CAFM-only training (no v10)."""
    x_gamma = inputs["x_gamma"]
    labels = inputs["labels"]
    gamma = inputs["gamma"]

    pred = gen(x_gamma, labels, gamma)

    _, logits_fake = dis_jvp(
        x=x_gamma, y=labels, t=gamma, dx=pred, dt=torch.ones_like(gamma),
    )
    gen_adv = lsgan_gen_adv_loss(logits_fake)
    gen_ot = pred.square().mean().mul(ot_scale)

    return {
        "loss/gen_adv": gen_adv,
        "loss/gen_ot": gen_ot,
        "loss/total_gen": gen_adv + gen_ot,
        "logits/fake": logits_fake.detach().mean(),
    }


def cafm_v10_gen_step(
    dis_jvp: Any, gen: Any, inputs: dict[str, Tensor],
    *, ot_scale: float = 0.0,
    lambda_v10: float = 0.1, v10_K: int = 1,
    v10_eps_radius: float = 0.3, v10_lr: float = 0.05,
) -> dict[str, Tensor]:
    """Combined CAFM + v10 generator loss. Used in Phase 2."""
    cafm_losses = cafm_gen_step(dis_jvp, gen, inputs, ot_scale=ot_scale)

    # v10 PGD mining on the same EqM target. Closure captures gen + (y, γ).
    labels, gamma, target = inputs["labels"], inputs["gamma"], inputs["target"]

    def fwd(x_perturbed: Tensor) -> Tensor:
        return gen(x_perturbed, labels, gamma)

    l_hard, v10_diag = v10_aux_loss(
        fwd, inputs["x_gamma"], target,
        K=v10_K, eps_radius=v10_eps_radius, lr=v10_lr,
    )

    out = dict(cafm_losses)
    out["loss/v10_hard"] = l_hard
    # base regression on the clean prediction, for diagnostic ratio
    pred_clean = gen(inputs["x_gamma"], labels, gamma)
    l_base = eqm_base_loss(pred_clean, target)
    out["loss/v10_base"] = l_base.detach()
    out["v10/ratio"] = l_hard.detach() / l_base.detach().clamp_min(1e-8)
    out.update(v10_diag)
    out["loss/total_gen"] = out["loss/total_gen"] + lambda_v10 * l_hard
    return out


__all__ = [
    "lsgan_dis_loss", "lsgan_gen_adv_loss",
    "prepare_eqm_inputs",
    "cafm_dis_step", "cafm_gen_step", "cafm_v10_gen_step",
]
