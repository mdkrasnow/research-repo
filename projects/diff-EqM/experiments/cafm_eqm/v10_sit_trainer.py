"""Phase 5: v10-augmented CAFM trainer for SiT (subclasses Lin's CAFM trainer).

Used for SiT head-to-head experiments (Phase 5). The base CAFM trainer is
Lin's `ContinuousAdversarialFlowTrainer` from the cloned AFM repo; we
override `training_step` to add v10 PGD hard-example mining on the
flow-matching velocity target.

Key difference vs CAFM-EqM:
- SiT target = velocity = noises - latents (NO c(γ) scaling).
- EqM target = (noises - latents) * c(γ).
- Otherwise the v10 mining mechanism is identical.

Usage:
    bash scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && \
        sbatch projects/diff-EqM/slurm/jobs/v10_cafm_sit_phase5.sbatch"

Activation: Phase 5 only (after workshop submission). Not run in Phase 1-4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_AFM = REPO_ROOT / "external" / "Adversarial-Flow-Models"
if str(EXTERNAL_AFM) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_AFM))

from .v10_mining import v10_aux_loss  # noqa: E402

# Lin's base trainer. Available only after `pip install -r requirements_cafm.txt`
# and after cloning the AFM repo + downloading SiT models.
try:
    from train_continuous_adversarial_flow_imagenet import (
        ContinuousAdversarialFlowTrainer,
    )
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "Lin's CAFM training script not importable. "
        "Verify external/Adversarial-Flow-Models exists and PYTHONPATH "
        "includes its root. Original: " + repr(e)
    )


class V10ContinuousAdversarialFlowTrainer(ContinuousAdversarialFlowTrainer):
    """CAFM trainer + v10 PGD hard-example mining on the gen step.

    Adds:
      - v10 PGD mining on the velocity target (noises - latents).
      - Combined loss: L_CAFM_G + λ_v10 * L_hard.
      - Diagnostics: v10/delta_norm_*, v10/l_hard, v10/ratio.

    Configuration via OmegaConf `cfg.v10`:
        lambda_v10: float = 0.1
        K: int = 1                  (FGSM-style)
        eps_radius: float = 0.3
        lr: float = 0.05
    """

    def training_step(
        self,
        step: int,
        *,
        latents: Tensor,
        labels: Tensor,
        noises: Tensor,
        timesteps: Tensor,
    ):
        loss_dict, visual_dict = super().training_step(
            step,
            latents=latents,
            labels=labels,
            noises=noises,
            timesteps=timesteps,
        )

        # Only augment generator steps.
        if self.is_dis_step(step):
            return loss_dict, visual_dict

        v10_cfg = self.config.get("v10", None)
        if v10_cfg is None or v10_cfg.get("lambda_v10", 0.0) <= 0:
            return loss_dict, visual_dict

        # SiT flow-matching target: velocity = noises - latents.
        latents_noised = self.interpolate(latents, noises, timesteps)
        velocity_real = self.velocity(latents, noises, timesteps)

        def fwd(x: Tensor) -> Tensor:
            return self.gen(x, labels, timesteps)

        l_hard, diag = v10_aux_loss(
            fwd,
            latents_noised,
            velocity_real,
            K=int(v10_cfg.get("K", 1)),
            eps_radius=float(v10_cfg.get("eps_radius", 0.3)),
            lr=float(v10_cfg.get("lr", 0.05)),
        )
        lam = float(v10_cfg.get("lambda_v10", 0.1))
        loss_dict["loss/v10_hard"] = l_hard
        # Augment total: standard CAFM gen loss + λ · L_hard.
        loss_dict["loss/total"] = loss_dict["loss/total"] + lam * l_hard
        for k, v in diag.items():
            loss_dict[k] = v

        return loss_dict, visual_dict


def main():
    V10ContinuousAdversarialFlowTrainer.create().entrypoint()


if __name__ == "__main__":
    main()
