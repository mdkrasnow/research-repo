"""Checkpoint-only energy-to-outcome monotonicity pilot.

At one fixed sampler depth, candidates for a held-out masked-recovery task are
ranked only by their model scalar energy.  They are then completed with the
unchanged sampler, and the hidden clean latent is used only to grade terminal
masked-region error.  Thus a positive correlation is an empirical prediction,
not a consequence of the gradient parameterization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_outcome_monotonicity.core import (paired_bootstrap_difference,
                                              shuffled_control, within_task_kendall)
from eval_masked_recovery import gd_recover, load_ema_model


VARIANTS = ("dot", "direct")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_energy(model: torch.nn.Module, state: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the scalar whose negative input gradient is the sampler field."""
    state = state.detach().requires_grad_(True)
    time = torch.ones((len(state),), device=state.device, dtype=state.dtype)
    with torch.enable_grad():
        _, energy = model(state, time, labels, get_energy=True, train=False)
    if not torch.is_tensor(energy) or energy.shape != (len(state),):
        raise ValueError(f"expected one scalar energy per sample, got {getattr(energy, 'shape', None)}")
    return energy.detach()


def masked_latent_error(recovered: torch.Tensor, clean: torch.Tensor,
                        keep: torch.Tensor) -> torch.Tensor:
    missing = 1 - keep
    return ((recovered - clean).square() * missing).flatten(1).sum(1) / missing.flatten(1).sum(1).clamp_min(1)


def load_bank(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    bank = torch.load(path, map_location="cpu", weights_only=True)
    required = {"endpoint_latents", "endpoint_labels", "endpoint_images", "pairs"}
    absent = required - set(bank)
    if absent:
        raise ValueError(f"bank is missing {sorted(absent)}")
    pairs = bank["pairs"].long()
    ids = torch.unique(pairs.flatten(), sorted=True)
    clean, labels = bank["endpoint_latents"][ids], bank["endpoint_labels"][ids]
    if len(clean) < 2:
        raise ValueError("bank has fewer than two distinct endpoints")
    return clean, labels


def evaluate_variant(model: torch.nn.Module, variant: str, clean: torch.Tensor,
                     labels: torch.Tensor, args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Generate shared-noise candidates and score their final recovery quality."""
    device = next(model.parameters()).device
    energy_rows, error_rows, oracle_rows = [], [], []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for start in range(0, len(clean), args.batch_size):
        x = clean[start:start + args.batch_size].to(device)
        y = labels[start:start + args.batch_size].to(device)
        batch = len(x)
        keep = (torch.rand((batch, 1, x.shape[-2], x.shape[-1]), generator=generator,
                           device=device) > args.mask_prob).to(x.dtype)
        energies, errors, oracle = [], [], []
        for _ in range(args.candidates):
            noise = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype)
            state = keep * x + (1 - keep) * noise
            # Score has no access to clean x or terminal outcome.
            energies.append(scalar_energy(model, state, y).cpu())
            oracle.append(masked_latent_error(state, x, keep).cpu())
            recovered = gd_recover(model, state, y, args.num_sampling_steps, args.stepsize,
                                   args.sampler, args.mu)
            errors.append(masked_latent_error(recovered, x, keep).cpu())
        energy_rows.append(torch.stack(energies, 1))
        error_rows.append(torch.stack(errors, 1))
        oracle_rows.append(torch.stack(oracle, 1))
    return {"energy": torch.cat(energy_rows).numpy(), "error": torch.cat(error_rows).numpy(),
            "oracle": torch.cat(oracle_rows).numpy()}


def main(args: argparse.Namespace) -> None:
    if args.candidates < 3:
        raise ValueError("at least three candidates are required for rank prediction")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean, labels = load_bank(args.bank)
    clean, labels = clean[:args.num_images], labels[:args.num_images]
    if len(clean) != args.num_images:
        raise ValueError(f"requested {args.num_images} tasks but bank contains {len(clean)}")
    models = {v: load_ema_model(str(getattr(args, f"{v}_checkpoint")), args.model,
              args.image_size // 8, args.num_classes, True, v, device) for v in VARIANTS}
    values = {v: evaluate_variant(models[v], v, clean, labels, args) for v in VARIANTS}
    scores = {v: within_task_kendall(values[v]["energy"], values[v]["error"]) for v in VARIANTS}
    oracle = {v: within_task_kendall(values[v]["oracle"], values[v]["error"]) for v in VARIANTS}
    shuffled = {v: shuffled_control(values[v]["energy"], values[v]["error"], seed=args.seed + i + 1)
                for i, v in enumerate(VARIANTS)}
    mean, ci, _ = paired_bootstrap_difference(scores["direct"], scores["dot"],
                                                replicates=args.bootstrap_replicates,
                                                seed=args.bootstrap_seed)
    primary_pass = bool(ci[0] > 0 and scores["direct"].mean() > 0 and oracle["direct"].mean() > .2 and
                        max(abs(shuffled[v].mean()) for v in VARIANTS) < .15)
    report = {
        "study": "energy_to_outcome_monotonicity", "pilot": True,
        "question": "Does lower scalar energy at a matched masked-recovery start rank lower terminal masked latent error?",
        "bank": str(args.bank), "bank_sha256": sha256(args.bank),
        "checkpoints": {v: {"path": str(getattr(args, f"{v}_checkpoint")),
                              "sha256": sha256(getattr(args, f"{v}_checkpoint"))} for v in VARIANTS},
        "protocol": {"num_images": args.num_images, "candidates_per_task": args.candidates,
                     "mask_prob": args.mask_prob, "steps": args.num_sampling_steps,
                     "stepsize": args.stepsize, "sampler": args.sampler, "seed": args.seed,
                     "bootstrap_replicates": args.bootstrap_replicates},
        "metrics": {v: {"energy_outcome_kendall_mean": float(scores[v].mean()),
                         "oracle_kendall_mean": float(oracle[v].mean()),
                         "shuffled_energy_kendall_mean": float(shuffled[v].mean()),
                         "mean_terminal_masked_latent_mse": float(values[v]["error"].mean())} for v in VARIANTS},
        "paired_direct_minus_dot": {"mean": mean, "ci95": list(ci)},
        "controls_pass": {"oracle_direct_gt_0p2": bool(oracle["direct"].mean() > .2),
                          "shuffled_abs_mean_lt_0p15": bool(max(abs(shuffled[v].mean()) for v in VARIANTS) < .15)},
        "pilot_rule": "PASS only if direct-dot lower bootstrap CI > 0, direct mean Kendall > 0, and both controls pass.",
        "pilot_verdict": "PASS" if primary_pass else "FAIL",
        "limitation": "One matched checkpoint per variant: this pilot cannot establish a seed-level claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **{f"{v}_{k}": a for v, d in values.items() for k, a in d.items()},
                        **{f"{v}_kendall": scores[v] for v in VARIANTS})
    print(json.dumps({"verdict": report["pilot_verdict"], "paired": report["paired_direct_minus_dot"],
                      "metrics": report["metrics"], "controls": report["controls_pass"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--dot-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="EqM-B/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mask-prob", type=float, default=.5)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=.0017)
    parser.add_argument("--sampler", choices=("gd", "ngd"), default="gd")
    parser.add_argument("--mu", type=float, default=.3)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
