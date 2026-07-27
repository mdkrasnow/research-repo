from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from download import find_model
from models import EqM_models
from energy_monotonicity.core import (
    cluster_bootstrap,
    get_effective_field,
    trajectory_metrics,
    trapezoid_line_integral,
)

VARIANTS = ("none", "dot", "direct")
CANONICAL_SIGNS = {"none": -1, "dot": -1, "direct": 1}


@dataclass(frozen=True)
class CheckpointRecord:
    variant: str
    epoch: int
    checkpoint_path: str
    run_path: str
    config_path: str | None
    checkpoint_step: int
    ema_available: bool
    weights_used: str
    model: str
    image_size: int
    num_classes: int
    uncond: bool
    corruption_mode: str
    architecture_summary: str
    parameter_count: int
    git_commit: str | None
    run_identifier: str
    sha256: str


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(type(value).__name__)


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _arg(checkpoint: dict[str, Any], name: str, default: Any = None) -> Any:
    args = checkpoint.get("args")
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def discover_checkpoints(run: Path, variant: str, epochs: list[int],
                         use_ema: bool = True) -> list[CheckpointRecord]:
    if not run.exists():
        raise FileNotFoundError(f"{variant} run does not exist: {run}")
    if run.is_file():
        specification = json.loads(run.read_text())
        if specification.get("variant") != variant:
            raise ValueError(f"run manifest {run} declares {specification.get('variant')}, "
                             f"requested {variant}")
        candidates = [Path(path) for path in specification["checkpoints"].values()]
    else:
        candidates = sorted(
            path for path in run.rglob("*.pt")
            if path.is_file() and (
                path.stem.lower().startswith("epoch") or path.stem.isdigit()
            )
        )
    by_epoch: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in candidates:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            continue
        epoch = state.get("epoch")
        if epoch is None and path.stem.lower().startswith("epoch"):
            epoch = int(path.stem[5:])
        if epoch not in epochs:
            continue
        ckpt_variant = _arg(state, "ebm")
        if ckpt_variant != variant:
            raise ValueError(
                f"checkpoint labeling failure: requested {variant}, {path} declares ebm={ckpt_variant!r}"
            )
        if epoch in by_epoch:
            raise ValueError(f"multiple checkpoints claim {variant} epoch {epoch}: "
                             f"{by_epoch[epoch][0]} and {path}")
        by_epoch[int(epoch)] = (path, state)

    missing = sorted(set(epochs) - set(by_epoch))
    if missing:
        raise FileNotFoundError(f"{variant} is missing requested epochs {missing} under {run}")

    records = []
    reference: dict[str, Any] | None = None
    for epoch in epochs:
        path, state = by_epoch[epoch]
        ema_available = "ema" in state
        if use_ema and not ema_available:
            raise ValueError(f"EMA required but unavailable: {path}")
        metadata = {
            "model": _arg(state, "model", "EqM-B/2"),
            "image_size": int(_arg(state, "image_size", 256)),
            "num_classes": int(_arg(state, "num_classes", 1000)),
            "uncond": bool(_arg(state, "uncond", True)),
            "corruption_mode": _arg(state, "corruption_mode", "gaussian"),
            "global_seed": int(_arg(state, "global_seed", -1)),
            "path_type": _arg(state, "path_type", "Linear"),
            "prediction": _arg(state, "prediction", "velocity"),
        }
        if reference is None:
            reference = metadata
        elif metadata != reference:
            raise ValueError(f"incompatible checkpoint metadata within {variant}: "
                             f"epoch {epoch}: {metadata} != {reference}")
        model = EqM_models[metadata["model"]](
            input_size=metadata["image_size"] // 8,
            num_classes=metadata["num_classes"],
            uncond=metadata["uncond"],
            ebm=variant,
        )
        weights = state["ema"] if use_ema else state.get("model", state)
        model.load_state_dict(weights, strict=True)
        parameter_count = sum(p.numel() for p in model.parameters())
        del model, weights
        config_paths = sorted(run.glob("*.json"))
        records.append(CheckpointRecord(
            variant=variant, epoch=epoch, checkpoint_path=str(path.resolve()),
            run_path=str((run.parent if run.is_file() else run).resolve()),
            config_path=str(run.resolve()) if run.is_file() else (
                str(config_paths[0].resolve()) if len(config_paths) == 1 else None
            ),
            checkpoint_step=int(state.get("step", -1)),
            ema_available=ema_available, weights_used="ema" if use_ema else "model",
            model=metadata["model"], image_size=metadata["image_size"],
            num_classes=metadata["num_classes"], uncond=metadata["uncond"],
            corruption_mode=metadata["corruption_mode"],
            architecture_summary=f"{metadata['model']} latent={metadata['image_size']//8} "
                                 f"ebm={variant} class_conditional=True uncond_time={metadata['uncond']}",
            parameter_count=parameter_count,
            git_commit=_arg(state, "git_sha", None),
            run_identifier=run.stem if run.is_file() else run.name,
            sha256=sha256_file(path),
        ))
    return records


def validate_cross_variant_manifest(records: list[CheckpointRecord]) -> None:
    expected = {(variant, epoch) for variant in VARIANTS for epoch in
                sorted({record.epoch for record in records})}
    found = {(record.variant, record.epoch) for record in records}
    if found != expected:
        raise ValueError(f"manifest is not rectangular: missing={sorted(expected-found)}, "
                         f"extra={sorted(found-expected)}")
    fields = ("model", "image_size", "num_classes", "uncond", "corruption_mode")
    for epoch in sorted({record.epoch for record in records}):
        epoch_records = [record for record in records if record.epoch == epoch]
        for field in fields:
            values = {getattr(record, field) for record in epoch_records}
            if len(values) != 1:
                raise ValueError(f"epoch {epoch} differs across variants on {field}: {values}")


def build_evaluation_bank(data_path: Path, output: Path, num_images: int,
                          noises_per_image: int, seed: int, image_size: int,
                          vae_name: str, device: torch.device,
                          force: bool) -> dict[str, torch.Tensor]:
    bank_path = output / "evaluation_bank.pt"
    metadata_path = output / "evaluation_bank.json"
    if bank_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        requested = {
            "num_images": num_images, "noises_per_image": noises_per_image,
            "seed": seed, "image_size": image_size, "vae": vae_name,
            "data_path": str(data_path.resolve()),
        }
        for key, value in requested.items():
            if metadata.get(key) != value:
                raise ValueError(f"cached evaluation bank {key}={metadata.get(key)!r}, "
                                 f"requested {value!r}; use a new output dir or --force")
        return torch.load(bank_path, map_location="cpu", weights_only=True)

    from diffusers.models import AutoencoderKL
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import transforms
    from train import center_crop_arr

    if data_path.name.lower() not in {"val", "validation", "test"}:
        raise ValueError(f"held-out dataset path must end in val/validation/test, got {data_path}")
    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = ImageFolder(data_path, transform=transform)
    if len(dataset) < num_images:
        raise ValueError(f"held-out dataset contains {len(dataset)} < {num_images}")
    selection_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=selection_generator)[:num_images]
    vae = AutoencoderKL.from_pretrained(vae_name).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    latents, labels = [], []
    latent_generator = torch.Generator(device=device).manual_seed(seed + 1)
    for start in range(0, num_images, 32):
        items = [dataset[int(index)] for index in indices[start:start + 32]]
        images = torch.stack([item[0] for item in items]).to(device)
        labels.extend(int(item[1]) for item in items)
        with torch.no_grad():
            distribution = vae.encode(images).latent_dist
            sample = distribution.sample(generator=latent_generator).mul(0.18215)
        latents.append(sample.cpu())
        if (start // 32 + 1) % 8 == 0:
            print(f"evaluation bank: encoded {min(start+32, num_images)}/{num_images}", flush=True)
    clean = torch.cat(latents)
    del vae
    noise_generator = torch.Generator().manual_seed(seed + 2)
    noise = torch.randn(
        (num_images, noises_per_image, *clean.shape[1:]),
        generator=noise_generator, dtype=torch.float32,
    )
    bank = {
        "clean": clean.float(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "indices": indices.long(),
        "noise": noise,
    }
    atomic_torch_save(bank_path, bank)
    atomic_json(metadata_path, {
        "version": 1, "data_path": str(data_path.resolve()),
        "dataset_class": "torchvision.datasets.ImageFolder",
        "split": data_path.name, "held_out": True, "augmentation": "none",
        "preprocessing": f"center_crop_arr({image_size}), ToTensor, Normalize(0.5,0.5)",
        "vae": vae_name, "vae_scale": 0.18215,
        "num_images": num_images, "noises_per_image": noises_per_image,
        "num_trajectories": num_images * noises_per_image,
        "seed": seed, "selection_seed": seed, "latent_seed": seed + 1,
        "noise_seed": seed + 2, "image_size": image_size,
        "image_indices": indices.tolist(), "labels": labels,
        "noise_storage": "evaluation_bank.pt",
        "clean_latent_storage": "evaluation_bank.pt",
        "bank_sha256": sha256_file(bank_path),
    })
    return bank


def load_model(record: CheckpointRecord, device: torch.device,
               dtype: torch.dtype) -> torch.nn.Module:
    model = EqM_models[record.model](
        input_size=record.image_size // 8, num_classes=record.num_classes,
        uncond=record.uncond, ebm=record.variant,
    )
    state = find_model(record.checkpoint_path)
    weights = state[record.weights_used] if record.weights_used in state else state
    model.load_state_dict(weights, strict=True)
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def evaluate_checkpoint(record: CheckpointRecord, bank: dict[str, torch.Tensor],
                        gamma: np.ndarray, output: Path, batch_size: int,
                        device: torch.device, dtype: torch.dtype,
                        force: bool) -> dict[str, Any]:
    cache_path = output / "cache" / f"{record.variant}_epoch{record.epoch:02d}.pt"
    if cache_path.exists() and not force:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached["checkpoint_sha256"] != record.sha256:
            raise ValueError(f"cache/checkpoint hash mismatch: {cache_path}")
        return cached
    model = load_model(record, device, dtype)
    clean = bank["clean"]
    noise = bank["noise"]
    labels = bank["labels"]
    num_images, noises_per_image = noise.shape[:2]
    trajectory_clean = clean[:, None].expand(-1, noises_per_image, -1, -1, -1).reshape(
        -1, *clean.shape[1:]
    )
    trajectory_noise = noise.reshape(-1, *clean.shape[1:])
    trajectory_labels = labels[:, None].expand(-1, noises_per_image).reshape(-1)
    trajectories = len(trajectory_labels)
    gammas = torch.tensor(gamma, dtype=dtype)
    all_fields = np.empty((trajectories, len(gamma), *clean.shape[1:]), dtype=np.float32)
    all_scalars = (
        np.empty((trajectories, len(gamma)), dtype=np.float64)
        if record.variant in {"dot", "direct"} else None
    )
    raw_shapes: set[tuple[int, ...]] = set()
    started = time.monotonic()
    for start in range(0, trajectories, batch_size):
        stop = min(start + batch_size, trajectories)
        x = trajectory_clean[start:stop].to(device=device, dtype=dtype)
        epsilon = trajectory_noise[start:stop].to(device=device, dtype=dtype)
        y = trajectory_labels[start:stop].to(device)
        for gamma_index, gamma_value in enumerate(gammas):
            z = ((1 - gamma_value) * epsilon + gamma_value * x).contiguous()
            tv = torch.full((stop-start,), float(gamma_value), device=device, dtype=dtype)
            result = get_effective_field(model, record.variant, z, y, tv)
            raw_shapes.add(result.raw_output_shape[1:])
            field = result.effective_field
            if not torch.isfinite(field).all():
                bad = (~torch.isfinite(field)).flatten(1).any(1).sum().item()
                print(f"WARNING {record.variant} epoch {record.epoch} gamma "
                      f"{float(gamma_value):.3f}: {bad} nonfinite fields", flush=True)
            all_fields[start:stop, gamma_index] = field.float().cpu().numpy()
            if all_scalars is not None:
                assert result.scalar_energy is not None
                all_scalars[start:stop, gamma_index] = (
                    result.scalar_energy.double().cpu().numpy()
                )
            del z, result, field
        if (start // batch_size + 1) % 8 == 0 or stop == trajectories:
            elapsed = time.monotonic() - started
            print(f"{record.variant} epoch {record.epoch}: {stop}/{trajectories} "
                  f"trajectories ({elapsed:.1f}s)", flush=True)

    points = (
        torch.tensor(gamma, dtype=torch.float64)[None, :, None, None, None] *
        trajectory_clean.double()[:, None] +
        (1 - torch.tensor(gamma, dtype=torch.float64)[None, :, None, None, None]) *
        trajectory_noise.double()[:, None]
    )
    raw_integral = trapezoid_line_integral(
        torch.from_numpy(all_fields).double(), points
    ).numpy()
    canonical_energy = CANONICAL_SIGNS[record.variant] * raw_integral
    c_gamma = np.minimum(1.0, 5.0 - 5.0 * gamma) * 4.0
    sampling_target = (
        (trajectory_clean.numpy() - trajectory_noise.numpy())[:, None] *
        c_gamma.reshape(1, -1, 1, 1, 1)
    )
    canonical_field = CANONICAL_SIGNS[record.variant] * all_fields
    flat_canonical_field = canonical_field.reshape(trajectories, len(gamma), -1)
    ideal_direction = (
        trajectory_noise.numpy() - trajectory_clean.numpy()
    ).reshape(trajectories, -1)
    per_gamma_alignment = (
        (flat_canonical_field * ideal_direction[:, None]).sum(2) /
        np.maximum(
            np.linalg.norm(flat_canonical_field, axis=2) *
            np.linalg.norm(ideal_direction, axis=1)[:, None],
            1e-30,
        )
    )
    metrics = trajectory_metrics(
        canonical_energy, gamma, canonical_field,
        epsilon_minus_x=trajectory_noise.numpy() - trajectory_clean.numpy(),
        target=-sampling_target,
    )
    validation = None
    if all_scalars is not None:
        scalar_difference = all_scalars - all_scalars[:, :1]
        discrepancy = np.abs(raw_integral - scalar_difference)
        energy_range = np.ptp(all_scalars, axis=1)
        validation = {
            "mean_absolute_discrepancy": float(discrepancy.mean()),
            "median_absolute_discrepancy": float(np.median(discrepancy)),
            "p95_absolute_discrepancy": float(np.quantile(discrepancy, 0.95)),
            "max_absolute_discrepancy": float(discrepancy.max()),
            "normalized_by_total_energy_range": float(
                (discrepancy.max(1) / np.maximum(energy_range, 1e-30)).mean()
            ),
        }
    result = {
        "variant": record.variant, "epoch": record.epoch,
        "checkpoint_sha256": record.sha256,
        "gamma": gamma, "raw_line_integral": raw_integral,
        "canonical_energy": canonical_energy, "scalar_energy": all_scalars,
        "fields": all_fields, "metrics": metrics,
        "image_ids": np.repeat(np.arange(num_images), noises_per_image),
        "noise_ids": np.tile(np.arange(noises_per_image), num_images),
        "validation": validation, "raw_output_shapes": sorted(raw_shapes),
        "directional_alignment_by_gamma": per_gamma_alignment,
        "canonical_energy_sign": CANONICAL_SIGNS[record.variant],
        "precision": str(dtype), "batch_size": batch_size,
    }
    atomic_torch_save(cache_path, result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def dense_validation(record: CheckpointRecord, bank: dict[str, torch.Tensor],
                     output: Path, batch_size: int, device: torch.device,
                     dtype: torch.dtype, subset: int, force: bool) -> dict[str, Any] | None:
    if record.variant == "none":
        return None
    diagnostic_bank = {
        "clean": bank["clean"][:subset],
        "labels": bank["labels"][:subset],
        "indices": bank["indices"][:subset],
        "noise": bank["noise"][:subset, :1],
    }
    summaries = {}
    for points in (21, 101):
        diagnostic = evaluate_checkpoint(
            record, diagnostic_bank, np.linspace(0, 1, points),
            output / "validation" / f"dense_{record.variant}_epoch{record.epoch:02d}_{points}",
            batch_size, device, dtype, force,
        )
        summaries[str(points)] = diagnostic["validation"]
    coarse = summaries["21"]["mean_absolute_discrepancy"]
    dense = summaries["101"]["mean_absolute_discrepancy"]
    summaries["convergence_pass"] = bool(dense < coarse)
    if not summaries["convergence_pass"]:
        raise RuntimeError(f"{record.variant} line-integral validation did not converge: "
                           f"21={coarse}, 101={dense}")
    atomic_json(output / "validation" / f"{record.variant}_epoch{record.epoch:02d}.json", summaries)
    return summaries


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def summarize(results: list[dict[str, Any]], bootstrap_replicates: int,
              bootstrap_seed: int, output: Path) -> tuple[list[dict[str, Any]],
                                                           list[dict[str, Any]],
                                                           dict[str, np.ndarray],
                                                           dict[str, Any]]:
    by_key = {(result["variant"], result["epoch"]): result for result in results}
    epochs = sorted({result["epoch"] for result in results})
    rows, paired_rows = [], []
    bootstrap_arrays: dict[str, np.ndarray] = {}
    bootstrap_draws = None
    for epoch in epochs:
        ordering = {
            variant: by_key[(variant, epoch)]["metrics"]["ordering_accuracy"]
            for variant in VARIANTS
        }
        image_ids = by_key[("none", epoch)]["image_ids"]
        boot, draws = cluster_bootstrap(ordering, image_ids, bootstrap_replicates, bootstrap_seed)
        if bootstrap_draws is None:
            bootstrap_draws = draws
        elif not np.array_equal(bootstrap_draws, draws):
            raise AssertionError("bootstrap resamples are not paired")
        for variant in VARIANTS:
            result = by_key[(variant, epoch)]
            metrics = result["metrics"]
            lo, hi = percentile_interval(boot[variant])
            rows.append({
                "variant": variant, "epoch": epoch,
                "ordering_accuracy": float(metrics["ordering_accuracy"].mean()),
                "ordering_ci_lower": lo, "ordering_ci_upper": hi,
                "adjacent_step_accuracy": float(metrics["adjacent_accuracy"].mean()),
                "perfect_trajectory_rate": float(metrics["perfect_trajectory"].mean()),
                "spearman_correlation": float(metrics["spearman"].mean()),
                "total_energy_drop": float(metrics["total_energy_drop"].mean()),
                "tie_rate": float(metrics["tie_rate"].mean()),
                "nan_rate": float(metrics["nan_rate"].mean()),
                "infinity_rate": float(metrics["inf_rate"].mean()),
                "zero_field_rate": float(metrics["zero_field_rate"].mean()),
                "violation_count": float(metrics["violation_count"].mean()),
                "violation_magnitude": float(metrics["violation_magnitude"].mean()),
                "directional_alignment": float(metrics["directional_alignment"].mean()),
                "field_target_mse": float(metrics["field_target_mse"].mean()),
            })
            bootstrap_arrays[f"{variant}_epoch{epoch:02d}"] = boot[variant]
        for left, right in (("direct", "dot"), ("direct", "none"), ("dot", "none")):
            differences = boot[left] - boot[right]
            lo, hi = percentile_interval(differences)
            paired_rows.append({
                "comparison": f"{left} - {right}", "epoch": epoch,
                "mean_difference": float(ordering[left].mean() - ordering[right].mean()),
                "ci_lower": lo, "ci_upper": hi,
            })
            bootstrap_arrays[f"{left}_minus_{right}_epoch{epoch:02d}"] = differences
    assert bootstrap_draws is not None
    bootstrap_arrays["cluster_draws"] = bootstrap_draws
    epoch8_pairs = {
        row["comparison"]: row for row in paired_rows if row["epoch"] == 8
    }
    if len(epoch8_pairs) != 3:
        verdict = {"verdict": "INCONCLUSIVE", "reason": "epoch 8 unavailable"}
    else:
        direct_dot = epoch8_pairs["direct - dot"]["ci_lower"]
        direct_none = epoch8_pairs["direct - none"]["ci_lower"]
        verdict = {
            "direct_minus_dot_ci_lower": direct_dot,
            "direct_minus_none_ci_lower": direct_none,
            "outperforms_dot": bool(direct_dot > 0),
            "noninferior_to_none_margin": 0.01,
            "noninferior_to_none": bool(direct_none > -0.01),
        }
        verdict["verdict"] = (
            "PASS" if verdict["outperforms_dot"] and verdict["noninferior_to_none"] else "FAIL"
        )
        verdict["reason"] = (
            "both preregistered confidence-interval conditions hold"
            if verdict["verdict"] == "PASS" else
            "one or both preregistered confidence-interval conditions fail"
        )
    return rows, paired_rows, bootstrap_arrays, verdict


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, newline="", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_per_trajectory(results: list[dict[str, Any]], output: Path) -> None:
    import pandas as pd
    frames = []
    for result in results:
        data = {
            "variant": result["variant"], "epoch": result["epoch"],
            "image_id": result["image_ids"], "noise_id": result["noise_ids"],
        }
        data.update(result["metrics"])
        frames.append(pd.DataFrame(data))
    frame = pd.concat(frames, ignore_index=True)
    path = output / "per_trajectory_metrics.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def make_plots(results: list[dict[str, Any]], summary: list[dict[str, Any]],
               output: Path, bootstrap_seed: int) -> dict[str, int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(summary)
    colors = {"none": "#4c78a8", "dot": "#f58518", "direct": "#54a24b"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANTS:
        part = frame[frame.variant == variant].sort_values("epoch")
        ax.plot(part.epoch, part.ordering_accuracy, marker="o", label=variant,
                color=colors[variant])
        ax.fill_between(part.epoch, part.ordering_ci_lower, part.ordering_ci_upper,
                        color=colors[variant], alpha=.18)
    ax.axvline(8, color="black", linestyle="--", alpha=.55, label="primary epoch 8")
    ax.set(xlabel="epoch", ylabel="pairwise ordering accuracy", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "ordering_accuracy_by_epoch.png", dpi=180)
    plt.close(fig)

    epoch8 = {r["variant"]: r for r in results if r["epoch"] == 8}
    exclusions: dict[str, int] = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(bootstrap_seed)
    for variant in VARIANTS:
        energy = epoch8[variant]["canonical_energy"]
        denominator = energy[:, 0] - energy[:, -1]
        threshold = max(1e-12, 1e-8 * np.nanmedian(np.abs(denominator)))
        valid = np.isfinite(denominator) & (np.abs(denominator) > threshold)
        exclusions[variant] = int((~valid).sum())
        normalized = (energy[valid] - energy[valid, -1:]) / denominator[valid, None]
        mean = normalized.mean(0)
        draws = rng.integers(0, len(normalized), size=(1000, len(normalized)))
        boot = normalized[draws].mean(1)
        lo, hi = np.quantile(boot, [0.025, .975], axis=0)
        ax.plot(epoch8[variant]["gamma"], mean, label=variant, color=colors[variant])
        ax.fill_between(epoch8[variant]["gamma"], lo, hi, color=colors[variant], alpha=.18)
    ax.set(xlabel="gamma (noise to clean)", ylabel="normalized canonical energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "mean_normalized_energy_epoch08.png", dpi=180)
    plt.close(fig)

    distribution = pd.concat([
        pd.DataFrame({"variant": variant,
                      "ordering_accuracy": epoch8[variant]["metrics"]["ordering_accuracy"]})
        for variant in VARIANTS
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=distribution, x="variant", y="ordering_accuracy",
                   palette=colors, cut=0, ax=ax)
    fig.tight_layout()
    fig.savefig(plots / "ordering_distribution_epoch08.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANTS:
        values = epoch8[variant]["directional_alignment_by_gamma"].mean(0)
        ax.plot(epoch8[variant]["gamma"], values, label=variant, color=colors[variant])
    ax.set(xlabel="gamma", ylabel="cos(canonical field, epsilon - x)", ylim=(-1, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "directional_alignment_by_gamma.png", dpi=180)
    plt.close(fig)
    return exclusions


def make_validation_plot(validation: dict[str, dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant in ("dot", "direct"):
        values = [validation[variant][str(points)]["mean_absolute_discrepancy"]
                  for points in (21, 101)]
        ax.plot([21, 101], values, marker="o", label=variant)
    ax.set(xlabel="gamma grid points", ylabel="mean absolute scalar/integral discrepancy",
           yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "plots" / "scalar_line_integral_discrepancy.png", dpi=180)
    plt.close(fig)


def make_examples(results: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epoch8 = {r["variant"]: r for r in results if r["epoch"] == 8}
    scores = {variant: epoch8[variant]["metrics"]["ordering_accuracy"] for variant in VARIANTS}
    direct_minus_dot = scores["direct"] - scores["dot"]
    all_mean = np.mean(np.stack(list(scores.values())), axis=0)
    selections = {
        "direct_outperforms_dot": np.argsort(-direct_minus_dot, kind="stable")[:5],
        "dot_outperforms_direct": np.argsort(direct_minus_dot, kind="stable")[:5],
        "all_models_well": np.argsort(-all_mean, kind="stable")[:5],
        "all_models_fail": np.argsort(all_mean, kind="stable")[:5],
    }
    example_dir = output / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(example_dir / "selection.json", {
        name: [int(index) for index in indices] for name, indices in selections.items()
    })
    for name, indices in selections.items():
        fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
        for ax, index in zip(axes, indices):
            for variant in VARIANTS:
                ax.plot(epoch8[variant]["gamma"],
                        epoch8[variant]["canonical_energy"][index],
                        label=variant)
            ax.set_title(f"trajectory {index}; direct-dot={direct_minus_dot[index]:.3f}")
            ax.set_ylabel("canonical implied E")
        axes[-1].set_xlabel("gamma")
        axes[0].legend(ncol=3)
        fig.tight_layout()
        fig.savefig(example_dir / f"{name}.png", dpi=160)
        plt.close(fig)


def write_report(output: Path, records: list[CheckpointRecord],
                 summary: list[dict[str, Any]], paired: list[dict[str, Any]],
                 validation: dict[str, Any], verdict: dict[str, Any],
                 exclusions: dict[str, int], args: argparse.Namespace) -> None:
    epoch8 = [row for row in summary if row["epoch"] == 8]
    epoch8_pairs = [row for row in paired if row["epoch"] == 8]
    available_epochs = sorted({row["epoch"] for row in summary})
    unavailable_epochs = sorted(set(range(1, 9)) - set(available_epochs))
    weakest = min(row["ordering_accuracy"] for row in epoch8) if epoch8 else math.nan
    lines = [
        "# Energy monotonicity of EqM parameterizations",
        "",
        "## Scientific question and hypothesis",
        "",
        "Does native scalar-energy parameterization improve held-out noise-to-data "
        "energy monotonicity over dot scalarization while remaining non-inferior "
        "to ordinary vector-field EqM by a 0.01 absolute margin?",
        "",
        "Epoch 8 is the sole primary checkpoint. Direct passes only when the paired "
        "image-cluster bootstrap lower CI for `direct-dot` is >0 and the lower CI "
        "for `direct-none` is >-0.01.",
        "",
        "## Corruption path and confirmed sign",
        "",
        "`z_gamma = gamma*x + (1-gamma)*epsilon` with 21 evenly spaced points. "
        "The repository trains its returned field toward `(x-epsilon)c(gamma)`, "
        "the sampling direction. Dot differentiates `z·f(z)`; direct returns "
        "`-grad E`. Therefore canonical decreasing energy uses fixed signs "
        "`none=-1`, `dot=-1`, `direct=+1` on the requested raw line integrals. "
        "These signs were derived from source before evaluation.",
        "",
        "## Evaluation bank",
        "",
        f"{args.num_images} held-out images, {args.noises_per_image} Gaussian draws "
        f"per image, seed {args.seed}, no augmentation, deterministic ordering, "
        "training-matched center crop/normalization/VAE encoding. Ground-truth "
        "labels are supplied; CFG is disabled.",
        "",
        "## Checkpoints and field definitions",
        "",
        f"{len(records)} EMA checkpoints were validated for variant, epoch, "
        "architecture, dataset configuration, and corruption setup. The full "
        "manifest with paths, hashes, steps, parameter counts, and run metadata "
        "is in `checkpoint_manifest.json`.",
        "",
        "- none: existing vector output, canonical path gradient = its negative.",
        "- dot: input gradient of per-sample `sum(z*f(z))`, never the raw vector.",
        "- direct: input gradient of exactly one native scalar per sample.",
        "",
        "## Primary metric and bootstrap",
        "",
        "Trajectory-level strict pairwise ordering accuracy over all 210 gamma "
        "pairs. Ties are failures. The 10,000-replicate paired bootstrap samples "
        "2,048 image clusters and always includes both associated noises, using "
        f"seed {args.bootstrap_seed}.",
        "",
        "## Epoch-8 primary results",
        "",
        "| variant | ordering | 95% CI | adjacent | perfect | Spearman | drop | ties | NaN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in epoch8:
        lines.append(
            f"| {row['variant']} | {row['ordering_accuracy']:.6f} | "
            f"[{row['ordering_ci_lower']:.6f}, {row['ordering_ci_upper']:.6f}] | "
            f"{row['adjacent_step_accuracy']:.6f} | {row['perfect_trajectory_rate']:.6f} | "
            f"{row['spearman_correlation']:.6f} | {row['total_energy_drop']:.6g} | "
            f"{row['tie_rate']:.6g} | {row['nan_rate']:.6g} |"
        )
    lines += [
        "",
        "| comparison | difference | paired 95% CI |",
        "|---|---:|---:|",
    ]
    for row in epoch8_pairs:
        lines.append(f"| {row['comparison']} | {row['mean_difference']:.6f} | "
                     f"[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |")
    lines += [
        "",
        f"## Verdict: {verdict['verdict']}",
        "",
        verdict["reason"] + ".",
    ]
    if weakest < 0.75:
        lines.append(
            f"The weakest epoch-8 absolute ordering accuracy is {weakest:.3f}; "
            "any comparative success is weak in absolute monotonicity."
        )
    lines += [
        "",
        "## Checkpoint learning curves",
        "",
        f"Available epochs in the requested 1-8 window are {available_epochs}; "
        f"epochs with no retained checkpoint are {unavailable_epochs}. All retained "
        "checkpoints use the same frozen bank. Machine-readable values are in "
        "`per_epoch_summary.csv` and paired differences in `paired_differences.csv`; "
        "all pre-8 epochs are diagnostic and were not used for selection.",
        "",
        "## Numerical validation",
        "",
        "Dot and direct raw scalar differences were compared with trapezoidal raw "
        "line integrals on 128 trajectories at 21 and 101 points.",
        "",
        "```json",
        json.dumps(validation, indent=2, default=_json_default),
        "```",
        "",
        "## Failures, exclusions, and limitations",
        "",
        f"Near-zero endpoint denominators excluded from only the normalized plot: "
        f"{exclusions}. They remain in the primary metric. NaN/tie/zero-field rates "
        "are retained in the CSV and per-trajectory parquet. Monotonicity is a "
        "landscape diagnostic; it does not establish generation or composition gains.",
        "",
        "## Concise interpretation for Yilun",
        "",
        f"Checkpoint-only held-out evaluation gives **{verdict['verdict']}** under "
        "the preregistered epoch-8 paired-bootstrap rule. See the epoch-8 table "
        "above for absolute scores and paired confidence intervals; no checkpoint "
        "selection, retraining, guidance, sampler change, or weight modification "
        "was performed.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--none-run", type=Path, required=True)
    parser.add_argument("--dot-run", type=Path, required=True)
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--num-images", type=int, default=2048)
    parser.add_argument("--noises-per-image", type=int, default=2)
    parser.add_argument("--num-gamma-points", type=int, default=21)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--bootstrap-seed", type=int, default=23456)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--validation-subset", type=int, default=128)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="build the fixed bank and run dense epoch-primary validation")
    parser.add_argument("--evaluate-only", nargs=2, metavar=("VARIANT", "EPOCH"),
                        help="evaluate and cache exactly one checkpoint")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="require all checkpoint caches and generate final outputs")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_gamma_points < 2:
        raise ValueError("--num-gamma-points must be at least 2")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("validation", "plots", "examples", "logs", "cache"):
        (output / directory).mkdir(exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gamma = np.linspace(0.0, 1.0, args.num_gamma_points, dtype=np.float64)

    print("discovering and validating checkpoints", flush=True)
    manifest_path = output / "checkpoint_manifest.json"
    if manifest_path.exists() and not args.force:
        records = [CheckpointRecord(**item) for item in json.loads(manifest_path.read_text())]
        requested = {(variant, epoch) for variant in VARIANTS for epoch in args.epochs}
        found = {(record.variant, record.epoch) for record in records}
        if found != requested:
            raise ValueError(
                f"cached manifest checkpoint set {sorted(found)} != requested {sorted(requested)}"
            )
        for record in records:
            if not Path(record.checkpoint_path).exists():
                raise FileNotFoundError(f"cached checkpoint disappeared: {record.checkpoint_path}")
    else:
        records = []
        for variant, run in zip(VARIANTS, (args.none_run, args.dot_run, args.direct_run)):
            records.extend(discover_checkpoints(run, variant, args.epochs, not args.raw_weights))
    validate_cross_variant_manifest(records)
    atomic_json(manifest_path, [asdict(record) for record in records])
    config = vars(args).copy()
    for runtime_key in ("prepare_only", "evaluate_only", "aggregate_only", "force"):
        config.pop(runtime_key, None)
    config.update({
        "gamma": gamma, "variants": VARIANTS, "primary_epoch": 8,
        "canonical_energy_signs": CANONICAL_SIGNS,
        "sign_source": "transport.training_losses and models.EqM.forward",
        "mixed_precision": False, "classifier_free_guidance": False,
        "weights_used": "raw" if args.raw_weights else "ema",
    })
    atomic_json(output / "config.json", config)
    reference = records[0]
    bank = build_evaluation_bank(
        args.data_path, output, args.num_images, args.noises_per_image,
        args.seed, reference.image_size, args.vae, device, args.force,
    )

    validation: dict[str, Any] = {}
    validation_path = output / "validation" / "summary.json"
    if args.aggregate_only:
        if not validation_path.exists():
            raise FileNotFoundError(f"missing prepared validation: {validation_path}")
        validation = json.loads(validation_path.read_text())
    elif not args.evaluate_only:
        for variant in ("dot", "direct"):
            record = next(record for record in records
                          if record.variant == variant and record.epoch == max(args.epochs))
            validation[variant] = dense_validation(
                record, bank, output, args.batch_size, device, dtype,
                min(args.validation_subset, args.num_images), args.force,
            )
        atomic_json(validation_path, validation)
        make_validation_plot(validation, output)
    if args.prepare_only:
        print(f"prepared evaluation bank and validation under {output}", flush=True)
        return

    results = []
    selected_records = sorted(records, key=lambda item: (item.epoch, VARIANTS.index(item.variant)))
    if args.evaluate_only:
        selected_variant, selected_epoch = args.evaluate_only
        if selected_variant not in VARIANTS:
            raise ValueError(f"unknown --evaluate-only variant {selected_variant}")
        selected_records = [
            record for record in selected_records
            if record.variant == selected_variant and record.epoch == int(selected_epoch)
        ]
        if len(selected_records) != 1:
            raise ValueError(f"--evaluate-only resolved {len(selected_records)} checkpoints")
    for record in selected_records:
        cache_path = output / "cache" / f"{record.variant}_epoch{record.epoch:02d}.pt"
        if args.aggregate_only:
            if not cache_path.exists():
                raise FileNotFoundError(f"missing checkpoint cache: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached["checkpoint_sha256"] != record.sha256:
                raise ValueError(f"cache/checkpoint hash mismatch: {cache_path}")
            results.append(cached)
        else:
            results.append(evaluate_checkpoint(
                record, bank, gamma, output, args.batch_size, device, dtype, args.force
            ))
    if args.evaluate_only:
        print(f"cached {selected_records[0].variant} epoch {selected_records[0].epoch}", flush=True)
        return
    summary, paired, bootstrap, verdict = summarize(
        results, args.bootstrap_replicates, args.bootstrap_seed, output
    )
    write_csv(output / "summary.csv", [row for row in summary if row["epoch"] == 8])
    write_csv(output / "per_epoch_summary.csv", summary)
    write_csv(output / "paired_differences.csv", paired)
    write_per_trajectory(results, output)
    np.savez_compressed(output / "bootstrap_results.npz", **bootstrap)
    atomic_json(output / "verdict.json", verdict)
    exclusions = make_plots(results, summary, output, args.bootstrap_seed)
    make_examples(results, output)
    write_report(output, records, summary, paired, validation, verdict, exclusions, args)
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
