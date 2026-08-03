"""Exact, resumable paired continuation trainer for the field-shaping study."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF

from masked_field_shaping.checkpointing import validate_base_checkpoint
from masked_field_shaping.corruption import bernoulli_pixel_corruption
from models import EqM_models
from transport import create_transport

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


SCIENTIFIC_FIELDS = (
    "base_checkpoint",
    "base_epoch",
    "continuation_epochs",
    "continuation_updates",
    "corruption_mode",
    "masked_example_probability",
    "min_mask_ratio",
    "max_mask_ratio",
    "mask_type",
    "mask_fill",
    "training_seed",
    "mask_seed",
    "preserve_optimizer_state",
    "model",
    "image_size",
    "num_classes",
    "vae",
    "global_batch_size",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "ema_decay",
    "path_type",
    "prediction",
    "loss_weight",
)


def canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class DeterministicImageFolder(Dataset):
    """ImageFolder with the repository crop/flip but restart-stable flip draws."""

    def __init__(self, root: str, image_size: int, seed: int):
        base = ImageFolder(root)
        self.samples = base.samples
        self.classes = base.classes
        self.loader = base.loader
        self.image_size = image_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = center_crop_arr(self.loader(path), self.image_size)
        # Stateless 50% horizontal flip. This is the same augmentation family
        # as train.py and makes mid-epoch retries independent of worker RNG.
        key = f"{self.seed}:{self.epoch}:{index}".encode("ascii")
        flip = hashlib.sha256(key).digest()[0] < 128
        if flip:
            image = TF.hflip(image)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, [0.5] * 3, [0.5] * 3)
        return tensor, target, index


@torch.no_grad()
def update_ema(ema_model, model, decay: float) -> None:
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, parameter in model_params.items():
        ema_params[name].mul_(decay).add_(parameter.data, alpha=1.0 - decay)


def _finite_gradient_norm(parameters) -> torch.Tensor:
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        contribution = parameter.grad.detach().float().square().sum()
        squared = contribution if squared is None else squared + contribution
    if squared is None:
        raise RuntimeError("no model gradients were produced")
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError(f"non-finite gradient norm: {norm.item()}")
    return norm


def freeze_module(module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def _capture_rank_rng(decision_generator, mask_generator) -> dict:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "decision": decision_generator.get_state(),
        "mask": mask_generator.get_state(),
    }


def _restore_rank_rng(state: dict, decision_generator, mask_generator) -> None:
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    decision_generator.set_state(state["decision"])
    mask_generator.set_state(state["mask"])


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _save_checkpoint(
    *,
    output_dir: Path,
    model,
    ema,
    optimizer,
    config,
    initial_state,
    step,
    continuation_updates_completed,
    next_epoch,
    next_batch,
    steps_per_epoch,
    decision_generator,
    mask_generator,
    final=False,
):
    local_rng = _capture_rank_rng(decision_generator, mask_generator)
    gathered = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
    dist.gather_object(local_rng, gathered, dst=0)
    if dist.get_rank() == 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": optimizer.state_dict(),
            "epoch": next_epoch,
            "step": step,
            "continuation_updates_completed": continuation_updates_completed,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "steps_per_epoch": steps_per_epoch,
            "rank_rng_states": gathered,
            "config": config,
            "config_sha256": canonical_hash({key: config[key] for key in SCIENTIFIC_FIELDS}),
            "initial_state": initial_state,
        }
        name = "final.pt" if final else f"step{step:07d}.pt"
        _atomic_torch_save(checkpoint, output_dir / "checkpoints" / name)
        _write_json(
            output_dir / "checkpoint_state.json",
            {
                "latest": str(output_dir / "checkpoints" / name),
                "step": step,
                "continuation_updates_completed": continuation_updates_completed,
                "next_epoch": next_epoch,
                "next_batch": next_batch,
                "final": final,
            },
        )
    dist.barrier()


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text())
    missing = [field for field in SCIENTIFIC_FIELDS if field not in config]
    if missing:
        raise ValueError(f"configuration missing required fields: {missing}")
    if config["corruption_mode"] == "gaussian":
        if config["masked_example_probability"] != 0.0:
            raise ValueError("Gaussian control must set masked_example_probability=0")
    elif config["corruption_mode"] == "gaussian_or_pixel_mask":
        if config["masked_example_probability"] != 0.25:
            raise ValueError("treatment masked_example_probability is locked to 0.25")
        if config["mask_type"] != "bernoulli_pixel" or config["mask_fill"] != "gaussian_noise":
            raise ValueError("treatment mask construction differs from the locked protocol")
    else:
        raise ValueError(f"unsupported corruption mode: {config['corruption_mode']}")
    return config


def main(config_path: str, resume_path: str | None = None) -> None:
    from diffusers.models import AutoencoderKL

    config = load_config(config_path)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_index = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    effective_batch = (
        int(config["per_device_batch_size"])
        * world_size
        * int(config["gradient_accumulation_steps"])
    )
    if effective_batch != int(config["global_batch_size"]):
        raise ValueError(
            f"effective global batch mismatch: {effective_batch} != {config['global_batch_size']}"
        )
    rank_seed = int(config["training_seed"]) * world_size + rank
    torch.manual_seed(rank_seed)
    decision_generator = torch.Generator(device=device).manual_seed(
        int(config["mask_seed"]) + 10_000 * rank
    )
    mask_generator = torch.Generator(device=device).manual_seed(
        int(config["mask_seed"]) + 1_000_003 + 10_000 * rank
    )

    output_dir = Path(config["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        resolved_path = output_dir / "resolved_configuration.json"
        if resolved_path.exists() and json.loads(resolved_path.read_text()) != config:
            raise RuntimeError("refusing to mutate an existing arm's resolved configuration")
        _write_json(resolved_path, config)
        (output_dir / "resume_command.txt").write_text(
            f"python -m torch.distributed.run --nproc_per_node={world_size} "
            f"-m masked_field_shaping.train_continuation --config {config_path} "
            f"--resume {output_dir}/checkpoints/<latest-valid>.pt\n"
        )
    dist.barrier()

    source_path = resume_path or config["base_checkpoint"]
    source = torch.load(source_path, map_location="cpu")
    if resume_path:
        expected_hash = canonical_hash({key: config[key] for key in SCIENTIFIC_FIELDS})
        if source.get("config_sha256") != expected_hash:
            raise RuntimeError("resume checkpoint scientific configuration mismatch")
        initial_state = source["initial_state"]
    else:
        initial_state = validate_base_checkpoint(source, int(config["base_epoch"]))

    model = EqM_models[config["model"]](
        input_size=int(config["image_size"]) // 8,
        num_classes=int(config["num_classes"]),
        uncond=True,
        ebm="none",
    ).to(device)
    ema = EqM_models[config["model"]](
        input_size=int(config["image_size"]) // 8,
        num_classes=int(config["num_classes"]),
        uncond=True,
        ebm="none",
    ).to(device)
    model.load_state_dict(source["model"])
    ema.load_state_dict(source["ema"])
    freeze_module(ema)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    if config["preserve_optimizer_state"]:
        optimizer.load_state_dict(source["opt"])
    model = DDP(model, device_ids=[device_index])
    model.train()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{config['vae']}").to(device)
    freeze_module(vae)
    if any(parameter.requires_grad for parameter in vae.parameters()):
        raise RuntimeError("VAE must remain frozen")

    dataset = DeterministicImageFolder(
        config["data_path"], int(config["image_size"]), int(config["training_seed"])
    )
    if len(dataset.classes) != 1000 or len(dataset) != int(config.get("expected_training_images", 1_281_167)):
        raise RuntimeError(
            f"unexpected ImageNet training set: classes={len(dataset.classes)} images={len(dataset)}"
        )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(config["training_seed"])
    )
    loader_generator = torch.Generator().manual_seed(91_337 + rank)
    loader = DataLoader(
        dataset,
        batch_size=int(config["per_device_batch_size"]),
        sampler=sampler,
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )
    accumulation_steps = int(config["gradient_accumulation_steps"])
    steps_per_epoch = len(loader) // accumulation_steps
    usable_batches_per_epoch = steps_per_epoch * accumulation_steps
    expected_updates = steps_per_epoch * int(config["continuation_epochs"])
    is_smoke = bool(config.get("smoke", False))
    if (not is_smoke and int(config["continuation_updates"]) != expected_updates) or (
        is_smoke and not 100 <= int(config["continuation_updates"]) <= expected_updates
    ):
        raise RuntimeError(
            f"continuation budget mismatch: config={config['continuation_updates']} computed={expected_updates}"
        )

    transport = create_transport(
        path_type=config["path_type"],
        prediction=config["prediction"],
        loss_weight=config["loss_weight"],
        corruption_mode="gaussian",
    )
    if resume_path:
        start_epoch = int(source["next_epoch"])
        start_batch = int(source["next_batch"])
        global_step = int(source["step"])
        completed_updates = int(source["continuation_updates_completed"])
        _restore_rank_rng(source["rank_rng_states"][rank], decision_generator, mask_generator)
    else:
        start_epoch = int(config["base_epoch"])
        start_batch = 0
        global_step = int(source["step"])
        completed_updates = 0

    if rank == 0:
        _write_json(
            output_dir / "initial_state.json",
            {
                **initial_state,
                "source_checkpoint": config["base_checkpoint"],
                "restored_optimizer": bool(config["preserve_optimizer_state"]),
                "restored_scheduler": False,
                "scheduler_limitation": "repository uses constant learning rate and serializes no scheduler",
                "world_size": world_size,
                "steps_per_epoch": steps_per_epoch,
            },
        )
    dist.barrier()

    metrics_path = output_dir / "training_metrics.jsonl"
    metrics_file = metrics_path.open("a", buffering=1) if rank == 0 else None
    began = time.time()
    interval_loss = 0.0
    interval_examples = 0
    interval_masked = 0
    interval_requested = 0.0
    interval_realized = 0.0
    last_loss = math.nan
    target_updates = int(config["continuation_updates"])
    target_epoch = int(config["base_epoch"]) + int(config["continuation_epochs"])
    stop = False
    final_next_epoch = start_epoch
    final_next_batch = start_batch

    for epoch in range(start_epoch, target_epoch):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        skip = start_batch if epoch == start_epoch else 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (pixels, labels, _indices) in enumerate(loader):
            if batch_index >= usable_batches_per_epoch:
                break
            if batch_index < skip:
                continue
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                clean_latent = vae.encode(pixels).latent_dist.sample().mul_(0.18215)

            selector = None
            masked_latent = None
            selected_count = 0
            requested_sum = 0.0
            realized_sum = 0.0
            if config["corruption_mode"] == "gaussian_or_pixel_mask":
                selector = torch.rand(
                    (pixels.shape[0],), device=device, generator=decision_generator
                ) < float(config["masked_example_probability"])
                pixel_mask = bernoulli_pixel_corruption(
                    pixels,
                    float(config["min_mask_ratio"]),
                    float(config["max_mask_ratio"]),
                    generator=mask_generator,
                )
                with torch.no_grad():
                    masked_latent = vae.encode(pixel_mask.corrupted).latent_dist.sample(
                        generator=mask_generator
                    ).mul_(0.18215)
                selected_count = int(selector.sum().item())
                if selected_count:
                    requested_sum = float(pixel_mask.requested_missing_ratio[selector].sum().item())
                    realized_sum = float(pixel_mask.realized_missing_ratio[selector].sum().item())

            model_kwargs = {"y": labels, "return_act": False, "train": True}
            accumulation_index = batch_index % accumulation_steps
            synchronizing = accumulation_index == accumulation_steps - 1
            sync_context = contextlib.nullcontext() if synchronizing else model.no_sync()
            with sync_context:
                losses = transport.training_losses(
                    model,
                    clean_latent,
                    model_kwargs,
                    corruption_endpoint=masked_latent,
                    corruption_selector=selector,
                )
                loss = losses["loss"].mean()
                scaled_loss = loss / accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss before step {global_step + 1}: {loss.item()}")
                scaled_loss.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step + 1}: {loss.item()}")
            last_loss = float(loss.detach())
            interval_loss += last_loss * pixels.shape[0]
            interval_examples += pixels.shape[0]
            interval_masked += selected_count
            interval_requested += requested_sum
            interval_realized += realized_sum
            if not synchronizing:
                continue
            next_step = global_step + 1
            gradient_norm = None
            if next_step % int(config["gradient_log_every"]) == 0:
                gradient_norm = _finite_gradient_norm(model.parameters())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model.module, float(config["ema_decay"]))

            global_step = next_step
            completed_updates += 1

            if completed_updates % int(config["log_every"]) == 0:
                values = torch.tensor(
                    [
                        interval_loss,
                        interval_examples,
                        interval_masked,
                        interval_requested,
                        interval_realized,
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(values)
                if rank == 0:
                    elapsed = time.time() - began
                    masked = values[2].item()
                    record = {
                        "time": time.time(),
                        "epoch": epoch,
                        "batch_in_epoch": batch_index,
                        "step": global_step,
                        "continuation_updates": completed_updates,
                        "loss": values[0].item() / values[1].item(),
                        "gradient_norm": float(gradient_norm) if gradient_norm is not None else None,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "effective_mask_fraction": masked / values[1].item(),
                        "mean_requested_mask_ratio": values[3].item() / masked if masked else None,
                        "mean_realized_mask_ratio": values[4].item() / masked if masked else None,
                        "updates_per_second": completed_updates / elapsed,
                        "max_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                    metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                    print(json.dumps(record, sort_keys=True), flush=True)
                interval_loss = 0.0
                interval_examples = 0
                interval_masked = 0
                interval_requested = 0.0
                interval_realized = 0.0

            next_epoch = epoch
            next_batch = batch_index + 1
            if next_batch >= usable_batches_per_epoch:
                next_epoch = epoch + 1
                next_batch = 0
            final_next_epoch = next_epoch
            final_next_batch = next_batch
            if completed_updates % int(config["checkpoint_every"]) == 0:
                _save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    config=config,
                    initial_state=initial_state,
                    step=global_step,
                    continuation_updates_completed=completed_updates,
                    next_epoch=next_epoch,
                    next_batch=next_batch,
                    steps_per_epoch=steps_per_epoch,
                    decision_generator=decision_generator,
                    mask_generator=mask_generator,
                )
            if completed_updates >= target_updates:
                stop = True
                break
        start_batch = 0
        if stop:
            break

    if completed_updates != target_updates:
        raise RuntimeError(f"training ended at {completed_updates}, expected exactly {target_updates}")
    _save_checkpoint(
        output_dir=output_dir,
        model=model,
        ema=ema,
        optimizer=optimizer,
        config=config,
        initial_state=initial_state,
        step=global_step,
        continuation_updates_completed=completed_updates,
        next_epoch=final_next_epoch,
        next_batch=final_next_batch,
        steps_per_epoch=steps_per_epoch,
        decision_generator=decision_generator,
        mask_generator=mask_generator,
        final=True,
    )
    if rank == 0:
        completion = {
            "status": "completed",
            "base_epoch": config["base_epoch"],
            "arm": config["arm"],
            "final_checkpoint": str(output_dir / "checkpoints" / "final.pt"),
            "optimizer_steps": completed_updates,
            "global_step": global_step,
            "final_training_loss": last_loss,
            "runtime_hours": (time.time() - began) / 3600.0,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json(output_dir / "completion.json", completion)
        metrics_file.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    main(arguments.config, arguments.resume)
