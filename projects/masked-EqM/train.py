# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for EqM using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import json
import logging
import os
from tqdm import tqdm
from models import EqM_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import torch.nn.functional as F

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new EqM model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    n_gpus = torch.cuda.device_count()
    # disable flash for energy training
    if args.ebm != 'none':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    print(f"Found {n_gpus} GPUs, trying to use device index {device}")
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}-ebm-{args.ebm}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        if args.wandb:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm
    ).to(device)

    # Note that parameter initialization is done within the EqM constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    resume_epoch = 0
    resume_step = None
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        try:
            if 'model' in state_dict.keys():
                model.load_state_dict(state_dict["model"])
                ema.load_state_dict(state_dict["ema"])
                opt.load_state_dict(state_dict["opt"])
                resume_epoch = int(state_dict.get("epoch", 0))
                resume_step = state_dict.get("step")
            else:
                model.load_state_dict(state_dict)
                ema.load_state_dict(state_dict)
        except RuntimeError as error:
            if args.ebm == 'direct':
                raise RuntimeError(
                    "--ebm direct requires a direct-energy checkpoint. "
                    "Vector-output EqM checkpoints have final_layer.* parameters "
                    "and cannot load into the energy_head.* architecture."
                ) from error
            raise

        ema = ema.to(device)
        model = model.to(device)
    requires_grad(ema, False)
    # The direct scalar head returns an input gradient produced by an
    # inner autograd.grad call.  DDP cannot reliably discover every scalar
    # head edge from that nested graph on the first iteration (the zero
    # initialized projection is especially important here), so enable its
    # unused-parameter traversal for this mode only.  Existing vector modes
    # retain their original DDP behavior and overhead.
    model = DDP(
        model,
        device_ids=[device],
        find_unused_parameters=(args.ebm == "direct"),
    )
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        corruption_mode=args.corruption_mode,
        mask_prob=args.mask_prob,
        fourier_cutoff=args.fourier_cutoff,
        blur_sigma=args.blur_sigma,
        downsample_factor=args.downsample_factor,
        gaussian_weight=args.gaussian_weight,
        mask_weight=args.mask_weight,
        blur_weight=args.blur_weight,
        fourier_weight=args.fourier_weight,
        downsample_weight=args.downsample_weight,
        structured_mask_weight=args.structured_mask_weight,
    )  # default: velocity;
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Epoch-boundary checkpoints from the original one-epoch screen predate
    # the explicit step field. Infer their completed step count from the
    # deterministic loader length; newer checkpoints carry the exact value.
    if resume_step is None:
        resume_step = resume_epoch * len(loader)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = int(resume_step)
    max_train_steps = (
        train_steps + args.max_steps
        if args.max_steps is not None
        else None
    )
    log_steps = 0
    running_loss = 0
    start_time = time()
    grad_metrics_file = None
    if rank == 0 and args.grad_log_every > 0:
        grad_metrics_path = os.path.join(experiment_dir, "gradient_metrics.jsonl")
        grad_metrics_file = open(
            grad_metrics_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward
    
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        reached_max_steps = False
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y, return_act=args.disp, train=True)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            grad_norm = None
            unclipped_grad_norm = None
            if args.max_grad_norm is not None:
                unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                    error_if_nonfinite=True,
                )
                grad_norm = unclipped_grad_norm
            elif args.grad_log_every > 0 and (train_steps + 1) % args.grad_log_every == 0:
                squared_norm = torch.zeros((), device=device)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        squared_norm += parameter.grad.detach().float().square().sum()
                grad_norm = squared_norm.sqrt()

            if grad_norm is not None and rank == 0 and args.grad_log_every > 0:
                head_squared_norm = torch.zeros((), device=device)
                backbone_squared_norm = torch.zeros((), device=device)
                for name, parameter in model.module.named_parameters():
                    if parameter.grad is None:
                        continue
                    contribution = parameter.grad.detach().float().square().sum()
                    if name.startswith("energy_head."):
                        head_squared_norm += contribution
                    else:
                        backbone_squared_norm += contribution
                next_step = train_steps + 1
                if args.grad_log_every > 0 and next_step % args.grad_log_every == 0:
                    record = {
                        "step": next_step,
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm.detach()),
                        "head_grad_norm": float(head_squared_norm.sqrt()),
                        "backbone_grad_norm": float(backbone_squared_norm.sqrt()),
                        "max_grad_norm": args.max_grad_norm,
                        "clipped": bool(
                            unclipped_grad_norm is not None
                            and unclipped_grad_norm > args.max_grad_norm
                        ),
                        "learning_rate": opt.param_groups[0]["lr"],
                    }
                    grad_metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save EqM checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "step": train_steps,
                        "epoch": epoch + 1,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if max_train_steps is not None and train_steps >= max_train_steps:
                reached_max_steps = True
                break

        if reached_max_steps:
            break

        # Save an epoch-boundary checkpoint regardless of step/ckpt_every
        # alignment -- resume via --ckpt does not restore train_steps/epoch/
        # RNG state, so matched multi-epoch comparisons rely on retraining
        # from scratch through a fixed epoch count and reading these off
        # directly, not on resuming from a step checkpoint mid-run.
        if rank == 0 and (args.save_epochs is None or epoch + 1 in args.save_epochs):
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args,
                "epoch": epoch + 1,
                "step": train_steps,
            }
            epoch_checkpoint_path = f"{checkpoint_dir}/epoch{epoch + 1:02d}.pt"
            torch.save(checkpoint, epoch_checkpoint_path)
            logger.info(f"Saved epoch checkpoint to {epoch_checkpoint_path}")
        dist.barrier()

    if grad_metrics_file is not None:
        grad_metrics_file.close()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train EqM-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--save-epochs", type=str, default=None,
                        help="Comma-separated epoch numbers to save; default saves every epoch")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--grad-log-every",
        type=int,
        default=0,
        help="write gradient diagnostics every N optimizer steps; 0 disables them",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="optional global gradient-norm clipping threshold",
    )
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional hard cap for bounded smoke/benchmark runs",
    )
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom EqM checkpoint")
    parser.add_argument("--disp", action="store_true",
                        help="Toggle to enable Dispersive Loss")
    parser.add_argument("--uncond", type=bool, default=True,
                        help="disable/enable noise conditioning")
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean", "direct"], default="none",
                        help="'direct' uses a scalar E_theta(x) and returns -grad_x E_theta(x)")

    parse_transport_args(parser)
    args = parser.parse_args()
    if args.grad_log_every < 0:
        parser.error("--grad-log-every must be non-negative")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    args.save_epochs = None if args.save_epochs is None else {
        int(value) for value in args.save_epochs.split(",") if value.strip()
    }
    main(args)
