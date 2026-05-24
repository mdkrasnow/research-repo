#!/usr/bin/env python3
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ImageNet 256x256 EqM training with optional DG-ANM (Differential-Geometry-Guided
Adversarial Negative Mining).

Fork of eqm-upstream/train.py with DG-ANM integration for latent-space mining.

Usage (vanilla EqM):
    torchrun --nproc_per_node=4 projects/diff-EqM/experiments/train_imagenet.py \
        --data-path /data/imagenet/train --model EqM-B/2 --epochs 80

Usage (DG-ANM):
    torchrun --nproc_per_node=4 projects/diff-EqM/experiments/train_imagenet.py \
        --data-path /data/imagenet/train --model EqM-B/2 --epochs 80 \
        --use-mining --gamma 2.0 --mine-every 4
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path setup: import from eqm-upstream/
# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
eqm_upstream = script_dir.parent / "eqm-upstream"
sys.path.insert(0, str(eqm_upstream))

import torch
# TF32 flags for A100 performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import EqM_models
from download import find_model
from transport import create_transport, Sampler
from train_utils import parse_transport_args
from diffusers.models import AutoencoderKL
try:
    import wandb_utils
except ImportError:
    wandb_utils = None


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """End DDP training."""
    dist.destroy_process_group()


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """Center cropping implementation from ADM."""
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
#                     DG-ANM: Geometry Estimation & Mining                      #
#################################################################################

def estimate_local_geometry(features, k=10, r=None):
    """
    Estimate local tangent/normal decomposition via feature-space PCA.

    Args:
        features: (B, D) feature vectors for a minibatch
        k: number of neighbors for local PCA
        r: tangent rank (if None, auto-select via 95% explained variance)

    Returns:
        P_T: (B, D, D) tangent projectors
        P_N: (B, D, D) normal projectors
    """
    B, D = features.shape
    device = features.device

    # Pairwise distances
    dists = torch.cdist(features, features)  # (B, B)
    # k-nearest neighbors (excluding self)
    _, nn_idx = dists.topk(k + 1, largest=False, dim=1)
    nn_idx = nn_idx[:, 1:]  # (B, k)

    P_T = torch.zeros(B, D, D, device=device)
    P_N = torch.zeros(B, D, D, device=device)

    for i in range(B):
        neighbors = features[nn_idx[i]]  # (k, D)
        local_mean = neighbors.mean(dim=0)
        centered = neighbors - local_mean  # (k, D)

        # Local covariance with regularization for numerical stability
        cov = (centered.T @ centered) / k  # (D, D)
        cov = cov + 1e-6 * torch.eye(D, device=device)  # prevent ill-conditioning

        # Eigendecomposition (ascending order)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)

        # Auto-select rank if not specified
        if r is None:
            cumvar = eigvals.cumsum(0) / (eigvals.sum() + 1e-8)
            r_local = max(1, (cumvar < 0.95).sum().item() + 1)
        else:
            r_local = r

        r_local = min(r_local, D)
        U_T = eigvecs[:, :r_local]  # (D, r)
        P_T[i] = U_T @ U_T.T
        P_N[i] = torch.eye(D, device=device) - P_T[i]

    return P_T, P_N


def mine_negatives(
    model, x, y, features, P_N, P_T,
    epsilon=0.1,
    mining_steps=3,
    mining_lr=0.01,
    lambda_N=1.0,
    lambda_T=1.0,
    lambda_W=0.1,
    device=None,
):
    """
    Generate adversarial negatives via projected gradient ascent in normal space.

    Operates in latent space: x is (B, 4, 32, 32) VAE latents.
    Geometry is estimated in feature space: features is (B, hidden_size).

    Args:
        model: EqM model (unwrapped, eval mode). Grad flows through delta only.
        x: anchor latents (B, 4, 32, 32)
        y: class labels (B,)
        features: (B, D) feature vectors at anchors
        P_N: (B, D, D) normal projectors
        P_T: (B, D, D) tangent projectors
        epsilon: perturbation budget (L2 ball radius in latent space)
        mining_steps: number of PGA iterations
        mining_lr: step size for signed gradient ascent
        lambda_N: normal displacement weight (maximize)
        lambda_T: tangent displacement weight (minimize)
        lambda_W: weak field weight

    Returns:
        x_neg: (B, 4, 32, 32) mined negatives (detached)
        mining_info: dict with diagnostics
    """
    B = x.shape[0]
    if device is None:
        device = x.device

    # Initialize perturbation
    delta = torch.randn_like(x) * 0.01
    delta.requires_grad_(True)

    t_ones = torch.ones(B, device=device)

    # Tracking
    delta_N_norm = torch.zeros(B, device=device)
    delta_T_norm = torch.zeros(B, device=device)
    field_norm = torch.zeros(B, device=device)

    for _ in range(mining_steps):
        x_neg = x.detach() + delta  # delta carries grad

        # Field at negative (grad flows through delta)
        field = model(x_neg, t_ones, y, train=False)
        field_norm = field.flatten(1).norm(dim=1)

        # Feature displacement (grad flows through delta)
        _, neg_acts = model(x_neg, t_ones, y, return_act=True)
        neg_features = neg_acts[-1].mean(dim=1)  # (B, hidden_size)

        delta_phi = neg_features - features.detach()  # (B, D)

        # Project to tangent/normal components
        delta_N = torch.bmm(P_N.detach(), delta_phi.unsqueeze(-1)).squeeze(-1)
        delta_T = torch.bmm(P_T.detach(), delta_phi.unsqueeze(-1)).squeeze(-1)

        delta_N_norm = delta_N.norm(dim=1)
        delta_T_norm = delta_T.norm(dim=1)

        # Mining objective: maximize normal, minimize tangent, encourage weak field
        L_normal = delta_N_norm ** 2
        L_tan = delta_T_norm ** 2
        L_weak = -field_norm

        mining_obj = (
            lambda_N * L_normal
            - lambda_T * L_tan
            + lambda_W * L_weak
        ).mean()

        # Gradient ascent on delta
        grad_delta = torch.autograd.grad(mining_obj, delta, retain_graph=False)[0]

        with torch.no_grad():
            delta = delta + mining_lr * grad_delta.sign()
            # Project back to epsilon ball (in latent space)
            delta_flat_norm = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            delta = delta * torch.clamp(epsilon / (delta_flat_norm + 1e-8), max=1.0)
            delta = delta.detach().requires_grad_(True)

    x_neg = (x + delta).detach()
    mining_info = {
        "avg_normal_component": delta_N_norm.mean().item(),
        "avg_tangent_component": delta_T_norm.mean().item(),
        "avg_field_norm_at_neg": field_norm.mean().item(),
    }
    return x_neg, mining_info


def dganm_negative_loss(
    model, x, x_neg, y,
    margin=5.0,
    rho=0.0,
    rollout_steps=5,
    rollout_step_size=0.01,
    device=None,
):
    """
    Compute the DG-ANM negative loss.

    L_neg = max(0, margin - |g_theta(x^-)|) + rho * L_traj

    This function is called with model in train mode (DDP-wrapped) so gradients
    synchronize across processes.

    Args:
        model: EqM model (DDP-wrapped, train mode)
        x: anchor latents (B, 4, 32, 32)
        x_neg: mined negatives (B, 4, 32, 32), detached
        y: class labels (B,)
        margin: minimum field norm at negatives
        rho: trajectory failure weight
        rollout_steps: steps for trajectory loss
        rollout_step_size: step size for trajectory rollout
    """
    B = x.shape[0]
    if device is None:
        device = x.device
    t_ones = torch.ones(B, device=device)

    # Field at negatives — through DDP model so grads sync
    field_at_neg = model(x_neg, t_ones, y, train=True)
    field_norm = field_at_neg.flatten(1).norm(dim=1)

    # Margin loss: enforce nontrivial restoring force at negatives
    loss_margin = F.relu(margin - field_norm).mean()

    # Trajectory failure loss (optional)
    loss_traj = torch.tensor(0.0, device=device)
    if rho > 0 and rollout_steps > 0:
        u = x_neg.detach().clone()
        for _ in range(rollout_steps):
            with torch.no_grad():
                grad = model(u.detach(), t_ones, y)
            u = u.detach() + grad * rollout_step_size
        loss_traj = F.mse_loss(u, x)

    total = loss_margin + rho * loss_traj
    return total


#################################################################################
#              v02 Score-Repulsion: cosine-contrastive on velocity              #
#################################################################################

def _v02_cosine_contrastive_step(
    model_ddp, model_module, x, y,
    train_eps, mining_epsilon, mining_steps, mining_lr,
    pos_sigma, lambda_pos, lambda_neg, neg_cos_margin,
    device,
):
    """v02 mining + loss in latent space. Returns (loss, diag).

    Differs from v01: (1) uniform t in (eps, 1-eps) instead of t=1,
    (2) PGA descends cosine similarity vs anchor velocity (no feature PCA),
    (3) cosine-contrastive loss with pos+neg legs instead of margin hinge.
    """
    B = x.shape[0]
    # FM forward path at random t (latent space). train_eps may be None when
    # the upstream transport handles its own eps; fall back to 1e-3.
    eps = train_eps if train_eps is not None else 1e-3
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps) + eps
    t_ = t.view(B, 1, 1, 1)
    x0 = torch.randn_like(x)
    x_t = (1.0 - t_) * x0 + t_ * x

    # --- PGA mine negatives by minimizing cos(v_neg, v_anchor) ---
    model_ddp.eval()
    with torch.no_grad():
        v_anchor_eval = model_module(x_t, t, y, train=False)
    delta = torch.randn_like(x_t) * 0.01
    delta.requires_grad_(True)
    for _ in range(mining_steps):
        v_neg = model_module(x_t.detach() + delta, t, y, train=False)
        obj = F.cosine_similarity(
            v_neg.flatten(1), v_anchor_eval.flatten(1).detach(), dim=1
        ).mean()
        grad_d = torch.autograd.grad(obj, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta = delta - mining_lr * grad_d.sign()  # descend cos
            flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            delta = delta * torch.clamp(mining_epsilon / (flat + 1e-8), max=1.0)
            delta = delta.detach().requires_grad_(True)
    x_neg = (x_t + delta).detach()
    x_pos = (x_t + pos_sigma * torch.randn_like(x_t)).detach()
    x_t_anchor = x_t.detach()
    model_ddp.train()

    # --- Cosine-contrastive loss; grads flow through DDP for sync ---
    v_a = model_ddp(x_t_anchor, t, y, train=True)
    v_p = model_ddp(x_pos, t, y, train=True)
    v_n = model_ddp(x_neg, t, y, train=True)
    sim_pos = F.cosine_similarity(v_p.flatten(1), v_a.flatten(1), dim=1)
    sim_neg = F.cosine_similarity(v_n.flatten(1), v_a.flatten(1), dim=1)
    l_pos = (1.0 - sim_pos).mean()
    l_neg = F.relu(sim_neg - neg_cos_margin).mean()
    loss_v02 = lambda_pos * l_pos + lambda_neg * l_neg

    diag = {
        "v02_pos_loss": l_pos.item(),
        "v02_neg_loss": l_neg.item(),
        "v02_neg_field_norm": v_n.flatten(1).norm(dim=1).mean().item(),
    }
    return loss_v02, x_neg, diag


#################################################################################
#       v09 Jacobian: random-noise finite-diff Jacobian regularizer             #
#################################################################################

def _v10_pgd_hard_example_step(
    model_ddp, model_module, x, y, transport,
    train_eps, v10_eps_radius, v10_K, v10_lr,
    device,
):
    """v10 PGD hard-example mining on EqM regression target.

    Mirror of dganm_variants/v10_hard_example.py adapted to IN-1K latents.

    Loss: L_aux = ||f(x_t + δ*) - target||²
        δ* = argmax_{||δ||₂ ≤ ε} ||f(x_t + δ) - target||²
        target = (x - x0) · c(t)  (EqM regression target)

    Returns (loss_v10, diag).
    """
    B = x.shape[0]
    eps = train_eps if train_eps is not None else 1e-3
    # FM forward path at random t — mirrors transport.training_losses internals
    # but exposes target externally for mining.
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps) + eps
    t_ = t.view(B, 1, 1, 1)
    x0 = torch.randn_like(x)
    x_t = (1.0 - t_) * x0 + t_ * x
    ut = (x - x0)
    ct = transport.get_ct(t).view(B, 1, 1, 1)
    target = ct * ut  # EqM energy-compatible regression target

    # --- PGD mine: maximize MSE(f(x_t + δ), target) ---
    model_ddp.eval()
    delta = torch.zeros_like(x_t).normal_(0.0, v10_eps_radius / 2.0)
    # project to L2 ball
    flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
    delta = delta * torch.clamp(v10_eps_radius / (flat + 1e-8), max=1.0)
    for _ in range(v10_K):
        delta = delta.detach().requires_grad_(True)
        pred_neg = model_module(x_t.detach() + delta, t, y, train=False)
        loss_adv = ((pred_neg - target) ** 2).mean()
        g = torch.autograd.grad(loss_adv, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta = delta + v10_lr * g.sign()
            flat = delta.flatten(1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            delta = delta * torch.clamp(v10_eps_radius / (flat + 1e-8), max=1.0)
    delta = delta.detach()
    model_ddp.train()

    # --- Hard-example loss; grads flow through DDP for sync ---
    pred_hard = model_ddp(x_t.detach() + delta, t, y, train=True)
    loss_v10 = F.mse_loss(pred_hard, target)

    diag = {
        "v10_hard_loss": loss_v10.item(),
        "v10_delta_norm": delta.flatten(1).norm(dim=1).mean().item(),
        "v10_field_norm": pred_hard.flatten(1).norm(dim=1).mean().item(),
    }
    return loss_v10, diag


def _v09_jacobian_step(model_ddp, x, y, train_eps, jac_sigma, jac_lambda, device):
    """Architecture-agnostic local-smoothness regularizer.

    L_jac = ||v(x_t + δ) - v(x_t)||² / σ²,  δ ~ N(0, σ²I).
    Encourages local Lipschitz continuity of velocity field.
    No mining, no cosine. 1 extra forward per call.
    """
    B = x.shape[0]
    eps = train_eps if train_eps is not None else 1e-3
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps) + eps
    t_ = t.view(B, 1, 1, 1)
    x0 = torch.randn_like(x)
    x_t = (1.0 - t_) * x0 + t_ * x

    delta = jac_sigma * torch.randn_like(x_t)
    v_anchor = model_ddp(x_t, t, y, train=True)
    v_pert = model_ddp(x_t + delta, t, y, train=True)
    diff = (v_pert - v_anchor).flatten(1)
    l_jac = (diff.pow(2).sum(dim=1) / (jac_sigma * jac_sigma + 1e-8)).mean()
    loss = jac_lambda * l_jac

    diag = {
        "v09_jac_loss": l_jac.item(),
        "v09_anchor_norm": v_anchor.flatten(1).norm(dim=1).mean().item(),
        "v09_aux_over_base": 0.0,  # filled by caller
    }
    return loss, diag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains an EqM model on ImageNet 256x256 with optional DG-ANM mining.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    n_gpus = torch.cuda.device_count()

    # Disable flash attention for energy-based training
    if args.ebm != 'none':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # Setup DDP
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, \
        f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    print(f"Found {n_gpus} GPUs, trying to use device index {device}")
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup experiment folder
    checkpoint_dir = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        mining_tag = "dganm" if args.use_mining else "vanilla"
        experiment_name = (
            f"{experiment_index:03d}-{model_string_name}-"
            f"{args.path_type}-{args.prediction}-{args.loss_weight}-{mining_tag}"
        )
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"DG-ANM mining: {'ENABLED (gamma={}, mine_every={})'.format(args.gamma, args.mine_every) if args.use_mining else 'DISABLED'}")

        entity = os.environ.get("ENTITY", "")
        project = os.environ.get("PROJECT", "")
        if args.wandb and entity and project and wandb_utils is not None:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm
    ).to(device)

    # EMA and optimizer
    ema = deepcopy(model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Load checkpoint if provided
    resume_epoch = 0
    resume_step = 0
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict["model"])
            ema.load_state_dict(state_dict["ema"])
            opt.load_state_dict(state_dict["opt"])
            resume_epoch = state_dict.get("epoch", 0)
            resume_step = state_dict.get("train_steps", 0)
            logger.info(f"Resuming from epoch {resume_epoch}, step {resume_step}")
        else:
            model.load_state_dict(state_dict)
            ema.load_state_dict(state_dict)
        ema = ema.to(device)
        model = model.to(device)

    requires_grad(ema, False)
    model = DDP(model, device_ids=[device])

    # Transport (for EqM base loss)
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    transport_sampler = Sampler(transport)

    # VAE (for encoding input images to latents)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.use_mining:
        logger.info(
            f"DG-ANM config: gamma={args.gamma}, epsilon={args.mining_epsilon}, "
            f"steps={args.mining_steps}, lr={args.mining_lr}, k={args.mining_k}, "
            f"mine_every={args.mine_every}, margin={args.neg_margin}, rho={args.neg_rho}"
        )

    # Setup data
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

    # Prepare models for training
    update_ema(ema, model.module, decay=0)  # Sync EMA weights
    model.train()
    ema.eval()

    # Monitoring variables
    train_steps = resume_step
    log_steps = 0
    running_loss = 0
    running_loss_base = 0
    running_loss_neg = 0
    running_mining_normal = 0
    running_mining_tangent = 0
    running_mining_field = 0
    start_time = time()

    # Sampling setup (for visualization during training)
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    # Cache for mined negatives (for mine_every > 1)
    cached_x_neg = None

    logger.info(f"Training for {args.epochs} epochs (resuming from epoch {resume_epoch})...")
    for epoch in range(resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            # =================================================================
            # 1. Standard EqM loss (unchanged from upstream)
            # =================================================================
            model_kwargs = dict(y=y, return_act=args.disp, train=True)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss_base = loss_dict["loss"].mean()
            loss = loss_base

            # =================================================================
            # 2. DG-ANM mining + negative loss (if enabled)
            # =================================================================
            loss_neg_val = 0.0
            mining_info = None

            if args.use_mining and args.gamma > 0:
                B = x.shape[0]

                if args.mining_flavor == "v09":
                    # v09: random-noise Jacobian regularizer. Architecture-agnostic.
                    if train_steps % args.mine_every == 0:
                        loss_v09, mining_info = _v09_jacobian_step(
                            model_ddp=model, x=x, y=y,
                            train_eps=args.train_eps,
                            jac_sigma=args.jac_sigma,
                            jac_lambda=args.jac_lambda,
                            device=device,
                        )
                        cached_x_neg = None
                        loss = loss + args.gamma * loss_v09
                        loss_neg_val = loss_v09.item()
                        mining_info["v09_aux_over_base"] = (loss_neg_val / max(loss_base.item(), 1e-8))
                elif args.mining_flavor == "v10":
                    # v10: PGD hard-example mining on EqM regression target.
                    # FGSM-style (K=1) per Briglia 2025; non-saturating per CIFAR
                    # Phase 0.3 PASS (ratio L_hard/L_clean stable 1.047-1.049).
                    if train_steps % args.mine_every == 0:
                        loss_v10, mining_info = _v10_pgd_hard_example_step(
                            model_ddp=model, model_module=model.module,
                            x=x, y=y, transport=transport,
                            train_eps=args.train_eps,
                            v10_eps_radius=args.v10_eps_radius,
                            v10_K=args.v10_K,
                            v10_lr=args.mining_lr,
                            device=device,
                        )
                        cached_x_neg = None
                        loss = loss + args.gamma * loss_v10
                        loss_neg_val = loss_v10.item()
                        mining_info["v10_aux_over_base"] = (loss_neg_val / max(loss_base.item(), 1e-8))
                elif args.mining_flavor == "v02":
                    # v02: cosine-contrastive on velocity at uniform t.
                    # Warmup: skip mining until base loss has trained the model
                    # enough that v(x_t) is input-sensitive. EqM-B/2 init has
                    # bias-dominated output; PGA gradient vanishes (cos≈1).
                    in_warmup = train_steps < args.mining_warmup_steps
                    if (not in_warmup) and train_steps % args.mine_every == 0:
                        loss_v02, x_neg_new, mining_info = _v02_cosine_contrastive_step(
                            model_ddp=model, model_module=model.module,
                            x=x, y=y,
                            train_eps=args.train_eps,
                            mining_epsilon=args.mining_epsilon,
                            mining_steps=args.mining_steps,
                            mining_lr=args.mining_lr,
                            pos_sigma=args.pos_sigma,
                            lambda_pos=args.lambda_pos,
                            lambda_neg=args.lambda_neg,
                            neg_cos_margin=args.neg_cos_margin,
                            device=device,
                        )
                        cached_x_neg = x_neg_new
                        loss = loss + args.gamma * loss_v02
                        loss_neg_val = loss_v02.item()
                    # else: skip aux this step (mine_every cadence; loss_neg=0)
                else:
                    # v01 (default): feature-PCA geometry + margin hinge at t=1.
                    if train_steps % args.mine_every == 0:
                        # --- Feature extraction (eval mode, no grad for model) ---
                        model.eval()
                        with torch.no_grad():
                            t_ones = torch.ones(B, device=device)
                            _, acts = model.module(x, t_ones, y, return_act=True)
                            features = acts[-1].mean(dim=1)  # (B, hidden_size)

                        # --- Geometry estimation ---
                        P_T, P_N = estimate_local_geometry(features, k=args.mining_k)

                        # --- Mine negatives (model.module in eval, delta has grad) ---
                        x_neg, mining_info = mine_negatives(
                            model.module, x, y, features, P_N, P_T,
                            epsilon=args.mining_epsilon,
                            mining_steps=args.mining_steps,
                            mining_lr=args.mining_lr,
                            lambda_N=args.lambda_N,
                            lambda_T=args.lambda_T,
                            lambda_W=args.lambda_W,
                            device=device,
                        )
                        cached_x_neg = x_neg.detach()

                        # Restore train mode
                        model.train()
                    else:
                        # Reuse cached negatives from last mining step
                        x_neg = cached_x_neg

                    # --- Negative loss (DDP model in train mode, grads sync) ---
                    if x_neg is not None:
                        model.train()
                        loss_neg = dganm_negative_loss(
                            model, x, x_neg, y,
                            margin=args.neg_margin,
                            rho=args.neg_rho,
                            device=device,
                        )
                        loss = loss + args.gamma * loss_neg
                        loss_neg_val = loss_neg.item()

            # =================================================================
            # 3. Optimize
            # =================================================================
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # =================================================================
            # 4. Logging
            # =================================================================
            running_loss += loss.item()
            running_loss_base += loss_base.item()
            running_loss_neg += loss_neg_val
            if mining_info is not None:
                # v01 keys: avg_normal_component / avg_tangent_component / avg_field_norm_at_neg.
                # v02 keys: v02_pos_loss / v02_neg_loss / v02_neg_field_norm.
                # Reuse the same accumulators; the log labels reflect the active flavor.
                if args.mining_flavor == "v09":
                    running_mining_normal += mining_info["v09_jac_loss"]
                    running_mining_tangent += mining_info["v09_aux_over_base"]
                    running_mining_field += mining_info["v09_anchor_norm"]
                elif args.mining_flavor == "v10":
                    running_mining_normal += mining_info["v10_hard_loss"]
                    running_mining_tangent += mining_info["v10_aux_over_base"]
                    running_mining_field += mining_info["v10_delta_norm"]
                elif args.mining_flavor == "v02":
                    running_mining_normal += mining_info["v02_pos_loss"]
                    running_mining_tangent += mining_info["v02_neg_loss"]
                    running_mining_field += mining_info["v02_neg_field_norm"]
                else:
                    running_mining_normal += mining_info["avg_normal_component"]
                    running_mining_tangent += mining_info["avg_tangent_component"]
                    running_mining_field += mining_info["avg_field_norm_at_neg"]
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce losses across processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                avg_loss_base = torch.tensor(running_loss_base / log_steps, device=device)
                dist.all_reduce(avg_loss_base, op=dist.ReduceOp.SUM)
                avg_loss_base = avg_loss_base.item() / dist.get_world_size()

                log_msg = (
                    f"(step={train_steps:07d}) "
                    f"Loss: {avg_loss:.4f}, "
                    f"Base: {avg_loss_base:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}"
                )

                wandb_dict = {
                    "train_loss": avg_loss,
                    "train_loss_base": avg_loss_base,
                    "train_steps_per_sec": steps_per_sec,
                }

                if args.use_mining and args.gamma > 0:
                    avg_loss_neg = running_loss_neg / log_steps
                    avg_mining_normal = running_mining_normal / max(1, log_steps // args.mine_every)
                    avg_mining_tangent = running_mining_tangent / max(1, log_steps // args.mine_every)
                    avg_mining_field = running_mining_field / max(1, log_steps // args.mine_every)

                    # Per-flavor diagnostic labels.
                    if args.mining_flavor == "v09":
                        diag_label = (
                            f", Aux: {avg_loss_neg:.4f}"
                            f", v09(jac={avg_mining_normal:.3f}"
                            f", aux/base={avg_mining_tangent:.3f}"
                            f", |v|={avg_mining_field:.3f})"
                        )
                    elif args.mining_flavor == "v10":
                        diag_label = (
                            f", Aux: {avg_loss_neg:.4f}"
                            f", v10(hard={avg_mining_normal:.4f}"
                            f", aux/base={avg_mining_tangent:.3f}"
                            f", ||δ||={avg_mining_field:.3f})"
                        )
                    elif args.mining_flavor == "v02":
                        diag_label = (
                            f", Aux: {avg_loss_neg:.4f}"
                            f", v02(pos={avg_mining_normal:.3f}"
                            f", neg={avg_mining_tangent:.3f}"
                            f", |v_neg|={avg_mining_field:.3f})"
                        )
                    else:
                        diag_label = (
                            f", Neg: {avg_loss_neg:.4f}"
                            f", Mining(N={avg_mining_normal:.3f}"
                            f", T={avg_mining_tangent:.3f}"
                            f", F={avg_mining_field:.3f})"
                        )
                    log_msg += diag_label
                    wandb_dict.update({
                        "train_loss_neg": avg_loss_neg,
                        "mining_normal_component": avg_mining_normal,
                        "mining_tangent_component": avg_mining_tangent,
                        "mining_field_norm_at_neg": avg_mining_field,
                    })

                logger.info(log_msg)
                if args.wandb and wandb_utils is not None:
                    wandb_utils.log(wandb_dict, step=train_steps)

                # Reset monitoring variables
                running_loss = 0
                running_loss_base = 0
                running_loss_neg = 0
                running_mining_normal = 0
                running_mining_tangent = 0
                running_mining_field = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    # Save final checkpoint
    model.eval()
    if rank == 0 and checkpoint_dir is not None:
        final_ckpt = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "train_steps": train_steps,
            "epoch": args.epochs,
        }
        final_path = f"{checkpoint_dir}/final.pt"
        torch.save(final_ckpt, final_path)
        logger.info(f"Saved final checkpoint to {final_path}")
    dist.barrier()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Upstream EqM args ---
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom EqM checkpoint")
    parser.add_argument("--disp", action="store_true",
                        help="Toggle to enable Dispersive Loss")
    parser.add_argument("--uncond", type=bool, default=True,
                        help="disable/enable noise conditioning")
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none",
                        help="energy formulation")

    # --- DG-ANM mining args ---
    parser.add_argument("--use-mining", action="store_true", default=False,
                        help="Enable DG-ANM adversarial negative mining")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Weight for negative loss (L = L_base + gamma * L_neg)")
    parser.add_argument("--mining-epsilon", type=float, default=0.1,
                        help="Perturbation L2 ball radius in latent space")
    parser.add_argument("--mining-steps", type=int, default=3,
                        help="Number of PGA iterations for mining")
    parser.add_argument("--mining-lr", type=float, default=0.01,
                        help="Step size for signed gradient ascent during mining")
    parser.add_argument("--mining-k", type=int, default=10,
                        help="Number of neighbors for local PCA geometry estimation")
    parser.add_argument("--lambda-N", type=float, default=1.0,
                        help="Normal component weight in mining objective")
    parser.add_argument("--lambda-T", type=float, default=1.0,
                        help="Tangent component weight in mining objective")
    parser.add_argument("--lambda-W", type=float, default=0.1,
                        help="Weak field weight in mining objective")
    parser.add_argument("--neg-margin", type=float, default=5.0,
                        help="Margin for negative loss: max(0, margin - |g(x^-)|)")
    parser.add_argument("--neg-rho", type=float, default=0.0,
                        help="Trajectory failure loss weight")
    parser.add_argument("--mine-every", type=int, default=1,
                        help="Mine negatives every K steps, reuse cached for others")

    # --- Mining flavor (v01 = original DG-ANM, v02 = cosine-contrastive) ---
    parser.add_argument("--mining-flavor", type=str, choices=["v01", "v02", "v09", "v10"], default="v01",
                        help="v01: feature-PCA geometry + margin hinge at t=1 (original DG-ANM). "
                             "v02: cosine-contrastive on velocity at uniform t (cancelled IN-1K, cosine saturated). "
                             "v09: random-noise Jacobian regularizer. "
                             "v10: PGD hard-example mining on EqM regression target (CIFAR Phase 0.3 PASS 13.40 vs 14.17).")
    parser.add_argument("--lambda-pos", type=float, default=1.0,
                        help="v02: weight on (1 - cos(v_pos, v_anchor)) leg")
    parser.add_argument("--lambda-neg", type=float, default=1.0,
                        help="v02: weight on relu(cos(v_neg, v_anchor) - margin) leg")
    parser.add_argument("--pos-sigma", type=float, default=0.05,
                        help="v02: stddev of Gaussian noise for x_pos around x_t (latent space)")
    parser.add_argument("--neg-cos-margin", type=float, default=0.0,
                        help="v02: subtract this from cos(v_neg, v_anchor) before relu")
    parser.add_argument("--mining-warmup-steps", type=int, default=0,
                        help="v02: skip mining until train_steps >= this. EqM-B/2 init has "
                             "bias-dominated output (|v|~200, cos≈1); needs ~5000 steps for "
                             "v(x_t) to become input-sensitive before mining is meaningful.")
    # --- v09 Jacobian regularizer args ---
    parser.add_argument("--jac-sigma", type=float, default=0.05,
                        help="v09: stddev of Gaussian perturbation in latent space")
    parser.add_argument("--jac-lambda", type=float, default=0.1,
                        help="v09: weight on (||v(x+δ)-v(x)||²/σ²) regularizer")
    # --- v10 PGD hard-example args ---
    parser.add_argument("--v10-K", type=int, default=1,
                        help="v10: PGD inner steps (FGSM K=1 per Briglia 2025)")
    parser.add_argument("--v10-eps-radius", type=float, default=0.3,
                        help="v10: L2 ball radius for δ (CIFAR Phase 0.3 PASS regime)")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
