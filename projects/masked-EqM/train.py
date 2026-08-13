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
from transport.utils import mean_flat
import wandb_utils
from fb_direct import ForwardBackwardsDirectTrainer
from fb_direct.exact_hvp import (
    exact_fwrev_backward,
    allreduce_fwrev_grads,
    fwrev_rank_sync_checksum,
    compute_wfb_gradient,
)
from fb_direct.adaptive_clip import adaptive_clip_update, adaptive_clip_threshold
from experiments.btm.fd import assert_no_double_backward
from experiments.btm.image_losses import (
    BTM_FD_MODES,
    BTM_MODES,
    BTMConfig,
    btm_eval_target_match,
    btm_loss,
)
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
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
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

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
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
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)

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

        if args.reset_adam_state:
            # WFB-EqM Stage 2 (2026-08-12, v5 finding + follow-up 2x2 factorial): v3-v5
            # showed delta_theta_norm (actual AdamW displacement, computed from BOTH m_t and
            # v_t) ~40x smaller for WFB than exact-direct, even though the field-update
            # direction was MORE consistently correct than baseline's (v5: 74% positive
            # cosine vs baseline's 49.8% coin-flip on a held-out real-data probe). WFB
            # changes the gradient's COORDINATES, not just its scale (g_wfb =
            # M^T(A+lambda I)^{-1/2}r is a rotation/spectral reshaping of g_raw = M^T r, per
            # Stage 1's cosine(g_raw,g_wfb)~0.43-0.49 -- nowhere near 1), so BOTH loaded Adam
            # moments (m_t ~ E[g], v_t ~ E[g^2]) are stale for the new geometry, not only
            # v_t. This flag is agnostic to backward mode -- used both for the WFB+reset arm
            # AND the exact-direct+reset control arm, to isolate "does resetting alone change
            # anything" from "does resetting fix WFB specifically" (2x2: {exact,wfb} x
            # {loaded,reset}). A full reset (not a rescale) is the principled choice, not a
            # hack: Adam's bias correction (v_hat = v/(1-beta2^t)) is specifically designed
            # to give an unbiased estimate from a fresh start -- this IS what "warm-starting a
            # new optimization phase" means, not an approximation. Preserves model weights,
            # global training step, data position, LR schedule, and EMA -- only clears the
            # per-parameter optimizer state (step/exp_avg/exp_avg_sq).
            opt.state.clear()
            logger.info(f"reset AdamW optimizer state (exp_avg/exp_avg_sq/step) to fresh "
                        f"(--reset-adam-state, arm={'wfb' if args.wfb_backward else 'exact' if args.exact_fwrev else 'double-backward'}).")
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
    btm_cfg = None
    if args.btm_mode is not None:
        btm_cfg = BTMConfig(
            mode=args.btm_mode,
            interpolant=args.btm_interpolant,
            tc=args.btm_tc,
            kappa=args.btm_kappa,
            fd_eps=args.fd_eps,
            fd_k=args.fd_k,
            fd_direction=args.fd_direction,
            energy_difference_fp32=args.energy_difference_fp32,
            fd_chunk=args.fd_chunk,
        )
        logger.info(f"BTM arm active: {btm_cfg}")
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
    ema_grad_norm_ema = [None]  # mutable cell: EMA of the unclipped grad norm scale
    grad_metrics_file = None
    if rank == 0 and args.grad_log_every > 0:
        grad_metrics_path = os.path.join(experiment_dir, "gradient_metrics.jsonl")
        grad_metrics_file = open(
            grad_metrics_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )

    # Fixed held-out diagnostic probe (WFB-EqM Stage 2 checklist, 2026-08-12, v2 after v4's
    # synthetic-noise probe was found uninformative -- both arms showed ~flat probe_delta_L
    # since independent random noise isn't a target real training has any reason to fit): a
    # SINGLE REAL image, VAE-encoded and run through the exact same transport sampling path
    # as training (transport.sample/plan/get_ct), fixed and reused identically every step.
    # Reproducible across arms: global RNG state entering this call is identical between arms
    # (same --global-seed, identical program flow up to this point -- the ARM-specific
    # backward-mode branch only executes inside the training loop, after this). Used ONLY to
    # measure one-step functional progress (held-out field-MSE delta L) and field-update/
    # residual alignment (cosine) -- never touches training data statistics or any .grad
    # beyond a pure eval-mode forward + z-grad-only probe.
    _probe_x, _probe_y_raw = next(iter(loader))
    _probe_x = _probe_x.to(device)
    probe_y = _probe_y_raw.to(device)
    with torch.no_grad():
        _probe_x1 = vae.encode(_probe_x).latent_dist.sample().mul_(0.18215)
    _probe_t, _probe_x0, _probe_x1 = transport.sample(_probe_x1)
    probe_t, probe_xt, probe_ut = transport.path_sampler.plan(_probe_t, _probe_x0, _probe_x1)
    probe_ut = probe_ut * transport.get_ct(probe_t)[:, None, None, None]
    probe_xt = probe_xt.detach()
    probe_t = probe_t.detach()
    probe_ut = probe_ut.detach()

    def probe_field(raw_model):
        was_training = raw_model.training
        raw_model.eval()
        if args.ebm == 'none':
            # Vector arms have no scalar energy to differentiate; the probe
            # field is the network output itself.  (Needed by --btm-mode
            # btm_vector, which is the corrected-BTM gold-standard baseline.)
            with torch.no_grad():
                out = raw_model(probe_xt.detach(), probe_t, probe_y).detach()
            raw_model.train(was_training)
            return out
        z = probe_xt.detach().clone().requires_grad_(True)
        E = raw_model(z, probe_t, probe_y, energy_only=True)
        g = torch.autograd.grad(E.sum(), z, create_graph=False)[0].detach()
        raw_model.train(was_training)
        return -g

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
            if args.exact_fwrev:
                # Exact forward-over-reverse gradient (fb_direct/exact_hvp.py):
                # mathematically identical to the double-backward path below,
                # computed as one dual-tensor forward + one first-order
                # backward. Replicates training_losses' sampling exactly
                # (same transport.sample / plan / get_ct sequence), then
                # bypasses loss.backward(). DDP allreduce hooks never fire on
                # this path -- guarded to world_size == 1 at startup.
                t_s, x0_s, x1_s = transport.sample(x)
                t_s, xt_s, ut_s = transport.path_sampler.plan(t_s, x0_s, x1_s)
                ut_s = ut_s * transport.get_ct(t_s)[:, None, None, None]
                opt.zero_grad()
                fwrev_stats = exact_fwrev_backward(
                    model.module, xt_s, t_s, y, ut_s, gp_lambda=args.gp_lambda,
                )
                if args.energy_zloss_lambda != 0.0:
                    # Separate light forward + ordinary (single-differentiation)
                    # backward: E depends on theta directly, no z-grad needed,
                    # so this is a plain first-order pass (~none's cost, not
                    # exact_fwrev's 2x). .backward() accumulates into the same
                    # .grad tensors exact_fwrev_backward just populated, matching
                    # loss.backward() accumulation semantics -- no extra opt/zero_grad.
                    E_reg = model.module(xt_s.detach(), t_s, y, energy_only=True)
                    zloss = args.energy_zloss_lambda * (E_reg ** 2).mean()
                    zloss.backward()
                    fwrev_stats["loss_zloss"] = float(zloss.detach())
                # DDP's bucketed hooks never fire on this path (no
                # loss.backward()); explicitly average grads across ranks.
                allreduce_fwrev_grads(model.module)
                if dist.get_world_size() > 1 and args.grad_log_every > 0 \
                        and (train_steps + 1) % args.grad_log_every == 0:
                    checksum = fwrev_rank_sync_checksum(model.module).to(device)
                    mn, mx = checksum.clone(), checksum.clone()
                    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
                    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
                    if float(mx - mn) > 1e-6:
                        raise RuntimeError(
                            f"exact-fwrev rank desync at step {train_steps + 1}: "
                            f"parameter checksum spread {float(mx - mn):.3e} -- "
                            "ranks have diverged; aborting before silent corruption."
                        )
                loss = torch.tensor(
                    fwrev_stats["loss_main"] + args.gp_lambda * fwrev_stats["loss_gp"],
                    device=device,
                )
            elif args.wfb_backward:
                # WFB-EqM Stage 2 (2026-08-12): Whitened Forward-Backward preconditioned
                # gradient (fb_direct/exact_hvp.py:compute_wfb_gradient) as a REPLACEMENT
                # backward operator, not layered on exact_fwrev. Same sampling sequence as
                # the exact-fwrev branch above (canonical residual convention). The model's
                # predicted field remains EXACTLY -grad_z E at all times -- only the
                # optimizer-supplied pseudo-gradient changes; see Stage 0/1 reports for the
                # mathematical justification and causal validation this rests on.
                t_s, x0_s, x1_s = transport.sample(x)
                t_s, xt_s, ut_s = transport.path_sampler.plan(t_s, x0_s, x1_s)
                ut_s = ut_s * transport.get_ct(t_s)[:, None, None, None]
                opt.zero_grad()
                wfb_result = compute_wfb_gradient(
                    model.module, xt_s, t_s, y, ut_s,
                    params=None, rho=args.wfb_rho, k=args.wfb_k, seed=train_steps,
                    alpha=args.wfb_alpha,
                )
                # compute_wfb_gradient/exact_field_vjp operate on the CANONICAL, UNRESCALED
                # residual r = field - ut (no loss-reduction factor). exact_fwrev_backward's
                # applied gradient uses w = (2/(B*D)) * (g+ut) = -(2/(B*D)) * r, i.e. the
                # SAME field_vjp machinery applied to r rescaled by 2/(B*D) -- proven exactly
                # (machine precision) in test_field_vjp_direct_sign_relation_to_fwrev_w and
                # test_wfb_gradient_matches_native_fwrev_scale. Since compute_wfb_gradient's
                # whole (A+lambda I)^{-1/2} chain is linear in r (A/lambda/lambda_max depend
                # only on M, never on r's scale), g_wfb(c*r) = c*g_wfb(r) exactly -- so the
                # IDENTICAL 2/(B*D) rescaling applied to exact-fwrev's w must be applied here
                # too, or the applied WFB gradient is off by ~(B*D)/2 (tens of thousands of x
                # for this checkpoint's latent shape) relative to ARM A's native scale, the
                # calibrated clip threshold, and AdamW's tuned hyperparameters -- silently
                # invalidating any Stage 2+ comparison (caught before any GPU step ran,
                # 2026-08-12, external review).
                B_wfb, D_wfb = xt_s.shape[0], xt_s[0].numel()
                wfb_native_scale = 2.0 / (B_wfb * D_wfb)
                for p, g in zip(wfb_result["params"], wfb_result["g_wfb"]):
                    p.grad = (wfb_native_scale * g).clone()
                # DDP bucketed hooks never fire on this path (no loss.backward());
                # reuses the exact-fwrev branch's averaging + rank-sync-checksum machinery
                # verbatim -- both paths populate .grad manually and skip loss.backward().
                allreduce_fwrev_grads(model.module)
                if dist.get_world_size() > 1 and args.grad_log_every > 0 \
                        and (train_steps + 1) % args.grad_log_every == 0:
                    checksum = fwrev_rank_sync_checksum(model.module).to(device)
                    mn, mx = checksum.clone(), checksum.clone()
                    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
                    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
                    if float(mx - mn) > 1e-6:
                        raise RuntimeError(
                            f"wfb-backward rank desync at step {train_steps + 1}: "
                            f"parameter checksum spread {float(mx - mn):.3e} -- "
                            "ranks have diverged; aborting before silent corruption."
                        )
                # loss_main == residual_rms**2 identity (validated in
                # matched_replay_jacobian_diagnostic.py's canonical_residual_and_validate)
                # -- directly comparable to the exact-fwrev arm's loss_main for logging.
                loss = torch.tensor(
                    wfb_result["r_norm"] ** 2 / wfb_result["r"].numel(), device=device,
                )
            elif args.btm_mode is not None:
                # Corrected Beckmann-Transport-Model arm.  The ONLY thing that
                # differs from the branch below is the (interpolant, target)
                # pair and, for the FD arms, that the parameter gradient is
                # obtained from scalar energy evaluations rather than from a
                # create_graph=True input-gradient graph.  Optimizer, clipping,
                # EMA, logging, checkpointing and DDP are shared verbatim.
                loss, btm_stats = btm_loss(
                    model, model.module, btm_cfg, transport, x, y,
                )
                opt.zero_grad()
                if args.btm_mode in BTM_FD_MODES:
                    # The backward runs inside the guard too, so the invariant
                    # "no mixed d_theta d_x phi anywhere in this training step"
                    # covers the whole step, not just loss construction.
                    with assert_no_double_backward():
                        loss.backward()
                else:
                    loss.backward()
            else:
                model_kwargs = dict(y=y, return_act=args.disp, train=True)
                loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()
                opt.zero_grad()
                loss.backward()
            grad_norm = None
            unclipped_grad_norm = None
            effective_max_grad_norm = args.max_grad_norm
            if args.adaptive_clip:
                # EMA-tracked threshold (NOT NFNets AGC -- that's a per-
                # parameter weight-ratio scheme for a different problem).
                # This targets the specific failure mode flagged in the
                # growing-clip-rate diagnostic (2026-08-10): a FIXED
                # threshold becomes progressively more binding if the
                # natural gradient scale drifts upward over training,
                # inflating the observed clip rate as an artifact of the
                # threshold being outgrown -- independent of whether the
                # landscape is actually getting rougher. Tracking the scale
                # and clipping at a constant MULTIPLE of it makes "clip
                # rate stays roughly constant" a design invariant instead
                # of something we hope holds; it does not by itself control
                # tail/curvature-driven events (see the curvature-probe
                # diagnostic for that).
                probe_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=True,
                )
                probe_val = float(probe_norm)
                ema_grad_norm_ema[0] = adaptive_clip_update(
                    ema_grad_norm_ema[0], probe_val, args.adaptive_clip_ema_decay,
                )
                effective_max_grad_norm = adaptive_clip_threshold(
                    ema_grad_norm_ema[0], args.adaptive_clip_factor,
                )
                unclipped_grad_norm = probe_norm
                if probe_val > effective_max_grad_norm:
                    scale = effective_max_grad_norm / (probe_val + 1e-30)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.detach().mul_(scale)
                grad_norm = unclipped_grad_norm
            elif args.max_grad_norm is not None:
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
                        "max_grad_norm": effective_max_grad_norm,
                        "clipped": bool(
                            unclipped_grad_norm is not None
                            and effective_max_grad_norm is not None
                            and unclipped_grad_norm > effective_max_grad_norm
                        ),
                        "learning_rate": opt.param_groups[0]["lr"],
                        "adaptive_clip": args.adaptive_clip,
                        "weight_decay": args.weight_decay,
                    }
                    if args.exact_fwrev:
                        record["loss_main"] = fwrev_stats["loss_main"]
                        record["loss_gp"] = fwrev_stats["loss_gp"]
                        record["gp_lambda"] = args.gp_lambda
                        record["field_norm"] = fwrev_stats["field_norm"]
                        record["target_norm"] = fwrev_stats["target_norm"]
                        if "loss_zloss" in fwrev_stats:
                            record["loss_zloss"] = fwrev_stats["loss_zloss"]
                            record["energy_zloss_lambda"] = args.energy_zloss_lambda
                    if args.wfb_backward:
                        # "applied" grad_norm/head_grad_norm/backbone_grad_norm above are
                        # already the WFB-preconditioned gradient (computed generically from
                        # .grad, which this branch populated with the NATIVE-SCALE g_wfb --
                        # see the 2/(B*D) rescaling note above). g_raw_norm_hypothetical is
                        # the CAUSAL diagnostic spec Section 9 requires: the SAME residual's
                        # raw M^T r, ALSO native-scaled here for direct comparability against
                        # ARM A's grad_norm and the calibrated clip threshold -- the ideal
                        # result is this trace still spiking past the clip threshold while
                        # grad_norm (applied) does not.
                        record["r_norm"] = wfb_result["r_norm"]
                        record["g_raw_norm_hypothetical"] = wfb_native_scale * wfb_result["g_raw_norm"]
                        record["wfb_native_scale"] = wfb_native_scale
                        record["lambda_max"] = wfb_result["lambda_max"]
                        record["lam"] = wfb_result["lam"]
                        record["T_eigmax"] = wfb_result["T_eigmax"]
                        record["m_lanczos"] = wfb_result["m"]
                        record["wfb_breakdown"] = wfb_result["breakdown"]
                        record["wfb_breakdown_reason"] = wfb_result["breakdown_reason"]
                        record["wfb_rho"] = args.wfb_rho
                        record["wfb_k"] = args.wfb_k
                        record["wfb_alpha"] = args.wfb_alpha
                    # Stage 2 checklist (2026-08-12, external review, requested before trusting
                    # any WFB-vs-exact comparison): |delta_theta| (actual AdamW displacement --
                    # NOT just the pre-clip pseudo-gradient; Adam's stale second-moment state
                    # from the loaded checkpoint could silently neutralize a correctly-directed
                    # but differently-scaled update), one-step held-out field-MSE delta L on a
                    # FIXED probe (the noisy single-batch training loss cannot answer "is this
                    # arm still learning"), and cos(field update, -residual) on that same probe
                    # (the field should keep moving toward reducing the residual even if the
                    # parameter-space gradient looks totally different under WFB).
                    if btm_cfg is not None:
                        record["btm_mode"] = btm_cfg.mode
                        record["btm_tc"] = btm_cfg.tc
                        record.update(btm_stats)
                        if args.btm_eval_every > 0 and \
                                (train_steps + 1) % args.btm_eval_every == 0:
                            # EVALUATION ONLY: exact grad_x phi vs the BTM
                            # target on the fixed probe batch.  This is how we
                            # find out whether an FD-trained potential learns
                            # the same conservative field the exact arm does.
                            was_training = model.module.training
                            model.module.eval()
                            try:
                                if btm_cfg.mode != "btm_vector":
                                    record.update(btm_eval_target_match(
                                        model.module, btm_cfg, transport,
                                        _probe_x1, probe_y))
                            finally:
                                model.module.train(was_training)
                    field_pre = probe_field(model.module)
                    probe_r = field_pre - probe_ut
                    probe_loss_pre = float(mean_flat((field_pre - probe_ut) ** 2).mean())
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    pre_flat = torch.cat([p.detach().reshape(-1) for p in trainable_params])
                    grad_metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            opt.step()
            update_ema(ema, model.module)

            if grad_norm is not None and rank == 0 and args.grad_log_every > 0 \
                    and (train_steps + 1) % args.grad_log_every == 0:
                post_flat = torch.cat([p.detach().reshape(-1) for p in trainable_params])
                delta_theta_norm = float((post_flat - pre_flat).norm())
                field_post = probe_field(model.module)
                probe_loss_post = float(mean_flat((field_post - probe_ut) ** 2).mean())
                field_delta = field_post - field_pre
                field_delta_norm = float(field_delta.norm())
                cos_val = float((field_delta * (-probe_r)).sum()
                                 / (field_delta_norm * probe_r.norm() + 1e-30))
                # Function-space progress per unit parameter movement (external review,
                # 2026-08-12): P_t = -<delta_s_t, r_t> is the (unnormalized) rate at which
                # this step's actual field motion reduces the probe residual; eta_func =
                # P_t / |delta_theta_t| asks "how much correct EqM field improvement per unit
                # parameter movement" -- the direct answer to whether WFB's smaller steps are
                # at least MORE efficient per unit of parameter motion, independent of
                # whether the optimizer state was reset (which controls step SIZE, not this
                # per-unit efficiency).
                P_t = float(-(field_delta * probe_r).sum())
                eta_func = P_t / (delta_theta_norm + 1e-30)
                checklist_record = {
                    "step": train_steps + 1,
                    "delta_theta_norm": delta_theta_norm,
                    "probe_loss_pre": probe_loss_pre,
                    "probe_loss_post": probe_loss_post,
                    "probe_delta_L": probe_loss_post - probe_loss_pre,
                    "probe_cos_field_update_vs_neg_r": cos_val,
                    "field_delta_norm": field_delta_norm,
                    "P_t": P_t,
                    "eta_func": eta_func,
                }
                grad_metrics_file.write(json.dumps(checklist_record, sort_keys=True) + "\n")

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


#################################################################################
#                     'forward-backwards-direct' Training Loop                  #
#################################################################################
# Kept as a SEPARATE function (rather than threaded into main() above) so the
# well-tested none/l2/dot/mean/direct path above is untouched byte-for-byte.
# theta (the EqM model) never appears on the LHS of an autograd graph here --
# see fb_direct/trainer.py for the algorithm. This is why theta is NOT DDP
# wrapped below (DDP requires participation in autograd to synchronize
# gradients); only phi (ForwardBackwardsDirectTrainer.phi) is DDP-wrapped,
# and the mapped theta update is then identical bit-for-bit on every rank
# because it is a deterministic function of the (DDP-synchronized) phi
# gradient. This distributed path is implemented per Section 9 of the spec
# but has only been exercised in single-process smoke runs in this
# environment (SLURM/multi-GPU here is remote-cluster-only per AGENTS.md);
# see documentation/forward-backwards-direct.md "Known limitations".
def main_forward_backwards_direct(args):
    assert torch.cuda.is_available() or args.allow_cpu, (
        "Training currently requires at least one GPU (or --allow-cpu for a local smoke test)."
    )
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    dist.init_process_group("gloo" if not torch.cuda.is_available() else "nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    use_cuda = torch.cuda.is_available()
    device = int(os.environ.get("LOCAL_RANK", 0)) if use_cuda else "cpu"
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.set_device(device)
    local_batch_size = int(args.global_batch_size // world_size)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_name = (
            f"{experiment_index:03d}-{model_string_name}-"
            f"{args.path_type}-{args.prediction}-{args.loss_weight}-ebm-{args.ebm}"
        )
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        fb_metrics_path = os.path.join(experiment_dir, "fb_direct_metrics.jsonl")
        fb_metrics_file = open(fb_metrics_path, "a", encoding="utf-8", buffering=1)
    else:
        logger = create_logger(None)
        checkpoint_dir = None
        fb_metrics_file = None

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm,
    ).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    fb_trainer = ForwardBackwardsDirectTrainer(
        model, lr=1e-4, max_grad_norm=args.max_grad_norm, device=device,
    )

    resume_epoch = 0
    resume_step = None
    if args.ckpt is not None:
        raw = torch.load(args.ckpt, map_location="cpu")
        if "phi" in raw or "theta" in raw:
            # A checkpoint produced by this mode (see fb_trainer.state_dict()).
            fb_trainer.load_state_dict(raw)
            resume_step = raw.get("step_count", 0)
        else:
            # Section 12: initializing from a plain 'direct' checkpoint.
            state_dict = raw["model"] if "model" in raw else raw
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[forward-backwards-direct] direct-checkpoint init: missing={missing} unexpected={unexpected}")
            fb_trainer.registry.tie_from_forward_()
        model.to(device)
    logger.info(f"[forward-backwards-direct] parameter coverage: {fb_trainer.coverage_report()}")

    if world_size > 1:
        fb_trainer.phi = DDP(fb_trainer.phi, device_ids=[device] if use_cuda else None, find_unused_parameters=False)

    transport = create_transport(
        args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps,
        corruption_mode=args.corruption_mode, mask_prob=args.mask_prob,
        fourier_cutoff=args.fourier_cutoff, blur_sigma=args.blur_sigma,
        downsample_factor=args.downsample_factor, gaussian_weight=args.gaussian_weight,
        mask_weight=args.mask_weight, blur_weight=args.blur_weight,
        fourier_weight=args.fourier_weight, downsample_weight=args.downsample_weight,
        structured_mask_weight=args.structured_mask_weight,
    )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"EqM Parameters (theta): {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"EqM Parameters (phi): {sum(p.numel() for p in fb_trainer.phi.parameters()):,}")

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed,
    )
    loader = DataLoader(
        dataset, batch_size=local_batch_size, shuffle=False, sampler=sampler,
        num_workers=args.num_workers, pin_memory=use_cuda, drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    if resume_step is None:
        resume_step = resume_epoch * len(loader)

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    train_steps = int(resume_step)
    max_train_steps = train_steps + args.max_steps if args.max_steps is not None else None
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training (forward-backwards-direct) for {args.epochs} epochs...")
    for epoch in range(resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        reached_max_steps = False
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            loss, diagnostics = fb_trainer.training_step(transport, x, y)
            update_ema(ema, model)

            if (
                args.fb_exact_audit_every > 0
                and (train_steps + 1) % args.fb_exact_audit_every == 0
            ):
                with torch.no_grad():
                    t_audit, x0_audit, x1_audit = transport.sample(x)
                    t_audit = t_audit.to(x)
                    t_audit, xt_audit, ut_audit = transport.path_sampler.plan(t_audit, x0_audit, x1_audit)
                    ut_audit = ut_audit * transport.get_ct(t_audit)[:, None, None, None]
                audit = fb_trainer.exact_field_audit(xt_audit, t_audit, y)
                diagnostics.update(audit)
                logger.info(f"(step={train_steps + 1:07d}) exact-field audit: {audit}")

                # Live PARAMETER-space gradient-alignment audit (Gate 1's
                # metric, but measured continuously during training rather
                # than once at a frozen checkpoint) -- distinguishes
                # "field/output agreement stays high but the update
                # direction is structurally biased" from "field agreement is
                # degrading too". Same cadence/cost tier as exact_field_audit
                # (one extra create_graph=True backward through theta).
                grad_audit = fb_trainer.exact_gradient_audit(xt_audit, t_audit, y, ut_audit)
                diagnostics.update(grad_audit)
                logger.info(f"(step={train_steps + 1:07d}) exact-gradient audit: {grad_audit}")

            if rank == 0 and fb_metrics_file is not None:
                record = {"step": train_steps + 1, **diagnostics}
                fb_metrics_file.write(json.dumps(record, sort_keys=True) + "\n")

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                if use_cuda:
                    torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device if use_cuda else "cpu")
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        **fb_trainer.state_dict(),
                        "ema": ema.state_dict(),
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

        if rank == 0 and (args.save_epochs is None or epoch + 1 in args.save_epochs):
            checkpoint = {
                **fb_trainer.state_dict(),
                "ema": ema.state_dict(),
                "args": args,
                "epoch": epoch + 1,
                "step": train_steps,
            }
            epoch_checkpoint_path = f"{checkpoint_dir}/epoch{epoch + 1:02d}.pt"
            torch.save(checkpoint, epoch_checkpoint_path)
            logger.info(f"Saved epoch checkpoint to {epoch_checkpoint_path}")
        dist.barrier()

    if rank == 0 and fb_metrics_file is not None:
        fb_metrics_file.close()

    model.eval()
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
    parser.add_argument(
        "--adaptive-clip", action="store_true",
        help="EMA-adaptive gradient-norm clipping instead of a fixed --max-grad-norm: "
             "clip to (EMA of unclipped grad norm) * --adaptive-clip-factor. Targets the "
             "growing-clip-rate diagnostic (2026-08-10): a fixed threshold becomes "
             "progressively more binding if the natural gradient scale drifts upward "
             "over training, independent of any real instability. Mutually exclusive "
             "with --max-grad-norm.",
    )
    parser.add_argument(
        "--adaptive-clip-factor", type=float, default=4.0,
        help="(requires --adaptive-clip) clip threshold = ema_grad_norm * this factor",
    )
    parser.add_argument(
        "--adaptive-clip-ema-decay", type=float, default=0.99,
        help="(requires --adaptive-clip) EMA decay for the tracked grad-norm scale",
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
    parser.add_argument("--ebm", type=str,
                        choices=["none", "l2", "dot", "mean", "direct", "forward-backwards-direct"],
                        default="none",
                        help="'direct' uses a scalar E_theta(x) and returns -grad_x E_theta(x). "
                             "'forward-backwards-direct' trains the same scalar-energy architecture "
                             "via an explicit reverse network (no create_graph=True double-backward); "
                             "see documentation/forward-backwards-direct.md")
    parser.add_argument(
        "--fb-exact-audit-every", type=int, default=0,
        help="(forward-backwards-direct only) run a diagnostic-only exact-field "
             "audit (torch.autograd.grad vs the reverse network) every N steps; 0 disables it",
    )
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="(forward-backwards-direct only) allow CPU-only training for local smoke tests",
    )
    parser.add_argument(
        "--exact-fwrev", action="store_true",
        help="(ebm='direct' only) compute the EXACT parameter gradient via "
             "Pearlmutter forward-over-reverse (torch.autograd.forward_ad dual "
             "pass + one first-order backward) instead of create_graph=True "
             "double-backward. Mathematically identical gradient, "
             "ordinary-training memory. See fb_direct/exact_hvp.py",
    )
    parser.add_argument(
        "--gp-lambda", type=float, default=0.0,
        help="(requires --exact-fwrev) gradient-penalty coefficient: adds "
             "gp_lambda * mean(||grad_z E||^2) to the loss, folded exactly into "
             "the same forward-over-reverse pass (smoothness regularizer "
             "targeting the mixed-Hessian conditioning)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="AdamW decoupled weight decay. Was hardcoded to 0 (2026-08-10 "
             "activation-scale diagnostic: median QKV/proj weight spectral "
             "norm grows +25.6%% monotonically over ~1.2M steps, identically "
             "in both fwrev arms -- GP touches mean(||grad_z E||^2), never "
             "raw weight scale, so it cannot arrest this. GPT-2/DeepNet-"
             "lineage fix: bound weight-norm growth directly.",
    )
    parser.add_argument(
        "--energy-zloss-lambda", type=float, default=0.0,
        help="(requires --exact-fwrev) adds energy_zloss_lambda * mean(E^2) "
             "to the loss via a separate light forward + ordinary backward "
             "(single differentiation, no z-grad needed -- gradients "
             "accumulate into the same .grad tensors as exact_fwrev_backward, "
             "matching loss.backward() accumulation semantics). Targets the "
             "companion 2026-08-10 finding: raw |E| grows +25-40%% over the "
             "same window, also GP-independent. Analog of Wortsman et al. "
             "2023's z-loss on output-logit divergence.",
    )

    parser.add_argument(
        "--wfb-backward", action="store_true",
        help="(ebm='direct' only, WFB-EqM Stage 2) compute the parameter gradient via "
             "Whitened Forward-Backward preconditioning: g_wfb = M^T(A+lambda I)^{-1/2}r, "
             "A = M M^T (mixed input-parameter Jacobian Gram operator), "
             "lambda = wfb_rho * lambda_max(A). Independent alternative to --exact-fwrev "
             "(not layered on top of it -- both require ebm='direct' but are mutually "
             "exclusive). See fb_direct/exact_hvp.py:compute_wfb_gradient and "
             "documentation/wfb-eqm-stage0-stage1-plan.md.",
    )
    parser.add_argument(
        "--wfb-rho", type=float, default=1e-4,
        help="(requires --wfb-backward) relative spectral damping: lambda = wfb_rho * "
             "lambda_max(A). Spec default 1e-4.",
    )
    parser.add_argument(
        "--wfb-k", type=int, default=12,
        help="(requires --wfb-backward) Lanczos iterations for the (A+lambda I)^{-1/2} r "
             "approximation. Stage 1 (job 38484995) found k=8's g_wfb_norm not fully "
             "converged vs k=12 (~7-12%% further decrease on severe batches) though "
             "T_eigmax converges by k=4 -- k=12 recommended as the production default.",
    )
    parser.add_argument(
        "--wfb-alpha", type=float, default=0.5,
        help="(requires --wfb-backward) power in g_alpha = M^T(A+lambda I)^{-alpha} r. "
             "0.5 is the original WFB formulation (default, preserves prior behavior). "
             "1.0 is full damped Gauss-Newton ('FBGN'): bounds the INDUCED FIELD update's "
             "per-mode gain sigma_i^2/(sigma_i^2+lambda) in [0,1], not just the parameter "
             "gradient's gain (which alpha=0.5 alone bounds). See Stage 2.5 diagnostic "
             "(job 38673961, 2026-08-12) for the empirical field-gain comparison.",
    )
    parser.add_argument(
        "--reset-adam-state", action="store_true",
        help="Reset AdamW's loaded exp_avg/exp_avg_sq/step to fresh right after --ckpt "
             "loading, regardless of backward mode (2026-08-12, generalized from the "
             "WFB-only --wfb-reset-adam-state for a proper 2x2 factorial: "
             "{exact,wfb} x {loaded,reset}). Stage 2 v3-v5 found WFB's actual parameter "
             "displacement ~40x smaller than exact-direct despite a MORE consistently "
             "correct field-update direction (held-out probe cosine 74%% positive vs "
             "baseline's 49.8%%) -- consistent with the loaded optimizer moments (both m_t "
             "and v_t, since WFB rotates/reshapes the gradient's coordinates, not just its "
             "scale) being stale for the new gradient geometry. Off by default.",
    )

    # ---------------- corrected-BTM arms (arXiv:2608.01692v2) ----------------
    # See documentation/btm-fd-scalar-plan-2026-08-13.md.  These replace the
    # sampling target only; they reuse the entire training stack below.
    parser.add_argument(
        "--btm-mode", type=str, default=None,
        choices=list(BTM_MODES),
        help="train a corrected Beckmann-Transport-Model arm instead of the "
             "legacy EqM target. 'btm_vector' requires --ebm none; every "
             "scalar arm requires --ebm direct. The FD arms "
             "(btm_scalar_fd_*) train the scalar potential using ONLY scalar "
             "energy evaluations -- no create_graph=True input-gradient path, "
             "enforced at runtime by fd.assert_no_double_backward().",
    )
    parser.add_argument(
        "--btm-interpolant", type=str, default="self_stopping",
        choices=["self_stopping", "linear", "eqm_legacy"],
        help="BTM interpolant. 'self_stopping' is the paper's Appendix-H "
             "piecewise choice (Idot_1 = 0); 'eqm_legacy' reproduces the "
             "inconsistent eq.(16) target as a negative control.",
    )
    parser.add_argument("--btm-tc", type=float, default=0.8,
                        help="breakpoint of the self-stopping interpolant; the "
                             "paper does not publish a value for its image "
                             "experiments, so this is swept on the toy first")
    parser.add_argument("--btm-kappa", type=float, default=0.8,
                        help="(--btm-interpolant eqm_legacy) c_t = (1-t)^kappa")
    parser.add_argument("--fd-eps", type=float, default=1e-3,
                        help="relative FD step: h = fd_eps * ||z||_2")
    parser.add_argument("--fd-k", type=int, default=1,
                        help="number of FD probe directions per sample")
    parser.add_argument("--fd-direction", type=str, default="rademacher",
                        choices=["rademacher"],
                        help="FD probe distribution (E[u u^T] = I/d)")
    parser.add_argument("--fd-chunk", type=int, default=None,
                        help="max rows per forward call for the 2KB FD batch")
    parser.add_argument(
        "--energy-difference-fp32", type=lambda s: str(s).lower() not in
        ("0", "false", "no"), default=True,
        help="promote low-precision energy evaluations to fp32 before the "
             "cancellation-prone FD subtraction (never demotes fp64)",
    )
    parser.add_argument("--btm-eval-every", type=int, default=0,
                        help="evaluate exact grad_x phi vs the BTM target on a "
                             "held-out batch every N steps (0 disables)")

    parse_transport_args(parser)
    args = parser.parse_args()
    if args.grad_log_every < 0:
        parser.error("--grad-log-every must be non-negative")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if args.adaptive_clip and args.max_grad_norm is not None:
        parser.error("--adaptive-clip and --max-grad-norm are mutually exclusive")
    if args.adaptive_clip_factor <= 0:
        parser.error("--adaptive-clip-factor must be positive")
    if not (0.0 < args.adaptive_clip_ema_decay < 1.0):
        parser.error("--adaptive-clip-ema-decay must be in (0, 1)")
    if args.exact_fwrev and args.ebm != "direct":
        parser.error("--exact-fwrev requires --ebm direct")
    if args.exact_fwrev and args.disp:
        parser.error("--exact-fwrev does not support --disp")
    if args.gp_lambda != 0.0 and not args.exact_fwrev:
        parser.error("--gp-lambda requires --exact-fwrev")
    if args.gp_lambda < 0:
        parser.error("--gp-lambda must be non-negative")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if args.energy_zloss_lambda != 0.0 and not args.exact_fwrev:
        parser.error("--energy-zloss-lambda requires --exact-fwrev")
    if args.energy_zloss_lambda < 0:
        parser.error("--energy-zloss-lambda must be non-negative")
    if args.wfb_backward and args.ebm != "direct":
        parser.error("--wfb-backward requires --ebm direct")
    if args.wfb_backward and args.exact_fwrev:
        parser.error("--wfb-backward and --exact-fwrev are mutually exclusive backward modes")
    if args.wfb_backward and args.disp:
        parser.error("--wfb-backward does not support --disp")
    if args.wfb_backward and args.gp_lambda != 0.0:
        parser.error("--gp-lambda requires --exact-fwrev, not supported with --wfb-backward")
    if args.wfb_backward and args.energy_zloss_lambda != 0.0:
        parser.error("--energy-zloss-lambda requires --exact-fwrev, not supported with --wfb-backward")
    if args.wfb_k < 1:
        parser.error("--wfb-k must be >= 1")
    if args.wfb_rho <= 0:
        parser.error("--wfb-rho must be positive")
    if args.wfb_alpha < 0:
        parser.error("--wfb-alpha must be >= 0 (0=direct, 0.5=WFB, 1=FBGN)")
    if args.btm_mode is not None:
        if args.btm_mode == "btm_vector" and args.ebm != "none":
            parser.error("--btm-mode btm_vector requires --ebm none "
                         "(it trains an unconstrained vector field)")
        if args.btm_mode != "btm_vector" and args.ebm != "direct":
            parser.error(f"--btm-mode {args.btm_mode} requires --ebm direct "
                         "(scalar energy head)")
        if args.exact_fwrev or args.wfb_backward:
            parser.error("--btm-mode is a self-contained training branch and "
                         "is mutually exclusive with --exact-fwrev/--wfb-backward")
        if args.disp:
            parser.error("--btm-mode does not support --disp")
        if args.fd_k < 1:
            parser.error("--fd-k must be >= 1")
        if args.fd_eps <= 0:
            parser.error("--fd-eps must be positive")
        if not 0.0 < args.btm_tc < 1.0:
            parser.error("--btm-tc must lie in (0, 1)")
    else:
        for flag, default in (("fd_eps", 1e-3), ("fd_k", 1)):
            if getattr(args, flag) != default:
                parser.error(f"--{flag.replace('_','-')} requires --btm-mode")
    if args.reset_adam_state and args.ckpt is None:
        parser.error("--reset-adam-state has no effect without --ckpt (no loaded optimizer state to reset)")
    args.save_epochs = None if args.save_epochs is None else {
        int(value) for value in args.save_epochs.split(",") if value.strip()
    }
    if args.ebm == "forward-backwards-direct":
        main_forward_backwards_direct(args)
    else:
        main(args)
