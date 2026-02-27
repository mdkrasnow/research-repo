import math
import sys
import collections
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from tabulate import tabulate

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import os.path as osp
import time
import numpy as np


def _custom_exception_hook(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, ipdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        ipdb.post_mortem(tb)


def hook_exception_ipdb():
    """Add a hook to ipdb when an exception is raised."""
    if not hasattr(_custom_exception_hook, 'origin_hook'):
        _custom_exception_hook.origin_hook = sys.excepthook
        sys.excepthook = _custom_exception_hook


def unhook_exception_ipdb():
    """Remove the hook to ipdb when an exception is raised."""
    assert hasattr(_custom_exception_hook, 'origin_hook')
    sys.excepthook = _custom_exception_hook.origin_hook

hook_exception_ipdb()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    val: float = 0
    avg: float = 0
    sum: float = 0
    sum2: float = 0
    std: float = 0
    count: float = 0
    tot_count: float = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum2 = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum2 += val * val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count
        self.std = (self.sum2 / self.count - self.avg * self.avg) ** 0.5

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        supervise_energy_landscape = True,
        use_innerloop_opt = True,
        show_inference_tqdm = True,
        baseline = False,
        sudoku = False,
        continuous = False,
        connectivity = False,
        shortest_path = False,
        mining_config = None,
    ):
        super().__init__()
        self.model = model
        self.inp_dim = self.model.inp_dim
        self.out_dim = self.model.out_dim
        self.out_shape = (self.out_dim, )
        self.self_condition = False
        self.supervise_energy_landscape = supervise_energy_landscape
        self.use_innerloop_opt = use_innerloop_opt

        # Mining configuration for CD-style training
        if mining_config is None:
            mining_config = {'strategy': 'adversarial', 'opt_steps': 2, 'noise_scale': 3.0}
        self.mining_config = mining_config
        self.mining_strategy = mining_config.get('strategy', 'adversarial')

        # Valid strategies: baseline NCE, random NCE, CD variants, TAM
        valid_strategies = ['none', 'random', 'adversarial', 'cd_langevin', 'cd_langevin_replay', 'cd_full', 'tam']
        assert self.mining_strategy in valid_strategies, \
            f"mining_strategy must be one of {valid_strategies}, got {self.mining_strategy}"

        # Initialize replay buffer for CD training if needed
        self.use_replay_buffer = mining_config.get('use_replay_buffer', False)
        if self.use_replay_buffer:
            from .replay_buffer import TBucketReplayBuffer
            buffer_size = mining_config.get('replay_buffer_size', 10000)
            num_buckets = mining_config.get('replay_buffer_buckets', 16)
            self.replay_buffer = TBucketReplayBuffer(
                buffer_size=buffer_size,
                num_buckets=num_buckets
            )
            self.replay_prob = mining_config.get('replay_sample_prob', 0.95)
        else:
            self.replay_buffer = None

        # Global step for energy loss scheduling (set by trainer)
        self.global_step = 0

        self.seq_length = seq_length
        self.objective = objective
        self.show_inference_tqdm = show_inference_tqdm
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.baseline = baseline
        self.sudoku = sudoku
        self.connectivity = connectivity
        self.continuous = continuous
        self.shortest_path = shortest_path

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Step size for optimizing
        register_buffer('opt_step_size', betas * torch.sqrt( 1 / (1 - alphas_cumprod)))
        # register_buffer('opt_step_size', 0.25 * torch.sqrt(alphas_cumprod) * torch.sqrt(1 / alphas_cumprod -1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)
        # whether to autonormalize

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, inp, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        with torch.enable_grad():
            model_output = self.model(inp, x, t)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, cond, x, t, x_self_cond = None, clip_denoised = False):
        preds = self.model_predictions(cond, x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            # x_start.clamp_(-6, 6)

            if self.continuous:
                sf = 2.0
            else:
                sf = 1.0

            x_start.clamp_(-sf, sf)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, cond, x, t, x_self_cond = None, clip_denoised = True, with_noise=False, scale=False):
        b, *_, device = *x.shape, x.device

        if type(t) == int:
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        else:
            batched_times = t
            noise = torch.randn_like(x)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(cond, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)

        # Don't scale inputs by expansion factor (Do that later)
        if not scale:
            model_mean = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape) * x_start

        if with_noise:
            pred_img = model_mean  + (0.5 * model_log_variance).exp() * noise
        else:
            pred_img = model_mean #  + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start

    def opt_step(self, inp, img, t, mask, data_cond, step=5, eval=True, sf=1.0, detach=True):
        with torch.enable_grad():
            for i in range(step):
                energy, grad = self.model(inp, img, t, return_both=True)
                img_new = img - extract(self.opt_step_size, t, grad.shape) * grad * sf  # / (i + 1) ** 0.5

                if mask is not None:
                    img_new = img_new * (1 - mask) + mask * data_cond

                if self.continuous:
                    sf = 2.0
                else:
                    sf = 1.0

                max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
                img_new = torch.clamp(img_new, -max_val, max_val)

                energy_new = self.model(inp, img_new, t, return_energy=True)
                if len(energy_new.shape) == 2:
                    bad_step = (energy_new > energy)[:, 0]
                elif len(energy_new.shape) == 1:
                    bad_step = (energy_new > energy)
                else:
                    raise ValueError('Bad shape!!!')

                # print("step: ", i, bad_step.float().mean())
                img_new[bad_step] = img[bad_step]

                if eval:
                    img = img_new.detach()
                else:
                    img = img_new

        return img

    def _opt_step_no_reject(self, inp, img, t, mask, data_cond, detach_output=True):
        """Single energy-descent step without greedy acceptance.

        detach_output=False needed inside PGD inner loop where gradients
        must flow back to the PGD variable y.
        """
        energy, grad = self.model(inp, img, t, return_both=True)
        img_new = img - extract(self.opt_step_size, t, grad.shape) * grad
        if mask is not None:
            img_new = img_new * (1 - mask) + mask * data_cond
        sf = 2.0 if self.continuous else 1.0
        max_val = extract(self.sqrt_alphas_cumprod, t, img_new.shape)[0, 0] * sf
        img_new = torch.clamp(img_new, -max_val, max_val)
        return img_new.detach() if detach_output else img_new

    def sample_negatives_langevin(self, inp, x_init, t, k_steps: int):
        """
        Sample negatives using Langevin dynamics (short-run MCMC).

        Implements the overdamped Langevin update:
            x_{k+1} = x_k - η·∇_x E(x_k) + sqrt(2ηT)·ε,  ε ~ N(0, I)
        where η = opt_step_size[t] and T is an effective temperature.

        Gradient convention: we compute ∂(E.mean())/∂x rather than routing
        through DiffusionWrapper's return_both path (which returns
        ∂E.sum()/∂x / B_norm — a training-specific normalization that should
        not enter the MCMC kernel). Using energy.mean() keeps the gradient
        scale independent of the fixed B_norm constant, so the drift/noise
        ratio is determined solely by the model's natural energy scale and the
        temperature parameter (langevin_sigma_multiplier), not by batch size.

        create_graph=False: no second-order computation needed during sampling.

        Parameter freeze: model weights are frozen so autograd only computes
        ∇_x E, not ∇_θ E (Du & Mordatch 2019).

        Returns negatives in xₜ space (same as x_init).
        """
        x = x_init.detach()

        step = extract(self.opt_step_size, t, x.shape)  # η: per-sample, broadcastable
        # sigma = T_mult * sqrt(2η).  T_mult (langevin_sigma_multiplier, default 0.1)
        # is a temperature parameter: T_eff = T_mult².  It governs how broadly the
        # chain explores relative to the drift field of energy.mean().  Keeping it
        # < 1 maintains SNR > 1 so the chain is drift-dominated, not a random walk.
        sigma_mult = self.mining_config.get('langevin_sigma_multiplier', 0.1)
        sigma = sigma_mult * torch.sqrt(2.0 * step)

        # Freeze model parameters during Langevin sampling (Du & Mordatch 2019).
        for p in self.model.parameters():
            p.requires_grad_(False)

        x_init_ref = x.clone()
        grad_norms = []

        for k in range(k_steps):
            # Explicitly enable grad on x — required because return_energy=True
            # does NOT set requires_grad internally (unlike return_both).
            x_g = x.requires_grad_(True)

            # Raw energy, no B_norm division.
            energy = self.model(inp, x_g, t, return_energy=True)  # [B, 1]

            # ∂(E.mean())/∂x = (1/B) * ∂E.sum()/∂x  — per-sample gradient at
            # natural scale, independent of B_norm.  create_graph=False: the
            # sampling graph must not accumulate into the training backward pass.
            grad = torch.autograd.grad(
                energy.mean(), x_g, create_graph=False
            )[0]  # [B, D]

            with torch.no_grad():
                # Langevin step: x ← x - η·∇(E.mean()) + σ·ε
                x = x_g.detach() - step * grad.detach() + sigma * torch.randn_like(x)
                x = torch.clamp(x, -2, 2)

                if k == 0 or k == k_steps - 1:
                    grad_norms.append(grad.detach().norm(dim=-1).mean().item())

        # Re-enable model parameter gradients for training.
        for p in self.model.parameters():
            p.requires_grad_(True)

        self._langevin_diag = {
            'grad_norm_first': grad_norms[0] if grad_norms else 0.0,
            'grad_norm_last': grad_norms[-1] if len(grad_norms) > 1 else grad_norms[0] if grad_norms else 0.0,
            'displacement': (x - x_init_ref).norm(dim=-1).mean().item(),
            'step_mean': step.mean().item(),
            'sigma_mean': sigma.mean().item(),
        }

        return x.detach()

    def sample_negatives_pgd(self, inp, x_init, t, noise_eps, k_steps: int, x_pos_ref=None):
        """
        Generate hard negatives via PGD ascent on the denoising MSE loss.

        Maximizes L(y) = ||model(inp, y, t) - noise_eps||²
        subject to ||y - x_init||₂ ≤ δ  (L2 ball projection after each step).

        When mining_objective='recovery_failure' and x_pos_ref is provided,
        maximizes ||opt_step(y) - x_pos||² instead (recovery-aligned mining).

        Step size η = δ / k_steps (standard PGD scaling).
        Stop-gradient: model parameters are frozen during PGD.

        Args:
            inp:       Conditioning input [B, inp_dim]
            x_init:    Starting point in xₜ space [B, out_dim]
            t:         Timestep indices [B]
            noise_eps: Ground-truth noise added to form x_t [B, out_dim]
            k_steps:   Number of PGD ascent steps

        Returns:
            Adversarial negatives [B, out_dim] (stop-grad, in xₜ space)
        """
        delta = self.mining_config.get('pgd_delta', 1.5)
        eta = self.mining_config.get('pgd_step_size') or (delta / max(k_steps, 1))

        y0 = x_init.detach().clone()
        y = y0.clone()

        # Freeze model parameters during PGD sampling (same convention as Langevin)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Record MSE at init (before any PGD steps) for diagnostic: is gap from init or PGD?
        # Must use enable_grad + requires_grad_(True) because model.forward calls autograd.grad internally.
        with torch.enable_grad():
            y0_g = y0.detach().requires_grad_(True)
            pred_init = self.model(inp.detach(), y0_g, t.detach())
            mse_neg_init = F.mse_loss(pred_init.detach(), noise_eps.detach(), reduction='none')
            mse_neg_init_val = mse_neg_init.reshape(mse_neg_init.shape[0], -1).mean(-1).mean().item()

        mining_obj = self.mining_config.get('mining_objective', 'denoising_mse')

        pgd_grad_norms = []
        for k in range(k_steps):
            y = y.detach().requires_grad_(True)
            if mining_obj == 'recovery_failure' and x_pos_ref is not None:
                # Recovery-aligned mining: maximize ||opt_step(y) - x_pos||²
                y_rec = self._opt_step_no_reject(inp.detach(), y, t.detach(), None, None, detach_output=False)
                diff = y_rec - x_pos_ref.detach()
                loss = (diff ** 2).reshape(diff.shape[0], -1).sum(-1).sum()
            else:
                pred = self.model(inp.detach(), y, t.detach())
                # Per-sample sum (not mean) so gradient scale is independent of out_dim
                mse = F.mse_loss(pred, noise_eps.detach(), reduction='none')  # [B, ...]
                loss = mse.reshape(mse.shape[0], -1).sum(-1).sum()            # scalar
            loss.backward()

            with torch.no_grad():
                grad = y.grad.detach()
                pgd_grad_norms.append(grad.norm(dim=-1).mean().item())
                y = y.detach() + eta * grad
                # L2 ball projection: project Δ = y - y0 onto ||Δ||₂ ≤ δ
                delta_y = y - y0
                norm = delta_y.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
                scale = (delta / norm).clamp(max=1.0)                       # [B, 1]
                y = y0 + delta_y * scale

        # Re-enable model parameter gradients for training
        for p in self.model.parameters():
            p.requires_grad_(True)

        with torch.enable_grad():
            y_g = y.detach().requires_grad_(True)
            pred_final = self.model(inp.detach(), y_g, t.detach())
            mse_neg_final = F.mse_loss(pred_final.detach(), noise_eps.detach(), reduction='none')
            mse_neg_final_val = mse_neg_final.reshape(mse_neg_final.shape[0], -1).mean(-1).mean().item()

        self._pgd_diag = {
            'grad_norm_first': pgd_grad_norms[0] if pgd_grad_norms else 0.0,
            'grad_norm_last': pgd_grad_norms[-1] if len(pgd_grad_norms) > 1 else pgd_grad_norms[0] if pgd_grad_norms else 0.0,
            'displacement': (y - y0).norm(dim=-1).mean().item(),
            'delta': delta,
            'eta': eta,
            'mse_neg_init': mse_neg_init_val,
            'mse_neg_final': mse_neg_final_val,
        }

        return y.detach()

    def _unroll_opt_steps(self, inp, x_init, t, mask, data_cond, k_steps: int):
        """
        Unroll k steps of opt_step from x_init (no gradients through trajectory).
        Returns list of intermediate states [x1, x2, ..., xk].
        Uses step=1 per call and eval=True so each iterate is detached.
        """
        iterates = []
        x = x_init.detach()
        for _ in range(k_steps):
            with torch.enable_grad():
                x = self.opt_step(inp, x, t, mask, data_cond, step=1, eval=True)
            x = x.detach()
            iterates.append(x)
        return iterates

    def _unroll_recovery_steps(self, inp, x_init, t, mask, data_cond, recovery_steps: int):
        """
        Unroll recovery steps from adversarial negative (no gradients through trajectory).
        Similar to _unroll_opt_steps but returns final state instead of list.

        Args:
            x_init: Starting state (typically y_neg from PGD)
            recovery_steps: Number of opt_steps to unroll (default 1-2)

        Returns: Final recovery state y_rec (detached, no gradients through unroll)
        """
        accept_all = self.mining_config.get('recovery_accept_all', False)
        x = x_init.detach()
        for _ in range(recovery_steps):
            with torch.enable_grad():
                if accept_all:
                    x = self._opt_step_no_reject(inp, x, t, mask, data_cond, detach_output=True)
                else:
                    x = self.opt_step(inp, x, t, mask, data_cond, step=1, eval=True)
            x = x.detach()
        return x

    def sample_negatives_tam(self, inp, x_pos, t, noise_eps, mask, data_cond,
                              anchor_step: int, pgd_k_steps: int, recovery_steps: int = 0):
        """
        Trajectory-Anchored Mining (TAM) with optional recovery:
        1. Unroll opt_step for anchor_step steps from x_pos -> x_anchor
        2. PGD from x_anchor with L2 ball centered at x_anchor
        3. (Optional) Recovery: unroll opt_steps from x_neg for recovery_steps

        Args:
            recovery_steps: Number of recovery unroll steps (default 0 = no recovery)

        Returns:
            - If recovery_steps > 0: (x_neg, x_rec) tuple
            - If recovery_steps == 0: x_neg (backward compatible)
        Stores diagnostics in self._tam_diag.
        """
        iterates = self._unroll_opt_steps(inp, x_pos, t, mask, data_cond, k_steps=anchor_step)
        x_anchor = iterates[-1]

        # Reuse existing PGD: x_init=x_anchor centers the L2 ball at the anchor
        mining_obj = self.mining_config.get('mining_objective', 'denoising_mse')
        x_pos_ref = x_pos if mining_obj == 'recovery_failure' else None
        x_neg = self.sample_negatives_pgd(inp, x_anchor, t, noise_eps, k_steps=pgd_k_steps, x_pos_ref=x_pos_ref)

        # Optional recovery unroll from x_neg
        if recovery_steps > 0:
            x_rec = self._unroll_recovery_steps(inp, x_neg, t, mask, data_cond, recovery_steps)
        else:
            x_rec = None

        # Diagnostics (same style as _pgd_diag)
        pgd_diag = getattr(self, '_pgd_diag', {})
        self._tam_diag = {
            'anchor_dist': (x_anchor - x_pos).norm(dim=-1).mean().item(),
            'neg_dist': (x_neg - x_pos).norm(dim=-1).mean().item(),
            'pgd_disp': (x_neg - x_anchor).norm(dim=-1).mean().item(),
            'mse_neg_init': pgd_diag.get('mse_neg_init', float('nan')),
            'mse_neg_final': pgd_diag.get('mse_neg_final', float('nan')),
        }

        # Return appropriately based on recovery_steps
        if recovery_steps > 0:
            return x_neg, x_rec
        else:
            return x_neg

    def compute_matrix_residual(self, A, X):
        """
        Compute matrix inversion residual: ||AX - I||_F

        Args:
            A: Input matrix (flattened, B x rank²)
            X: Predicted inverse (flattened, B x rank²)

        Returns:
            residual: Frobenius norm ||AX - I||_F per sample (B,)
        """
        import math
        batch_size = A.shape[0]
        rank = int(math.sqrt(A.shape[1]))

        # Reshape to matrices
        A_mat = A.view(batch_size, rank, rank)
        X_mat = X.view(batch_size, rank, rank)

        # Compute AX
        AX = torch.bmm(A_mat, X_mat)

        # Compute residual: ||AX - I||_F
        I = torch.eye(rank, device=A.device).unsqueeze(0).expand(batch_size, -1, -1)
        residual = torch.norm(AX - I, p='fro', dim=(1, 2))

        return residual

    def generate_peripheral_samples(self, inp, x_start, t, noise, mining_config):
        """Generate peripheral-distribution (PD) samples for OEST* energy-barrier training.

        Adapts the peripheral sample generation from OEST (Outlier Exposure via Smooth
        Transformations) to the matrix inversion domain. OEST generates near-OOD
        "peripheral" samples via simple transforms of in-distribution data, then trains
        an energy barrier between ID and peripheral samples.

        References:
            - Ming et al., "Revisiting Energy-Based Model for Out-of-Distribution
              Detection", arXiv:2412.03058, 2024. §IV-B: peripheral distribution
              generated via simple data transformations (cutout, rotation, noise, etc.)
            - Hendrycks & Mazeika, "Deep Anomaly Detection with Outlier Exposure",
              ICLR 2019. Original OE framework using auxiliary OOD data.
            - Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
              Energy score E(x) = -T*log Σ exp(f_i(x)/T) for OOD detection.

        Domain adaptation: IRED's training data uses A = UU^T + UU^T + 0.5*I (dataset.py:455),
        while OOD uses +0.1*I (dataset.py:453). We generate peripheral samples by reducing
        the diagonal regularization to bridge this gap, creating matrices with intermediate
        conditioning (between training's κ(0.5) and OOD's κ(0.1)).

        Args:
            inp: Input matrices A, flattened [B, rank²]
            x_start: Ground-truth inverses A⁻¹, flattened [B, rank²]
            t: Diffusion timesteps [B]
            noise: Noise tensor [B, seq_len] (reused for consistency)
            mining_config: Dict with peripheral_* hyperparameters

        Returns:
            inp_pd: Peripheral input matrices [B, rank²]
            x_pd_noisy: Noisy peripheral inverses at timestep t [B, seq_len]
        """
        transform = mining_config.get('peripheral_transform', 'condition')
        batch_size = inp.shape[0]
        rank_sq = inp.shape[1]
        rank = int(math.sqrt(rank_sq))

        if transform == 'condition':
            # Reduce diagonal regularization: A_pd = A - (0.5 - eps)*I
            # where eps ~ Uniform(eps_min, eps_max). This creates matrices with
            # intermediate conditioning between ID (0.5) and OOD (0.1).
            # The resulting diagonal regularization is eps (ranging 0.15-0.4),
            # bridging the distribution gap per OEST §IV-B.
            eps_min = mining_config.get('peripheral_eps_min', 0.15)
            eps_max = mining_config.get('peripheral_eps_max', 0.4)
            eps = torch.empty(batch_size, 1, 1, device=inp.device).uniform_(eps_min, eps_max)

            A_mat = inp.view(batch_size, rank, rank)
            I_mat = torch.eye(rank, device=inp.device).unsqueeze(0)  # [1, rank, rank]

            # Subtract (0.5 - eps)*I to reduce conditioning from 0.5*I to eps*I
            delta = (0.5 - eps) * I_mat  # [B, rank, rank]
            A_pd_mat = A_mat - delta

            # Recompute inverse of the peripheral matrix
            # Use torch.linalg.inv for batched inversion
            A_pd_inv_mat = torch.linalg.inv(A_pd_mat)

            inp_pd = A_pd_mat.reshape(batch_size, rank_sq)
            x_pd_start = A_pd_inv_mat.reshape(batch_size, rank_sq)

        elif transform == 'corrupt_solution':
            # Keep input A unchanged; add rank-1 perturbation to the inverse.
            # y_pd = A⁻¹ + σ*(u @ v^T) where u,v are random unit vectors.
            # This creates structurally near-correct but wrong solutions.
            sigma = mining_config.get('peripheral_corrupt_sigma', 0.1)
            u = torch.randn(batch_size, rank, 1, device=inp.device)
            v = torch.randn(batch_size, 1, rank, device=inp.device)
            u = u / (u.norm(dim=1, keepdim=True) + 1e-8)
            v = v / (v.norm(dim=2, keepdim=True) + 1e-8)
            perturbation = sigma * torch.bmm(u, v)  # [B, rank, rank]

            X_mat = x_start.view(batch_size, rank, rank)
            X_pd_mat = X_mat + perturbation

            inp_pd = inp.clone()
            x_pd_start = X_pd_mat.reshape(batch_size, rank_sq)

        elif transform == 'row_swap':
            # Swap 1-2 random rows of the inverse matrix.
            # Cheap structural corruption that preserves element magnitudes.
            X_mat = x_start.view(batch_size, rank, rank).clone()
            for _ in range(torch.randint(1, 3, (1,)).item()):
                idx = torch.randint(0, rank, (2,))
                X_mat[:, idx[0]], X_mat[:, idx[1]] = X_mat[:, idx[1]].clone(), X_mat[:, idx[0]].clone()

            inp_pd = inp.clone()
            x_pd_start = X_mat.reshape(batch_size, rank_sq)
        else:
            raise ValueError(f"Unknown peripheral transform: {transform}")

        # Noise peripheral targets to the same timestep t (shared noise schedule)
        x_pd_noisy = self.q_sample(x_start=x_pd_start, t=t, noise=noise)

        return inp_pd, x_pd_noisy

    @torch.no_grad()
    def p_sample_loop(self, batch_size, shape, inp, cond, mask, return_traj=False):
        device = self.betas.device

        if hasattr(self.model, 'randn'):
            img = self.model.randn(batch_size, shape, inp, device)
        else:
            img = torch.randn((batch_size, *shape), device=device)

        x_start = None


        if self.show_inference_tqdm:
            iterator = tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps)
        else:
            iterator = reversed(range(0, self.num_timesteps))

        preds = []

        for t in iterator:
            self_cond = x_start if self.self_condition else None
            batched_times = torch.full((img.shape[0],), t, device = inp.device, dtype = torch.long)

            cond_val = None
            if mask is not None:
                cond_val = self.q_sample(x_start = inp, t = batched_times, noise = torch.zeros_like(inp))
                img = img * (1 - mask) + cond_val * mask

            img, x_start = self.p_sample(inp, img, t, self_cond, scale=False, with_noise=self.baseline)

            if mask is not None:
                img = img * (1 - mask) + cond_val * mask

            # if t < 50:

            if self.sudoku:
                step = 20
            else:
                step = 5

            if self.use_innerloop_opt:
                if t < 1:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)
                else:
                    img = self.opt_step(inp, img, batched_times, mask, cond_val, step=step, sf=1.0)

                img = img.detach()

            if self.continuous:
                sf = 2.0
            elif self.shortest_path:
                sf = 0.1
            else:
                sf = 1.0

            # This clip threshold needs to be adjust to be larger for generalizations settings
            max_val = extract(self.sqrt_alphas_cumprod, batched_times, x_start.shape)[0, 0] * sf

            img = torch.clamp(img, -max_val, max_val)

            # Correctly scale output
            img_unscaled = self.predict_start_from_noise(img, batched_times, torch.zeros_like(img))
            preds.append(img_unscaled)

            batched_times_prev = batched_times - 1

            if t != 0:
                img = extract(self.sqrt_alphas_cumprod, batched_times_prev, img_unscaled.shape) * img_unscaled
            # img, _, _ = self.q_posterior(img_unscaled, img, batched_times)

        if return_traj:
            return torch.stack(preds, dim=0)
        else:
            return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    @torch.no_grad()
    def sample(self, x, label, mask, batch_size = 16, return_traj=False):
        # seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(batch_size, self.out_shape, x, label, mask, return_traj=return_traj)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, inp, x_start, mask, t, noise = None):
        b, *c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        if mask is not None:
            # Mask out inputs
            x_cond = self.q_sample(x_start = inp, t = t, noise = torch.zeros_like(noise))
            x = x * (1 - mask) + mask * x_cond

        # predict and take gradient step

        model_out = self.model(inp, x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if mask is not None:
            # Mask out targets
            model_out = model_out * (1 - mask) + mask * target

        loss = F.mse_loss(model_out, target, reduction = 'none')

        if self.shortest_path:
            mask1 = (x_start > 0)
            mask2 = torch.logical_not(mask1)
            # mask1, mask2 = mask1.float(), mask2.float()
            weight = mask1 * 10 + mask2 * 0.5
            # loss = (loss * weight) / weight.sum() * target.numel()
            loss = loss * weight

        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss_mse = loss

        if self.supervise_energy_landscape:
            use_ired_contrastive = self.mining_config.get('use_ired_contrastive_loss', False)
            if use_ired_contrastive:
                # Reuse the same noisy sample from the denoising path to reduce variance
                data_sample = x
                noise = torch.randn_like(x_start)  # still needed for negative init below
            else:
                noise = torch.randn_like(x_start)
                data_sample = self.q_sample(x_start = x_start, t = t, noise = noise)

            if mask is not None:
                data_cond = self.q_sample(x_start = x_start, t = t, noise = torch.zeros_like(noise))
                data_sample = data_sample * (1 - mask) + mask * data_cond

            # Add a noise contrastive estimation term with samples drawn from the data distribution
            #noise = torch.randn_like(x_start)

            # Optimize a sample using gradient descent on energy landscape
            xmin_noise = self.q_sample(x_start = x_start, t = t, noise = 3.0 * noise)

            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond
            else:
                data_cond = None

            if self.sudoku:
                s = x_start.size()
                x_start_im = x_start.view(-1, 9, 9, 9).argmax(dim=-1)
                randperm = torch.randint(0, 9, x_start_im.size(), device=x_start_im.device)

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_im = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                xmin_noise_im = F.one_hot(xmin_noise_im.long(), num_classes=9)
                xmin_noise_im = (xmin_noise_im - 0.5) * 2

                xmin_noise_rescale = xmin_noise_im.view(-1, 729)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.connectivity:
                s = x_start.size()
                x_start_im = x_start.view(-1, 12, 12)
                randperm = (torch.randint(0, 1, x_start_im.size(), device=x_start_im.device) - 0.5) * 2

                rand_mask = (torch.rand(x_start_im.size(), device=x_start_im.device) < 0.05).float()

                xmin_noise_rescale = x_start_im * (1 - rand_mask) + randperm * (rand_mask)

                loss_opt = torch.ones(1)

                loss_scale = 0.05
            elif self.shortest_path:
                x_start_list = x_start.argmax(dim=2)
                classes = x_start.size(2)
                rand_vals = torch.randint(0, classes, x_start_list.size()).to(x_start.device)

                x_start_neg = torch.cat([rand_vals[:, :1], x_start_list[:, 1:]], dim=1)
                x_start_neg_oh = F.one_hot(x_start_neg[:, :, 0].long(), num_classes=classes)[:, :, :, None]
                xmin_noise_rescale = (x_start_neg_oh - 0.5) * 2

                loss_opt = torch.ones(1)

                loss_scale = 0.5
            else:
                # Mining strategy for continuous tasks (matrix inversion, addition, etc.)
                if self.mining_strategy == 'none':
                    # No negative mining: skip energy supervision
                    xmin_noise_rescale = x_start
                    loss_opt = torch.zeros(1, device=x_start.device)
                    loss_scale = 0.0  # Zero out energy loss contribution

                elif self.mining_strategy == 'random':
                    # Random negative mining: use random samples from the data distribution
                    xmin_noise_rescale = torch.randn_like(x_start)
                    xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)
                    loss_opt = torch.zeros(1, device=x_start.device)
                    loss_scale = 0.5

                elif self.mining_strategy == 'adversarial':
                    # Adversarial negative mining: gradient-based hard negatives via opt_step (baseline)
                    opt_steps = self.mining_config.get('opt_steps', 2)
                    xmin_noise = self.opt_step(inp, xmin_noise, t, mask, data_cond, step=opt_steps, sf=1.0)
                    xmin = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                    loss_opt = torch.pow(xmin_noise - xmin, 2).mean()

                    xmin_noise = xmin_noise.detach()
                    xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
                    xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)
                    # Default loss_scale=0.5 for NCE path; allow config override for CD path (q205)
                    loss_scale = self.mining_config.get('energy_loss_weight', 0.5)

                elif self.mining_strategy in ['cd_langevin', 'cd_langevin_replay', 'cd_full']:
                    # CD-style training with Langevin sampling
                    opt_steps = self.mining_config.get('opt_steps', 10)
                    noise_scale = self.mining_config.get('noise_scale', 1.5)

                    # Initialize negatives: replay-or-fresh in xₜ space
                    if self.use_replay_buffer and torch.rand(()).item() < self.replay_prob:
                        x_init = self.replay_buffer.sample(t, self.num_timesteps, x_start.device)
                        if x_init is None:
                            x_init = self.q_sample(x_start=x_start, t=t, noise=noise_scale * torch.randn_like(x_start))
                    else:
                        x_init = self.q_sample(x_start=x_start, t=t, noise=noise_scale * torch.randn_like(x_start))

                    # Sample negatives: PGD (q217+) or Langevin (default)
                    if self.mining_config.get('use_pgd_negatives', False):
                        # q218+: center PGD ball on the positive noisy sample (data_sample)
                        # so δ-constraint is relative to the manifold, not a far noise init.
                        # q217: center=x_init (far noise init), q218: center=data_sample
                        pgd_center = data_sample if self.mining_config.get('pgd_center_on_positive', False) else x_init
                        xmin_noise = self.sample_negatives_pgd(inp, pgd_center, t, noise, k_steps=opt_steps)
                    else:
                        xmin_noise = self.sample_negatives_langevin(inp, x_init, t, k_steps=opt_steps)

                    # Add to replay buffer for future use (still in xₜ space)
                    if self.use_replay_buffer:
                        self.replay_buffer.add(xmin_noise, t, self.num_timesteps)

                    # Convert to x₀ space for residual filtering
                    xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
                    xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)

                    loss_opt = torch.zeros(1, device=x_start.device)
                    loss_scale = self.mining_config.get('energy_loss_weight', 0.05)

                elif self.mining_strategy == 'tam':
                    # Trajectory-Anchored Mining: anchor PGD on intermediate opt_step iterates
                    anchor_step = self.mining_config.get('tam_anchor_step', 2)
                    pgd_k = self.mining_config.get('opt_steps', 3)
                    recovery_steps = self.mining_config.get('recovery_steps', 0) if self.mining_config.get('use_recovery_loss', False) else 0

                    tam_out = self.sample_negatives_tam(
                        inp, data_sample, t, noise, mask, x_start,
                        anchor_step=anchor_step,
                        pgd_k_steps=pgd_k,
                        recovery_steps=recovery_steps,
                    )

                    # Unpack TAM output (handles both recovery and non-recovery cases)
                    if recovery_steps > 0:
                        xmin_noise, xmin_noise_rec = tam_out
                    else:
                        xmin_noise = tam_out
                        xmin_noise_rec = None

                    # TAM output is already in x_t space, no need for q_sample re-wrap
                    xmin_noise_rescale = self.predict_start_from_noise(xmin_noise, t, torch.zeros_like(xmin_noise))
                    xmin_noise_rescale = torch.clamp(xmin_noise_rescale, -2, 2)
                    loss_opt = torch.zeros(1, device=x_start.device)
                    loss_scale = self.mining_config.get('energy_loss_weight', 0.05)

                else:
                    raise ValueError(f"Unknown mining strategy: {self.mining_strategy}")

            # Re-noise for non-CD and non-TAM strategies only
            # For CD: xmin_noise is already refined in xₜ space from Langevin
            # For TAM: xmin_noise is already in xₜ space from PGD
            if self.mining_strategy not in ['cd_langevin', 'cd_langevin_replay', 'cd_full', 'tam']:
                xmin_noise = self.q_sample(x_start=xmin_noise_rescale, t=t, noise=noise)

            if mask is not None:
                xmin_noise = xmin_noise * (1 - mask) + mask * data_cond

            # Compute energy of both distributions
            inp_concat = torch.cat([inp, inp], dim=0)
            # CRITICAL: Detach negative samples to prevent gradient flow through sampler
            x_concat = torch.cat([data_sample, xmin_noise.detach()], dim=0)
            t_concat = torch.cat([t, t], dim=0)
            energy = self.model(inp_concat, x_concat, t_concat, return_energy=True)

            # Choose loss based on strategy
            use_cd_loss = self.mining_config.get('use_cd_loss', False)
            use_ired_contrastive = self.mining_config.get('use_ired_contrastive_loss', False)
            _gc_mse_pos = _gc_mse_neg = 0.0  # populated only in gradient-contrastive path

            if use_ired_contrastive:
                # IRED-style contrastive loss: softplus(E_pos - E_neg)
                # Enforces E_pos < E_neg via a stable logistic form.
                # Unlike the CD (E_neg - E_pos) term which fights the architecture,
                # this directly pushes positives to lower energy than negatives.
                energy_pos, energy_neg = torch.chunk(energy, 2, 0)

                temp = self.mining_config.get('contrastive_temperature', 1.0)
                detach_epos = self.mining_config.get('detach_epos_contrastive', False)
                e_pos = energy_pos.squeeze(-1)  # [B]
                e_neg = energy_neg.squeeze(-1)  # [B]
                # detach_epos: only E_neg receives contrastive gradient; E_pos grows freely from denoising
                e_pos_logit = e_pos.detach() if detach_epos else e_pos
                logits = (e_pos_logit - e_neg) / temp  # want e_pos < e_neg
                loss_energy = F.softplus(logits).unsqueeze(-1)  # [B, 1]

                # Energy magnitude regularization (same as CD path)
                energy_reg_weight = self.mining_config.get('energy_reg_weight', 0.1)
                loss_energy_reg = energy_reg_weight * (energy_pos.pow(2) + energy_neg.pow(2))  # [B,1]

            elif use_cd_loss:
                # Architecture-aware energy loss for IRED.
                #
                # IMPORTANT: In this architecture, the denoising prediction IS ∂E/∂x
                # (DiffusionWrapper returns autograd.grad([E], [x])). This means:
                #   E = ||fc4(h)||²  →  ∂E/∂x = 2·fc4(h)·(∂fc4(h)/∂x)
                #
                # Standard CD minimizes (E_pos - E_neg), which pushes E_pos → 0, which
                # collapses fc4(h_pos) → 0, which collapses ∂E/∂x_pos → 0, which collapses
                # the denoising prediction to zero → MSE → 1.0.
                #
                # Using (E_neg - E_pos) instead:
                # → minimizing drives E_neg down and E_pos UP
                # → E_pos = ||fc4(h_pos)||² stays large → ∂E/∂x_pos stays nonzero → denoising works
                # → E_neg gets lower energy (negatives pulled toward model distribution)
                #
                # This is NOT standard CD, but it is compatible with the IRED architecture
                # where the energy gradient is the prediction. The denoising loss dominates
                # and the energy loss provides an auxiliary shaping signal on the landscape.
                energy_pos, energy_neg_raw = torch.chunk(energy, 2, 0)
                energy_neg = energy_neg_raw  # Gradients flow to θ, not through x_neg

                # IRED-aware energy shaping.
                # In this architecture E = ||fc4(h)||², so E_pos must stay large
                # (otherwise ∂E/∂x → 0 and denoising collapses).
                # Use margin ranking: relu(margin + E_neg - E_pos)
                # This penalizes only when E_neg is too close to E_pos (within margin),
                # and is bounded (zero loss when ordering is satisfied by margin).
                cd_margin = self.mining_config.get('cd_margin', 1.0)
                loss_energy = F.relu(cd_margin + energy_neg - energy_pos)  # [B,1]

                # Energy magnitude regularization: L2-penalize energy values to prevent
                # them diverging to arbitrary magnitudes while preserving the sign of
                # (E_pos - E_neg). Du & Mordatch (2019) §3.2 use this regularizer:
                #   λ_reg * (E_pos² + E_neg²)
                # It is added OUTSIDE the residual filter and timestep mask so that
                # the magnitude is always bounded regardless of which samples are kept.
                # Effective weight in the final loss = loss_scale * energy_reg_weight
                # (e.g., 0.05 * 0.1 = 0.005 per unit E²). Keep energy_reg_weight small
                # (default 0.1) so the regularizer doesn't dominate the CD objective.
                energy_reg_weight = self.mining_config.get('energy_reg_weight', 0.1)
                loss_energy_reg = energy_reg_weight * (energy_pos.pow(2) + energy_neg.pow(2))  # [B,1]

                # Residual filtering (false-negative removal)
                use_residual_filter = self.mining_config.get('use_residual_filter', False)
                if use_residual_filter:
                    # Compute residuals in x₀ space
                    residuals = self.compute_matrix_residual(inp, xmin_noise_rescale)  # (B,)

                    # Get threshold (fixed or quantile)
                    tau = self.mining_config.get('residual_tau', None)
                    if tau is None:
                        # Fallback: batch quantile
                        q = self.mining_config.get('residual_filter_quantile', 0.3)
                        tau = torch.quantile(residuals.detach(), q).item()

                    # Keep negatives with residual >= tau (filter false negatives)
                    keep = (residuals >= tau).float().unsqueeze(1)  # (B,1)
                    loss_energy = loss_energy * keep

                # Energy loss scheduling
                use_energy_schedule = self.mining_config.get('use_energy_schedule', False)
                if use_energy_schedule:
                    warmup = self.mining_config.get('energy_loss_warmup_steps', 20000)
                    max_w = self.mining_config.get('energy_loss_max_weight', 0.05)
                    step = self.global_step
                    loss_scale = max_w * min(1.0, step / warmup)

                # Timestep range filtering
                use_timestep_range = self.mining_config.get('use_timestep_range', False)
                if use_timestep_range:
                    lo, hi = self.mining_config.get('energy_loss_timestep_range', [0.2, 0.8])
                    tmin = int(lo * (self.num_timesteps - 1))
                    tmax = int(hi * (self.num_timesteps - 1))
                    t_mask = ((t >= tmin) & (t <= tmax)).float().unsqueeze(1)  # (B,1)

                    # Apply mask to loss_energy (per-sample), not to loss_scale (scalar)
                    loss_energy = loss_energy * t_mask

            elif self.mining_config.get('use_gradient_contrastive', False):
                # Gradient-contrastive: shape the gradient FIELD, not energy values.
                # Instead of E_pos < E_neg, enforce: denoising_error(x_pos) < denoising_error(x_neg).
                # Loss: softplus((MSE_pos - MSE_neg) / T)
                # - MSE_pos = per-sample ||pred_pos - ε||² (already computed as loss_mse)
                # - MSE_neg = per-sample ||pred_neg - ε||² where pred_neg = model(inp, x_neg, t)
                # Detach MSE_pos from contrastive so denoising is only trained by regular MSE loss.
                # Contrastive only pushes pred_neg AWAY from ε at negative sample locations.
                pred_neg = self.model(inp, xmin_noise.detach(), t)
                mse_pos_gc = reduce(loss_mse, 'b ... -> b', 'mean').detach()  # [B], no contrastive grad
                mse_neg_gc = reduce(
                    F.mse_loss(pred_neg, target, reduction='none'), 'b ... -> b', 'mean'
                )  # [B]
                gc_temp = self.mining_config.get('gc_temperature', 1.0)
                loss_energy = F.softplus((mse_pos_gc - mse_neg_gc) / gc_temp).unsqueeze(-1)  # [B,1]
                loss_energy_reg = torch.zeros_like(loss_energy)
                # Store for GC diagnostic
                _gc_mse_pos = mse_pos_gc.mean().item()
                _gc_mse_neg = mse_neg_gc.mean().item()
                energy_pos = energy_neg = None  # not used in GC path

            else:
                # Original NCE (baseline - for comparison)
                energy_pos, energy_neg = torch.chunk(energy, 2, 0)
                energy_stack = torch.cat([energy_pos, energy_neg], dim=-1)
                target = torch.zeros(energy_pos.size(0)).to(energy_stack.device)
                loss_energy = F.cross_entropy(-1 * energy_stack, target.long(), reduction='none')[:, None]

            # === RECOVERY LOSS (TAM-CTL) ===
            loss_rec_weighted = 0.0
            if self.mining_strategy == 'tam' and self.mining_config.get('use_recovery_loss', False):
                if xmin_noise_rec is not None:
                    recovery_target_mode = self.mining_config.get('recovery_target', 'noise')
                    recovery_loss_weight = self.mining_config.get('recovery_loss_weight', 0.1)

                    # Compute denoising target
                    if recovery_target_mode == 'implied_noise':
                        # Geometrically correct: ε_impl = (x_rec - √ᾱ·x₀) / √(1-ᾱ)
                        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
                        sqrt_1mab = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                        eps_impl = (xmin_noise_rec.detach() - sqrt_ab * x_start) / (sqrt_1mab + 1e-8)
                        rec_target = eps_impl.detach()
                    else:
                        rec_target = noise

                    pred_rec = self.model(inp, xmin_noise_rec.detach(), t)
                    loss_rec = F.mse_loss(pred_rec, rec_target, reduction='none')
                    loss_rec = reduce(loss_rec, 'b ... -> b', 'mean')
                    loss_rec_weighted = loss_rec * recovery_loss_weight

                    # Trainable ranking loss: recovery should denoise better than negative
                    if self.mining_config.get('use_recovery_rank_loss', False):
                        margin = self.mining_config.get('recovery_rank_margin', 0.1)
                        pred_neg_rank = self.model(inp, xmin_noise.detach(), t)
                        if recovery_target_mode == 'implied_noise':
                            eps_impl_neg = (xmin_noise.detach() - sqrt_ab * x_start) / (sqrt_1mab + 1e-8)
                            target_neg = eps_impl_neg.detach()
                        else:
                            target_neg = noise
                        loss_neg_rank = reduce(F.mse_loss(pred_neg_rank, target_neg, reduction='none'), 'b ... -> b', 'mean')
                        rank_loss = F.relu(margin + loss_rec - loss_neg_rank)
                        rank_weight = self.mining_config.get('recovery_rank_weight', 0.1)
                        loss_rec_weighted = loss_rec_weighted + rank_loss * rank_weight

                    # Timestep weighting: w(t) = (t/T)^gamma — upweight noisier timesteps
                    aux_gamma = self.mining_config.get('aux_timestep_gamma', 0.0)
                    if aux_gamma > 0:
                        t_frac = t.float() / max(float(self.num_timesteps - 1), 1.0)
                        aux_w = t_frac.pow(aux_gamma)  # [B]
                        loss_rec_weighted = loss_rec_weighted * aux_w

            # === SCORE SMOOTHING LOSS ===
            # Penalize ||f(x_ref + ξ, t) - f(x_ref, t)||² where ξ ~ N(0, σ²I)
            # Encourages a smoother score field for better OOD generalization.
            loss_smooth = torch.tensor(0.0, device=x_start.device)
            smooth_sigma = self.mining_config.get('score_smooth_sigma', 0.0)
            smooth_weight = self.mining_config.get('score_smooth_weight', 0.0)
            if smooth_sigma > 0 and smooth_weight > 0:
                xi = torch.randn_like(data_sample) * smooth_sigma
                x_ref = data_sample.detach()
                pred_clean = self.model(inp, x_ref, t)
                pred_perturbed = self.model(inp, x_ref + xi, t)
                loss_smooth = reduce(
                    F.mse_loss(pred_perturbed, pred_clean.detach(), reduction='none'),
                    'b ... -> b', 'mean'
                ) * smooth_weight  # [B]

            # === PERIPHERAL DISTRIBUTION LOSS (OEST*) ===
            # Energy-barrier loss from OEST* (Ming et al., arXiv:2412.03058, Eq. 12):
            #   L_energy* = -α * log σ((E(x_per) - E(x_in)) / β)
            # where σ is the sigmoid function, α is the loss weight, and β controls
            # the sigmoid temperature. This loss trains an energy barrier between
            # in-distribution and peripheral (near-OOD) samples, encouraging
            # E(x_peripheral) > E(x_in) without requiring explicit OOD data.
            #
            # Theoretical basis (OEST* Theorem 1): If an energy barrier exists between
            # ID and peripheral samples, it generalizes to true OOD samples under
            # Lipschitz continuity of the energy function.
            #
            # Adaptation to IRED: OEST was designed for classifiers where
            # E(x) = -T*log Σ exp(f_i(x)/T) (Liu et al., NeurIPS 2020). IRED has a
            # direct scalar energy E(x,y) = ||fc4(h)||², so we apply the OEST* loss
            # directly to IRED's raw energy values without the LogSumExp transform.
            loss_peripheral_scalar = torch.tensor(0.0, device=x_start.device)
            if self.mining_config.get('use_peripheral_loss', False):
                pd_alpha = self.mining_config.get('peripheral_alpha', 0.2)
                pd_beta = self.mining_config.get('peripheral_beta', 10.0)

                # Generate peripheral samples and noise to same timestep t
                inp_pd, x_pd_noisy = self.generate_peripheral_samples(
                    inp, x_start, t, noise, self.mining_config
                )

                # Compute energies: E(x_in) from positive sample, E(x_per) from peripheral
                # Reuse energy_pos if available (from CD/NCE path), otherwise compute fresh
                if energy_pos is not None:
                    E_in = energy_pos.squeeze(-1).mean()  # scalar
                else:
                    E_in = self.model(inp, data_sample.detach(), t, return_energy=True).squeeze(-1).mean()

                E_pd = self.model(inp_pd, x_pd_noisy.detach(), t, return_energy=True).squeeze(-1).mean()

                # OEST* Eq. 12: L = -α * log(σ((E_pd - E_in) / β))
                # Numerically stable via F.logsigmoid
                e_gap = (E_pd - E_in) / pd_beta
                loss_peripheral_scalar = -pd_alpha * F.logsigmoid(e_gap)

            # Combine losses.
            # Shape fix: loss_mse is [B, seq_len] (per-token denoising error),
            # loss_energy is [B, 1] (per-sample CD loss). Adding them directly would
            # broadcast energy to [B, seq_len], making each of the seq_len MSE terms
            # carry its own copy of the energy gradient — effectively multiplying the
            # energy loss weight by seq_len. Instead, reduce MSE to [B] first so that
            # the energy and MSE losses are combined at the per-sample level, then
            # averaged over the batch. This gives each loss term its intended weight.
            loss_mse_reduced = reduce(loss_mse, 'b ... -> b', 'mean')  # [B, seq_len] → [B]
            loss_energy_reduced = loss_energy.squeeze(-1)               # [B, 1] → [B]

            # Total loss: L = L_mse + λ_energy * L_energy + λ_rec * L_rec + L_peripheral
            # Peripheral loss is a scalar (batch-averaged energy gap), broadcasts to [B].
            loss = loss_mse_reduced + loss_scale * loss_energy_reduced + loss_peripheral_scalar  # [B]
            if isinstance(loss_rec_weighted, torch.Tensor):
                loss = loss + loss_rec_weighted  # Add recovery loss if TAM-CTL enabled
            if isinstance(loss_smooth, torch.Tensor) and loss_smooth.dim() > 0:
                loss = loss + loss_smooth  # Add score smoothing loss

            # Add energy magnitude regularization unconditionally (not masked by
            # residual filter or timestep range, since magnitude bounding should
            # apply regardless of which samples are selected as hard negatives).
            # Effective weight = loss_scale * energy_reg_weight (e.g. 0.05 * 0.1 = 0.005).
            if use_cd_loss or use_ired_contrastive:
                loss_reg_reduced = loss_energy_reg.squeeze(-1)  # [B, 1] → [B]
                loss = loss + loss_scale * loss_reg_reduced

            # Increment global step for scheduling
            self.global_step += 1

            # === LOSS COMPONENT LOGGING ===
            if self.global_step % 100 == 0:
                with torch.no_grad():
                    _l_mse = loss_mse_reduced.mean().item()
                    _l_energy_raw = loss_energy_reduced.mean().item() if isinstance(loss_energy_reduced, torch.Tensor) else 0.0
                    _l_energy_wtd = (loss_scale * loss_energy_reduced).mean().item() if isinstance(loss_energy_reduced, torch.Tensor) else 0.0
                    _l_rec = loss_rec_weighted.mean().item() if isinstance(loss_rec_weighted, torch.Tensor) else 0.0
                    _l_total = loss.mean().item()
                    _extras = ""
                    _l_smooth = loss_smooth.mean().item() if isinstance(loss_smooth, torch.Tensor) and loss_smooth.dim() > 0 else 0.0
                    if _l_smooth > 0:
                        _extras += f" smooth={_l_smooth:.6f}"
                    if use_cd_loss or use_ired_contrastive:
                        _e_pos = energy_pos.mean().item()
                        _e_neg = energy_neg.mean().item()
                        _extras = f" E_pos={_e_pos:.4f} E_neg={_e_neg:.4f} E_gap={_e_neg - _e_pos:.4f}"
                    print(
                        f"[LOSS-DIAG step={self.global_step}] "
                        f"mse={_l_mse:.6f} energy_raw={_l_energy_raw:.6f} energy_wtd={_l_energy_wtd:.6f} "
                        f"rec={_l_rec:.6f} total={_l_total:.6f}{_extras}",
                        flush=True
                    )

            # === DIAGNOSTIC LOGGING ===
            use_gc = self.mining_config.get('use_gradient_contrastive', False)
            if self.global_step % 100 == 0:
                with torch.no_grad():
                    neg_dist = (xmin_noise - data_sample).norm(dim=-1).mean().item()
                    model_out_norm = model_out.norm(dim=-1).mean().item()
                    pred_std = model_out.std().item()
                    pred_abs = model_out.abs().mean().item()
                    tgt_std_val = noise.std().item()
                    tgt_abs_val = noise.abs().mean().item()
                    pred_f = model_out.reshape(b, -1)
                    tgt_f = noise.reshape(b, -1)
                    cos_sim = F.cosine_similarity(pred_f, tgt_f, dim=-1).mean().item()
                    diag = getattr(self, '_langevin_diag', {})
                    mse_val = loss_mse_reduced.mean().item()

                if use_gc:
                    use_pgd = self.mining_config.get('use_pgd_negatives', False)
                    use_tam = self.mining_strategy == 'tam'
                    pgd_diag = getattr(self, '_pgd_diag', {})
                    tam_diag = getattr(self, '_tam_diag', {})

                    if use_tam:
                        # TAM-specific diagnostics (and TAM-CTL recovery diagnostics if enabled)
                        neg_tag = "TAM-GC"
                        tam_extra = (
                            f" anchor_dist={tam_diag.get('anchor_dist', float('nan')):.4f}"
                            f" pgd_disp={tam_diag.get('pgd_disp', float('nan')):.4f}"
                            f" mse_neg_init={tam_diag.get('mse_neg_init', float('nan')):.6f}"
                            f" mse_neg_final={tam_diag.get('mse_neg_final', float('nan')):.6f}"
                        )

                        # Add recovery diagnostics if TAM-CTL is enabled
                        recovery_extra = ""
                        if self.mining_config.get('use_recovery_loss', False) and xmin_noise_rec is not None:
                            with torch.no_grad():
                                mse_pos_val = _gc_mse_pos  # Already have this
                                mse_neg_val = _gc_mse_neg  # Already have this
                                pred_rec = self.model(inp, xmin_noise_rec.detach(), t)
                                mse_rec_val = F.mse_loss(pred_rec, noise).item()
                                rec_dist = (xmin_noise_rec - xmin_noise).norm(dim=-1).mean().item()
                                recovery_extra = (
                                    f" mse_rec={mse_rec_val:.6f}"
                                    f" rec_dist={rec_dist:.4f}"
                                    f" neg_vs_rec={mse_neg_val - mse_rec_val:.6f}"
                                )

                        print(
                            f"[{neg_tag}-DIAG step={self.global_step}] "
                            f"mse_pos={_gc_mse_pos:.6f} mse_neg={_gc_mse_neg:.6f} "
                            f"mse_gap={(_gc_mse_neg - _gc_mse_pos):.6f} "
                            f"pred_norm={model_out_norm:.4f} pred_std={pred_std:.4f} pred_abs={pred_abs:.4f} "
                            f"tgt_std={tgt_std_val:.4f} tgt_abs={tgt_abs_val:.4f} cos={cos_sim:.4f} "
                            f"neg_dist={neg_dist:.4f}"
                            f"{tam_extra}"
                            f"{recovery_extra}",
                            flush=True
                        )
                    else:
                        # PGD or GC diagnostics
                        neg_tag = "PGD-GC" if use_pgd else "GC"
                        pgd_extra = (
                            f" pgd_grad0={pgd_diag.get('grad_norm_first', 0):.4f}"
                            f" pgd_gradK={pgd_diag.get('grad_norm_last', 0):.4f}"
                            f" pgd_disp={pgd_diag.get('displacement', 0):.4f}"
                            f" mse_neg_init={pgd_diag.get('mse_neg_init', 0):.4f}"
                            f" mse_neg_final={pgd_diag.get('mse_neg_final', 0):.4f}"
                            f" delta={pgd_diag.get('delta', 0):.3f}"
                            f" eta={pgd_diag.get('eta', 0):.4f}"
                        ) if use_pgd else ""
                        print(
                            f"[{neg_tag}-DIAG step={self.global_step}] "
                            f"mse_pos={_gc_mse_pos:.6f} mse_neg={_gc_mse_neg:.6f} "
                            f"mse_gap={(_gc_mse_neg - _gc_mse_pos):.6f} "
                            f"pred_norm={model_out_norm:.4f} pred_std={pred_std:.4f} pred_abs={pred_abs:.4f} "
                            f"tgt_std={tgt_std_val:.4f} tgt_abs={tgt_abs_val:.4f} cos={cos_sim:.4f} "
                            f"neg_dist={neg_dist:.4f}"
                            f"{pgd_extra}",
                            flush=True
                        )
                # === TAM-CTL RECOVERY DIAGNOSTICS (independent of GC/CD path) ===
                if self.mining_strategy == 'tam' and self.mining_config.get('use_recovery_loss', False):
                    if xmin_noise_rec is not None:
                        with torch.no_grad():
                            tam_diag = getattr(self, '_tam_diag', {})
                            # Compute MSE at each stage (use noise, not target, which may be reassigned)
                            pred_pos = model_out  # Already have from forward pass
                            mse_pos_diag = F.mse_loss(pred_pos, noise).item()
                            # Model forward pass requires gradients internally, so enable them even in no_grad context
                            with torch.enable_grad():
                                pred_neg = self.model(inp, xmin_noise.detach(), t)
                            mse_neg_diag = F.mse_loss(pred_neg.detach(), noise).item()
                            with torch.enable_grad():
                                pred_rec = self.model(inp, xmin_noise_rec.detach(), t)
                            mse_rec_diag = F.mse_loss(pred_rec.detach(), noise).item()
                            rec_dist_diag = (xmin_noise_rec - xmin_noise).norm(dim=-1).mean().item()

                            # Enhanced geometry diagnostics
                            dist_rec_to_pos = (xmin_noise_rec - data_sample).norm(dim=-1).mean().item()
                            sqrt_ab_d = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
                            sqrt_1mab_d = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                            eps_impl_d = (xmin_noise_rec.detach() - sqrt_ab_d * x_start) / (sqrt_1mab_d + 1e-8)
                            noise_norm_d = noise.norm(dim=-1).mean().item()
                            eps_impl_gap = (eps_impl_d - noise).norm(dim=-1).mean().item()
                            eps_impl_gap_rel = eps_impl_gap / (noise_norm_d + 1e-8)

                            # Dynamics: does recovery point toward basin?
                            rec_dir = xmin_noise_rec - xmin_noise
                            pos_dir = data_sample - xmin_noise
                            cos_recovery = F.cosine_similarity(
                                rec_dir.reshape(rec_dir.shape[0], -1),
                                pos_dir.reshape(pos_dir.shape[0], -1), dim=-1
                            ).mean().item()

                            # Recovery gain ratio
                            rec_gain = (mse_neg_diag - mse_rec_diag) / (mse_neg_diag + 1e-8)
                            # PGD radius actual
                            pgd_radius = tam_diag.get('pgd_disp', float('nan'))

                            # Rank loss (if enabled)
                            rank_extra = ""
                            if self.mining_config.get('use_recovery_rank_loss', False):
                                with torch.enable_grad():
                                    _pr = self.model(inp, xmin_noise.detach(), t)
                                _rec_target_d = noise  # approximate
                                if self.mining_config.get('recovery_target', 'noise') == 'implied_noise':
                                    _rec_target_d = (xmin_noise_rec.detach() - sqrt_ab_d * x_start) / (sqrt_1mab_d + 1e-8)
                                _neg_target_d = noise
                                if self.mining_config.get('recovery_target', 'noise') == 'implied_noise':
                                    _neg_target_d = (xmin_noise.detach() - sqrt_ab_d * x_start) / (sqrt_1mab_d + 1e-8)
                                _rl_rec = F.mse_loss(pred_rec.detach(), _rec_target_d.detach()).item()
                                _rl_neg = F.mse_loss(_pr.detach(), _neg_target_d.detach()).item()
                                _m = self.mining_config.get('recovery_rank_margin', 0.1)
                                _rank_raw = max(0.0, _m + _rl_rec - _rl_neg)
                                rank_extra = f" rank_raw={_rank_raw:.6f}"

                            print(
                                f"[TAM-CTL-DIAG step={self.global_step}] "
                                f"mse_pos={mse_pos_diag:.6f} mse_neg={mse_neg_diag:.6f} mse_rec={mse_rec_diag:.6f} "
                                f"neg_vs_rec={mse_neg_diag - mse_rec_diag:.6f} rec_gain={rec_gain:.4f} "
                                f"rec_dist={rec_dist_diag:.4f} pgd_radius={pgd_radius:.4f} "
                                f"anchor_dist={tam_diag.get('anchor_dist', float('nan')):.4f} "
                                f"dist_rec_pos={dist_rec_to_pos:.4f} eps_gap_rel={eps_impl_gap_rel:.4f} "
                                f"cos_rec={cos_recovery:.4f}{rank_extra}",
                                flush=True
                            )

                else:
                    # Energy-based path: compute gradE and log energy stats
                    x_pos_diag = data_sample.detach().requires_grad_(True)
                    e_diag = self.model(inp.detach(), x_pos_diag, t.detach(), return_energy=True)
                    grad_E_x = torch.autograd.grad(e_diag.sum(), x_pos_diag, create_graph=False)[0]
                    grad_E_norm = grad_E_x.norm(dim=-1).mean().item()
                    with torch.no_grad():
                        e_pos_mean = energy_pos.mean().item()  # type: ignore[union-attr]
                        e_neg_mean = energy_neg.mean().item()  # type: ignore[union-attr]
                        margin = e_neg_mean - e_pos_mean
                        energy_wtd = (loss_scale * loss_energy_reduced).mean().item()
                        tag = "IRED-CL" if use_ired_contrastive else "CD"
                        print(
                            f"[{tag}-DIAG step={self.global_step}] "
                            f"E_pos={e_pos_mean:.4f} E_neg={e_neg_mean:.4f} "
                            f"margin={margin:.4f} E_wtd={energy_wtd:.6f} "
                            f"MSE={mse_val:.6f} "
                            f"gradE={grad_E_norm:.4f} "
                            f"pred_norm={model_out_norm:.4f} pred_std={pred_std:.4f} pred_abs={pred_abs:.4f} "
                            f"tgt_std={tgt_std_val:.4f} tgt_abs={tgt_abs_val:.4f} cos={cos_sim:.4f} "
                            f"neg_dist={neg_dist:.4f} "
                            f"lang_grad0={diag.get('grad_norm_first',0):.4f} "
                            f"lang_gradK={diag.get('grad_norm_last',0):.4f} "
                            f"lang_disp={diag.get('displacement',0):.4f} "
                            f"step={diag.get('step_mean',0):.4f} "
                            f"sigma={diag.get('sigma_mean',0):.4f}",
                            flush=True
                        )
                # === PERIPHERAL DISTRIBUTION DIAGNOSTICS ===
                if self.mining_config.get('use_peripheral_loss', False):
                    with torch.no_grad():
                        print(
                            f"[PD-DIAG step={self.global_step}] "
                            f"E_in={E_in.item():.4f} E_pd={E_pd.item():.4f} "
                            f"E_gap={E_pd.item() - E_in.item():.4f} "
                            f"L_pd={loss_peripheral_scalar.item():.6f}",
                            flush=True
                        )
            # === END DIAGNOSTIC LOGGING ===

            return loss.mean(), (loss_mse.mean(), loss_energy.mean(), loss_opt.mean())
        else:
            loss = loss_mse

            # Increment global step for scheduling
            self.global_step += 1

            if self.global_step % 100 == 0:
                with torch.no_grad():
                    pred_norm = model_out.norm(dim=-1).mean().item()
                    pred_std  = model_out.std().item()
                    pred_abs  = model_out.abs().mean().item()
                    tgt_std   = target.std().item()
                    tgt_abs   = target.abs().mean().item()
                    pred_f = model_out.reshape(b, -1)
                    tgt_f  = target.reshape(b, -1)
                    cos_sim = F.cosine_similarity(pred_f, tgt_f, dim=-1).mean().item()
                print(
                    f"[MSE-DIAG step={self.global_step}] "
                    f"MSE={loss_mse.mean().item():.6f} "
                    f"pred_norm={pred_norm:.4f} pred_std={pred_std:.4f} pred_abs={pred_abs:.4f} "
                    f"tgt_std={tgt_std:.4f} tgt_abs={tgt_abs:.4f} cos={cos_sim:.4f}",
                    flush=True
                )

            return loss.mean(), (loss_mse.mean(), -1, -1)

    def forward(self, inp, target, mask, *args, **kwargs):
        b, *c = target.shape
        device = target.device
        if len(c) == 1:
            self.out_dim = c[0]
            self.out_shape = c
        else:
            self.out_dim = c[-1]
            self.out_shape = c

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(inp, target, mask, t, *args, **kwargs)

# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        validation_batch_size = None,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        data_workers = None,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        metric = 'mse',
        cond_mask = False,
        validation_dataset = None,
        extra_validation_datasets = None,
        extra_validation_every_mul = 10,
        evaluate_first = False,
        latent = False,
        autoencode_model = None
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model

        # Conditioning on mask

        self.cond_mask = cond_mask

        # Whether to do reasoning in the latent space

        self.latent = latent

        if autoencode_model is not None:
            self.autoencode_model = autoencode_model.cuda()

        # sampling and training hyperparameters
        self.out_dim = self.model.out_dim

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.extra_validation_every_mul = extra_validation_every_mul

        self.batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size if validation_batch_size is not None else train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # Evaluation metric.
        self.metric = metric
        self.data_workers = data_workers

        if self.data_workers is None:
            self.data_workers = 2  # Use small number of workers to avoid DataLoader freeze on cluster

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = False, num_workers = self.data_workers)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.validation_dataset = validation_dataset

        if self.validation_dataset is not None:
            dl = DataLoader(self.validation_dataset, batch_size = validation_batch_size, shuffle=False, pin_memory=False, num_workers = self.data_workers)
            dl = self.accelerator.prepare(dl)
            self.validation_dl = dl
        else:
            self.validation_dl = None

        self.extra_validation_datasets = extra_validation_datasets

        if self.extra_validation_datasets is not None:
            self.extra_validation_dls = dict()
            for key, dataset in self.extra_validation_datasets.items():
                dl = DataLoader(dataset, batch_size = validation_batch_size, shuffle=False, pin_memory=False, num_workers = self.data_workers)
                dl = self.accelerator.prepare(dl)
                self.extra_validation_dls[key] = dl
        else:
            self.extra_validation_dls = None

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.evaluate_first = evaluate_first

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        if osp.isfile(milestone):
            milestone_file = milestone
        else:
            milestone_file = str(self.results_folder / f'model-{milestone}.pt')
        data = torch.load(milestone_file)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if self.evaluate_first:
            milestone = self.step // self.save_and_sample_every
            self.evaluate(device, milestone)
            self.evaluate_first = False  # hack: later we will use this flag as a bypass signal to determine whether we want to run extra validation.

        end_time = time.time()
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process, dynamic_ncols = True) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                end_tiem = time.time()
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)

                    if self.cond_mask:
                        inp, label, mask = data
                        inp, label, mask = inp.float().to(device), label.float().to(device), mask.float().to(device)
                    elif self.latent:
                        inp, label, label_gt, mask_latent = data
                        mask_latent = mask_latent.float().to(device)
                        inp, label, label_gt = inp.float().to(device), label.float().to(device), label_gt.float().to(device)
                        mask = None
                    else:
                        inp, label = data
                        inp, label = inp.float().to(device), label.float().to(device)
                        mask = None

                    data_time = time.time() - end_time; end_time = time.time()

                    with self.accelerator.autocast():
                        loss, (loss_denoise, loss_energy, loss_opt) = self.model(inp, label, mask)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                nn_time = time.time() - end_time; end_time = time.time()
                pbar.set_description(f'loss: {total_loss:.4f} loss_denoise: {loss_denoise:.4f} loss_energy: {loss_energy:.4f} loss_opt: {loss_opt:.4f} data_time: {data_time:.2f} nn_time: {nn_time:.2f}')

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    # if True:
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every

                        self.save(milestone)

                        if self.latent:
                            self.evaluate(device, milestone, inp=inp, label=label_gt, mask=mask_latent)
                        else:
                            self.evaluate(device, milestone, inp=inp, label=label, mask=mask)


                pbar.update(1)

        accelerator.print('training complete')

    def evaluate(self, device, milestone, inp=None, label=None, mask=None):
        print('Running Evaluation...')
        self.ema.ema_model.eval()

        if inp is not None and label is not None:
            with torch.no_grad():
                # batches = num_to_groups(self.num_samples, self.batch_size)

                if self.latent:
                    all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0)), range(1)))
                else:
                    all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0)), range(1)))
                    # all_samples_list = list(map(lambda n: self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0), return_traj=True), range(1)))
                # all_samples_list = list(map(lambda n: self.model.sample(inp, label, mask, batch_size=inp.size(0)), range(1)))
                # all_samples_list = [self.model.sample(inp, batch_size=inp.size(0))]

                all_samples = torch.cat(all_samples_list, dim = 0)

                print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (Train)')
                if self.metric == 'mse':
                    all_samples = torch.cat(all_samples_list, dim = 0)
                    mse_error = (all_samples - label).pow(2).mean()
                    rows = [('mse_error', mse_error)]
                    print(tabulate(rows))
                elif self.metric == 'bce':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                elif self.metric == 'sudoku':
                    assert len(all_samples_list) == 1
                    summary = sudoku_accuracy(all_samples_list[0], label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                elif self.metric == 'sort':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    summary.update(sort_accuracy(all_samples_list[0], label, mask))
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'sort-2':
                    assert len(all_samples_list) == 1
                    summary = sort_accuracy_2(all_samples_list[0], label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'shortest-path-1d':
                    assert len(all_samples_list) == 1
                    summary = binary_classification_accuracy_4(all_samples_list[0], label)
                    summary.update(shortest_path_1d_accuracy(all_samples_list[0], label, mask, inp))
                    rows = [[k, v] for k, v in summary.items()]
                elif self.metric == 'sudoku_latent':
                    sample = all_samples_list[0].view(-1, 9, 9, 3).permute(0, 3, 1, 2).contiguous() * 4
                    prediction = self.autoencode_model.decode(sample)
                    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 729)

                    assert len(all_samples_list) == 1
                    summary = sudoku_accuracy(prediction, label, mask)
                    rows = [[k, v] for k, v in summary.items()]
                    print(tabulate(rows))
                else:
                    raise NotImplementedError()

        if self.validation_dl is not None:
            self._run_validation(self.validation_dl, device, milestone, prefix = 'Validation')

        if (self.step % (self.save_and_sample_every * self.extra_validation_every_mul) == 0 and self.extra_validation_dls is not None) or self.evaluate_first:
            for key, extra_dl in self.extra_validation_dls.items():
                self._run_validation(extra_dl, device, milestone, prefix = key)

    def _run_validation(self, dl, device, milestone, prefix='Validation'):
        meters = collections.defaultdict(AverageMeter)
        with torch.no_grad():
            for i, data in enumerate(tqdm(dl, total=len(dl), desc=f'running on the validation dataset (ID: {prefix})')):
                if self.cond_mask:
                    inp, label, mask = map(lambda x: x.float().to(device), data)
                elif self.latent:
                    inp, label, label_gt, mask = map(lambda x: x.float().to(device), data)
                else:
                    inp, label = map(lambda x: x.float().to(device), data)
                    mask = None

                if self.latent:
                    # Masking doesn't make sense in the latent space
                    # samples = self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0))
                    samples = self.ema.ema_model.sample(inp, label, None, batch_size=inp.size(0))
                else:
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    # samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))
                    samples = self.ema.ema_model.sample(inp, label, mask, batch_size=inp.size(0))

                # np.savez("sudoku.npz", inp=inp.detach().cpu().numpy(), label=label.detach().cpu().numpy(), mask=mask.detach().cpu().numpy(), samples=samples.detach().cpu().numpy())
                # import pdb
                # pdb.set_trace()
                # print("here")
                if self.metric == 'sudoku':
                    # samples_traj = samples
                    summary = sudoku_accuracy(samples[-1], label, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'sudoku_latent':
                    sample = samples.view(-1, 9, 9, 3).permute(0, 3, 1, 2).contiguous() * 4
                    prediction = self.autoencode_model.decode(sample)
                    prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, 729)
                    summary = sudoku_accuracy(prediction, label_gt, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                elif self.metric == 'sort':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(sort_accuracy(samples, label, mask))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'sort-2':
                    summary = sort_accuracy_2(samples, label, mask)
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'shortest-path-1d':
                    summary = binary_classification_accuracy_4(samples, label)
                    summary.update(shortest_path_1d_accuracy(samples, label, mask, inp))
                    # summary.update(shortest_path_1d_accuracy_closed_loop(samples, label, mask, inp, self.ema.ema_model.sample))
                    for k, v in summary.items():
                        meters[k].update(v, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'mse':
                    # all_samples = torch.cat(all_samples_list, dim = 0)
                    mse_error = (samples - label).pow(2).mean()
                    meters['mse'].update(mse_error, n=inp.size(0))
                    if i > 20:
                        break
                elif self.metric == 'bce':
                    summary = binary_classification_accuracy_4(samples, label)
                    for k, v in summary.items():
                        meters[k].update(v, n=samples.shape[0])
                    if i > 20:
                        break
                else:
                    raise NotImplementedError()

            rows = [[k, v.avg] for k, v in meters.items()]
            print(f'Validation Result @ Iteration {self.step}; Milestone = {milestone} (ID: {prefix})')
            print(tabulate(rows))


as_float = lambda x: float(x.item())


@torch.no_grad()
def binary_classification_accuracy(pred: torch.Tensor, label: torch.Tensor, name: str = '', saturation: bool = True) -> dict[str, float]:
    r"""Compute the accuracy of binary classification.

    Args:
        pred: the prediction, of the same shape as ``label``.
        label: the label, of the same shape as ``pred``.
        name: the name of this monitor.
        saturation: whether to check the saturation of the prediction. Saturation
            is defined as :math:`1 - \min(pred, 1 - pred)`

    Returns:
        a dict of monitor values.
    """
    if name != '':
        name = '/' + name
    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    acc = label.float().eq((pred > 0.5).float())
    if saturation:
        sat = 1 - (pred - (pred > 0.5).float()).abs()
        return {
            prefix: as_float(acc.float().mean()),
            prefix + '/saturation/mean': as_float(sat.mean()),
            prefix + '/saturation/min': as_float(sat.min())
        }
    return {prefix: as_float(acc.float().mean())}


@torch.no_grad()
def binary_classification_accuracy_4(pred: torch.Tensor, label: torch.Tensor, name: str = '') -> dict[str, float]:
    if name != '':
        name = '/' + name

    # table = list()
    # table.append(('pred', pred[0].squeeze()))
    # table.append(('label', label[0].squeeze()))
    # print(tabulate(table))

    prefix = 'accuracy' + name
    pred = pred.view(-1)  # Binary accuracy
    label = label.view(-1)
    numel = pred.numel()

    gt_0_pred_0 = ((label < 0.0) & (pred < 0.0)).sum() / numel
    gt_0_pred_1 = ((label < 0.0) & (pred >= 0.0)).sum() / numel
    gt_1_pred_0 = ((label > 0.0) & (pred < 0.0)).sum() / numel
    gt_1_pred_1 = ((label > 0.0) & (pred >= 0.0)).sum() / numel

    accuracy = gt_0_pred_0 + gt_1_pred_1
    balanced_accuracy = sum([
        gt_0_pred_0 / ((label < 0.0).float().sum() / numel),
        gt_1_pred_1 / ((label >= 0.0).float().sum() / numel),
    ]) / 2

    return {
        prefix + '/gt_0_pred_0': as_float(gt_0_pred_0),
        prefix + '/gt_0_pred_1': as_float(gt_0_pred_1),
        prefix + '/gt_1_pred_0': as_float(gt_1_pred_0),
        prefix + '/gt_1_pred_1': as_float(gt_1_pred_1),
        prefix + '/accuracy': as_float(accuracy),
        prefix + '/balance_accuracy': as_float(balanced_accuracy),
    }


@torch.no_grad()
def sudoku_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = '') -> dict[str, float]:
    if name != '':
        name = '/' + name

    pred = pred.view(-1, 9, 9, 9).argmax(dim=-1)
    label = label.view(-1, 9, 9, 9).argmax(dim=-1)

    correct = (pred == label).float()
    mask = mask.view(-1, 9, 9, 9)[:, :, :, 0]
    mask_inverse = 1 - mask

    accuracy = (correct * mask_inverse).sum() / mask_inverse.sum()

    return {
        'accuracy': as_float(accuracy),
        'consistency': as_float(sudoku_consistency(pred)),
        'board_accuracy': as_float(sudoku_score(pred))
    }


def sudoku_consistency(pred: torch.Tensor) -> bool:
    pred_onehot = F.one_hot(pred, num_classes=9)

    all_row_correct = (pred_onehot.sum(dim=1) == 1).all(dim=-1).all(dim=-1)
    all_col_correct = (pred_onehot.sum(dim=2) == 1).all(dim=-1).all(dim=-1)

    blocked = pred_onehot.view(-1, 3, 3, 3, 3, 9)
    all_block_correct = (blocked.sum(dim=(2, 4)) == 1).all(dim=-1).all(dim=-1).all(dim=-1)

    return (all_row_correct & all_col_correct & all_block_correct).float().mean()


def sudoku_score(pred: torch.Tensor) -> bool:
    valid_mask = torch.ones_like(pred)

    pred_sum_axis_1 = pred.sum(dim=1, keepdim=True)
    pred_sum_axis_2 = pred.sum(dim=2, keepdim=True)

    # Use the sum criteria from the SAT-Net paper
    axis_1_mask = (pred_sum_axis_1 == 36)
    axis_2_mask = (pred_sum_axis_2 == 36)

    valid_mask = valid_mask * axis_1_mask.float() * axis_2_mask.float()

    valid_mask = valid_mask.view(-1, 3, 3, 3, 3)
    grid_mask = pred.view(-1, 3, 3, 3, 3).sum(dim=(2, 4), keepdim=True) == 36

    valid_mask = valid_mask * grid_mask.float()

    return valid_mask.mean()


def sort_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    array = (label[:, 0, ..., 2] * 0.5 + 0.5).sum(dim=-1).cpu()
    pred = pred.cpu()
    for t in range(pred.shape[1]):
        pred_xy = pred[:, t, ..., -1].reshape(pred.shape[0], -1).argmax(dim=-1)
        pred_x = torch.div(pred_xy, pred.shape[2], rounding_mode='floor')
        pred_y = pred_xy % pred.shape[2]
        # swap x and y
        next_array = array.clone()
        next_array.scatter_(1, pred_y.unsqueeze(1), array.gather(1, pred_x.unsqueeze(1)))
        next_array.scatter_(1, pred_x.unsqueeze(1), array.gather(1, pred_y.unsqueeze(1)))
        array = next_array

    ground_truth = torch.arange(pred.shape[2] - 1, -1, -1, device=array.device).unsqueeze(0).repeat(pred.shape[0], 1)
    elem_close = (array - ground_truth).abs() < 0.1
    element_correct = elem_close.float().mean()
    array_correct = elem_close.all(dim=-1).float().mean()
    return {
        'element_correct': as_float(element_correct),
        'array_correct': as_float(array_correct),
    }


def sort_accuracy_2(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    array = label[:, 0, :, 0].clone().cpu()  # B x N
    pred = pred.cpu()
    for t in range(pred.shape[1]):
        pred_x = pred[:, t, :, 1].argmax(dim=-1)  # B x N
        pred_y = pred[:, t, :, 2].argmax(dim=-1)  # B x N
        # swap x and y
        next_array = array.clone()
        next_array.scatter_(1, pred_y.unsqueeze(1), array.gather(1, pred_x.unsqueeze(1)))
        next_array.scatter_(1, pred_x.unsqueeze(1), array.gather(1, pred_y.unsqueeze(1)))
        array = next_array

    # stupid_impl_array = label[:, 0, :, 0].clone()  # B x N
    # for b in range(pred.shape[0]):
    #     for t in range(pred.shape[1]):
    #         pred_x = pred[b, t, :, 1].argmax(dim=-1)2
    #         pred_y = pred[b, t, :, 2].argmax(dim=-1)
    #         # swap x and y
    #         u, v = stupid_impl_array[b, pred_y].clone(), stupid_impl_array[b, pred_x].clone()
    #         stupid_impl_array[b, pred_x], stupid_impl_array[b, pred_y] = u, v

    # assert (array == stupid_impl_array).all(), 'Inconsistent implementation'
    # print('Consistent implementation!!')

    elem_close = torch.abs(array - label[:, -1, :, 0].cpu()) < 1e-5
    element_correct = elem_close.float().mean()
    array_correct = elem_close.all(dim=-1).float().mean()

    pred_first_action = pred[:, 0, :, 1:3].argmax(dim=-2).cpu()
    label_first_action = label[:, 0, :, 1:3].argmax(dim=-2).cpu()
    first_action_correct = (pred_first_action == label_first_action).all(dim=-1).float().mean()

    return {
        'element_accuracy' + name: as_float(element_correct),
        'array_accuracy' + name: as_float(array_correct),
        'first_action_accuracy' + name: as_float(first_action_correct)
    }


def shortest_path_1d_accuracy(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, inp: torch.Tensor, name: str = ''):
    if name != '':
        name = '/' + name

    pred_argmax = pred[:, :, :, -1].argmax(-1)
    label_argmax = label[:, :, :, -1].argmax(-1)

    argmax_accuracy = (pred_argmax == label_argmax).float().mean()

    # vis_array = torch.stack([pred_argmax, label_argmax], dim=1)
    # table = list()
    # for i in range(len(vis_array)):
    #     table.append((vis_array[i, 0].cpu().tolist(), vis_array[i, 1].cpu().tolist()))
    # print(tabulate(table))

    pred_argmax_first = pred_argmax[:, 0]
    label_argmax_first = label_argmax[:, 0]

    first_action_accuracy = (pred_argmax_first == label_argmax_first).float().mean()

    first_action_s = inp[:, :, 0, 1].argmax(dim=-1)
    first_action_t = pred_argmax_first
    first_action_feasibility = (inp[
        torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
        first_action_s,
        first_action_t,
        0
    ] > 0).float().cpu()

    final_t = label_argmax[:, -1]
    first_action_accuracy_2 = first_action_distance_accuracy(inp[..., 0], first_action_s, final_t, first_action_t).float().cpu()
    first_action_accuracy_2 = first_action_accuracy_2 * first_action_feasibility

    return {
        'argmax_accuracy' + name: as_float(argmax_accuracy),
        'first_action_accuracy' + name: as_float(first_action_accuracy),
        'first_action_feasibility' + name: as_float(first_action_feasibility.mean()),
        'first_action_accuracy_2' + name: as_float(first_action_accuracy_2.mean()),
    }


def get_shortest_batch(edges: torch.Tensor) -> torch.Tensor:
    """ Return the length of shortest path between nodes. """
    b = edges.shape[0]
    n = edges.shape[1]

    # n + 1 indicates unreachable.
    shortest = torch.ones((b, n, n), dtype=torch.float32, device=edges.device) * (n + 1)
    shortest[torch.where(edges == 1)] = 1
    # Make sure that shortest[x, x] = 0
    shortest -= shortest * torch.eye(n).unsqueeze(0).to(shortest.device)
    shortest = shortest

    # Floyd Algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j:
                    shortest[:, i, j] = torch.min(shortest[:, i, j], shortest[:, i, k] + shortest[:, k, j])
    return shortest


def first_action_distance_accuracy(edge: torch.Tensor, s: torch.Tensor, t: torch.Tensor, pred: torch.Tensor):
    shortest = get_shortest_batch(edge.detach().cpu())
    b = edge.shape[0]
    b_arrange = torch.arange(b, dtype=torch.int64, device=edge.device)
    return shortest[b_arrange, pred, t] < shortest[b_arrange, s, t]


def shortest_path_1d_accuracy_closed_loop(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, inp: torch.Tensor, sample_fn, name: str = '', execution_steps: int = 1):
    assert execution_steps in (1, 2), 'Only 1-step and 2-step execution is supported'
    b, t, n, _ = pred.shape
    failed = torch.zeros(b, dtype=torch.bool, device='cpu')
    succ = torch.zeros(b, dtype=torch.bool, device='cpu')

    for i in range(8 // execution_steps):
        pred_argmax = pred[:, :, :, -1].argmax(-1)
        pred_argmax_first = pred_argmax[:, 0]
        pred_argmax_second = pred_argmax[:, 1]
        target_argmax = inp[:, :, 0, 3].argmax(dim=-1)

        first_action_s = inp[:, :, 0, 1].argmax(dim=-1)
        first_action_t = pred_argmax_first
        first_action_feasibility = (inp[
            torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
            first_action_s,
            first_action_t,
            0
        ] > 0).cpu()
        last_t = first_action_t

        failed |= ~(first_action_feasibility.to(torch.bool))
        succ |= (first_action_t == target_argmax).cpu() & ~failed

        print(f'Step {i} (F) s={first_action_s[0].item()}, t={first_action_t[0].item()}, goal={target_argmax[0].item()}, feasible={first_action_feasibility[0].item()}')

        if execution_steps >= 2:
            second_action_s = first_action_t
            second_action_t = pred_argmax_second
            second_action_feasibility = (inp[
                torch.arange(inp.shape[0], dtype=torch.int64, device=inp.device),
                second_action_s,
                second_action_t,
                0
            ] > 0).cpu()
            failed |= ~(second_action_feasibility.to(torch.bool))
            succ |= (second_action_t == target_argmax).cpu() & ~failed
            last_t = second_action_t

            print(f'Step {i} (S) s={second_action_s[0].item()}, t={second_action_t[0].item()}, goal={target_argmax[0].item()}, feasible={second_action_feasibility[0].item()}')

        inp_clone = inp.clone()
        inp_clone[:, :, :, 1] = 0
        inp_clone[torch.arange(b, dtype=torch.int64, device=inp.device), last_t, :, 1] = 1
        inp = inp_clone
        pred = sample_fn(inp, label, mask, batch_size=inp.size(0))

    return {
        'closed_loop_success_rate' + name: as_float(succ.float().mean()),
    }

