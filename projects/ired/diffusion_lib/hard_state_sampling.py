"""
Task-agnostic hard-state sampling for IRED training (q242+).

Three scoring functions that identify "hard" candidate y-states, plus
a per-timestep normalization helper. All scores are detached from the
computation graph to prevent the model from gaming the hardness metric.
"""

import torch


def compute_replay_uncertainty(model, inp, x_states, t, n_eval=2):
    """Sampler A: Score states by stochastic evaluation disagreement.

    Runs n_eval forward passes through the model (in train mode for
    dropout/stochastic variation) and measures pairwise disagreement.

    Args:
        model: DiffusionWrapper — callable(inp, x, t) -> pred
        inp: condition tensor [B, inp_dim]
        x_states: candidate y-states [B, seq_len]
        t: timesteps [B]
        n_eval: number of stochastic forward passes

    Returns:
        hardness: [B] tensor (detached)
    """
    # Note: cannot use torch.no_grad() because the DiffusionWrapper model
    # internally computes torch.autograd.grad(energy, x) which requires grad.
    preds = [model(inp, x_states.detach().requires_grad_(True), t).detach() for _ in range(n_eval)]
    hardness = (preds[0] - preds[1]).pow(2).mean(dim=-1)  # [B]
    return hardness.detach()


def compute_trajectory_divergence(opt_step_fn, inp, x_states, t, mask, data_cond):
    """Sampler B: H_traj(y) = |T(T(y)) - sg(T(y))|^2.

    Measures how much a second refinement step changes the output.
    Uses the existing _opt_step_no_reject infrastructure.

    Args:
        opt_step_fn: callable(inp, x, t, mask, data_cond) -> x_new
        inp, x_states, t, mask, data_cond: standard IRED tensors

    Returns:
        hardness: [B] tensor (detached)
    """
    x = x_states.detach()
    x1 = opt_step_fn(inp, x, t, mask, data_cond)       # T(y)
    x2 = opt_step_fn(inp, x1, t, mask, data_cond)       # T(T(y))
    hardness = (x2 - x1.detach()).pow(2).mean(dim=-1)   # [B], sg on x1
    return hardness.detach()


def compute_local_instability(model, inp, x_states, t, xi_std=0.1):
    """Sampler C: H_instab(y) = |g(x, y+xi1, k) - g(x, y+xi2, k)|^2.

    Measures local sensitivity of the model's prediction to small
    perturbations of the candidate state.

    Args:
        model: DiffusionWrapper — callable(inp, x, t) -> pred
        inp: condition tensor [B, inp_dim]
        x_states: candidate y-states [B, seq_len]
        t: timesteps [B]
        xi_std: standard deviation of perturbation noise

    Returns:
        hardness: [B] tensor (detached)
    """
    x = x_states.detach()
    xi1 = torch.randn_like(x) * xi_std
    xi2 = torch.randn_like(x) * xi_std
    pred1 = model(inp, (x + xi1).requires_grad_(True), t).detach()
    pred2 = model(inp, (x + xi2).requires_grad_(True), t).detach()
    hardness = (pred1 - pred2).pow(2).mean(dim=-1)  # [B]
    return hardness.detach()


def normalize_hardness_per_timestep(hardness, t, num_timesteps, eps=1e-8):
    """Normalize hardness scores within timestep buckets.

    Without normalization, early (high-noise) timesteps would dominate
    hardness rankings. This z-scores within 10 buckets.

    Args:
        hardness: [B] raw hardness scores
        t: [B] timesteps
        num_timesteps: total number of diffusion timesteps
        eps: numerical stability

    Returns:
        h_norm: [B] normalized hardness in ~[0, 1] (clamped)
    """
    h_norm = torch.zeros_like(hardness)
    num_buckets = min(10, num_timesteps)
    bucket_size = max(1, num_timesteps // num_buckets)

    for b_start in range(0, num_timesteps, bucket_size):
        b_end = b_start + bucket_size
        mask = (t >= b_start) & (t < b_end)
        if mask.sum() < 2:
            continue
        vals = hardness[mask]
        mu, std = vals.mean(), vals.std()
        if std > eps:
            h_norm[mask] = ((vals - mu) / (std + eps)).clamp(0, 3) / 3.0
        # else: leave as 0 (uniform within bucket)

    return h_norm
