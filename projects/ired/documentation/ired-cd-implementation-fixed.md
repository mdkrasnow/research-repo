# IRED-CD Implementation Guide (CORRECTED)

**CRITICAL FIXES APPLIED**: This document contains the literature-aligned, correct implementation of CD-style EBM training for IRED.

**Key Corrections**:
1. ✅ Fixed CD loss gradient flow (detach x_neg, NOT energy_neg)
2. ✅ Fixed replay buffer space mismatch (store xₜ, not x₀)
3. ✅ Added timestep-bucketed replay buffer
4. ✅ Use wrapper gradient (avoid recomputing autograd)
5. ✅ Fixed energy scheduling shape issues
6. ✅ Renamed to IRED-CD (not DCD unless parameter-free diffusion)

---

## Naming Convention Changes

**Old (confusing)**:
- `mining_strategy: adversarial` → actually energy descent

**New (clear)**:
- `mining_strategy: none` → baseline (no mining)
- `mining_strategy: nce_random` → old random NCE
- `mining_strategy: cd_langevin` → CD + Langevin sampling
- `mining_strategy: cd_langevin_replay` → + replay buffer
- `mining_strategy: cd_full` → + residual filter + schedule

---

## A) Langevin Sampler (CORRECTED)

**File**: `diffusion_lib/denoising_diffusion_pytorch_1d.py`

```python
def sample_negatives_langevin(self, inp, x_init, t, k_steps: int):
    """
    Sample negatives using Langevin dynamics (short-run MCMC).

    CRITICAL: Uses wrapper gradient to avoid recomputing autograd.
    Returns negatives in xₜ space (same as x_init).

    Args:
        inp: Input conditioning (e.g., matrix A)
        x_init: Initial negative sample at timestep t (xₜ space)
        t: Diffusion timestep
        k_steps: Number of Langevin steps

    Returns:
        x_neg: Refined negative at timestep t (xₜ space)
    """
    x = x_init.detach()

    step = extract(self.opt_step_size, t, x.shape)  # per-sample, broadcastable
    sigma_mult = self.mining_config.get('langevin_sigma_multiplier', 1.0)
    sigma = sigma_mult * torch.sqrt(2.0 * step)

    for _ in range(k_steps):
        # Use wrapper to get both energy and gradient in one call
        energy, grad = self.model(inp, x, t, return_both=True)

        with torch.no_grad():
            # Langevin step: x ← x - η∇E + σ·N(0,I)
            x = x - step * grad + sigma * torch.randn_like(x)
            x = torch.clamp(x, -2, 2)  # Keep in reasonable range

    return x.detach()
```

**Why this is correct**:
- Uses existing wrapper gradient (efficient, consistent)
- Returns in xₜ space (matches initialization)
- Short-run sampling with CD-style updates (literature-supported)

---

## B) Replay Buffer (CORRECTED - Timestep-Bucketed)

**New file**: `diffusion_lib/replay_buffer.py`

```python
import torch
from collections import deque
import math

class TBucketReplayBuffer:
    """
    Timestep-bucketed FIFO replay buffer for persistent CD.

    Stores negatives in xₜ space, bucketed by timestep to avoid
    distribution mismatch when sampling.

    Literature: Persistent contrastive divergence (Tieleman 2008),
    Implicit Generation and Modeling with EBMs (Du & Mordatch, NeurIPS 2019)
    """

    def __init__(self, buffer_size=10000, num_buckets=16):
        """
        Args:
            buffer_size: Max samples per bucket
            num_buckets: Number of timestep buckets (e.g., 16 for T=10)
        """
        self.num_buckets = num_buckets
        self.buckets = [deque(maxlen=buffer_size) for _ in range(num_buckets)]

    def _bucket(self, t, num_timesteps: int):
        """Map timestep t ∈ [0, T-1] to bucket ∈ [0, B-1]."""
        b = (t.float() / max(1, num_timesteps - 1) * (self.num_buckets - 1)).long()
        return torch.clamp(b, 0, self.num_buckets - 1)

    def add(self, x_neg_xt, t, num_timesteps: int):
        """
        Add negatives to buffer.

        Args:
            x_neg_xt: Negatives in xₜ space (B, dim)
            t: Timesteps (B,)
            num_timesteps: Total diffusion timesteps
        """
        b = self._bucket(t, num_timesteps)
        x_cpu = x_neg_xt.detach().cpu()

        for i in range(x_cpu.shape[0]):
            self.buckets[b[i].item()].append(x_cpu[i])

    def sample(self, t, num_timesteps: int, device):
        """
        Sample from buffer, matching timestep buckets.

        Args:
            t: Timesteps to sample for (B,)
            num_timesteps: Total diffusion timesteps
            device: Target device

        Returns:
            samples: (B, dim) or None if buffer too small
        """
        b = self._bucket(t, num_timesteps)
        out = []

        for i in range(t.shape[0]):
            bucket = self.buckets[b[i].item()]
            if len(bucket) == 0:
                return None  # Buffer not ready yet
            j = torch.randint(0, len(bucket), (1,)).item()
            out.append(bucket[j])

        return torch.stack(out).to(device)

    def __len__(self):
        return sum(len(b) for b in self.buckets)
```

**Why this is correct**:
- Stores negatives in **xₜ space** (matches sampling space)
- Timestep-bucketed (avoids distribution mismatch)
- FIFO per bucket (persistent chains)

---

## C) Negative Initialization (CORRECTED)

**In `p_losses` method, before sampling**:

```python
# Initialize negatives: replay-or-fresh in xₜ space
noise_scale = self.mining_config.get('noise_scale', 1.0)  # 1.0–1.5, NOT 3.0
x_init = self.q_sample(x_start=x_start, t=t, noise=noise_scale * torch.randn_like(x_start))

# Try replay buffer (persistent chains)
if self.use_replay_buffer and torch.rand(()) < self.replay_prob:
    x_replay = self.replay_buffer.sample(t, self.num_timesteps, x_start.device)
    if x_replay is not None:
        x_init = x_replay

# Sample negatives with Langevin (in xₜ space)
opt_steps = self.mining_config.get('opt_steps', 10)
x_neg = self.sample_negatives_langevin(inp, x_init, t, k_steps=opt_steps)

# Add to replay buffer for future use (still in xₜ space)
if self.use_replay_buffer:
    self.replay_buffer.add(x_neg, t, self.num_timesteps)
```

**Why this is correct**:
- Initializes in xₜ space, samples in xₜ space, stores in xₜ space (consistent!)
- Replay buffer maintains persistent chains
- `noise_scale=1.0-1.5` (not 3.0) for better initialization

---

## D) CD-Style Energy Loss (CORRECTED - Critical Gradient Flow Fix!)

**THIS IS THE KEY FIX**: Detach `x_neg`, NOT `energy_neg`!

```python
# Compute energies for positives and negatives
E_pos = self.model(inp, data_sample, t, return_energy=True)  # [B,1]

# CRITICAL: Stop gradient through SAMPLER, not through ENERGY EVAL
x_neg_detached = x_neg.detach()  # ← detach the sample
E_neg = self.model(inp, x_neg_detached, t, return_energy=True)  # ← energy gradients flow!

# CD-style energy difference loss
loss_energy = (E_pos - E_neg)  # [B,1]
```

**Why this is correct**:
- Gradients flow from `E_neg` to EBM parameters (learning happens!)
- Gradients do NOT flow through sampling trajectory (avoids higher-order terms)
- This is exactly CD/short-run EBM training (Du & Mordatch NeurIPS 2019)

**What was wrong before**:
```python
# ❌ WRONG: kills gradients entirely
loss_energy = energy_pos - energy_neg.detach()  # no learning from negatives!
```

---

## E) Residual Filtering (CORRECTED)

**Helper function** (unchanged):

```python
def compute_matrix_residual(self, A, X):
    """
    Compute matrix inversion residual: ||AX - I||_F

    Args:
        A: Input matrix (flattened, B x rank²)
        X: Predicted inverse (flattened, B x rank²)

    Returns:
        residual: Frobenius norm per sample (B,)
    """
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
```

**In `p_losses`, after computing energies**:

```python
use_residual_filter = self.mining_config.get('use_residual_filter', False)

if use_residual_filter:
    # IMPORTANT: Compute residuals in x₀ space (for task oracle)
    # Need to convert x_neg to x₀ for residual check
    x_neg_x0 = self.predict_start_from_noise(x_neg, t, torch.zeros_like(x_neg))
    x_neg_x0 = torch.clamp(x_neg_x0, -2, 2)

    # Compute residuals
    residuals = self.compute_matrix_residual(inp, x_neg_x0)  # (B,)

    # Get threshold (fixed or quantile)
    tau = self.mining_config.get('residual_tau', None)
    if tau is None:
        # Fallback: batch quantile (can be noisy)
        q = self.mining_config.get('residual_filter_quantile', 0.3)
        tau = torch.quantile(residuals.detach(), q).item()

    # Keep negatives with residual >= tau (filter out false negatives)
    keep = (residuals >= tau).float().unsqueeze(1)  # (B,1)
    loss_energy = loss_energy * keep
```

**Why this is correct**:
- Computes residuals in x₀ space (where inversion is defined)
- Filters false negatives (too close to correct solutions)
- Debiasing hard negatives (Chuang et al., NeurIPS 2020)

---

## F) Energy Loss Scheduling (CORRECTED - Shape Fix)

**CRITICAL**: Don't multiply `loss_scale` (scalar) by `t_mask` (per-sample). Mask the loss directly!

```python
# Base energy loss weight
loss_scale = self.mining_config.get('energy_loss_weight', 0.05)

# Optional: warmup schedule
use_energy_schedule = self.mining_config.get('use_energy_schedule', False)
if use_energy_schedule:
    warmup = self.mining_config.get('energy_loss_warmup_steps', 20000)
    max_w = self.mining_config.get('energy_loss_max_weight', 0.05)
    step = getattr(self, "global_step", 0)
    loss_scale = max_w * min(1.0, step / warmup)

# Optional: timestep range filtering
use_timestep_range = self.mining_config.get('use_timestep_range', False)
if use_timestep_range:
    lo, hi = self.mining_config.get('energy_loss_timestep_range', [0.2, 0.8])
    tmin = int(lo * (self.num_timesteps - 1))
    tmax = int(hi * (self.num_timesteps - 1))
    t_mask = ((t >= tmin) & (t <= tmax)).float().unsqueeze(1)  # (B,1)

    # Apply mask to loss_energy (per-sample), not to loss_scale (scalar)
    loss_energy = loss_energy * t_mask
```

**Final loss combination**:

```python
# Combine MSE + CD energy loss
loss = loss_mse + loss_scale * loss_energy.mean()
```

**Why this is correct**:
- `loss_scale` is scalar warmup coefficient
- `t_mask` is per-sample (B,1) timestep filter
- Masking happens on `loss_energy`, not on scalar `loss_scale`

**What was wrong before**:
```python
# ❌ WRONG: shape mismatch or silent broadcasting bug
loss_scale = loss_scale * t_mask  # scalar × (B,1) = shape error or broadcast mess
```

---

## G) Integration in GaussianDiffusion1D.__init__

```python
# Mining configuration for CD-style training
if mining_config is None:
    mining_config = {'strategy': 'none'}
self.mining_config = mining_config
self.mining_strategy = mining_config.get('strategy', 'none')

# Validate strategy
valid_strategies = ['none', 'nce_random', 'cd_langevin', 'cd_langevin_replay', 'cd_full']
assert self.mining_strategy in valid_strategies, \
    f"mining_strategy must be one of {valid_strategies}, got {self.mining_strategy}"

# Initialize replay buffer if needed
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
```

---

## H) Unified p_losses Patch

If you want, I can provide a **minimal diff patch** for your existing `p_losses` method that:
- Preserves your variable names (`xmin_noise`, `xmin_noise_rescale`, etc.)
- Fixes only the critical bugs
- Adds the new CD path as a conditional branch

Would you like me to:
1. Read your current `p_losses` implementation
2. Create a unified diff patch that you can apply directly

---

## Updated Config Files

### q202_cd_langevin.json (renamed from q202_dcd_cdloss.json)

```json
{
  "experiment_name": "q202_cd_langevin",
  "description": "IRED-CD Stage 1: CD loss + Langevin sampling",
  "investigation": "ired-cd",
  "ablation_stage": "cd_langevin",
  "mining_strategy": "cd_langevin",
  "rank": 20,
  "ood": false,
  "diffusion_steps": 10,
  "batch_size": 2048,
  "train_steps": 100000,
  "learning_rate": 0.0001,
  "mining_opt_steps": 10,
  "mining_noise_scale": 1.5,
  "langevin_sigma_multiplier": 1.0,
  "use_langevin": true,
  "use_replay_buffer": false,
  "use_cd_loss": true,
  "use_residual_filter": false,
  "use_energy_schedule": false,
  "energy_loss_weight": 0.05,
  "output_dir": "results/ds_inverse/cd_langevin",
  "notes": "Corrected CD loss (detach x_neg, not energy_neg). Langevin sigma=sqrt(2*eta). k=10 steps. No replay buffer yet."
}
```

### q203_cd_replay.json

```json
{
  "experiment_name": "q203_cd_replay",
  "description": "IRED-CD Stage 2: CD loss + Langevin + Replay buffer",
  "investigation": "ired-cd",
  "ablation_stage": "cd_replay",
  "mining_strategy": "cd_langevin_replay",
  "rank": 20,
  "ood": false,
  "diffusion_steps": 10,
  "batch_size": 2048,
  "train_steps": 100000,
  "learning_rate": 0.0001,
  "mining_opt_steps": 10,
  "mining_noise_scale": 1.5,
  "langevin_sigma_multiplier": 1.0,
  "use_langevin": true,
  "use_replay_buffer": true,
  "replay_buffer_size": 10000,
  "replay_buffer_buckets": 16,
  "replay_sample_prob": 0.95,
  "use_cd_loss": true,
  "use_residual_filter": false,
  "use_energy_schedule": false,
  "energy_loss_weight": 0.05,
  "output_dir": "results/ds_inverse/cd_replay",
  "notes": "Timestep-bucketed replay buffer (16 buckets). Stores negatives in xₜ space. p_replay=0.95."
}
```

### q204_cd_full.json

```json
{
  "experiment_name": "q204_cd_full",
  "description": "IRED-CD Stage 3: Full implementation (CD + Langevin + Replay + Residual filter + Schedule)",
  "investigation": "ired-cd",
  "ablation_stage": "cd_full",
  "mining_strategy": "cd_full",
  "rank": 20,
  "ood": false,
  "diffusion_steps": 10,
  "batch_size": 2048,
  "train_steps": 100000,
  "learning_rate": 0.0001,
  "mining_opt_steps": 10,
  "mining_noise_scale": 1.5,
  "langevin_sigma_multiplier": 1.0,
  "use_langevin": true,
  "use_replay_buffer": true,
  "replay_buffer_size": 10000,
  "replay_buffer_buckets": 16,
  "replay_sample_prob": 0.95,
  "use_cd_loss": true,
  "use_residual_filter": true,
  "residual_filter_quantile": 0.3,
  "use_energy_schedule": true,
  "use_timestep_range": true,
  "energy_loss_warmup_steps": 20000,
  "energy_loss_max_weight": 0.05,
  "energy_loss_timestep_range": [0.2, 0.8],
  "output_dir": "results/ds_inverse/cd_full",
  "notes": "Full IRED-CD. Residual filtering (30th percentile) prevents false negatives. Energy schedule (warmup 20K steps, max_weight=0.05). Timestep range filtering [0.2T, 0.8T]."
}
```

---

## Critical Takeaways

### ✅ What's Correct Now

1. **CD loss gradient flow**: Detach `x_neg`, gradients flow from `E_neg`
2. **Replay buffer**: Timestep-bucketed, stores/samples xₜ negatives
3. **Langevin sampler**: Uses wrapper gradient, returns xₜ space
4. **Residual filter**: Computes in x₀ space, filters in energy loss
5. **Energy scheduling**: Masks `loss_energy`, not `loss_scale`

### ❌ What Was Wrong Before

1. Detaching `energy_neg` → killed all learning from negatives
2. Replay buffer space mismatch → distribution shift
3. No timestep bucketing → negative sampling noise
4. Recomputing autograd → inefficient, inconsistent
5. Shape mismatch in scheduling → silent broadcasting bugs

### 📚 Literature Support

- **CD-style training**: Du & Mordatch (NeurIPS 2019)
- **Persistent chains**: Tieleman (2008)
- **Short-run MCMC**: Learning Non-Convergent Short-Run MCMC (NeurIPS 2019)
- **False negative debiasing**: Chuang et al. (NeurIPS 2020)

---

## Next Steps

1. **Create replay_buffer.py** with TBucketReplayBuffer
2. **Add sample_negatives_langevin()** to GaussianDiffusion1D
3. **Patch p_losses** with corrected CD loss logic

**Would you like me to**:
- Read your current `p_losses` method
- Generate a unified diff patch
- Show exactly what lines to change

This will make implementation much faster and ensure correctness!
