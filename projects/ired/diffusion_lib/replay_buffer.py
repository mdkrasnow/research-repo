"""
Timestep-bucketed replay buffer for persistent contrastive divergence.

Literature:
- Persistent Contrastive Divergence (Tieleman, 2008)
- Implicit Generation and Modeling with Energy Based Models (Du & Mordatch, NeurIPS 2019)
"""

import torch
from collections import deque


class TBucketReplayBuffer:
    """
    Timestep-bucketed FIFO replay buffer for persistent CD.

    Stores negatives in xₜ space, bucketed by timestep to avoid
    distribution mismatch when sampling.
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

        # Return without requires_grad: the Langevin sampler calls x_init.detach()
        # at entry, which would strip requires_grad anyway. The model wrapper sets
        # requires_grad_(True) on x inside sample_negatives_langevin as needed.
        samples = torch.stack(out).to(device)
        return samples

    def __len__(self):
        return sum(len(b) for b in self.buckets)
