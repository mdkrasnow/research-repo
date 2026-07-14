"""
Structured start-state corruption for EqM training x0.
Extends Transport.sample()'s pure-Gaussian x0 to Bernoulli mask, Fourier
low-pass, and Gaussian-blur corruption of the VAE latent, per
structured-start-state proposal (see projects/masked-EqM/CLAUDE.md).
Bernoulli masking follows MAE-style random masking (He et al.). Fourier
and blur corruption operate on the VAE latent (not raw pixels) since
that's what's available at this choke point -- documented deviation from
pixel-space corruption, see CLAUDE.md.

Blur corruption (added 2026-07-14, second structured-start-state family
per Yilun's generalization-of-masking question): a fixed-sigma isotropic
Gaussian blur kernel applied depthwise (per latent channel, no cross-channel
mixing) via conv2d, replacing high-frequency content with a smoothed
version of x1 -- structurally analogous to mask_corrupt (coarse structure
kept, detail destroyed) but via low-pass spatial smoothing instead of
random erasure. Unlike fourier_corrupt, no noise is injected into the
suppressed band; the blurred latent itself is z0 (deterministic given x1
and sigma), matching how a real deblurring task would present a corrupted
image. sigma is calibrated (see documentation/blur-calibration.md) to match
the clean-vs-corrupted MSE of the existing p=0.5 mask task, so the two
structured-start families present the field with comparably hard corruption.
"""
import math
import torch as th
import torch.nn.functional as F


def mask_corrupt(x1, mask_prob):
    """z0 = m*x1 + (1-m)*eps; m ~ Bernoulli(1-mask_prob), shared across
    the 4 latent channels (spatial patch mask, not per-channel)."""
    m = (th.rand_like(x1[:, :1]) > mask_prob).float()
    eps = th.randn_like(x1)
    return m * x1 + (1 - m) * eps


def _gaussian_kernel1d(sigma, device, dtype):
    """Odd-length 1D Gaussian kernel, radius = ceil(3*sigma), truncated +
    renormalized (standard scipy/PIL convention)."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = th.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = th.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def blur_corrupt(x1, sigma):
    """z0 = GaussianBlur_sigma(x1), applied per-channel (depthwise conv,
    no cross-channel mixing) via separable 1D convs with reflect padding.
    Deterministic given x1 (no noise injected) -- unlike mask/fourier
    corruption, the suppressed high-frequency band is simply discarded,
    not noise-replaced, matching a real deblurring corruption process."""
    C = x1.shape[1]
    kernel1d = _gaussian_kernel1d(sigma, x1.device, x1.dtype)
    radius = (kernel1d.numel() - 1) // 2
    kernel_h = kernel1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    kernel_v = kernel1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.pad(x1, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel_h, groups=C)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel_v, groups=C)
    return x


def _radial_lowpass_mask(hw, cutoff, device):
    """Build a [H,W] radial low-pass mask; cutoff in (0,1] is the fraction
    of max radius kept (frequencies within cutoff*max_radius pass through)."""
    h, w = hw
    fy = th.fft.fftfreq(h, device=device).view(h, 1).expand(h, w)
    fx = th.fft.fftfreq(w, device=device).view(1, w).expand(h, w)
    radius = th.sqrt(fy ** 2 + fx ** 2)
    max_radius = radius.max()
    return (radius <= cutoff * max_radius).float()


def fourier_corrupt(x1, cutoff):
    """Radial low-pass corruption of the latent's FFT: keep frequencies
    within `cutoff` fraction of Nyquist from x1, replace the rest with
    noise spectrum, invert."""
    x1_fft = th.fft.fft2(x1, norm="ortho")
    mask = _radial_lowpass_mask(x1.shape[-2:], cutoff, x1.device)
    eps_fft = th.fft.fft2(th.randn_like(x1), norm="ortho")
    z0_fft = mask * x1_fft + (1 - mask) * eps_fft
    return th.fft.ifft2(z0_fft, norm="ortho").real


def mixture_sample(x1, gaussian_weight, mask_weight, fourier_weight,
                    mask_prob, fourier_cutoff, blur_weight=0.0, blur_sigma=1.0):
    """Per-sample arm draw (categorical over normalized weights) across
    the batch; dispatches each sample to gaussian/mask/fourier/blur corruption.
    Gaussian arm = th.randn_like(x1), identical to baseline Transport.sample.
    Keeps a local arm_idx tensor (not returned in v1) so a future return-value
    change for diagnostics is a one-line addition, not a rewrite."""
    weights = th.tensor(
        [gaussian_weight, mask_weight, fourier_weight, blur_weight],
        device=x1.device, dtype=th.float,
    )
    arm_idx = th.multinomial(weights, x1.shape[0], replacement=True)

    gaussian_x0 = th.randn_like(x1)
    mask_x0 = mask_corrupt(x1, mask_prob)
    fourier_x0 = fourier_corrupt(x1, fourier_cutoff)
    blur_x0 = blur_corrupt(x1, blur_sigma)

    view_shape = (-1, *([1] * (x1.dim() - 1)))
    x0 = th.where((arm_idx == 0).view(view_shape), gaussian_x0, mask_x0)
    x0 = th.where((arm_idx == 2).view(view_shape), fourier_x0, x0)
    x0 = th.where((arm_idx == 3).view(view_shape), blur_x0, x0)
    return x0


if __name__ == "__main__":
    x1 = th.randn(2, 4, 32, 32)
    for fn, kwargs in [
        (mask_corrupt, dict(mask_prob=0.5)),
        (fourier_corrupt, dict(cutoff=0.25)),
        (blur_corrupt, dict(sigma=1.5)),
    ]:
        x0 = fn(x1, **kwargs)
        assert x0.shape == x1.shape
        assert x0.dtype == x1.dtype
        assert x0.device == x1.device
    x0 = mixture_sample(x1, gaussian_weight=1.0, mask_weight=1.0, fourier_weight=1.0,
                         mask_prob=0.5, fourier_cutoff=0.25, blur_weight=1.0, blur_sigma=1.5)
    assert x0.shape == x1.shape
    assert x0.dtype == x1.dtype
    assert x0.device == x1.device
    print("transport/corruption.py smoke checks passed")
