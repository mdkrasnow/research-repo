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


def _block_mask_single(h, w, area_frac, device):
    """Keep-mask with 1-2 axis-aligned rectangular block(s) zeroed out,
    total zeroed area ~= area_frac*h*w. Random aspect ratio per block,
    random placement. Large-contiguous-region family of structured_mask_corrupt."""
    m = th.ones(h, w, device=device)
    n_blocks = 1 if th.rand((), device=device).item() < 0.5 else 2
    remaining = area_frac
    for i in range(n_blocks):
        this_frac = remaining / (n_blocks - i)
        target_area = max(1.0, this_frac * h * w)
        log_aspect = (th.rand((), device=device).item() - 0.5) * 2 * math.log(2.0)
        aspect = math.exp(log_aspect)
        bh = int(round(math.sqrt(target_area * aspect)))
        bw = int(round(math.sqrt(target_area / aspect)))
        bh = min(max(bh, 1), h)
        bw = min(max(bw, 1), w)
        top = int(th.randint(0, h - bh + 1, (1,), device=device).item())
        left = int(th.randint(0, w - bw + 1, (1,), device=device).item())
        m[top:top + bh, left:left + bw] = 0.0
        remaining -= this_frac
    return m


def _patch_mask_single(h, w, area_frac, patch_size, device):
    """Keep-mask over non-overlapping patch_size x patch_size cells (MAE-style
    coarse patchify), a random subset of cells zeroed until area_frac of grid
    covered. Medium-patches family of structured_mask_corrupt."""
    ph = math.ceil(h / patch_size)
    pw = math.ceil(w / patch_size)
    n_cells = ph * pw
    n_mask_cells = int(round(area_frac * n_cells))
    n_mask_cells = min(max(n_mask_cells, 0), n_cells)
    cell_mask = th.ones(n_cells, device=device)
    if n_mask_cells > 0:
        perm = th.randperm(n_cells, device=device)
        cell_mask[perm[:n_mask_cells]] = 0.0
    cell_mask = cell_mask.view(ph, pw)
    m = cell_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    return m[:h, :w]


def _region_mask_single(h, w, area_frac, device):
    """Keep-mask with one irregular 4-connected region zeroed out, grown via
    random walk/flood-fill from a random seed until area_frac of grid covered.
    Irregular-connected-region family of structured_mask_corrupt (spatial
    only -- no Fourier-domain operations)."""
    target_area = int(round(area_frac * h * w))
    target_area = min(max(target_area, 1), h * w)
    occupied = th.zeros(h, w, dtype=th.bool, device=device)
    seed_y = int(th.randint(0, h, (1,), device=device).item())
    seed_x = int(th.randint(0, w, (1,), device=device).item())
    occupied[seed_y, seed_x] = True
    frontier = [(seed_y, seed_x)]
    count = 1
    while count < target_area and frontier:
        pick = int(th.randint(0, len(frontier), (1,), device=device).item())
        y, x = frontier[pick]
        neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        order = th.randperm(4, device=device).tolist()
        grown = False
        for oi in order:
            ny, nx = neighbors[oi]
            if 0 <= ny < h and 0 <= nx < w and not occupied[ny, nx]:
                occupied[ny, nx] = True
                frontier.append((ny, nx))
                count += 1
                grown = True
                break
        if not grown:
            frontier.pop(pick)
    return (~occupied).float()


def make_structured_mask(x1, mask_prob, block_weight=0.35, patch_weight=0.35,
                          region_weight=0.20, elementwise_weight=0.10, patch_size=4):
    """Per-sample categorical draw over 4 structured-mask families (large
    contiguous block(s), medium non-overlapping patches, one irregular
    connected region, minority elementwise Bernoulli) producing a spatial
    keep-mask m (1=visible, 0=masked-out), shared across latent channels like
    mask_corrupt. All families operate purely spatially -- no Fourier-domain
    operations. Returns m, shape [B,1,H,W], dtype matching x1."""
    b = x1.shape[0]
    h, w = x1.shape[-2:]
    device = x1.device
    weights = th.tensor(
        [block_weight, patch_weight, region_weight, elementwise_weight],
        device=device, dtype=th.float,
    )
    arm_idx = th.multinomial(weights, b, replacement=True)
    masks = []
    for i in range(b):
        arm = int(arm_idx[i].item())
        if arm == 0:
            m = _block_mask_single(h, w, mask_prob, device)
        elif arm == 1:
            m = _patch_mask_single(h, w, mask_prob, patch_size, device)
        elif arm == 2:
            m = _region_mask_single(h, w, mask_prob, device)
        else:
            m = (th.rand(h, w, device=device) > mask_prob).float()
        masks.append(m)
    return th.stack(masks, dim=0).unsqueeze(1).to(x1.dtype)


def structured_mask_corrupt(x1, mask_prob, block_weight=0.35, patch_weight=0.35,
                             region_weight=0.20, elementwise_weight=0.10, patch_size=4):
    """z0 = m*x1 + (1-m)*eps, m from make_structured_mask -- structured
    counterpart to mask_corrupt, same masked-fraction severity (mask_prob)
    but with spatially contiguous corruption instead of pure elementwise
    Bernoulli, per Stage 2 structured-mask variant proposal
    (documentation/stage2_structured_mask_proposal_2026-07-22.md)."""
    m = make_structured_mask(x1, mask_prob, block_weight, patch_weight,
                              region_weight, elementwise_weight, patch_size)
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


def downsample_corrupt(x1, factor):
    """z0 = Upsample(Downsample(x1, 1/factor), original_size), bilinear both
    ways -- removes high-frequency detail while preserving coarse structure,
    a third variant on the mask/blur "coarse-structure-kept, detail-destroyed"
    theme (spatial resampling instead of erasure or smoothing). Deterministic
    given x1 and factor, like blur_corrupt -- no noise injected."""
    h, w = x1.shape[-2:]
    small = F.interpolate(x1, scale_factor=1.0 / factor, mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


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
                    mask_prob, fourier_cutoff, blur_weight=0.0, blur_sigma=1.0,
                    downsample_weight=0.0, downsample_factor=4.0,
                    structured_mask_weight=0.0):
    """Per-sample arm draw (categorical over normalized weights) across
    the batch; dispatches each sample to exactly one of
    gaussian/mask/fourier/blur/downsample/structured_mask corruption
    (categorical draw, never blended within a sample). Gaussian arm =
    th.randn_like(x1), identical to baseline Transport.sample. Keeps a local
    arm_idx tensor (not returned in v1) so a future return-value change for
    diagnostics is a one-line addition, not a rewrite."""
    weights = th.tensor(
        [gaussian_weight, mask_weight, fourier_weight, blur_weight,
         downsample_weight, structured_mask_weight],
        device=x1.device, dtype=th.float,
    )
    arm_idx = th.multinomial(weights, x1.shape[0], replacement=True)

    gaussian_x0 = th.randn_like(x1)
    mask_x0 = mask_corrupt(x1, mask_prob)
    fourier_x0 = fourier_corrupt(x1, fourier_cutoff)
    blur_x0 = blur_corrupt(x1, blur_sigma)
    downsample_x0 = downsample_corrupt(x1, downsample_factor)
    structured_mask_x0 = structured_mask_corrupt(x1, mask_prob)

    view_shape = (-1, *([1] * (x1.dim() - 1)))
    x0 = th.where((arm_idx == 0).view(view_shape), gaussian_x0, mask_x0)
    x0 = th.where((arm_idx == 2).view(view_shape), fourier_x0, x0)
    x0 = th.where((arm_idx == 3).view(view_shape), blur_x0, x0)
    x0 = th.where((arm_idx == 4).view(view_shape), downsample_x0, x0)
    x0 = th.where((arm_idx == 5).view(view_shape), structured_mask_x0, x0)
    return x0


if __name__ == "__main__":
    x1 = th.randn(2, 4, 32, 32)
    for fn, kwargs in [
        (mask_corrupt, dict(mask_prob=0.5)),
        (fourier_corrupt, dict(cutoff=0.25)),
        (blur_corrupt, dict(sigma=1.5)),
        (downsample_corrupt, dict(factor=4.0)),
        (structured_mask_corrupt, dict(mask_prob=0.5)),
    ]:
        x0 = fn(x1, **kwargs)
        assert x0.shape == x1.shape
        assert x0.dtype == x1.dtype
        assert x0.device == x1.device
    x0 = mixture_sample(x1, gaussian_weight=1.0, mask_weight=1.0, fourier_weight=1.0,
                         mask_prob=0.5, fourier_cutoff=0.25, blur_weight=1.0, blur_sigma=1.5,
                         downsample_weight=1.0, downsample_factor=4.0,
                         structured_mask_weight=1.0)
    assert x0.shape == x1.shape
    assert x0.dtype == x1.dtype
    assert x0.device == x1.device

    # structured-mask-specific checks: masked fraction near target, and each
    # family individually produces a valid binary keep-mask of the right shape
    for area_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        m = make_structured_mask(th.randn(64, 4, 32, 32), area_frac)
        assert m.shape == (64, 1, 32, 32)
        assert th.all((m == 0) | (m == 1))
        observed_masked_frac = 1.0 - m.mean().item()
        assert abs(observed_masked_frac - area_frac) < 0.15, \
            f"masked fraction {observed_masked_frac} far from target {area_frac}"

    assert _block_mask_single(32, 32, 0.5, th.device("cpu")).shape == (32, 32)
    assert _patch_mask_single(32, 32, 0.5, 4, th.device("cpu")).shape == (32, 32)
    assert _region_mask_single(32, 32, 0.5, th.device("cpu")).shape == (32, 32)
    for m in [
        _block_mask_single(32, 32, 0.5, th.device("cpu")),
        _patch_mask_single(32, 32, 0.5, 4, th.device("cpu")),
        _region_mask_single(32, 32, 0.5, th.device("cpu")),
    ]:
        assert th.all((m == 0) | (m == 1))

    print("transport/corruption.py smoke checks passed")
