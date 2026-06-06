"""v17 MorphismGym — controlled image-morphism testbed for UNKNOWN morphism discovery (Track A).

Why this exists (pivot from v13-v16): CIFAR/crop ladders showed augmentation discovery has no headroom
over crop for a KNOWN GENERIC nuisance. The fix is not more crop tuning — it is a testbed where valid
morphisms are REAL but HIDDEN, and crucially where a LABEL-FREE anchor can separate valid from invalid.

The gym's load-bearing property: shapes are rendered from latent factors, so the set of valid images is a
low-dim MANIFOLD. A VALID morphism (translation/rotation/scale/hue/brightness/stroke) maps a rendered shape
to ANOTHER rendered shape with different latents -> stays ON the manifold (and inside the broad ANCHOR
support). An INVALID decoy (object-erasing crop, big shear, background-only, shape-warp, occlusion, color
collapse) leaves the manifold -> OFF anchor. Therefore a frozen, label-free anchor (random-conv / AE /
pixel-stats + energy distance) CAN tell valid from invalid here, unlike generic CIFAR crop. That is the
whole thesis this ladder tests.

Discovery rules honored: NO pretrained semantic encoders; GT latent factors used for EVALUATION ONLY
(`latent_*` helpers, never read by the policy or anchor); no held-out labels in discovery.

This module = dataset + morphisms + decoys + anchors (importable). Policy in v17_policy.py, metrics in
v17_eval_metrics.py, orchestration in v17_run_*.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cpu")
H = W = 32
SHAPES = ["circle", "square", "triangle", "star", "ring", "cross"]
NSHAPE = len(SHAPES)

# ---------------------------------------------------------------------------
# Latent factors (eval-only ground truth). z = [cx, cy, rot, log_scale, hue, bright, stroke, bg]
# Indices:
CX, CY, ROT, SCL, HUE, BRT, STK, BG = range(8)
NLAT = 8
LAT_NAMES = ["cx", "cy", "rot", "scale", "hue", "bright", "stroke", "bg"]


def _hsv_to_rgb(h, s, v):
    """h,s,v in [0,1] tensors [B] -> rgb [B,3]. Differentiable."""
    h6 = (h % 1.0) * 6.0
    c = v * s
    x = c * (1 - (h6 % 2 - 1).abs())
    m = v - c
    z = torch.zeros_like(h)
    r = torch.where(h6 < 1, c, torch.where(h6 < 2, x, torch.where(h6 < 3, z,
        torch.where(h6 < 4, z, torch.where(h6 < 5, x, c)))))
    g = torch.where(h6 < 1, x, torch.where(h6 < 2, c, torch.where(h6 < 3, c,
        torch.where(h6 < 4, x, torch.where(h6 < 5, z, z)))))
    b = torch.where(h6 < 1, z, torch.where(h6 < 2, z, torch.where(h6 < 3, x,
        torch.where(h6 < 4, c, torch.where(h6 < 5, c, x)))))
    return torch.stack([r + m, g + m, b + m], 1)


def _sdf(shape_id, u):
    """Signed distance for a unit shape in shape-frame coords u:[B,H,W,2]. <0 inside."""
    ux, uy = u[..., 0], u[..., 1]
    r = torch.sqrt(ux * ux + uy * uy + 1e-9)
    theta = torch.atan2(uy, ux)
    if shape_id == 0:    # circle
        return r - 1.0
    if shape_id == 1:    # square
        return torch.maximum(ux.abs(), uy.abs()) - 1.0
    if shape_id == 2:    # triangle (3-fold, smooth angular radius)
        rad = 1.0 + 0.30 * torch.cos(3 * theta + math.pi)
        return r - rad
    if shape_id == 3:    # star (5-fold)
        rad = 0.85 + 0.45 * torch.cos(5 * theta)
        return r - rad
    if shape_id == 4:    # ring (same outer as circle; fill carved by stroke band below)
        return r - 1.0
    if shape_id == 5:    # cross (union of two bars)
        a = torch.maximum(ux.abs() - 1.0, uy.abs() - 0.32)
        b = torch.maximum(ux.abs() - 0.32, uy.abs() - 1.0)
        return torch.minimum(a, b)
    raise ValueError(shape_id)


def render(z, shape_ids, eps=0.04):
    """Render a batch of shapes. z:[B,8] latents (see indices), shape_ids:[B] long. -> img [B,3,H,W] in [0,1].

    Differentiable in z, but we only use forward. Latents act geometrically so morphisms have known latent
    deltas (used for eval-only coverage)."""
    B = z.size(0)
    dev = z.device
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=dev),
                            torch.linspace(-1, 1, W, device=dev), indexing="ij")
    grid = torch.stack([xs, ys], -1).unsqueeze(0).expand(B, H, W, 2)  # [B,H,W,2]
    cx = z[:, CX].view(B, 1, 1)
    cy = z[:, CY].view(B, 1, 1)
    rot = z[:, ROT].view(B, 1, 1)
    scale = torch.exp(z[:, SCL]).view(B, 1, 1).clamp(0.15, 2.5)
    cos, sin = torch.cos(-rot), torch.sin(-rot)
    px = grid[..., 0] - cx
    py = grid[..., 1] - cy
    ux = (cos * px - sin * py) / (scale * 0.5)
    uy = (sin * px + cos * py) / (scale * 0.5)
    u = torch.stack([ux, uy], -1)

    img = torch.zeros(B, 3, H, W, device=dev)
    # background gray
    bg = z[:, BG].view(B, 1, 1, 1).clamp(0, 1)
    img = img + bg
    # foreground color
    hue = z[:, HUE]
    val = z[:, BRT].clamp(0.15, 1.0)
    color = _hsv_to_rgb(hue, torch.ones_like(hue) * 0.9, val)  # [B,3]
    stroke = z[:, STK].clamp(0.04, 0.5).view(B, 1, 1)

    for sid in range(NSHAPE):
        m = (shape_ids == sid)
        if not m.any():
            continue
        sdf = _sdf(sid, u)  # [B,H,W]
        fill = torch.sigmoid(-sdf / eps)  # inside mass
        band = torch.exp(-(sdf ** 2) / (2 * stroke ** 2))  # outline band of width ~stroke
        if sid == 4:  # ring: keep only the band (hollow)
            fg = band
        else:
            fg = torch.maximum(fill, band)  # filled + thick outline
        fg = fg * m.view(B, 1, 1).float()
        col = color.view(B, 3, 1, 1)
        img = img * (1 - fg.unsqueeze(1)) + col * fg.unsqueeze(1)
    return img.clamp(0, 1)


# ---------------------------------------------------------------------------
# Latent distributions are TASK-DRIVEN. A TaskSpec marks which factors are HIDDEN (narrow in VISIBLE,
# broadened in ANCHOR, with a disjoint tail reserved for HELDOUT eval). Non-hidden factors share one
# moderate range across all three regimes (varied but not a discriminating axis). The morphism families
# whose latent action lies along the hidden factors are the task's "true families" (eval-only).

# factor -> (full_lo, full_hi): the broad valid range; visible is a narrow sub-band, heldout a tail band.
_FULL = {
    CX: (-0.32, 0.32), CY: (-0.32, 0.32), ROT: (-1.30, 1.30), SCL: (-0.32, 0.32),
    HUE: (0.0, 1.0), BRT: (0.45, 0.95), STK: (0.06, 0.30), BG: (0.08, 0.55),
}
# narrow visible band for a HIDDEN factor (low end of full range)
_VIS = {
    CX: (-0.10, 0.10), CY: (-0.10, 0.10), ROT: (-0.17, 0.17), SCL: (-0.05, 0.05),
    HUE: (0.0, 0.15), BRT: (0.78, 0.92), STK: (0.10, 0.15), BG: (0.10, 0.20),
}
# heldout tail (disjoint from visible, inside full) for a HIDDEN factor
_HO = {
    CX: (0.18, 0.32), CY: (0.18, 0.32), ROT: (0.70, 1.25), SCL: (0.16, 0.30),
    HUE: (0.45, 0.78), BRT: (0.48, 0.62), STK: (0.22, 0.30), BG: (0.40, 0.55),
}
# moderate shared range for NON-hidden factors (same in vis/anchor/heldout)
_SHARED = {
    CX: (-0.08, 0.08), CY: (-0.08, 0.08), ROT: (-0.15, 0.15), SCL: (-0.05, 0.05),
    HUE: (0.0, 0.18), BRT: (0.78, 0.92), STK: (0.10, 0.15), BG: (0.10, 0.20),
}

# which morphism family/families move each factor (eval-only "true family" mapping)
FACTOR_FAMILY = {CX: "translate_x", CY: "translate_y", ROT: "rotate", SCL: "scale",
                 HUE: "hue", BRT: "bright", STK: "stroke"}


class TaskSpec:
    def __init__(self, name, hidden_factors):
        self.name = name
        self.hidden = list(hidden_factors)         # factors broadened from visible

    def true_families(self):
        return [FACTOR_FAMILY[f] for f in self.hidden if f in FACTOR_FAMILY]

    def _ranges(self, regime):
        r = {}
        for f in range(NLAT):
            if f in self.hidden:
                r[f] = {"visible": _VIS, "anchor": _FULL, "heldout": _HO}[regime][f]
            else:
                r[f] = _SHARED[f]
        return r

    def sample(self, n, regime, seed=0):
        g = torch.Generator().manual_seed(seed)
        z = torch.zeros(n, NLAT)
        rng = self._ranges(regime)
        for f in range(NLAT):
            lo, hi = rng[f]
            z[:, f] = torch.rand(n, generator=g) * (hi - lo) + lo
        sid = torch.randint(0, NSHAPE, (n,), generator=g)
        return z, sid


# Task presets (Phase 1 discovery matrix)
TASKS = {
    "single_rotation":  TaskSpec("single_rotation", [ROT]),
    "single_hue":       TaskSpec("single_hue", [HUE]),
    "single_scale":     TaskSpec("single_scale", [SCL]),
    "multi_independent": TaskSpec("multi_independent", [ROT, HUE, SCL]),
    "multi_composed":   TaskSpec("multi_composed", [ROT, SCL, CX]),
    "decoy_pressure":   TaskSpec("decoy_pressure", [ROT, HUE]),
    # impossible_control: heldout factor (BG) has NO valid morphism that reaches it -> unreachable.
    "impossible_control": TaskSpec("impossible_control", [BG]),
    # full gym (calibration default)
    "full":             TaskSpec("full", [CX, CY, ROT, SCL, HUE, BRT, STK]),
}


def make_dataset(n, regime, seed=0, task="full"):
    spec = task if isinstance(task, TaskSpec) else TASKS[task]
    z, sid = spec.sample(n, regime, seed=seed)
    with torch.no_grad():
        img = render(z, sid)
    return img, z, sid


# ---------------------------------------------------------------------------
# MORPHISMS. Each family: apply(img, mag) -> img, and latent_delta(mag) -> dz (eval-only coverage).
# Families are differentiable in `mag` (a per-image scalar tensor [B]) so the policy can learn magnitudes.

def _affine(img, theta):  # theta:[B,2,3]
    grid = F.affine_grid(theta, list(img.size()), align_corners=False)
    return F.grid_sample(img, grid, align_corners=False, padding_mode="border")


def m_translate_x(img, mag):
    B = img.size(0); th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 0, 2] = -mag  # grid coords: shift content by +mag
    return _affine(img, th)


def m_translate_y(img, mag):
    B = img.size(0); th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 1, 2] = -mag
    return _affine(img, th)


def m_rotate(img, mag):
    B = img.size(0); c, s = torch.cos(mag), torch.sin(mag)
    th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = c; th[:, 0, 1] = -s; th[:, 1, 0] = s; th[:, 1, 1] = c
    return _affine(img, th)


def m_scale(img, mag):  # mag = log-scale; >0 zoom in (bigger object)
    B = img.size(0); inv = torch.exp(-mag)
    th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = inv; th[:, 1, 1] = inv
    return _affine(img, th)


def m_hue(img, mag):
    """Proper hue rotation: rotate RGB about the gray axis (1,1,1)/sqrt3 by angle mag*2pi (Rodrigues).
    Preserves luminance and saturation -> output stays on the rendered-color manifold (valid)."""
    B = img.size(0)
    ang = mag * 2 * math.pi
    cos, sin = torch.cos(ang), torch.sin(ang)
    # R = cos*I + sin*[k]_x + (1-cos)*k k^T, with k=(1,1,1)/sqrt3 -> k k^T = 1/3 * ones
    kx = 1.0 / math.sqrt(3.0)  # each off-diagonal cross-term coefficient
    oc = (1 - cos) / 3.0
    a = cos + oc
    b = oc - kx * sin
    c = oc + kx * sin
    M = torch.stack([torch.stack([a, b, c], 1),
                     torch.stack([c, a, b], 1),
                     torch.stack([b, c, a], 1)], 1)  # [B,3,3] circulant
    flat = img.view(B, 3, -1)
    out = torch.bmm(M, flat).view_as(img)
    return out.clamp(0, 1)


def m_bright(img, mag):
    return (img * (1.0 + mag.view(-1, 1, 1, 1))).clamp(0, 1)


def m_stroke(img, mag):
    """Approximate stroke-thickness as soft dilation(+)/erosion(-). Differentiable via blend weight."""
    k = 3
    dil = F.max_pool2d(img, k, 1, k // 2)
    ero = -F.max_pool2d(-img, k, 1, k // 2)
    w = mag.view(-1, 1, 1, 1)
    pos = torch.clamp(w, 0, 1); neg = torch.clamp(-w, 0, 1)
    return (img * (1 - pos - neg) + dil * pos + ero * neg).clamp(0, 1)


# latent deltas (eval-only): how a family of magnitude `m` (python float) moves GT latents
def latent_delta(fam, m):
    dz = torch.zeros(NLAT)
    if fam == "translate_x": dz[CX] = m
    elif fam == "translate_y": dz[CY] = m
    elif fam == "rotate": dz[ROT] = m
    elif fam == "scale": dz[SCL] = m
    elif fam == "hue": dz[HUE] = m
    elif fam == "bright": dz[BRT] = m
    elif fam == "stroke": dz[STK] = m * 0.4
    return dz


# VALID families (latent-aligned) + magnitude ranges (used to scale a [-1,1] policy magnitude)
VALID_FAMILIES = {
    "translate_x": (m_translate_x, 0.35),
    "translate_y": (m_translate_y, 0.35),
    "rotate":      (m_rotate, 1.3),
    "scale":       (m_scale, 0.33),
    "hue":         (m_hue, 0.5),
    "bright":      (m_bright, 0.30),
    "stroke":      (m_stroke, 1.0),
}

# ---------------------------------------------------------------------------
# DECOYS — invalid transforms that LEAVE the rendered-shape manifold. No valid latent action.

def d_crop_erase(img, mag):  # aggressive center zoom that cuts the object out of frame
    B = img.size(0); z = 2.2 + 1.5 * torch.clamp(mag, 0, 1)
    th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = 1.0 / z; th[:, 1, 1] = 1.0 / z
    th[:, 0, 2] = 0.6; th[:, 1, 2] = 0.6  # off-center so object leaves frame
    return _affine(img, th)


def d_big_shear(img, mag):
    B = img.size(0); s = 1.2 + 1.5 * torch.clamp(mag, 0, 1)
    th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 0, 1] = s
    return _affine(img, th)


def d_bg_shortcut(img, mag):  # replace foreground with background shade (object erased)
    bg = img.mean(dim=(2, 3), keepdim=True)
    a = torch.clamp(mag, 0.3, 1).view(-1, 1, 1, 1)
    return img * (1 - a) + bg * a


def d_shape_warp(img, mag):  # strong sinusoidal warp -> changes shape identity
    B, _, h, w = img.size()
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=img.device),
                            torch.linspace(-1, 1, w, device=img.device), indexing="ij")
    amp = 0.25 + 0.35 * torch.clamp(mag, 0, 1)
    grid = torch.stack([xs + amp.view(-1, 1, 1) * torch.sin(4 * ys),
                        ys + amp.view(-1, 1, 1) * torch.sin(4 * xs)], -1)
    if grid.dim() == 3:
        grid = grid.unsqueeze(0).expand(B, h, w, 2)
    return F.grid_sample(img, grid, align_corners=False, padding_mode="border")


def d_occlude(img, mag):  # paint a black box over the center
    out = img.clone()
    s = int(6 + 8 * float(torch.clamp(mag, 0, 1).mean()))
    c = H // 2
    out[:, :, c - s // 2:c + s // 2, c - s // 2:c + s // 2] = 0.0
    return out


def d_color_collapse(img, mag):  # desaturate to gray (extreme color collapse)
    gray = img.mean(1, keepdim=True).expand_as(img)
    a = torch.clamp(mag, 0.4, 1).view(-1, 1, 1, 1)
    return img * (1 - a) + gray * a


DECOY_FAMILIES = {
    "crop_erase":     (d_crop_erase, 1.0),
    "big_shear":      (d_big_shear, 1.0),
    "bg_shortcut":    (d_bg_shortcut, 1.0),
    "shape_warp":     (d_shape_warp, 1.0),
    "occlude":        (d_occlude, 1.0),
    "color_collapse": (d_color_collapse, 1.0),
}


def apply_family(name, img, mag_unit):
    """Apply family by name with a UNIT magnitude in [-1,1] (or [0,1] for decoys), scaled to its range."""
    if name in VALID_FAMILIES:
        fn, rng = VALID_FAMILIES[name]
        return fn(img, mag_unit * rng)
    fn, rng = DECOY_FAMILIES[name]
    return fn(img, mag_unit.abs() * rng)


# ---------------------------------------------------------------------------
# ANCHORS (label-free, NO pretrained semantics). Used for 0C selection + discovery objective.

class PixelStatsAnchor(nn.Module):
    """Downsampled pixels + per-channel color histogram stats. No learning."""
    def __init__(self, ds=8):
        super().__init__(); self.ds = ds
    def forward(self, x):
        p = F.adaptive_avg_pool2d(x, self.ds).flatten(1)
        mu = x.mean(dim=(2, 3)); sd = x.std(dim=(2, 3))
        return torch.cat([p, mu, sd], 1)
    fg = forward


class RandomConvAnchor(nn.Module):
    """Frozen random conv stack (in-domain, untrained — NOT a pretrained semantic encoder)."""
    def __init__(self, seed=777):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (0.1 if p.dim() == 1 else 0.3))
        for p in self.parameters():
            p.requires_grad_(False)
    def forward(self, x):
        h = self.net(x)                                    # [B,64,h,w]
        flat = h.flatten(1)
        # Label-free off-manifold statistics (NO semantics): catch the "stealth" decoys that low-level
        # conv features miss. (a) spatial std -> flatness (bg-erase); (b) chroma = spread across RGB means
        # -> grayscale (color-collapse). Both ~0 for collapsed images, high for rendered shapes.
        sp_std = h.flatten(2).std(2)                       # [B,64]
        ch_mean = x.mean(dim=(2, 3))                       # [B,3] per-channel mean
        chroma = ch_mean.std(1, keepdim=True)              # [B,1] colorfulness
        sat = (x.amax(1) - x.amin(1)).flatten(1).mean(1, keepdim=True)  # [B,1] per-pixel chroma -> sat
        return torch.cat([flat, 4.0 * sp_std, 30.0 * chroma, 30.0 * sat], 1)
    fg = forward


class TinyAE(nn.Module):
    """Small in-domain autoencoder trained FROM SCRATCH on anchor images (no labels). Bottleneck = anchor."""
    def __init__(self, zdim=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(64 * 4 * 4, zdim))
        self.dec = nn.Sequential(nn.Linear(zdim, 64 * 4 * 4), nn.ReLU(),
                                 nn.Unflatten(1, (64, 4, 4)),
                                 nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.enc(x)
    fg = forward

    def recon(self, x):
        return self.dec(self.enc(x))


def train_tiny_ae(imgs, steps=400, bs=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    ae = TinyAE().to(imgs.device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    n = imgs.size(0)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        xb = imgs[idx]
        loss = F.mse_loss(ae.recon(xb), xb)
        opt.zero_grad(); loss.backward(); opt.step()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def build_anchor(kind, anchor_imgs, seed=777):
    if kind == "pixel":
        return PixelStatsAnchor().to(anchor_imgs.device)
    if kind == "randomconv":
        return RandomConvAnchor(seed=seed).to(anchor_imgs.device)
    if kind == "ae":
        return train_tiny_ae(anchor_imgs, seed=seed)
    raise ValueError(kind)


def energy_distance(a, b):
    """ED between feature sets a:[N,D], b:[M,D]. >=0, 0 iff same distribution."""
    return (2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean())


class AnchorScorer:
    """Wraps a label-free encoder + PCA-WHITENED anchor statistics. Whitening (fit on anchor features)
    equalizes dimension contributions so no raw conv dim dominates the distance — the proven recipe from
    rungs 12-14 / _se2_discovery. Exposes:
      - ed(x): distribution energy distance of batch x's features to the anchor (DISCOVERY signal).
      - nn_dist / validity(x): per-image distance to anchor (eval metric).
    NO labels, NO morphism IDs used."""
    def __init__(self, encoder, anchor_imgs, k_pca=48, k=8, ref_n=512, ae_recon_weight=0.0,
                 use_stats=True, stats_weight=1.5):
        self.enc = encoder
        self.k = k
        self.ae_recon_weight = ae_recon_weight
        self.use_stats = use_stats
        self.stats_weight = stats_weight
        with torch.no_grad():
            fa = encoder(anchor_imgs)
            self.mu = fa.mean(0, keepdim=True)
            kq = min(k_pca, fa.size(1), fa.size(0))
            _, S, V = torch.pca_lowrank(fa - self.mu, q=kq)
            self.proj = V[:, :kq]                              # [D,kq]
            self.whiten = 1.0 / (S[:kq] / math.sqrt(max(fa.size(0) - 1, 1)) + 1e-4)
            if use_stats:
                st = self._raw_stats(anchor_imgs)
                self.st_mu = st.mean(0, keepdim=True)
                self.st_sd = st.std(0, keepdim=True) + 1e-6
            self.anchor_feats = self.feats(anchor_imgs)
            self.ref = self.anchor_feats[:ref_n]

    def _raw_stats(self, x):
        """Label-free structure/color stats that survive whitening (catch stealth collapse decoys).
        chroma (colorfulness), saturation (per-pixel channel spread), edge-energy (spatial structure)."""
        ch_mean = x.mean(dim=(2, 3))                       # [B,3]
        chroma = ch_mean.std(1, keepdim=True)              # colorfulness
        sat = (x.amax(1) - x.amin(1)).flatten(1).mean(1, keepdim=True)
        gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().flatten(1).mean(1, keepdim=True)
        gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().flatten(1).mean(1, keepdim=True)
        return torch.cat([chroma, sat, gx + gy], 1)        # [B,3]

    def feats(self, x, grad=False):
        f = self.enc.fg(x) if grad else self.enc(x)
        w = ((f - self.mu) @ self.proj) * self.whiten
        if self.use_stats:
            st = self._raw_stats(x)
            st = (st - self.st_mu) / self.st_sd * self.stats_weight
            w = torch.cat([w, st], 1)
        return w

    def ed(self, x, grad=False, ref=None):
        f = self.feats(x, grad=grad)
        r = self.ref if ref is None else ref
        return energy_distance(f, r)

    def nn_dist(self, x, grad=False):
        f = self.feats(x, grad=grad)
        d = torch.cdist(f, self.ref)
        kk = min(self.k, self.ref.size(0))
        return d.topk(kk, dim=1, largest=False).values.mean(1)  # [B]

    def validity(self, x):
        s = -self.nn_dist(x)
        if self.ae_recon_weight > 0 and hasattr(self.enc, "recon"):
            with torch.no_grad():
                rec = ((self.enc.recon(x) - x) ** 2).flatten(1).mean(1)
            s = s - self.ae_recon_weight * rec
        return s
