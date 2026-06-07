"""Multi-family morphism discovery for the EqM bridge — the v17 MorphismGym recipe ported to CIFAR.

Supersedes the single-operator v12/v13 bridge. v17 (symmetry-discovery, Phase 0-3 + MNIST/dSprites
transfer) validated: a label-free PCA-WHITENED random-conv energy-distance anchor (+ chroma/edge stats)
separates valid (on-manifold) morphisms from invalid decoys, and an EMA-reward BANDIT over candidate
families (valid + decoys) learns WHICH morphisms are valid and at what magnitude — beating random valid
augmentation and matching/beating a known oracle on an EqM-lite proxy. Here that policy is discovered
offline against real CIFAR, FROZEN, and used as EqM augmentation.

Self-contained (torch only). Images are CIFAR in [-1,1].
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- morphisms (act on [-1,1])
def _affine(img, theta):
    grid = F.affine_grid(theta, list(img.size()), align_corners=False)
    return F.grid_sample(img, grid, align_corners=False, padding_mode="reflection")


def m_translate_x(img, mag):
    B = img.size(0); th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 0, 2] = -mag
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


def m_scale(img, mag):
    B = img.size(0); inv = torch.exp(-mag)
    th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = inv; th[:, 1, 1] = inv
    return _affine(img, th)


def m_hue(img, mag):
    B = img.size(0); ang = mag * 2 * math.pi
    cos, sin = torch.cos(ang), torch.sin(ang)
    kx = 1.0 / math.sqrt(3.0); oc = (1 - cos) / 3.0
    a = cos + oc; b = oc - kx * sin; c = oc + kx * sin
    M = torch.stack([torch.stack([a, b, c], 1), torch.stack([c, a, b], 1),
                     torch.stack([b, c, a], 1)], 1)
    return torch.bmm(M, img.view(B, 3, -1)).view_as(img).clamp(-1, 1)


def m_bright(img, mag):
    return (img + mag.view(-1, 1, 1, 1) * 0.6).clamp(-1, 1)


def d_crop_erase(img, mag):
    B = img.size(0); z = 2.2 + 1.5 * torch.clamp(mag, 0, 1)
    th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = 1.0 / z; th[:, 1, 1] = 1.0 / z; th[:, 0, 2] = 0.6; th[:, 1, 2] = 0.6
    return _affine(img, th)


def d_big_shear(img, mag):
    B = img.size(0); s = 1.2 + 1.5 * torch.clamp(mag, 0, 1)
    th = torch.zeros(B, 2, 3, device=img.device)
    th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 0, 1] = s
    return _affine(img, th)


def d_color_collapse(img, mag):
    gray = img.mean(1, keepdim=True).expand_as(img)
    al = torch.clamp(mag, 0.4, 1).view(-1, 1, 1, 1)
    return img * (1 - al) + gray * al


VALID_FAMILIES = {"translate_x": (m_translate_x, 0.30), "translate_y": (m_translate_y, 0.30),
                  "rotate": (m_rotate, 0.6), "scale": (m_scale, 0.30),
                  "hue": (m_hue, 0.5), "bright": (m_bright, 1.0)}
DECOY_FAMILIES = {"crop_erase": (d_crop_erase, 1.0), "big_shear": (d_big_shear, 1.0),
                  "color_collapse": (d_color_collapse, 1.0)}
ALL_FAMILIES = list(VALID_FAMILIES) + list(DECOY_FAMILIES)


def apply_family(name, img, mag_unit):
    if name in VALID_FAMILIES:
        fn, rng = VALID_FAMILIES[name]; return fn(img, mag_unit * rng)
    fn, rng = DECOY_FAMILIES[name]; return fn(img, mag_unit.abs() * rng)


# --------------------------------------------------------------------------- anchor (PCA-whitened ED)
class RandomConvAnchor(nn.Module):
    def __init__(self, seed=777):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (0.1 if p.dim() == 1 else 0.3))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h = self.net(x); flat = h.flatten(1)
        sp_std = h.flatten(2).std(2)
        ch_mean = x.mean(dim=(2, 3)); chroma = ch_mean.std(1, keepdim=True)
        sat = (x.amax(1) - x.amin(1)).flatten(1).mean(1, keepdim=True)
        return torch.cat([flat, 4.0 * sp_std, 30.0 * chroma, 30.0 * sat], 1)

    fg = forward


def energy_distance(a, b):
    return 2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean()


class _AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                               nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 16, 64))
        self.d = nn.Sequential(nn.Linear(64, 64 * 16), nn.ReLU(), nn.Unflatten(1, (64, 4, 4)),
                               nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
                               nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                               nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        return self.d(self.e(x))


def _valid_nuisance_aug(x, gen):
    """MILD valid nuisances (flip/small affine/photometric) -- the manifold the robust AE should ACCEPT.
    Decoys (extreme shear/erase/collapse) are NOT in this set -> stay off-manifold -> high recon error."""
    B = x.size(0)
    flip = torch.rand(B, generator=gen) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), torch.flip(x, dims=[3]), x)
    ang = (torch.rand(B, generator=gen) * 2 - 1) * math.radians(15)
    sc = torch.exp((torch.rand(B, generator=gen) * 2 - 1) * 0.15)
    tx = (torch.rand(B, generator=gen) * 2 - 1) * 0.18
    ty = (torch.rand(B, generator=gen) * 2 - 1) * 0.18
    c, s = torch.cos(ang), torch.sin(ang)
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = c / sc; th[:, 0, 1] = -s / sc; th[:, 1, 0] = s / sc; th[:, 1, 1] = c / sc
    th[:, 0, 2] = tx; th[:, 1, 2] = ty
    grid = F.affine_grid(th, list(x.size()), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
    x = m_bright(x, (torch.rand(B, generator=gen) * 2 - 1) * 0.5)
    x = m_hue(x, (torch.rand(B, generator=gen) * 2 - 1) * 0.5)
    return x


def train_robust_ae(real, steps=1500, seed=0):
    """AE trained on valid-nuisance-augmented CIFAR. Its recon error is LOW for valid morphisms (in-manifold)
    and HIGH for destructive decoys -- complements the random-conv ED (which catches crop/collapse but
    misses shear). Together (max-z) they separate valid from ALL decoys on CIFAR (sep>0, proxy-validated)."""
    torch.manual_seed(seed)
    ae = _AE().to(real.device); opt = torch.optim.Adam(ae.parameters(), 1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        xb = _valid_nuisance_aug(real[torch.randint(0, real.size(0), (128,), generator=g)], g)
        loss = F.mse_loss(ae(xb), xb)
        opt.zero_grad(); loss.backward(); opt.step()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def ae_recon(ae, x):
    return ((ae(x) - x) ** 2).flatten(1).mean(1)  # [B]


class AnchorScorer:
    def __init__(self, anchor_imgs, k_pca=48, ref_n=512, seed=777, use_stats=True, stats_weight=1.5):
        self.enc = RandomConvAnchor(seed=seed).to(anchor_imgs.device)
        self.use_stats = use_stats; self.stats_weight = stats_weight
        with torch.no_grad():
            fa = self.enc(anchor_imgs); self.mu = fa.mean(0, keepdim=True)
            kq = min(k_pca, fa.size(1), fa.size(0))
            _, S, V = torch.pca_lowrank(fa - self.mu, q=kq)
            self.proj = V[:, :kq]
            self.whiten = 1.0 / (S[:kq] / math.sqrt(max(fa.size(0) - 1, 1)) + 1e-4)
            if use_stats:
                st = self._raw_stats(anchor_imgs)
                self.st_mu = st.mean(0, keepdim=True); self.st_sd = st.std(0, keepdim=True) + 1e-6
            self.ref = self.feats(anchor_imgs)[:ref_n]

    def _raw_stats(self, x):
        ch_mean = x.mean(dim=(2, 3)); chroma = ch_mean.std(1, keepdim=True)
        sat = (x.amax(1) - x.amin(1)).flatten(1).mean(1, keepdim=True)
        gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().flatten(1).mean(1, keepdim=True)
        gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().flatten(1).mean(1, keepdim=True)
        return torch.cat([chroma, sat, gx + gy], 1)

    def feats(self, x, grad=False):
        f = self.enc.fg(x) if grad else self.enc(x)
        w = ((f - self.mu) @ self.proj) * self.whiten
        if self.use_stats:
            st = (self._raw_stats(x) - self.st_mu) / self.st_sd * self.stats_weight
            w = torch.cat([w, st], 1)
        return w

    def ed(self, x, grad=False):
        return energy_distance(self.feats(x, grad=grad), self.ref)


# --------------------------------------------------------------------------- policy + discovery (bandit)
class MorphismPolicy(nn.Module):
    def __init__(self, families, depth=1, init_logstd=-1.0):
        super().__init__()
        self.families = list(families); self.K = len(self.families); self.depth = depth
        self.logits = nn.Parameter(torch.zeros(self.K))
        self.mag_mu = nn.Parameter(torch.zeros(self.K))
        self.mag_logstd = nn.Parameter(torch.full((self.K,), init_logstd))

    def family_weights(self):
        return torch.softmax(self.logits, 0)

    def grouped_layer(self, img, gen=None):
        B = img.size(0); w = self.family_weights()
        fam_idx = torch.multinomial(w, B, replacement=True, generator=gen)
        eps = torch.randn(B, generator=gen, device=img.device)
        out = img.clone(); pre = torch.zeros(B, device=img.device)
        for j, fam in enumerate(self.families):
            m = (fam_idx == j)
            if not m.any():
                continue
            pre_j = self.mag_mu[j] + torch.exp(self.mag_logstd[j]) * eps[m]
            idx = m.nonzero(as_tuple=True)[0]
            out = out.index_copy(0, idx, apply_family(fam, img[m], torch.tanh(pre_j)))
            pre = pre.index_copy(0, idx, pre_j)
        return out, fam_idx, pre

    def grouped_transform(self, img, gen=None):
        out = img; pre_layers = []; fam_last = None
        for _ in range(self.depth):
            out, fam_idx, pre = self.grouped_layer(out, gen=gen)
            pre_layers.append(pre); fam_last = fam_idx
        return out, fam_last, torch.stack(pre_layers, 1)

    @torch.no_grad()
    def sample_transform(self, img, gen=None):
        return self.grouped_transform(img, gen=gen)[0]


def discover(policy, visible, scorer, steps=400, lr=0.05, bs=128, a_move=1.0, a_div=0.3,
             a_bound=0.1, move_margin=0.6, use_anchor=True, use_diversity=True, use_bounds=True,
             seed=0, ema=0.9, ae=None, ae_weight=50.0):
    """Optional `ae` (robust AE) adds an off-manifold recon term: combined validity = conv-ED + ae_weight*
    recon. The conv-ED catches crop/collapse; the AE catches shear; together they reject all decoys on
    natural CIFAR (proxy-validated, sep>0). ae_weight scales recon to conv-ED units."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam([{"params": [policy.mag_mu, policy.mag_logstd], "lr": lr},
                            {"params": [policy.logits], "lr": lr * 3}])
    n = visible.size(0); g = torch.Generator().manual_seed(seed + 1)
    K = policy.K; reward = torch.zeros(K); seen = torch.zeros(K); d = {}
    for step in range(steps):
        xb = visible[torch.randint(0, n, (bs,), generator=g)]
        out, fam_idx, pre = policy.grouped_transform(xb, gen=g)
        with torch.no_grad():
            f0 = scorer.feats(xb)
        fT = scorer.feats(out, grad=True)
        ed = energy_distance(fT, scorer.ref); move = (fT - f0).norm(dim=1).mean()
        rec_b = ae_recon(ae, out) if ae is not None else None      # [B] per-image, differentiable
        L = torch.zeros((), device=xb.device)
        if use_anchor:
            L = L + ed; d["L_anchor"] = float(ed)
            if ae is not None:
                L = L + ae_weight * rec_b.mean(); d["L_ae"] = float(rec_b.mean())
        L = L + a_move * F.relu(move_margin - move); d["move"] = float(move)
        if use_bounds:
            L = L + a_bound * (F.relu(pre.abs() - 2.0) ** 2).mean()
        with torch.no_grad():
            fdet = fT.detach(); rec_d = rec_b.detach() if rec_b is not None else None
            for j in range(K):
                m = (fam_idx == j)
                if m.sum() >= 8:
                    edj = float(energy_distance(fdet[m], scorer.ref)) if use_anchor else 0.0
                    aej = float(rec_d[m].mean()) * ae_weight if (use_anchor and rec_d is not None) else 0.0
                    movej = float((fdet[m] - f0[m]).norm(dim=1).mean())
                    rj = -(edj + aej) + 0.2 * min(movej / move_margin, 2.0)
                    reward[j] = ema * reward[j] + (1 - ema) * rj if seen[j] > 0 else rj
                    seen[j] += 1
        w = policy.family_weights(); L = L + (-(w * reward.detach()).sum())
        if use_diversity:
            L = L + a_div * (w * (w + 1e-9).log()).sum()
        opt.zero_grad(); L.backward(); opt.step()
    fw = {f: float(x) for f, x in zip(policy.families, policy.family_weights())}
    eff = {f: float(policy.family_weights()[i] * policy.mag_mu[i].abs())
           for i, f in enumerate(policy.families)}
    tot = sum(eff.values()) + 1e-9
    decoy_usage = sum(eff[f] for f in DECOY_FAMILIES) / tot
    return {"final": d, "family_weights": fw,
            "mag_mu": {f: float(m) for f, m in zip(policy.families, policy.mag_mu.detach())},
            "reward": {f: float(r) for f, r in zip(policy.families, reward)},
            "decoy_usage": decoy_usage,
            "effective_usage": {f: eff[f] / tot for f in policy.families}}
