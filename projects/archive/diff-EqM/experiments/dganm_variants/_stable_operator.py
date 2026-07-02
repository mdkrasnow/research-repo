"""Frozen-anchor stable-generator operator discovery for the EqM bridge (v12).

Implements the toy recipe (rungs 9-14) at image scale:
  - operator = a LOW-DIM image group action: spatial affine warp by M = matrix_exp(A), A a learned 2x2
    generator (rotation/scale/shear; no translation). Applied via affine_grid + grid_sample. This is the
    image analog of the toy's latent linear group generator, kept low-dim so exp(A)+stability is honest.
  - frozen anchor = a FROZEN random-conv feature extractor (non-co-adapting, cheap, no external weights).
    Discovery minimizes energy-distance between anchor-features of T(x) and of fresh real images, so T is
    pulled to map real images to the (feature) data manifold. NOT the live EqM field -> no co-adaptation.
  - stability reg = det(M)~1 + well-conditioned (cond->1), i.e. an isometric image group action.
  - move term = keep T non-identity (||T(x)-x|| ~ a small reference affine), so it doesn't collapse to I.

Operator is discovered ONCE, FROZEN, then used as EqM augmentation in the v12 step_fn.
Operator-quality diagnostics (anchor loss, det, cond, ||M-I||, feature-shift consistency) are the honest
signal — per the toy lesson that recall/coverage is confounded.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def affine_warp(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply 2x2 linear image transform M (no translation) via grid_sample. x: [B,3,H,W]."""
    B = x.size(0)
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, :2, :2] = M.unsqueeze(0)
    grid = F.affine_grid(theta, list(x.size()), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


class RandomConvAnchor(nn.Module):
    """FROZEN random-conv feature extractor. Non-co-adapting manifold reference (no external weights)."""
    def __init__(self, seed: int = 12345):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (0.1 if p.dim() == 1 else 0.3))
        for p in self.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def features(self, x):
        f = self.net(x)
        return f.flatten(1)

    def features_grad(self, x):
        # params are frozen (requires_grad_(False)); WITHOUT no_grad so gradient FLOWS to the input x.
        # Required for operator discovery: with @no_grad the anchor term contributes ZERO gradient to A,
        # so the operator was driven only by move+stability (sign/angle = init seed). (bug fixed 2026-06-04)
        return self.net(x).flatten(1)


def _energy_distance(a, b):
    return 2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean()


def discover_stable_affine(real_batch_fn, device, *, steps=600, batch=128, lr=2e-3,
                           lam_move=0.5, lam_det=1.0, lam_cond=1.0, ref_deg=15.0,
                           anchor_seed=12345, init_seed=0, mix_pi=0.5):
    """Discover a frozen stable affine operator against a frozen random-conv anchor.

    real_batch_fn(n) -> [n,3,H,W] real image batch (values in model's range). Returns (M_frozen, diag).

    Objective = MIXTURE-anchor (proxy-validated 2026-06-04): augmentation forms a MIXTURE
    (1-pi)*P_real + pi*T(P_real), not a replacement, so we pull that mixture's anchor features toward
    the real distribution rather than pulling T(real) alone. This is direction-sensitive without labels
    (fixes the sign-gauge ambiguity of the old T-only objective). diag["A_gen"] is returned so the EqM
    step can augment with the whole one-parameter GROUP exp(t*A), t~U(-1,1), not one arbitrary orientation.
    """
    torch.manual_seed(init_seed)
    anchor = RandomConvAnchor(seed=anchor_seed).to(device).eval()
    A = torch.nn.Parameter((0.05 * torch.randn(2, 2, device=device)))
    opt = torch.optim.Adam([A], lr=lr)

    # NON-LEAKING move band: hinge on pixel-distance in [floor(=5deg), cap(=45deg)] rotations -- non-collapse
    # + non-blowup only. NOT a hard target at ref_deg (that would hand the operator its magnitude). The
    # anchor term (now grad-flowing) must find the actual angle/direction. Scale-free (normalized by floor).
    with torch.no_grad():
        def _rot_move(deg):
            th = math.radians(deg); R = torch.tensor([[math.cos(th), -math.sin(th)],
                                                      [math.sin(th), math.cos(th)]], device=device)
            xr0 = real_batch_fn(min(batch, 64)).to(device)
            return (affine_warp(xr0, R) - xr0).flatten(1).norm(dim=1).mean()
        move_floor = _rot_move(5.0); move_cap = _rot_move(45.0); target_move = _rot_move(ref_deg)  # report-only
        af0 = anchor.features(real_batch_fn(min(batch, 64)).to(device))
        af1 = anchor.features(real_batch_fn(min(batch, 64)).to(device))
        anchor0 = float(_energy_distance(af0, af1))   # baseline anchor distance (real vs real)

    def move_pen(mv):
        return torch.relu((move_floor - mv) / move_floor) ** 2 + torch.relu((mv - move_cap) / move_floor) ** 2

    diag_hist = []
    for s in range(steps):
        xr = real_batch_fn(batch).to(device)
        M = torch.matrix_exp(A)
        Tx = affine_warp(xr, M)
        # MIXTURE-anchor: match (1-pi)*real_feats + pi*T_feats to fresh real feats (mass-correct, direction-sensitive)
        fR = anchor.features(real_batch_fn(batch).to(device))                 # eval-side anchor (no grad ok)
        f_vis = anchor.features(xr); fT = anchor.features_grad(Tx)            # fT GRAD-flowing -> trains A
        n_aug = max(1, int(mix_pi * batch)); n_vis = max(0, batch - n_aug)
        iv = torch.randperm(f_vis.size(0), device=device)[:min(n_vis, f_vis.size(0))]
        ia = torch.randperm(fT.size(0), device=device)[:min(n_aug, fT.size(0))]
        f_mix = torch.cat([f_vis[iv], fT[ia]], 0)
        anchor_loss = _energy_distance(f_mix, fR)
        move = (Tx - xr).flatten(1).norm(dim=1).mean()
        det = torch.det(M); sv = torch.linalg.svdvals(M)
        cond = (sv.max() / sv.min().clamp_min(1e-6) - 1.0) ** 2
        loss = (anchor_loss + lam_move * move_pen(move)
                + lam_det * (torch.log(det.abs() + 1e-8)) ** 2 + lam_cond * cond)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == steps - 1:
            diag_hist.append((s, float(anchor_loss), float(move)))

    with torch.no_grad():
        M = torch.matrix_exp(A).detach()
        xr = real_batch_fn(batch).to(device); Tx = affine_warp(xr, M)
        fT = anchor.features(Tx); fR = anchor.features(real_batch_fn(batch).to(device))
        sv = torch.linalg.svdvals(M)
        # feature-shift consistency: cosine between per-sample feature displacements (coherent op -> high)
        dF = anchor.features(Tx) - anchor.features(xr)
        dF = dF / (dF.norm(dim=1, keepdim=True) + 1e-8)
        shift_consistency = float((dF @ dF.mean(0, keepdim=True).T).mean())
        diag = {
            "M": M.tolist(),
            "A_gen": A.detach().tolist(),   # generator: EqM aug uses exp(t*A), t~U(-1,1) (the GROUP)
            "angle_deg": float(torch.atan2(M[1, 0], M[0, 0]) * 180 / math.pi),
            "det": float(torch.det(M)),
            "cond": float(sv.max() / sv.min().clamp_min(1e-6)),
            "sv": [float(v) for v in sv],
            "off_identity": float((M - torch.eye(2, device=device)).norm()),
            "anchor_baseline_real_real": anchor0,
            "anchor_final_T_real": float(_energy_distance(fT, fR)),
            "move": float((Tx - xr).flatten(1).norm(dim=1).mean()),
            "target_move": float(target_move),
            "feature_shift_consistency": shift_consistency,
            "history": diag_hist,
        }
    return M, diag
