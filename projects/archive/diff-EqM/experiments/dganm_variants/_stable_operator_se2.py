"""SE(2)-style homogeneous-affine stable-generator discovery for the v13 EqM bridge.

v12 (`_stable_operator.py`) used a 2x2 linear generator (rotation/scale/shear, NO translation) — a poor
match for CIFAR, whose useful nuisance transforms are translation/crop. This sibling uses a 3x3
HOMOGENEOUS affine generator:

    A2 = learned (2,3) ; A = [[A2],[0,0,0]] ; M = matrix_exp(A)   # M is affine, last row [0,0,1]

so M can express TRANSLATION (M[:2,2]) + scale/shear/small-rotation. Acts on images via affine_grid +
grid_sample. Same CORRECTED discovery recipe validated in the SE2 proxy (results_cifar_se2_proxy.json):
  - frozen GRAD-FLOWING random-conv anchor (gradient must reach the operator — the v12 no-grad bug)
  - MIXTURE-anchor objective: (1-pi)*P_real + pi*T(P_real) ~= P_real  (mass-correct, direction-sensitive)
  - broad NON-LEAKING move hinge (bounds image move in [floor,cap], does NOT target a magnitude)
  - stability reg: det(linear)~1, cond(linear)->1; translation bounded by the move hinge
  - NO live-EqM closure (operator discovered once, frozen, used as augmentation)

Returns (M3_frozen, diag); diag["A_gen"] is the (2,3) generator so the EqM step can augment with the
whole one-parameter GROUP exp(t*A), t~U(-1,1).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from ._stable_operator import RandomConvAnchor, _energy_distance

PXN_DEFAULT = 2.0 / 32.0   # 1px in affine_grid normalized coords for a 32px image (rescaled per image size)


def affine_warp3(x: torch.Tensor, M3: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 homogeneous affine M3 to images x [B,C,H,W] via grid_sample."""
    B = x.size(0)
    theta = M3[:2, :].unsqueeze(0).repeat(B, 1, 1)
    grid = F.affine_grid(theta, list(x.size()), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


def _build_M(A2: torch.Tensor, device) -> torch.Tensor:
    A = torch.cat([A2, torch.zeros(1, 3, device=device)], 0)
    return torch.matrix_exp(A)


def discover_stable_se2(real_batch_fn, device, *, steps=600, batch=128, lr=2e-3,
                        lam_move=0.5, lam_det=1.0, lam_cond=1.0,
                        move_floor_px=2.0, move_cap_px=10.0, img_size=32,
                        anchor_seed=12345, init_seed=0, mix_pi=0.5):
    """Discover a frozen stable SE(2) affine operator vs a frozen grad-flowing random-conv anchor.

    real_batch_fn(n) -> [n,3,H,W] real image batch. Returns (M3_frozen, diag).
    move_floor_px/move_cap_px define a BROAD non-leaking image-move band (not a target magnitude).
    """
    torch.manual_seed(init_seed)
    anchor = RandomConvAnchor(seed=anchor_seed).to(device).eval()
    A2 = torch.nn.Parameter(0.02 * torch.randn(2, 3, device=device))
    opt = torch.optim.Adam([A2], lr=lr)
    pxn = 2.0 / img_size

    def trans_px(dx):
        M = torch.eye(3, device=device); M[0, 2] = dx * pxn
        return M

    with torch.no_grad():
        xr0 = real_batch_fn(min(batch, 64)).to(device)
        move_floor = (affine_warp3(xr0, trans_px(move_floor_px)) - xr0).flatten(1).norm(dim=1).mean()
        move_cap = (affine_warp3(xr0, trans_px(move_cap_px)) - xr0).flatten(1).norm(dim=1).mean()
        af0 = anchor.features(real_batch_fn(min(batch, 64)).to(device))
        af1 = anchor.features(real_batch_fn(min(batch, 64)).to(device))
        anchor0 = float(_energy_distance(af0, af1))

    def move_pen(mv):
        return torch.relu((move_floor - mv) / move_floor) ** 2 + torch.relu((mv - move_cap) / move_floor) ** 2

    m = 256; n_aug = max(1, int(mix_pi * m)); n_vis = max(0, m - n_aug)
    diag_hist = []
    for s in range(steps):
        xr = real_batch_fn(batch).to(device)
        M = _build_M(A2, device); Tx = affine_warp3(xr, M)
        L = M[:2, :2]; sv = torch.linalg.svdvals(L); det = torch.det(L)
        fR = anchor.features(real_batch_fn(batch).to(device))
        f_vis = anchor.features(xr); fT = anchor.features_grad(Tx)   # fT GRAD-flowing -> trains A2
        iv = torch.randperm(f_vis.size(0), device=device)[:min(n_vis, f_vis.size(0))]
        ia = torch.randperm(fT.size(0), device=device)[:min(n_aug, fT.size(0))]
        f_mix = torch.cat([f_vis[iv], fT[ia]], 0)
        anchor_loss = _energy_distance(f_mix, fR)
        move = (Tx - xr).flatten(1).norm(dim=1).mean()
        cond = (sv.max() / sv.min().clamp_min(1e-6) - 1.0) ** 2
        loss = (anchor_loss + lam_move * move_pen(move)
                + lam_det * (torch.log(det.abs() + 1e-8)) ** 2 + lam_cond * cond)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == steps - 1:
            diag_hist.append((s, float(anchor_loss), float(move)))

    with torch.no_grad():
        M = _build_M(A2.detach(), device)
        L = M[:2, :2]; sv = torch.linalg.svdvals(L)
        xr = real_batch_fn(batch).to(device); Tx = affine_warp3(xr, M)
        fT = anchor.features(Tx); fR = anchor.features(real_batch_fn(batch).to(device))
        dF = anchor.features(Tx) - anchor.features(xr)
        dF = dF / (dF.norm(dim=1, keepdim=True) + 1e-8)
        shift_consistency = float((dF @ dF.mean(0, keepdim=True).T).mean())
        diag = {
            "M": M.tolist(),
            "A_gen": A2.detach().tolist(),
            "tx_px": float(M[0, 2] / pxn), "ty_px": float(M[1, 2] / pxn),
            "det": float(torch.det(L)),
            "cond": float(sv.max() / sv.min().clamp_min(1e-6)),
            "sv": [float(v) for v in sv],
            "lin_off_identity": float((L - torch.eye(2, device=device)).norm()),
            "anchor_baseline_real_real": anchor0,
            "anchor_final_T_real": float(_energy_distance(fT, fR)),
            "move": float((Tx - xr).flatten(1).norm(dim=1).mean()),
            "move_floor": float(move_floor), "move_cap": float(move_cap),
            "feature_shift_consistency": shift_consistency,
            "history": diag_hist,
        }
    return M, diag
