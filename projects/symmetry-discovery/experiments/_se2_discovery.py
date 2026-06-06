"""Shared SE(2) translation discovery for v14 rungs D/E (unsupervised, frozen-anchor, mixture objective).

Validated recipe (rungs A-C): frozen GRAD-FLOWING fine random-conv anchor + MIXTURE-anchor objective
((1-pi)P_real + pi*T(P_real) ~= P_real) + broad NON-LEAKING move hinge + det/cond stability. Discovers
either a SINGLE translation operator (v13-style) or a translation DISTRIBUTION (mean + per-axis std).
No labels, no held-out targets. The distribution's spread EMERGES (Rung B: single-op is underdetermined
on a translation-spread anchor; Rung C: a distribution covers the 2D region a point cannot).
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F

PXN = 2.0 / 32.0


def trans_M(dx, dy, dev):
    M = torch.eye(3, device=dev); M[0, 2] = dx * PXN; M[1, 2] = dy * PXN
    return M

def warp(x, M):
    th = M[:2, :].unsqueeze(0).repeat(x.size(0), 1, 1)
    return F.grid_sample(x, F.affine_grid(th, list(x.size()), align_corners=False),
                         align_corners=False, padding_mode="reflection")

def warp_txty(x, txty):  # per-image translation, txty:[B,2] px
    B = x.size(0); th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 0, 2] = txty[:, 0] * PXN; th[:, 1, 2] = txty[:, 1] * PXN
    return F.grid_sample(x, F.affine_grid(th, list(x.size()), align_corners=False),
                         align_corners=False, padding_mode="reflection")


class FrozenConv(nn.Module):
    """FINE frozen random-conv (stride [2,1,1] -> 16x16): small translations are visible (not sub-cell)."""
    def __init__(self, seed=777):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,1,1), nn.ReLU(),
                                 nn.Conv2d(64,64,3,1,1), nn.ReLU())
        with torch.no_grad():
            for p in self.net.parameters(): p.copy_(torch.randn(p.shape, generator=g)*(0.1 if p.dim()==1 else 0.3))
        for p in self.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x): return self.net(x).flatten(1)
    def fg(self, x): return self.net(x).flatten(1)   # grad flows to input


def _ed_t(a, b): return 2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean()
def _build1(A2, dev): return torch.matrix_exp(torch.cat([A2, torch.zeros(1,3,device=dev)], 0))


def _setup(imgs, k_pca=32, seed=777):
    dev = imgs.device; enc = FrozenConv(seed=seed).to(dev)
    with torch.no_grad():
        a = enc(imgs); mu = a.mean(0, keepdim=True)
        _, _, Vp = torch.pca_lowrank(a - mu, q=min(k_pca, a.size(1))); proj = Vp[:, :k_pca]
        m2 = (warp(imgs, trans_M(2, 0, dev)) - imgs).flatten(1).norm(dim=1).mean()
        m10 = (warp(imgs, trans_M(10, 0, dev)) - imgs).flatten(1).norm(dim=1).mean()
    def Pg(x): return (enc.fg(x) - mu) @ proj
    def P(x):
        with torch.no_grad(): return (enc(x) - mu) @ proj
    anchor = P(imgs)
    return dev, enc, P, Pg, anchor, float(m2), float(m10)


def _move_pen(mv, m2, m10):
    return torch.relu((m2 - mv) / m2) ** 2 + torch.relu((mv - m10) / m2) ** 2


def discover_single(imgs, steps=500, lr=5e-3, seed=2):
    dev, enc, P, Pg, anchor, m2, m10 = _setup(imgs)
    torch.manual_seed(seed); A2 = torch.nn.Parameter(0.02*torch.randn(2,3,device=dev)); opt=torch.optim.Adam([A2],lr=lr)
    n = imgs.size(0)
    for s in range(steps):
        xv = imgs[torch.randperm(n)[:128]]
        M = _build1(A2, dev); Tx = warp(xv, M); L = M[:2,:2]
        mv = (Tx-xv).flatten(1).norm(dim=1).mean(); sv = torch.linalg.svdvals(L); det = torch.det(L)
        fmix = torch.cat([Pg(xv).detach()[:128], Pg(Tx)[:128]], 0)
        ai = torch.randperm(anchor.size(0))[:256]
        loss = _ed_t(fmix, anchor[ai]) + 0.5*_move_pen(mv,m2,m10) + (torch.log(det.abs()+1e-8))**2 + (sv.max()/sv.min().clamp_min(1e-6)-1)**2
        opt.zero_grad(); loss.backward(); opt.step()
    M = _build1(A2.detach(), dev)
    return torch.tensor([float(M[0,2]/PXN), float(M[1,2]/PXN)])   # (tx,ty) px


def discover_distribution(imgs, steps=500, lr=5e-3, seed=4):
    """Returns (mean_txty[2], std_txty[2]) in px — a Gaussian translation distribution."""
    dev, enc, P, Pg, anchor, m2, m10 = _setup(imgs)
    torch.manual_seed(seed)
    meanA = torch.nn.Parameter(0.02*torch.randn(2,3,device=dev))
    logsig = torch.nn.Parameter(torch.tensor([-1.0,-1.0], device=dev))
    opt = torch.optim.Adam([meanA, logsig], lr=lr); n = imgs.size(0)
    for s in range(steps):
        xv = imgs[torch.randperm(n)[:128]]
        dx = torch.exp(logsig[0])*torch.randn(()); dy = torch.exp(logsig[1])*torch.randn(())
        A2c = torch.stack([torch.stack([meanA[0,0],meanA[0,1],meanA[0,2]+dx]),
                           torch.stack([meanA[1,0],meanA[1,1],meanA[1,2]+dy])])
        M = _build1(A2c, dev); Tx = warp(xv, M); L = M[:2,:2]
        mv = (Tx-xv).flatten(1).norm(dim=1).mean(); sv = torch.linalg.svdvals(L); det = torch.det(L)
        fmix = torch.cat([Pg(xv).detach()[:128], Pg(Tx)[:128]], 0)
        ai = torch.randperm(anchor.size(0))[:256]
        loss = _ed_t(fmix, anchor[ai]) + 0.5*_move_pen(mv,m2,m10) + (torch.log(det.abs()+1e-8))**2 + (sv.max()/sv.min().clamp_min(1e-6)-1)**2
        opt.zero_grad(); loss.backward(); opt.step()
    mean = torch.tensor([float(meanA[0,2].detach()/PXN), float(meanA[1,2].detach()/PXN)])
    std = torch.tensor([float(torch.exp(logsig[0]).detach()/PXN), float(torch.exp(logsig[1]).detach()/PXN)])
    return mean, std


# ---- v14 (beat-crop) augmentation POLICY: qθ over a 2-gen SE(2) Lie basis ----
_Z = None
def _zrow(dev):
    return torch.zeros(1, 3, device=dev)

def sample_policy_M(A1, A2, logsig, dev):
    z = torch.randn(2, device=dev) * torch.exp(logsig)
    return torch.matrix_exp(z[0]*torch.cat([A1, _zrow(dev)]) + z[1]*torch.cat([A2, _zrow(dev)]))

def policy_txty(A1, A2, logsig, n, dev):
    """Sample n per-image (tx,ty) px draws from the policy (for augmentation)."""
    z = torch.randn(n, 2, device=dev) * torch.exp(logsig)              # [n,2] coeffs
    # M_i translation = (z_i1*A1 + z_i2*A2)[:, 2] to first order is not exact; use exact exp per draw is slow.
    # Spatial translation of exp(sum z*A) for our near-pure-translation generators ~ linear in z; use the
    # generators' translation columns directly (A*[:,2]) -> tx,ty = z1*A1[:,2:] + z2*A2[:,2:] (px via /PXN).
    t = z[:, :1]*A1[:, 2].unsqueeze(0) + z[:, 1:2]*A2[:, 2].unsqueeze(0)   # [n,2] in normalized units
    return t / PXN

def discover_policy(imgs, steps=400, lr=5e-3, entropy=True, eig2_floor=2.0,
                    scorer=None, labels=None, lam_util=0.0, seed=3):
    """Learn qθ over M=exp(z1*A1+z2*A2), z~N(0,diag(exp(logsig)^2)). anchor + move hinge + stability,
    optional 2D induced-covariance entropy floor, optional UTILITY (make augs the scorer finds hard,
    on-manifold). Returns (A1,A2,logsig). Utility uses TRAINING labels only (no held-out)."""
    dev, enc, P, Pg, anchor, m2, m10 = _setup(imgs)
    torch.manual_seed(seed)
    A1 = torch.nn.Parameter(0.02*torch.randn(2,3,device=dev)); A2 = torch.nn.Parameter(0.02*torch.randn(2,3,device=dev))
    logsig = torch.nn.Parameter(torch.tensor([-0.7,-0.7], device=dev))
    opt = torch.optim.Adam([A1,A2,logsig], lr=lr); n = imgs.size(0); Zr = _zrow(dev)
    for s in range(steps):
        idx = torch.randperm(n)[:128]; xv = imgs[idx]
        z = torch.randn(2, device=dev)*torch.exp(logsig)
        M = torch.matrix_exp(z[0]*torch.cat([A1,Zr])+z[1]*torch.cat([A2,Zr])); Tx = warp(xv, M); L = M[:2,:2]
        mv = (Tx-xv).flatten(1).norm(dim=1).mean(); sv = torch.linalg.svdvals(L); det = torch.det(L)
        fmix = torch.cat([Pg(xv).detach()[:128], Pg(Tx)[:128]], 0)
        loss = _ed_t(fmix, anchor[torch.randperm(anchor.size(0))[:256]]) + 0.5*_move_pen(mv,m2,m10) \
               + (torch.log(det.abs()+1e-8))**2 + (sv.max()/sv.min().clamp_min(1e-6)-1)**2
        if entropy:
            tt = []
            for _ in range(20):
                zz = torch.randn(2, device=dev)*torch.exp(logsig)
                Mk = torch.matrix_exp(zz[0]*torch.cat([A1,Zr])+zz[1]*torch.cat([A2,Zr]))
                tt.append(torch.stack([Mk[0,2]/PXN, Mk[1,2]/PXN]))
            ev = torch.linalg.eigvalsh(torch.cov(torch.stack(tt,0).T)).clamp_min(0)
            loss = loss + 0.5*torch.relu(eig2_floor - ev.min())**2
        if scorer is not None and lam_util > 0 and labels is not None:
            import torch.nn.functional as F
            yb = labels[idx]
            ce = F.cross_entropy(scorer(Tx), yb)        # make augs the scorer finds HARD (on-manifold via anchor)
            loss = loss - lam_util*ce
        opt.zero_grad(); loss.backward(); opt.step()
    return A1.detach(), A2.detach(), logsig.detach()
