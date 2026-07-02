"""v14 Rung C — single-operator vs operator-DISTRIBUTION proxy (the crux of the v13 failure mode).

v13 found ONE clean translation; FID barely moved because crop aug's value is a high-entropy 2D shift
DISTRIBUTION, not one point. Here the held-out gap is a 2D translation BLOCK (tx,ty ~ U[3,5]^2) that a
single operator can only touch at one point but an operator-distribution can cover.

Setup: real CIFAR, frozen random-conv -> PCA32 features. Per-image random translations build the pools:
  visible = U([-6,6]^2) \ held-quadrant ; held = U([3,5]^2) (2D block) ; anchor = visible ∪ held (unlabeled).

Arms (energy-distance to the held BLOCK in common PCA space; + support diagnostics):
  BASE                       visible as-is
  KNOWN_CROP_DISTRIBUTION    per-image random U([3,5]^2) crop (oracle distribution; positive control)
  SINGLE_DISCOVERED_SE2      discover ONE A (v13-style) -> one (tx,ty) point
  RANDOM_SINGLE_SE2          one random A
  DISCOVERED_2GEN_SE2        discover A1,A2; sample t1,t2~U(-1,1) -> exp(t1A1+t2A2) (2D subgroup)
  RANDOM_2GEN_SE2            random A1,A2 sampled the same way
  DISCOVERED_DISTRIBUTION_SE2  discover mean A + per-axis log-sigma on (tx,ty); sample A=mean+sigma*eps

Discovery: mixture-anchor ((1-pi)P_vis + pi*T(P_vis) ~= anchor), grad-flowing encoder, broad move hinge,
det/cond stability. NO entropy floor first — test whether 2D spread EMERGES from matching the 2D anchor
(Rung B showed single-op is underdetermined, so a distribution should *want* to spread). No held labels.

PASS: KNOWN_CROP_DIST < BASE; DISCOVERED_DISTRIBUTION < SINGLE_DISCOVERED and < RANDOM_2GEN and < BASE;
and the discovered distribution has HIGH-RANK 2D support (cov eig-ratio not ~0; not one diagonal line).
"""
import argparse, json, math, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEV = torch.device("cpu")
K_PCA = 32; PXN = 2.0 / 32.0
N_IMG = 300; SET_N = 900
LAM_MOVE = 0.5; LAM_DET = 1.0; LAM_COND = 1.0


def trans_M(dx, dy):
    M = torch.eye(3, device=DEV); M[0, 2] = dx * PXN; M[1, 2] = dy * PXN
    return M

def warp(x, M):
    th = M[:2, :].unsqueeze(0).repeat(x.size(0), 1, 1)
    return F.grid_sample(x, F.affine_grid(th, list(x.size()), align_corners=False),
                         align_corners=False, padding_mode="reflection")

def warp_per_img(x, txty):  # txty: [B,2] px -> per-image translation
    B = x.size(0); th = torch.zeros(B, 2, 3, device=DEV)
    th[:, 0, 0] = 1; th[:, 1, 1] = 1; th[:, 0, 2] = txty[:, 0] * PXN; th[:, 1, 2] = txty[:, 1] * PXN
    return F.grid_sample(x, F.affine_grid(th, list(x.size()), align_corners=False),
                         align_corners=False, padding_mode="reflection")


class FrozenConv(nn.Module):
    # FINE encoder: stride [2,1,1] -> 16x16 spatial (cell=2px), so SMALL translations (2.5-5px) are
    # visible in features. A stride-8 (4x4) encoder makes sub-cell shifts invisible -> energy-distance
    # noise floor (the coarse-feature trap; also why v13's coarse anchor saw translation weakly).
    def __init__(self, seed=777):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,1,1), nn.ReLU(),
                                 nn.Conv2d(64,64,3,1,1), nn.ReLU())
        with torch.no_grad():
            for p in self.net.parameters(): p.copy_(torch.randn(p.shape, generator=g)*(0.1 if p.dim()==1 else 0.3))
        for p in self.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x): return self.net(x).flatten(1)
    def fg(self, x): return self.net(x).flatten(1)


def ed(a, b): return float(2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean())
def ed_t(a, b): return 2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean()


def sample_region(n, region):
    # WIDE 2D test: visible = near-center disk (r<=1.5, ~canonical jitter); held = a wide ANNULUS
    # (r in [2.5,5], ALL angles) — a genuine 2D area no single (tx,ty) point can cover but a
    # distribution can. anchor = visible ∪ held. This rewards COVERAGE, not a single well-placed point.
    if region == "held":
        r = 2.5 + 2.5 * torch.rand(n, device=DEV)
        th = 2 * math.pi * torch.rand(n, device=DEV)
        return torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
    # visible: small disk r<=1.5
    r = 1.5 * torch.rand(n, device=DEV).sqrt()
    th = 2 * math.pi * torch.rand(n, device=DEV)
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_se2_distribution.json")
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    steps = 250 if a.quick else 600
    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                 if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    imgs = torch.stack([ds[i][0] for i in torch.randperm(len(ds))[:N_IMG].tolist()]).to(DEV)
    enc = FrozenConv().to(DEV)

    def pool(region, passes=3):
        fs = [enc(warp_per_img(imgs, sample_region(imgs.size(0), region))) for _ in range(passes)]
        return torch.cat(fs, 0)
    vis_raw = pool("visible"); held_raw = pool("held"); anchor_raw = torch.cat([vis_raw, held_raw], 0)
    mu = anchor_raw.mean(0, keepdim=True)
    _,_,Vp = torch.pca_lowrank(anchor_raw-mu, q=K_PCA); proj = Vp[:, :K_PCA]
    def P(f): return (f-mu)@proj
    def Pg(x): return (enc.fg(x)-mu)@proj
    vis, held, anchor = P(vis_raw), P(held_raw), P(anchor_raw)
    def sub(t, n=SET_N): return t[torch.randperm(t.size(0))[:min(n,t.size(0))]]
    held_e, anchor_e = sub(held), sub(anchor)
    base_gap = ed(sub(vis), held_e)
    with torch.no_grad():
        m2 = (warp(imgs, trans_M(2,0))-imgs).flatten(1).norm(dim=1).mean()
        m10 = (warp(imgs, trans_M(10,0))-imgs).flatten(1).norm(dim=1).mean()
    def move_pen(mv): return torch.relu((m2-mv)/m2)**2 + torch.relu((mv-m10)/m2)**2

    def build1(A2): return torch.matrix_exp(torch.cat([A2, torch.zeros(1,3,device=DEV)],0))

    # ---- known crop distribution (oracle positive control) ----
    def known_features():
        return P(enc(warp_per_img(imgs, sample_region(imgs.size(0), "held"))))
    known_gap = ed(sub(known_features()), held_e)

    # ---- generic apply helpers for eval ----
    def apply_single(M):  return P(enc(warp(sub(imgs, 256), M)))
    def apply_2gen(A1, A2, draws=4):
        out = []
        xb = sub(imgs, 256)
        for _ in range(draws):
            t1 = (torch.rand(())*2-1).item(); t2 = (torch.rand(())*2-1).item()
            out.append(P(enc(warp(xb, torch.matrix_exp(t1*torch.cat([A1,torch.zeros(1,3,device=DEV)])+t2*torch.cat([A2,torch.zeros(1,3,device=DEV)]))))))
        return torch.cat(out, 0)
    def apply_dist(meanA, logsig, draws=4):
        out = []; xb = sub(imgs, 256)
        for _ in range(draws):
            A2 = meanA.clone()
            A2[0,2] += torch.exp(logsig[0])*torch.randn(()); A2[1,2] += torch.exp(logsig[1])*torch.randn(())
            out.append(P(enc(warp(xb, build1(A2)))))
        return torch.cat(out, 0)

    # ---- support diagnostics: sample many ops, cov of (tx,ty) ----
    def support_of(sampler, n=64):
        pts = torch.stack([sampler() for _ in range(n)], 0)  # [n,2] tx,ty px
        c = torch.cov(pts.T) if pts.size(0) > 1 else torch.zeros(2,2)
        evs = torch.linalg.eigvalsh(c).clamp_min(0)
        e1, e2 = float(evs.max()), float(evs.min())
        logdet = float(torch.log(c.det().clamp_min(1e-12)))
        return {"tx_mean": float(pts[:,0].mean()), "ty_mean": float(pts[:,1].mean()),
                "tx_std": float(pts[:,0].std()), "ty_std": float(pts[:,1].std()),
                "eig1": e1, "eig2": e2, "eig_ratio": e2/max(e1,1e-9), "logdet_cov": logdet}

    # ===== discovery routines (mixture-anchor) =====
    def disc_single():
        torch.manual_seed(2); A2 = torch.nn.Parameter(0.02*torch.randn(2,3,device=DEV)); opt=torch.optim.Adam([A2],lr=5e-3)
        for s in range(steps):
            xv = warp_per_img(imgs[torch.randperm(imgs.size(0))[:128]], sample_region(128,"visible"))
            M = build1(A2); Tx = warp(xv, M); L=M[:2,:2]
            mv=(Tx-xv).flatten(1).norm(dim=1).mean(); sv=torch.linalg.svdvals(L); det=torch.det(L)
            fmix=torch.cat([Pg(xv).detach()[:128], Pg(Tx)[:128]],0)
            loss=ed_t(fmix, sub(anchor,256))+LAM_MOVE*move_pen(mv)+LAM_DET*(torch.log(det.abs()+1e-8))**2+LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1)**2
            opt.zero_grad(); loss.backward(); opt.step()
        return A2.detach()

    def disc_2gen():
        torch.manual_seed(3); A1=torch.nn.Parameter(0.02*torch.randn(2,3,device=DEV)); A2=torch.nn.Parameter(0.02*torch.randn(2,3,device=DEV))
        opt=torch.optim.Adam([A1,A2],lr=5e-3); Z=torch.zeros(1,3,device=DEV)
        for s in range(steps):
            xv = warp_per_img(imgs[torch.randperm(imgs.size(0))[:128]], sample_region(128,"visible"))
            t1=(torch.rand(())*2-1); t2=(torch.rand(())*2-1)
            M=torch.matrix_exp(t1*torch.cat([A1,Z])+t2*torch.cat([A2,Z])); Tx=warp(xv,M); L=M[:2,:2]
            mv=(Tx-xv).flatten(1).norm(dim=1).mean(); sv=torch.linalg.svdvals(L); det=torch.det(L)
            fmix=torch.cat([Pg(xv).detach()[:128], Pg(Tx)[:128]],0)
            loss=ed_t(fmix, sub(anchor,256))+LAM_MOVE*move_pen(mv)+LAM_DET*(torch.log(det.abs()+1e-8))**2+LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1)**2
            opt.zero_grad(); loss.backward(); opt.step()
        return A1.detach(), A2.detach()

    def disc_dist():
        torch.manual_seed(4); meanA=torch.nn.Parameter(0.02*torch.randn(2,3,device=DEV))
        logsig=torch.nn.Parameter(torch.tensor([-1.0,-1.0],device=DEV))  # init small spread
        opt=torch.optim.Adam([meanA,logsig],lr=5e-3)
        for s in range(steps):
            xv = warp_per_img(imgs[torch.randperm(imgs.size(0))[:128]], sample_region(128,"visible"))
            A2=meanA.clone()
            eps=torch.randn(2,device=DEV)
            A2=A2 + torch.zeros_like(A2)
            dx=torch.exp(logsig[0])*torch.randn(()); dy=torch.exp(logsig[1])*torch.randn(())
            A2c=meanA.clone(); A2c=torch.stack([torch.stack([meanA[0,0],meanA[0,1],meanA[0,2]+dx]),
                                                torch.stack([meanA[1,0],meanA[1,1],meanA[1,2]+dy])])
            M=build1(A2c); Tx=warp(xv,M); L=M[:2,:2]
            mv=(Tx-xv).flatten(1).norm(dim=1).mean(); sv=torch.linalg.svdvals(L); det=torch.det(L)
            fmix=torch.cat([Pg(xv).detach()[:128], Pg(Tx)[:128]],0)
            loss=ed_t(fmix, sub(anchor,256))+LAM_MOVE*move_pen(mv)+LAM_DET*(torch.log(det.abs()+1e-8))**2+LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1)**2
            opt.zero_grad(); loss.backward(); opt.step()
        return meanA.detach(), logsig.detach()

    res = {"config": {"steps": steps, "base_gap": base_gap, "known_gap": known_gap}}
    # SINGLE discovered + random
    As = disc_single(); Ms = build1(As)
    torch.manual_seed(11); Ar = torch.zeros(2,3,device=DEV); Ar[:,:2]=0.1*torch.randn(2,2,device=DEV); Ar[:,2]=0.1*torch.randn(2,device=DEV); Mr=build1(Ar)
    def samp_single(M): return lambda: torch.tensor([float(M[0,2]/PXN), float(M[1,2]/PXN)])
    res["SINGLE_DISCOVERED_SE2"]={"gap": ed(apply_single(Ms), held_e), **support_of(samp_single(Ms))}
    res["RANDOM_SINGLE_SE2"]={"gap": ed(apply_single(Mr), held_e), **support_of(samp_single(Mr))}
    # 2GEN discovered + random
    A1,A2 = disc_2gen()
    torch.manual_seed(12); R1=torch.zeros(2,3,device=DEV); R1[:,2]=0.15*torch.randn(2,device=DEV); R2=torch.zeros(2,3,device=DEV); R2[:,2]=0.15*torch.randn(2,device=DEV)
    Z=torch.zeros(1,3,device=DEV)
    def samp_2gen(B1,B2):
        def f():
            t1=(torch.rand(())*2-1); t2=(torch.rand(())*2-1); M=torch.matrix_exp(t1*torch.cat([B1,Z])+t2*torch.cat([B2,Z]))
            return torch.tensor([float(M[0,2]/PXN), float(M[1,2]/PXN)])
        return f
    res["DISCOVERED_2GEN_SE2"]={"gap": ed(apply_2gen(A1,A2), held_e), **support_of(samp_2gen(A1,A2))}
    res["RANDOM_2GEN_SE2"]={"gap": ed(apply_2gen(R1,R2), held_e), **support_of(samp_2gen(R1,R2))}
    # DISTRIBUTION discovered
    meanA, logsig = disc_dist()
    def samp_dist():
        dx=torch.exp(logsig[0])*torch.randn(()); dy=torch.exp(logsig[1])*torch.randn(())
        return torch.tensor([float(meanA[0,2]/PXN+dx/PXN), float(meanA[1,2]/PXN+dy/PXN)])
    res["DISCOVERED_DISTRIBUTION_SE2"]={"gap": ed(apply_dist(meanA,logsig), held_e),
                                        **support_of(samp_dist), "logsig":[float(logsig[0]),float(logsig[1])]}

    # ---- PRIMARY metric: coverage in TRANSLATION (tx,ty) space (we control it -> unconfounded by coarse
    # features / energy-distance floor). target = the useful crop region (annulus). An arm's value = how
    # well its INDUCED (tx,ty) distribution covers the target. Single point can't; a distribution can. ----
    def ed_2d(a, b): return float(2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean())
    def collect(sampler, n=256): return torch.stack([sampler() for _ in range(n)], 0)
    target_tt = sample_region(512, "held")                 # annulus = useful crop region to cover
    base_cover = ed_2d(sample_region(256, "visible"), target_tt)
    known_cover = ed_2d(sample_region(256, "held"), target_tt)   # oracle samples target -> ~0
    samplers = {"SINGLE_DISCOVERED_SE2": samp_single(Ms), "RANDOM_SINGLE_SE2": samp_single(Mr),
                "DISCOVERED_2GEN_SE2": samp_2gen(A1,A2), "RANDOM_2GEN_SE2": samp_2gen(R1,R2),
                "DISCOVERED_DISTRIBUTION_SE2": samp_dist}
    for k, sm in samplers.items():
        res[k]["cover_ed"] = ed_2d(collect(sm), target_tt)
    res["config"].update({"base_cover": base_cover, "known_cover": known_cover})

    order=list(samplers.keys())
    print(f"=== v14 Rung C — single vs distribution ===")
    print(f"COVERAGE in (tx,ty) space: base={base_cover:.3f}  known_crop_dist={known_cover:.3f}  (lower=covers target better)")
    print(f"{'arm':28s} {'cover':>6s} {'featgap':>7s} {'tx_s':>5s} {'ty_s':>5s} {'eig1':>5s} {'eig2':>5s} {'erat':>5s}")
    for k in order:
        r=res[k]; print(f"{k:28s} {r['cover_ed']:6.3f} {r['gap']:7.3f} {r['tx_std']:5.2f} {r['ty_std']:5.2f} {r['eig1']:5.1f} {r['eig2']:5.2f} {r['eig_ratio']:5.2f}")
    sd=res["SINGLE_DISCOVERED_SE2"]["cover_ed"]; dd=res["DISCOVERED_DISTRIBUTION_SE2"]; r2=res["RANDOM_2GEN_SE2"]["cover_ed"]
    known_ok = known_cover < base_cover - 1e-3
    beats = dd["cover_ed"] < sd and dd["cover_ed"] < r2 and dd["cover_ed"] < base_cover
    high_rank = dd["eig_ratio"] > 0.15 and dd["eig2"] > 0.5
    print(f"\nKNOWN_CROP_DIST<base: {known_ok} | DIST covers better than single & random2gen & base: {beats} | DIST high-rank 2D: {high_rank} (erat={dd['eig_ratio']:.2f})")
    ok = known_ok and beats and high_rank
    print("RUNG C:", "PASS — discovered distribution covers the 2D crop region better than single/random, w/ 2D support"
          if ok else "FAIL — distribution does not cover better / collapsed (interpret)")
    json.dump(res, open(a.out_json,"w"), indent=2); print(f"wrote {a.out_json}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
