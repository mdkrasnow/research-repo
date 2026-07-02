"""v14 (beat-crop) Rung B — policy support / diversity.

Learn an augmentation POLICY qθ over a 2-generator SE(2) Lie basis (M = exp(z1*A1+z2*A2), z~N(0,diag(σ^2)))
against a frozen grad-flowing anchor. Test whether it has HIGH-RANK 2D support, beats a single op and a
random policy, stays bounded/on-manifold — and whether an ENTROPY floor is needed for the spread.

CIFAR + frozen fine random-conv -> PCA. Target = wide 2D annulus (no single point covers it). Coverage
measured in TRANSLATION (tx,ty) space (clean, vs the feature-ED noise floor seen earlier).

Arms:
  BASE                              visible near-center
  KNOWN_CROP                        oracle: sample the target region (positive control)
  RANDOM_POLICY                     random 2-gen + random coeff std
  SINGLE_DISCOVERED_SE2             one discovered op (v13)
  DISCOVERED_POLICY_ANCHOR_ONLY     policy, anchor+move+stability, NO entropy term
  DISCOVERED_POLICY_ANCHOR_ENTROPY  policy + entropy floor (cannot collapse to a line)

PASS: discovered policy (either) high-rank 2D support, covers better than single & random & base, bounded.
"""
import argparse, json, math, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEV = torch.device("cpu"); PXN = 2.0/32.0; K_PCA = 32; N_IMG = 300
Z = torch.zeros(1, 3, device=DEV)


def warp(x, M):
    th = M[:2, :].unsqueeze(0).repeat(x.size(0), 1, 1)
    return F.grid_sample(x, F.affine_grid(th, list(x.size()), align_corners=False), align_corners=False, padding_mode="reflection")
def warp_txty(x, tt):
    B=x.size(0); th=torch.zeros(B,2,3,device=DEV); th[:,0,0]=1; th[:,1,1]=1; th[:,0,2]=tt[:,0]*PXN; th[:,1,2]=tt[:,1]*PXN
    return F.grid_sample(x, F.affine_grid(th, list(x.size()), align_corners=False), align_corners=False, padding_mode="reflection")
def trans_M(dx,dy):
    M=torch.eye(3,device=DEV); M[0,2]=dx*PXN; M[1,2]=dy*PXN; return M

class FrozenConv(nn.Module):
    def __init__(self, seed=777):
        super().__init__(); g=torch.Generator().manual_seed(seed)
        self.net=nn.Sequential(nn.Conv2d(3,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,1,1),nn.ReLU(),nn.Conv2d(64,64,3,1,1),nn.ReLU())
        with torch.no_grad():
            for p in self.net.parameters(): p.copy_(torch.randn(p.shape,generator=g)*(0.1 if p.dim()==1 else 0.3))
        for p in self.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def forward(self,x): return self.net(x).flatten(1)
    def fg(self,x): return self.net(x).flatten(1)

def ed(a,b): return float(2*torch.cdist(a,b).mean()-torch.cdist(a,a).mean()-torch.cdist(b,b).mean())
def ed_t(a,b): return 2*torch.cdist(a,b).mean()-torch.cdist(a,a).mean()-torch.cdist(b,b).mean()

def region(n, kind):
    if kind=="held":  # annulus r in [2.5,5]
        r=2.5+2.5*torch.rand(n,device=DEV); th=2*math.pi*torch.rand(n,device=DEV)
        return torch.stack([r*torch.cos(th), r*torch.sin(th)],1)
    r=1.5*torch.rand(n,device=DEV).sqrt(); th=2*math.pi*torch.rand(n,device=DEV)  # visible disk
    return torch.stack([r*torch.cos(th), r*torch.sin(th)],1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-json",default="results_cifar_policy.json")
    ap.add_argument("--quick",action="store_true"); a=ap.parse_args()
    steps=250 if a.quick else 600
    torch.manual_seed(0)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root=next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data")) if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))),"data")
    ds=datasets.CIFAR10(root,train=True,download=False,transform=tf)
    imgs=torch.stack([ds[i][0] for i in torch.randperm(len(ds))[:N_IMG].tolist()]).to(DEV)
    enc=FrozenConv().to(DEV)
    def poolf(kind,passes=3): return torch.cat([enc(warp_txty(imgs,region(imgs.size(0),kind))) for _ in range(passes)],0)
    anchor_raw=torch.cat([poolf("visible"),poolf("held")],0); mu=anchor_raw.mean(0,keepdim=True)
    _,_,Vp=torch.pca_lowrank(anchor_raw-mu,q=K_PCA); proj=Vp[:,:K_PCA]
    def P(x):
        with torch.no_grad(): return (enc(x)-mu)@proj
    def Pg(x): return (enc.fg(x)-mu)@proj
    anchor=(anchor_raw-mu)@proj
    def sub(t,n=800): return t[torch.randperm(t.size(0))[:min(n,t.size(0))]]
    with torch.no_grad():
        m2=(warp(imgs,trans_M(2,0))-imgs).flatten(1).norm(dim=1).mean(); m10=(warp(imgs,trans_M(10,0))-imgs).flatten(1).norm(dim=1).mean()
    def move_pen(mv): return torch.relu((m2-mv)/m2)**2+torch.relu((mv-m10)/m2)**2

    target_tt=region(512,"held"); base_cover=ed(region(256,"visible"),target_tt); known_cover=ed(region(256,"held"),target_tt)

    def learn_policy(entropy, steps=steps, seed=3):
        torch.manual_seed(seed)
        A1=torch.nn.Parameter(0.02*torch.randn(2,3,device=DEV)); A2=torch.nn.Parameter(0.02*torch.randn(2,3,device=DEV))
        logsig=torch.nn.Parameter(torch.tensor([-0.7,-0.7],device=DEV))
        opt=torch.optim.Adam([A1,A2,logsig],lr=5e-3)
        eig2_floor=2.0  # px^2: smaller eigenvalue of induced (tx,ty) covariance must exceed this -> 2D support
        for s in range(steps):
            xv=warp_txty(imgs[torch.randperm(imgs.size(0))[:128]], region(128,"visible"))
            z=torch.randn(2,device=DEV)*torch.exp(logsig)
            M=torch.matrix_exp(z[0]*torch.cat([A1,Z])+z[1]*torch.cat([A2,Z])); Tx=warp(xv,M); L=M[:2,:2]
            mv=(Tx-xv).flatten(1).norm(dim=1).mean(); sv=torch.linalg.svdvals(L); det=torch.det(L)
            fmix=torch.cat([Pg(xv).detach()[:128],Pg(Tx)[:128]],0)
            loss=ed_t(fmix,sub(anchor,256))+0.5*move_pen(mv)+(torch.log(det.abs()+1e-8))**2+(sv.max()/sv.min().clamp_min(1e-6)-1)**2
            if entropy:
                # induced (tx,ty) 2D covariance floor: penalize anisotropic/low-rank support (the real diversity)
                K=24; tt=[]
                for _ in range(K):
                    zz=torch.randn(2,device=DEV)*torch.exp(logsig)
                    Mk=torch.matrix_exp(zz[0]*torch.cat([A1,Z])+zz[1]*torch.cat([A2,Z]))
                    tt.append(torch.stack([Mk[0,2]/PXN, Mk[1,2]/PXN]))
                T=torch.stack(tt,0); c=torch.cov(T.T); ev=torch.linalg.eigvalsh(c).clamp_min(0)
                loss=loss+0.5*torch.relu(eig2_floor-ev.min())**2
            opt.zero_grad(); loss.backward(); opt.step()
        return A1.detach(),A2.detach(),logsig.detach()

    def samp_policy(A1,A2,logsig):
        def f():
            z=torch.randn(2)*torch.exp(logsig); M=torch.matrix_exp(z[0]*torch.cat([A1,Z])+z[1]*torch.cat([A2,Z]))
            return torch.tensor([float(M[0,2]/PXN),float(M[1,2]/PXN)])
        return f
    def samp_single(M): return lambda: torch.tensor([float(M[0,2]/PXN),float(M[1,2]/PXN)])

    def support(sampler,n=128):
        pts=torch.stack([sampler() for _ in range(n)],0); c=torch.cov(pts.T) if n>1 else torch.zeros(2,2)
        ev=torch.linalg.eigvalsh(c).clamp_min(0)
        return {"tx_std":float(pts[:,0].std()),"ty_std":float(pts[:,1].std()),"eig1":float(ev.max()),
                "eig2":float(ev.min()),"eig_ratio":float(ev.min()/ev.max().clamp_min(1e-9)),
                "cover_ed":ed(pts,target_tt),"max_abs_tx":float(pts.abs().max())}

    # single discovered (reuse policy with tiny entropy -> ~single). Simpler: discover one op via 1-gen.
    torch.manual_seed(2); As=torch.nn.Parameter(0.02*torch.randn(2,3,device=DEV)); opt=torch.optim.Adam([As],lr=5e-3)
    for s in range(steps):
        xv=warp_txty(imgs[torch.randperm(imgs.size(0))[:128]],region(128,"visible"))
        M=torch.matrix_exp(torch.cat([As,Z])); Tx=warp(xv,M); L=M[:2,:2]
        mv=(Tx-xv).flatten(1).norm(dim=1).mean(); sv=torch.linalg.svdvals(L); det=torch.det(L)
        fmix=torch.cat([Pg(xv).detach()[:128],Pg(Tx)[:128]],0)
        loss=ed_t(fmix,sub(anchor,256))+0.5*move_pen(mv)+(torch.log(det.abs()+1e-8))**2+(sv.max()/sv.min().clamp_min(1e-6)-1)**2
        opt.zero_grad(); loss.backward(); opt.step()
    Ms=torch.matrix_exp(torch.cat([As.detach(),Z]))

    res={"config":{"steps":steps,"base_cover":base_cover,"known_cover":known_cover}}
    res["SINGLE_DISCOVERED_SE2"]=support(samp_single(Ms))
    torch.manual_seed(12); Rp=(0.15*torch.randn(2,3,device=DEV),0.15*torch.randn(2,3,device=DEV),torch.tensor([0.0,0.0],device=DEV))
    res["RANDOM_POLICY"]=support(samp_policy(*Rp))
    res["DISCOVERED_POLICY_ANCHOR_ONLY"]=support(samp_policy(*learn_policy(False)))
    res["DISCOVERED_POLICY_ANCHOR_ENTROPY"]=support(samp_policy(*learn_policy(True)))

    order=["SINGLE_DISCOVERED_SE2","RANDOM_POLICY","DISCOVERED_POLICY_ANCHOR_ONLY","DISCOVERED_POLICY_ANCHOR_ENTROPY"]
    print("=== v14 (beat-crop) Rung B — policy support ===")
    print(f"coverage(tx,ty): base={base_cover:.3f} known_crop={known_cover:.3f} (lower=better)")
    print(f"{'arm':34s} {'cover':>6s} {'tx_s':>5s} {'ty_s':>5s} {'eig1':>5s} {'eig2':>5s} {'erat':>5s} {'maxtx':>6s}")
    for k in order:
        r=res[k]; print(f"{k:34s} {r['cover_ed']:6.3f} {r['tx_std']:5.2f} {r['ty_std']:5.2f} {r['eig1']:5.1f} {r['eig2']:5.2f} {r['eig_ratio']:5.2f} {r['max_abs_tx']:6.1f}")
    ao=res["DISCOVERED_POLICY_ANCHOR_ONLY"]; ae=res["DISCOVERED_POLICY_ANCHOR_ENTROPY"]
    sd=res["SINGLE_DISCOVERED_SE2"]["cover_ed"]; rp=res["RANDOM_POLICY"]["cover_ed"]
    best=min(ao["cover_ed"],ae["cover_ed"]); best_arm=ae if ae["cover_ed"]<=ao["cover_ed"] else ao
    known_ok=known_cover<base_cover-1e-3
    beats=best<sd and best<rp and best<base_cover
    high_rank=best_arm["eig_ratio"]>0.15 and best_arm["eig2"]>0.5
    bounded=best_arm["max_abs_tx"]<12.0
    print(f"\nknown<base:{known_ok} | policy covers<single&random&base:{beats} | high-rank 2D:{high_rank} | bounded:{bounded}")
    ok=known_ok and beats and high_rank and bounded
    print("RUNG B:", "PASS — discovered policy has high-rank 2D support, beats single+random, bounded"
          if ok else "FAIL — policy low-rank/doesn't beat controls/unbounded (interpret)")
    json.dump(res,open(a.out_json,"w"),indent=2); print(f"wrote {a.out_json}")
    return ok


if __name__=="__main__":
    raise SystemExit(0 if main() else 1)
