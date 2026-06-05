"""CIFAR-appropriate fast proxy: SE(2) homogeneous-affine operator (WITH translation) vs the rotation-only v12.

v12 killed only the rotation-2x2 architecture (no translation/crop) + used known ROTATION as a mismatched
positive control. CIFAR benefits from crop/translation, not rotation. Here:
  - operator = SE(2)-style 3x3 affine M=matrix_exp(A), A's top 2 rows learned + bottom row 0 -> M is affine
    on homogeneous coords, can express TRANSLATION (M[:2,2]) + scale/shear/rotation. Acts via grid_sample.
  - INJECTED GAP IS TRANSLATION (the CIFAR-relevant nuisance), not rotation: visible x-translations exclude
    a held band {+3,+4,+5}px; held-out = that band; anchor = full (unlabeled) translation distribution.
  - discovery uses the CORRECTED recipe (lesson from the v12 proxy bug): frozen GRAD-FLOWING encoder
    (anchor gradient must reach the operator), MIXTURE objective ((1-pi)P_vis + pi*T(P_vis) ~= anchor),
    broad NON-LEAKING move hinge (bounds magnitude, does NOT target the held +4px), stability reg.

Arms (energy-distance to held-out in a common PCA32(frozen-conv) space):
  BASE                    visible features as-is (floor)
  KNOWN_TRANSLATE_CROP    true +4px translation (POSITIVE control; must beat BASE -> harness valid)
  KNOWN_ROTATION          true 20deg rotation (WRONG transform for a translation gap; should NOT fill it)
  RANDOM_SE2              random stable 3x3 affine, move-matched (negative control)
  DISCOVERED_SE2          T-only objective, learned 3x3
  DISCOVERED_SE2_MIXTURE  mixture-anchor objective, learned 3x3 (the treatment)

Success: KNOWN_TRANSLATE_CROP < BASE; DISCOVERED_SE2_MIXTURE < BASE and < RANDOM_SE2; sane op diagnostics
(translation in band, det~1, cond bounded, non-identity, anchor improves). No held-out labels in discovery.
"""
import argparse, json, math, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_PCA = 32
PXN = 2.0 / 32.0                                  # 1px in affine_grid normalized coords (32px image)
HELD_TX = [3.0, 4.0, 5.0]                         # held-out x-translations (px) — one-sided gap
VISIBLE_TX = [float(t) for t in range(-6, 7) if t not in (3, 4, 5)]
N_IMG = 1500; SET_N = 1200
LAM_MOVE = 0.5; LAM_DET = 1.0; LAM_COND = 1.0; LAM_TRANS = 0.0


def trans_mat3(dx_px, dy_px, device):            # pure translation (homogeneous 3x3)
    M = torch.eye(3, device=device); M[0, 2] = dx_px * PXN; M[1, 2] = dy_px * PXN
    return M

def rot_mat3(deg, device):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    M = torch.eye(3, device=device); M[0, 0] = c; M[0, 1] = -s; M[1, 0] = s; M[1, 1] = c
    return M

def affine_warp3(x, M3):                          # apply 3x3 homogeneous affine via grid_sample
    B = x.size(0); theta = M3[:2, :].unsqueeze(0).repeat(B, 1, 1)
    grid = F.affine_grid(theta, list(x.size()), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


class FrozenConv(nn.Module):
    def __init__(self, seed=777):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU(),
                                 nn.Conv2d(64,64,3,2,1), nn.ReLU())
        with torch.no_grad():
            for p in self.net.parameters(): p.copy_(torch.randn(p.shape, generator=g)*(0.1 if p.dim()==1 else 0.3))
        for p in self.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x): return self.net(x).flatten(1)
    def feat_grad(self, x): return self.net(x).flatten(1)   # params frozen, grad FLOWS to input


def energy_distance(a, b):
    return float(2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean())

def ed_t(a, b):
    return 2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_cifar_se2_proxy.json")
    ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    steps = 300 if args.quick else 1500
    n_img = 400 if args.quick else N_IMG
    torch.manual_seed(0)
    print(f"device={DEV} k_pca={K_PCA} held_tx={HELD_TX}px visible_tx={VISIBLE_TX}")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    data_root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                      if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(data_root, train=True, download=False, transform=tf)
    sel = torch.randperm(len(ds))[:n_img].tolist()
    imgs = torch.stack([ds[i][0] for i in sel]).to(DEV)
    enc = FrozenConv().to(DEV)

    def feat(x): return enc(x)
    # eval pools: TRANSLATED views (x-translation)
    vis_imgs = torch.cat([affine_warp3(imgs, trans_mat3(t,0,DEV)) for t in VISIBLE_TX], 0)
    held_imgs = torch.cat([affine_warp3(imgs, trans_mat3(t,0,DEV)) for t in HELD_TX], 0)
    all_imgs = torch.cat([vis_imgs, held_imgs], 0)
    mu = feat(all_imgs).mean(0, keepdim=True)
    _,_,Vp = torch.pca_lowrank(feat(all_imgs)-mu, q=K_PCA); proj = Vp[:, :K_PCA]
    def P(f): return (f - mu) @ proj
    def Pg(x): return (enc.feat_grad(x) - mu) @ proj            # grad-flowing
    vis = P(feat(vis_imgs)); held = P(feat(held_imgs)); anchor = P(feat(all_imgs))
    known_tc = P(feat(affine_warp3(imgs, trans_mat3(4.0,0,DEV))))   # true +4px -> held band (POS control)
    known_rot = P(feat(affine_warp3(imgs, rot_mat3(20.0,DEV))))     # wrong transform

    def sub(t,n=SET_N): return t[torch.randperm(t.size(0))[:min(n,t.size(0))]]
    held_e=sub(held); anchor_e=sub(anchor); vis_e=sub(vis)
    base_gap=energy_distance(vis_e,held_e)
    ktc_gap=energy_distance(sub(known_tc),held_e); krot_gap=energy_distance(sub(known_rot),held_e)
    # NON-LEAKING move band: hinge [2px,10px] image move (held is +4px, inside band -> not handed the answer)
    with torch.no_grad():
        mfloor=(affine_warp3(imgs,trans_mat3(2,0,DEV))-imgs).flatten(1).norm(dim=1).mean()
        mcap  =(affine_warp3(imgs,trans_mat3(10,0,DEV))-imgs).flatten(1).norm(dim=1).mean()
    def move_pen(mv): return torch.relu((mfloor-mv)/mfloor)**2 + torch.relu((mv-mcap)/mfloor)**2

    results={"config":{"k_pca":K_PCA,"held_tx":HELD_TX,"n_img":n_img,"steps":steps,
                       "mfloor":float(mfloor),"mcap":float(mcap),"device":str(DEV)},
             "BASE":{"gap_heldout":base_gap},
             "KNOWN_TRANSLATE_CROP":{"gap_heldout":ktc_gap},
             "KNOWN_ROTATION":{"gap_heldout":krot_gap}}

    def lin(M3): return M3[:2,:2]
    def report(M3,name):
        with torch.no_grad():
            Tv=P(feat(affine_warp3(sub(vis_imgs),M3))); L=lin(M3); sv=torch.linalg.svdvals(L)
            return {"gap_heldout":energy_distance(Tv,held_e),"gap_anchor":energy_distance(Tv,anchor_e),
                    "gap_visible":energy_distance(Tv,vis_e),
                    "tx_px":float(M3[0,2]/PXN),"ty_px":float(M3[1,2]/PXN),
                    "det":float(torch.det(L)),"cond":float(sv.max()/sv.min().clamp_min(1e-6)),
                    "lin_off_id":float((L-torch.eye(2,device=DEV)).norm())}

    def build_M(A2):                                # A2: (2,3) -> full 3x3 affine generator -> matrix_exp
        A=torch.cat([A2, torch.zeros(1,3,device=DEV)],0); return torch.matrix_exp(A)

    # RANDOM_SE2
    torch.manual_seed(1); Ar=torch.zeros(2,3,device=DEV); Ar[:, :2]=0.15*torch.randn(2,2,device=DEV); Ar[:,2]=0.15*torch.randn(2,device=DEV)
    results["RANDOM_SE2"]=report(build_M(Ar),"rand")

    def discover(mixture):
        torch.manual_seed(2 if not mixture else 4)
        A2=torch.nn.Parameter(0.02*torch.randn(2,3,device=DEV)); opt=torch.optim.Adam([A2],lr=5e-3)
        m=256; pi=0.5; n_aug=int(pi*m); n_vis=m-n_aug
        for s in range(steps):
            bi=imgs[torch.randperm(imgs.size(0))[:128]]
            tv=VISIBLE_TX[torch.randint(len(VISIBLE_TX),(1,)).item()]
            xv=affine_warp3(bi, trans_mat3(tv,0,DEV))
            M=build_M(A2); Tx=affine_warp3(xv,M); L=lin(M)
            mv=(Tx-xv).flatten(1).norm(dim=1).mean(); sv=torch.linalg.svdvals(L); det=torch.det(L)
            reg=LAM_MOVE*move_pen(mv)+LAM_DET*(torch.log(det.abs()+1e-8))**2+LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1.0)**2
            if mixture:
                f_vis=Pg(xv).detach(); f_aug=Pg(Tx)
                iv=torch.randperm(f_vis.size(0))[:n_vis]; ia=torch.randperm(f_aug.size(0))[:n_aug]
                fmix=torch.cat([f_vis[iv],f_aug[ia]],0); loss=ed_t(fmix,sub(anchor,m))+reg
            else:
                loss=ed_t(Pg(Tx),sub(anchor,256))+reg
            opt.zero_grad(); loss.backward(); opt.step()
        return build_M(A2.detach())
    results["DISCOVERED_SE2"]=report(discover(False),"disc")
    results["DISCOVERED_SE2_MIXTURE"]=report(discover(True),"mix")

    order=["BASE","KNOWN_TRANSLATE_CROP","KNOWN_ROTATION","RANDOM_SE2","DISCOVERED_SE2","DISCOVERED_SE2_MIXTURE"]
    print(f"\nbase={base_gap:.3f} known_tc={ktc_gap:.3f} known_rot={krot_gap:.3f} (move band {float(mfloor):.1f}-{float(mcap):.1f})")
    print(f"{'arm':26s} {'gap_HO':>7s} {'gap_anch':>8s} {'gap_vis':>7s} {'tx_px':>6s} {'ty_px':>6s} {'det':>5s} {'cond':>6s} {'off':>5s}")
    print("-"*86)
    for a in order:
        r=results[a]
        print(f"{a:26s} {r.get('gap_heldout',0):7.3f} {r.get('gap_anchor',float('nan')):8.3f} "
              f"{r.get('gap_visible',float('nan')):7.3f} {r.get('tx_px',float('nan')):6.2f} {r.get('ty_px',float('nan')):6.2f} "
              f"{r.get('det',float('nan')):5.2f} {r.get('cond',float('nan')):6.2f} {r.get('lin_off_id',float('nan')):5.2f}")
    mix=results["DISCOVERED_SE2_MIXTURE"]["gap_heldout"]; dsc=results["DISCOVERED_SE2"]["gap_heldout"]
    rnd=results["RANDOM_SE2"]["gap_heldout"]
    gate_pos = ktc_gap < base_gap
    win = (mix < base_gap) and (mix < rnd)
    print(f"\nGATE KNOWN_TRANSLATE_CROP<base: {gate_pos} (pos control valid?) | KNOWN_ROTATION<base: {krot_gap<base_gap} (should be ~no)")
    print(f"DISCOVERED_SE2_MIXTURE<base: {mix<base_gap} | <random: {mix<rnd} | mixture<T-only: {mix<dsc}")
    print("VERDICT:", "PASS — SE2 mixture discovers useful translation (build v13)" if (gate_pos and win)
          else ("HARNESS INVALID (pos control failed)" if not gate_pos else "NEGATIVE (operator/objective insufficient)"))
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(results, open(args.out_json,"w"), indent=2); print(f"wrote {args.out_json}")

if __name__ == "__main__": main()
