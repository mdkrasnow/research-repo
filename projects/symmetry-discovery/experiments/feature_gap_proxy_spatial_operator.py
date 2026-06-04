"""Architecture-correct fast proxy: SPATIAL Lie-generator operator vs the flat-PCA baseline.

Prior proxy negative was an ARCHITECTURE failure: dense M=exp(A) on flattened random-conv PCA vectors
cannot represent an image rotation (coherent/simple in spatial coords, nonlinear after flatten+PCA).
Here the operator acts on IMAGE COORDINATES (spatial affine M=exp(A), 2x2, via grid_sample) — the right
prior for image transforms — discovered against a frozen feature anchor, evaluated in a COMMON frozen
feature PCA space so all arms are apples-to-apples.

Setup: real CIFAR; known transform = rotation; visible angles exclude held-out band {+15,+20,+25}°;
frozen random-conv encoder phi; common eval space = PCA32(phi(.)) fit on the full (anchor) distribution.
A +~20° affine maps several visible angles INTO the held band, so a coherent spatial operator CAN close
the gap; the question is whether anchor-discovery finds it.

Arms (all evaluated as energy-distance to held-out features in the common PCA32 space):
  BASE                      visible features as-is (floor)
  KNOWN_AUG                 true +20° rotation of base imgs (positive control; ~0 gap)
  PCA_LINEAR_DISCOVERED     OLD arch: dense M=exp(A) in PCA32 (negative-architecture baseline)
  SPATIAL_RANDOM            random spatial affine M=exp(A) (2x2), move-matched (negative control)
  SPATIAL_DISCOVERED        learned spatial affine M=exp(A) vs frozen anchor + stability + move (treatment)
  SPATIAL_DISCOVERED_RESIDUAL  same, anchor sampling weighted toward far-from-visible support (targets gap)

No live-EqM closure. No held-out labels in discovery. Diagnostics mandatory (det/cond/move/angle).
"""
import argparse, json, math, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_PCA = 32
HELD = [15.0, 20.0, 25.0]
VISIBLE = [float(a) for a in range(-30, 31, 5) if a not in (15, 20, 25)]
N_IMG = 1500; SET_N = 1200
LAM_MOVE = 0.5; LAM_DET = 1.0; LAM_COND = 1.0


def rot_mat(deg, device):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    return torch.tensor([[c, -s], [s, c]], device=device)

def affine_warp(x, M):
    B = x.size(0); theta = torch.zeros(B, 2, 3, device=x.device); theta[:, :2, :2] = M.unsqueeze(0)
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

def energy_distance(a, b):
    return float(2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean())

def ed_t(a, b):  # differentiable
    return 2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_spatial_proxy.json")
    ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    steps = 300 if args.quick else 1500
    n_img = 400 if args.quick else N_IMG
    torch.manual_seed(0)
    print(f"device={DEV} k_pca={K_PCA} held={HELD}")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    ds = datasets.CIFAR10("data", train=True, download=False, transform=tf)
    sel = torch.randperm(len(ds))[:n_img].tolist()
    imgs = torch.stack([ds[i][0] for i in sel]).to(DEV)
    enc = FrozenConv().to(DEV)

    def feat_imgs(x): return enc(x)
    # eval pools (raw features)
    vis_imgs = torch.cat([affine_warp(imgs, rot_mat(a, DEV)) for a in VISIBLE], 0)
    held_imgs = torch.cat([affine_warp(imgs, rot_mat(a, DEV)) for a in HELD], 0)
    all_imgs = torch.cat([vis_imgs, held_imgs], 0)
    vis_raw = feat_imgs(vis_imgs); held_raw = feat_imgs(held_imgs); all_raw = feat_imgs(all_imgs)
    mu = all_raw.mean(0, keepdim=True)
    _, _, Vp = torch.pca_lowrank(all_raw - mu, q=K_PCA); proj = Vp[:, :K_PCA]
    def P(f): return (f - mu) @ proj
    vis = P(vis_raw); held = P(held_raw); anchor = P(all_raw)
    known = P(feat_imgs(affine_warp(imgs, rot_mat(20.0, DEV))))   # true +20 -> held band

    def sub(t, n=SET_N): return t[torch.randperm(t.size(0))[:min(n, t.size(0))]]
    held_e = sub(held); anchor_e = sub(anchor); vis_e = sub(vis)
    base_gap = energy_distance(vis_e, held_e); known_gap = energy_distance(sub(known), held_e)
    # image-space move target = a ~20deg rotation change; pca-space target = centroid gap
    with torch.no_grad():
        img_move_target = (affine_warp(imgs, rot_mat(20.0, DEV)) - imgs).flatten(1).norm(dim=1).mean()
        pca_move_target = (held.mean(0) - vis.mean(0)).norm()
    # residual-anchor weights: anchor points far from visible distribution (unlabeled gap focus)
    with torch.no_grad():
        d2vis = torch.cdist(anchor, sub(vis, 600)).min(1).values
        res_w = (d2vis / d2vis.sum())

    results = {"config": {"k_pca": K_PCA, "held": HELD, "n_img": n_img, "steps": steps,
                          "img_move_target": float(img_move_target), "pca_move_target": float(pca_move_target),
                          "device": str(DEV)},
               "BASE": {"gap_heldout": base_gap}, "KNOWN_AUG": {"gap_heldout": known_gap}}

    def report_spatial(M, name):
        with torch.no_grad():
            Tv = P(feat_imgs(affine_warp(sub(vis_imgs), M)))
            sv = torch.linalg.svdvals(M); I = torch.eye(2, device=DEV)
            return {"gap_heldout": energy_distance(Tv, held_e), "gap_anchor": energy_distance(Tv, anchor_e),
                    "gap_visible": energy_distance(Tv, vis_e), "angle_deg": float(torch.atan2(M[1,0],M[0,0])*180/math.pi),
                    "det": float(torch.det(M)), "cond": float(sv.max()/sv.min().clamp_min(1e-6)),
                    "off_identity": float((M-I).norm())}

    # SPATIAL_RANDOM
    torch.manual_seed(1); Ar = 0.3*torch.randn(2,2,device=DEV)
    results["SPATIAL_RANDOM"] = report_spatial(torch.matrix_exp(Ar), "rand")

    # SPATIAL_DISCOVERED (+ residual variant)
    def discover_spatial(residual):
        torch.manual_seed(2); A = torch.nn.Parameter(0.05*torch.randn(2,2,device=DEV))
        opt = torch.optim.Adam([A], lr=5e-3)
        for s in range(steps):
            bi = imgs[torch.randperm(imgs.size(0))[:128]]
            va = VISIBLE[torch.randint(len(VISIBLE),(1,)).item()]
            xv = affine_warp(bi, rot_mat(va, DEV))
            M = torch.matrix_exp(A); Tx = affine_warp(xv, M)
            ft = P(feat_imgs(Tx))
            if residual:
                aidx = torch.multinomial(res_w, 256, replacement=True); ab = anchor[aidx]
            else:
                ab = sub(anchor, 256)
            mv = (Tx - xv).flatten(1).norm(dim=1).mean()
            sv = torch.linalg.svdvals(M); det = torch.det(M)
            loss = ed_t(ft, ab) + LAM_MOVE*(mv/img_move_target - 1.0)**2 \
                   + LAM_DET*(torch.log(det.abs()+1e-8))**2 + LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1.0)**2
            opt.zero_grad(); loss.backward(); opt.step()
        return torch.matrix_exp(A.detach())
    results["SPATIAL_DISCOVERED"] = report_spatial(discover_spatial(False), "disc")
    results["SPATIAL_DISCOVERED_RESIDUAL"] = report_spatial(discover_spatial(True), "disc_res")

    # PCA_LINEAR_DISCOVERED (old arch: dense M in PCA space)
    torch.manual_seed(3); Ad = torch.nn.Parameter(0.02*torch.randn(K_PCA,K_PCA,device=DEV))
    opt = torch.optim.Adam([Ad], lr=5e-3)
    for s in range(steps):
        vb = sub(vis,256); ab = sub(anchor,256); M = torch.matrix_exp(Ad); Tv = vb@M.T
        sv = torch.linalg.svdvals(M); det = torch.det(M); mv=(vb@M.T-vb).norm(dim=1).mean()
        loss = ed_t(Tv,ab)+LAM_MOVE*(mv/pca_move_target-1.0)**2+LAM_DET*(torch.log(det.abs()+1e-8))**2+LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1.0)**2
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Mp = torch.matrix_exp(Ad.detach()); Tv = sub(vis)@Mp.T; svp=torch.linalg.svdvals(Mp)
        results["PCA_LINEAR_DISCOVERED"] = {"gap_heldout": energy_distance(Tv, held_e),
            "gap_anchor": energy_distance(Tv, anchor_e), "det": float(torch.det(Mp)),
            "cond": float(svp.max()/svp.min().clamp_min(1e-6)), "off_identity": float((Mp-torch.eye(K_PCA,device=DEV)).norm())}

    order = ["BASE","KNOWN_AUG","PCA_LINEAR_DISCOVERED","SPATIAL_RANDOM","SPATIAL_DISCOVERED","SPATIAL_DISCOVERED_RESIDUAL"]
    print(f"\nbase_gap={base_gap:.3f} known_gap={known_gap:.3f} (img_move_target={img_move_target:.2f})")
    print(f"{'arm':30s} {'gap_HO':>7s} {'gap_anch':>8s} {'gap_vis':>7s} {'angle':>6s} {'det':>5s} {'cond':>6s} {'off_id':>6s}")
    print("-"*88)
    for a in order:
        r = results[a]
        print(f"{a:30s} {r.get('gap_heldout',0):7.3f} {r.get('gap_anchor',float('nan')):8.3f} "
              f"{r.get('gap_visible',float('nan')):7.3f} {r.get('angle_deg',float('nan')):6.1f} "
              f"{r.get('det',float('nan')):5.2f} {r.get('cond',float('nan')):6.2f} {r.get('off_identity',float('nan')):6.2f}")
    sd = results["SPATIAL_DISCOVERED"]["gap_heldout"]; pl = results["PCA_LINEAR_DISCOVERED"]["gap_heldout"]
    sr = results["SPATIAL_RANDOM"]["gap_heldout"]; rr = results["SPATIAL_DISCOVERED_RESIDUAL"]["gap_heldout"]
    print(f"\nGATE known<base: {known_gap<base_gap} | SPATIAL_DISC<base: {sd<base_gap} | <random: {sd<sr} | <PCA_linear: {sd<pl}")
    print(f"residual<base: {rr<base_gap} | residual<spatial_disc: {rr<sd}")
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(results, open(args.out_json,"w"), indent=2); print(f"wrote {args.out_json}")

if __name__ == "__main__": main()
