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

GAUGE FIX (2026-06-04): a learned group generator A and -A define the SAME one-parameter group;
sign is a gauge. Old eval applied only M=exp(A) (one orientation) and called -23.7° "wrong direction"
vs a +15..25° held band -- but exp(-A)=+23.7° is free. So we now evaluate the discovered operator as a
GROUP/ORBIT (forward, inverse, bidir, exp(tA)), not a single directed map. Also the old anchor objective
T(P_vis) ~= P_anchor_full was mass-mismatched: augmentation makes a MIXTURE (1-pi)P_vis + pi T(P_vis),
not a replacement. New MIXTURE objective matches the mixture to the full anchor -> direction-sensitive on
a one-sided gap WITHOUT held-out labels.

Arms (all evaluated as energy-distance to held-out features in the common PCA32 space):
  BASE                         visible features as-is (floor)
  KNOWN_AUG                    true +20° rotation of base imgs (positive control; ~0 gap)
  PCA_LINEAR_DISCOVERED        OLD arch: dense M=exp(A) in PCA32 (negative-architecture baseline)
  SPATIAL_RANDOM               random spatial affine M=exp(A) (2x2), move-matched (negative control)
  SPATIAL_RANDOM_BIDIR         random A evaluated as orbit {M,M^-1} (negative control, gauge-fair)
  SPATIAL_DISCOVERED_FORWARD   T-only discovery, M=exp(A)        (= old SPATIAL_DISCOVERED)
  SPATIAL_DISCOVERED_INVERSE   T-only discovery, M^-1=exp(-A)    (gauge flip of the same group)
  SPATIAL_DISCOVERED_BIDIR     T-only discovery, orbit {M,M^-1}
  SPATIAL_DISCOVERED_ORBIT     T-only discovery, orbit exp(tA), t in {-1,-.5,.5,1}
  SPATIAL_DISCOVERED_RESIDUAL  T-only, anchor sampling weighted toward far-from-visible support
  MIXTURE_SINGLE               mixture-anchor objective, single direction M=exp(A)
  MIXTURE_BIDIR                mixture-anchor objective, orbit {M,M^-1}

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
    def feat_grad(self, x): return self.net(x).flatten(1)  # params frozen, grad FLOWS to input (for discovery)

def energy_distance(a, b):
    return float(2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean())

def ed_t(a, b):  # differentiable
    return 2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_spatial_proxy_v2.json")
    ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    steps = 300 if args.quick else 1500
    n_img = 400 if args.quick else N_IMG
    torch.manual_seed(0)
    print(f"device={DEV} k_pca={K_PCA} held={HELD}")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    data_root = next((d for d in ("data", "../../../data", os.path.expanduser("~/Desktop/research-repo/data"))
                      if os.path.isdir(os.path.join(d, "cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(data_root, train=True, download=False, transform=tf)
    sel = torch.randperm(len(ds))[:n_img].tolist()
    imgs = torch.stack([ds[i][0] for i in sel]).to(DEV)
    enc = FrozenConv().to(DEV)

    def feat_imgs(x): return enc(x)              # no-grad, for eval
    def Pg(x): return (enc.feat_grad(x) - mu) @ proj   # grad FLOWS to x, for discovery
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
    # NON-LEAKING move band: a broad hinge [5deg, 45deg] in pixel-distance, NOT centered on the held band
    # (held=[15,20,25]). The old code targeted a 20deg rotation == held-band center, which HANDED the
    # operator its magnitude; with the anchor term now grad-flowing, discovery must find angle+sign itself.
    with torch.no_grad():
        move_floor = (affine_warp(imgs, rot_mat(5.0, DEV)) - imgs).flatten(1).norm(dim=1).mean()
        move_cap   = (affine_warp(imgs, rot_mat(45.0, DEV)) - imgs).flatten(1).norm(dim=1).mean()
        img_move_target = (affine_warp(imgs, rot_mat(20.0, DEV)) - imgs).flatten(1).norm(dim=1).mean()  # report-only
        pca_move_target = (held.mean(0) - vis.mean(0)).norm()
    def move_pen(mv):  # zero inside [floor,cap], quadratic outside (scale-free); non-collapse + non-blowup
        return torch.relu((move_floor - mv)/move_floor)**2 + torch.relu((mv - move_cap)/move_floor)**2
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

    def eval_ops(Ms, name):
        # gauge-fair orbit eval: apply each op to a DISJOINT equal chunk of visible imgs (balanced
        # counts -> total stays SET_N, no double-count inflation), pool features, energy distance.
        with torch.no_grad():
            base = sub(vis_imgs); chunks = torch.chunk(base, len(Ms))
            Tx = torch.cat([affine_warp(c, M) for c, M in zip(chunks, Ms)], 0)
            ft = P(feat_imgs(Tx))
            return {"gap_heldout": energy_distance(ft, held_e), "gap_anchor": energy_distance(ft, anchor_e),
                    "gap_visible": energy_distance(ft, vis_e), "n_ops": len(Ms)}

    # SPATIAL_RANDOM
    torch.manual_seed(1); Ar = 0.3*torch.randn(2,2,device=DEV)
    results["SPATIAL_RANDOM"] = report_spatial(torch.matrix_exp(Ar), "rand")
    results["SPATIAL_RANDOM_BIDIR"] = eval_ops([torch.matrix_exp(Ar), torch.matrix_exp(-Ar)], "randb")

    # SPATIAL_DISCOVERED (+ residual variant)
    def discover_spatial(residual):
        torch.manual_seed(2); A = torch.nn.Parameter(0.05*torch.randn(2,2,device=DEV))
        opt = torch.optim.Adam([A], lr=5e-3)
        for s in range(steps):
            bi = imgs[torch.randperm(imgs.size(0))[:128]]
            va = VISIBLE[torch.randint(len(VISIBLE),(1,)).item()]
            xv = affine_warp(bi, rot_mat(va, DEV))
            M = torch.matrix_exp(A); Tx = affine_warp(xv, M)
            ft = Pg(Tx)   # GRAD-flowing features: anchor term now actually trains A
            if residual:
                aidx = torch.multinomial(res_w, 256, replacement=True); ab = anchor[aidx]
            else:
                ab = sub(anchor, 256)
            mv = (Tx - xv).flatten(1).norm(dim=1).mean()
            sv = torch.linalg.svdvals(M); det = torch.det(M)
            loss = ed_t(ft, ab) + LAM_MOVE*move_pen(mv) \
                   + LAM_DET*(torch.log(det.abs()+1e-8))**2 + LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1.0)**2
            opt.zero_grad(); loss.backward(); opt.step()
        return A.detach()

    # mixture-anchor objective: (1-pi)P_vis + pi*T(P_vis) ~= P_anchor_full.  Direction-sensitive on a
    # one-sided gap: only the sign that FILLS missing anchor support lowers the mixture's energy distance.
    def discover_mixture(bidir, pi=0.5):
        torch.manual_seed(4 if not bidir else 5)
        A = torch.nn.Parameter(0.05*torch.randn(2,2,device=DEV)); opt = torch.optim.Adam([A], lr=5e-3)
        m = 256; n_aug = int(pi*m); n_vis = m - n_aug
        for s in range(steps):
            bi = imgs[torch.randperm(imgs.size(0))[:192]]
            va = VISIBLE[torch.randint(len(VISIBLE),(1,)).item()]
            xv = affine_warp(bi, rot_mat(va, DEV))
            M = torch.matrix_exp(A)
            if bidir:
                Minv = torch.matrix_exp(-A); h = xv.size(0)//2
                Tx = torch.cat([affine_warp(xv[:h], M), affine_warp(xv[h:], Minv)], 0)
            else:
                Tx = affine_warp(xv, M)
            f_vis = P(feat_imgs(xv)).detach(); f_aug = Pg(Tx)   # f_aug GRAD-flowing
            iv = torch.randperm(f_vis.size(0))[:n_vis]; ia = torch.randperm(f_aug.size(0))[:n_aug]
            fmix = torch.cat([f_vis[iv], f_aug[ia]], 0); ab = sub(anchor, m)
            mv = (Tx - xv[:Tx.size(0)]).flatten(1).norm(dim=1).mean()
            sv = torch.linalg.svdvals(M); det = torch.det(M)
            loss = ed_t(fmix, ab) + LAM_MOVE*move_pen(mv) \
                   + LAM_DET*(torch.log(det.abs()+1e-8))**2 + LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1.0)**2
            opt.zero_grad(); loss.backward(); opt.step()
        return A.detach()

    A_disc = discover_spatial(False)
    M_d = torch.matrix_exp(A_disc); Minv_d = torch.matrix_exp(-A_disc)
    results["SPATIAL_DISCOVERED_FORWARD"] = report_spatial(M_d, "fwd")
    results["SPATIAL_DISCOVERED_INVERSE"] = report_spatial(Minv_d, "inv")
    results["SPATIAL_DISCOVERED_BIDIR"]   = eval_ops([M_d, Minv_d], "bidir")
    results["SPATIAL_DISCOVERED_ORBIT"]   = eval_ops([torch.matrix_exp(t*A_disc) for t in (-1.0,-0.5,0.5,1.0)], "orbit")
    results["SPATIAL_DISCOVERED_RESIDUAL"] = report_spatial(torch.matrix_exp(discover_spatial(True)), "disc_res")
    A_m = discover_mixture(False); results["MIXTURE_SINGLE"] = report_spatial(torch.matrix_exp(A_m), "mix1")
    A_mb = discover_mixture(True); results["MIXTURE_BIDIR"] = eval_ops([torch.matrix_exp(A_mb), torch.matrix_exp(-A_mb)], "mixb")

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

    order = ["BASE","KNOWN_AUG","PCA_LINEAR_DISCOVERED","SPATIAL_RANDOM","SPATIAL_RANDOM_BIDIR",
             "SPATIAL_DISCOVERED_FORWARD","SPATIAL_DISCOVERED_INVERSE","SPATIAL_DISCOVERED_BIDIR",
             "SPATIAL_DISCOVERED_ORBIT","SPATIAL_DISCOVERED_RESIDUAL","MIXTURE_SINGLE","MIXTURE_BIDIR"]
    print(f"\nbase_gap={base_gap:.3f} known_gap={known_gap:.3f} (img_move_target={img_move_target:.2f})")
    print(f"{'arm':30s} {'gap_HO':>7s} {'gap_anch':>8s} {'gap_vis':>7s} {'angle':>6s} {'det':>5s} {'cond':>6s} {'off_id':>6s}")
    print("-"*88)
    for a in order:
        r = results[a]
        print(f"{a:30s} {r.get('gap_heldout',0):7.3f} {r.get('gap_anchor',float('nan')):8.3f} "
              f"{r.get('gap_visible',float('nan')):7.3f} {r.get('angle_deg',float('nan')):6.1f} "
              f"{r.get('det',float('nan')):5.2f} {r.get('cond',float('nan')):6.2f} {r.get('off_identity',float('nan')):6.2f}")
    fwd = results["SPATIAL_DISCOVERED_FORWARD"]["gap_heldout"]; inv = results["SPATIAL_DISCOVERED_INVERSE"]["gap_heldout"]
    bid = results["SPATIAL_DISCOVERED_BIDIR"]["gap_heldout"]; orb = results["SPATIAL_DISCOVERED_ORBIT"]["gap_heldout"]
    pl = results["PCA_LINEAR_DISCOVERED"]["gap_heldout"]; sr = results["SPATIAL_RANDOM"]["gap_heldout"]
    srb = results["SPATIAL_RANDOM_BIDIR"]["gap_heldout"]
    m1 = results["MIXTURE_SINGLE"]["gap_heldout"]; mb = results["MIXTURE_BIDIR"]["gap_heldout"]
    print(f"\nGATE known<base: {known_gap<base_gap}")
    print(f"GAUGE TEST  fwd<base:{fwd<base_gap} inv<base:{inv<base_gap} bidir<base:{bid<base_gap} orbit<base:{orb<base_gap}")
    print(f"  -> inverse better than forward? {inv<fwd}  (if yes: old 'wrong direction' was a gauge artifact)")
    print(f"  -> best orbit arm vs random_bidir control: bidir<rand_bidir={bid<srb} orbit<rand_bidir={orb<srb}")
    print(f"MIXTURE     single<base:{m1<base_gap} (<random:{m1<sr}) | bidir<base:{mb<base_gap} (<rand_bidir:{mb<srb})")
    best_disc = min(fwd, inv, bid, orb, m1, mb)
    print(f"BEST discovered arm gap_HO={best_disc:.3f}  base={base_gap:.3f}  known={known_gap:.3f}")
    print("VERDICT:", "GROUP/MIXTURE FIX WORKS (discovered beats base+random control)"
          if (best_disc < base_gap and best_disc < srb) else
          "STILL NEGATIVE (architecture ok, discovery/objective insufficient even gauge-fair)")
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(results, open(args.out_json,"w"), indent=2); print(f"wrote {args.out_json}")

if __name__ == "__main__": main()
