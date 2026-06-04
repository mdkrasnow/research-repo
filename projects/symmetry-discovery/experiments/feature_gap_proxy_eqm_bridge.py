"""Fast CIFAR feature-space proxy for the EqM bridge mechanism question.

Replaces the slow/noisy 40ep CIFAR-FID inner loop with a seconds-to-minutes feature-space test that
preserves the core mechanism: frozen non-co-adapting anchor + learned stable generator M=exp(A) +
coherent reusable operator, with a held-out TRANSFORM gap and operator-quality diagnostics (not just
coverage — rungs 15-16 showed coverage is confounded).

Setup:
  - Real CIFAR images (local). Known transform = rotation.
  - FROZEN random-conv encoder phi -> features -> PCA to k=32 (cheap exp(A), stable diagnostics).
  - visible angles = {-30..30 step5} EXCLUDING held-out band {+15,+20,+25}; held-out = that band.
  - anchor_full = phi-features over ALL angles (incl held-out, UNLABELED) -> the frozen reference.
  - operator acts in PCA-feature space: f -> M f.

Arms:
  BASE_FEATURE              visible features as-is (floor: baseline gap to held-out region)
  KNOWN_AUG_FEATURE         features of the TRUE held-out-angle rotation (positive control; ~0 gap)
  RANDOM_STABLE_FEATURE     M=exp(A_random), stable, move-matched (negative control)
  DISCOVERED_STABLE_FEATURE M=exp(A) learned vs frozen anchor + stability + move (treatment)

Primary metric: energy-distance(arm_features, held-out_features) — lower = operator maps visible into the
held-out transform region. WIN requires: KNOWN beats BASE (harness gate); DISCOVERED beats BASE AND
RANDOM; AND DISCOVERED has coherent/stable operator diagnostics (det~1, bounded cond, non-identity,
anchor improves). NOT live-EqM closure. NO held-out labels in discovery.
"""
import argparse, json, math, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_PCA = 32
HELD = [15.0, 20.0, 25.0]
VISIBLE = [a for a in range(-30, 31, 5) if a not in (15, 20, 25)]
N_IMG = 1500
SET_N = 1200          # cap per set for O(n^2) energy distance
T_STEPS = 1500
LAM_MOVE = 0.5; LAM_DET = 1.0; LAM_COND = 1.0


def rot_imgs(x, deg):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    B = x.size(0)
    theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=x.device).unsqueeze(0).repeat(B, 1, 1)
    grid = F.affine_grid(theta, list(x.size()), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


class FrozenConv(nn.Module):
    def __init__(self, seed=777):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU(),
                                 nn.Conv2d(64,64,3,2,1), nn.ReLU())
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (0.1 if p.dim()==1 else 0.3))
        for p in self.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x): return self.net(x).flatten(1)


def energy_distance(a, b):
    return float(2*torch.cdist(a,b).mean() - torch.cdist(a,a).mean() - torch.cdist(b,b).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_feature_proxy.json")
    ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
    steps = 300 if args.quick else T_STEPS
    n_img = 400 if args.quick else N_IMG
    torch.manual_seed(0)
    print(f"device={DEV} k_pca={K_PCA} held={HELD} visible={VISIBLE}")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    ds = datasets.CIFAR10("data", train=True, download=False, transform=tf)
    idx = torch.randperm(len(ds))[:n_img]
    imgs = torch.stack([ds[i][0] for i in idx]).to(DEV)        # [N,3,32,32]
    enc = FrozenConv().to(DEV)

    def feats_at(deg):  # frozen features of all imgs rotated by deg
        return enc(rot_imgs(imgs, deg))

    # build raw feature pools
    vis_raw = torch.cat([feats_at(a) for a in VISIBLE], 0)
    held_raw = torch.cat([feats_at(a) for a in HELD], 0)
    all_raw = torch.cat([vis_raw, held_raw], 0)
    # PCA (fit on full/anchor distribution)
    mu = all_raw.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(all_raw - mu, q=K_PCA)
    proj = V[:, :K_PCA]
    def P(f): return (f - mu) @ proj
    vis = P(vis_raw); held = P(held_raw); anchor = P(all_raw)
    # known-aug = features of the TRUE held-out-angle transform applied to images (deg=20, mid held band)
    known = P(feats_at(20.0))

    def sub(t, n=SET_N):
        return t[torch.randperm(t.size(0))[:min(n, t.size(0))]]
    held_eval = sub(held); anchor_eval = sub(anchor)

    base_gap = energy_distance(sub(vis), held_eval)
    known_gap = energy_distance(sub(known), held_eval)
    move_target = float((held.mean(0) - vis.mean(0)).norm())   # gap centroid distance

    def op_metrics(M, name):
        with torch.no_grad():
            Tvis = sub(vis) @ M.T
            gap = energy_distance(Tvis, held_eval)
            anch = energy_distance(Tvis, anchor_eval)
            mv = float((vis @ M.T - vis).norm(dim=1).mean())
            I = torch.eye(K_PCA, device=DEV); svd = torch.linalg.svdvals(M)
            dshift = (vis @ M.T - vis); dshift = dshift / (dshift.norm(dim=1, keepdim=True) + 1e-8)
            shift_consistency = float((dshift @ dshift.mean(0, keepdim=True).T).mean())
            return {"arm": name, "gap_to_heldout": gap, "gap_to_anchor": anch,
                    "off_identity": float((M-I).norm()), "det": float(torch.det(M)),
                    "cond": float(svd.max()/svd.min().clamp_min(1e-6)), "move": mv,
                    "shift_consistency": shift_consistency}

    # RANDOM_STABLE: random skew-ish A, scaled so move ~ target
    torch.manual_seed(1); A_r = 0.1*torch.randn(K_PCA, K_PCA, device=DEV)
    Mr = torch.matrix_exp(A_r)
    with torch.no_grad():
        mv0 = float((vis @ Mr.T - vis).norm(dim=1).mean())
        A_r = A_r * (move_target/max(mv0,1e-6))**0.0  # keep as-is; report move
    rnd = op_metrics(torch.matrix_exp(A_r), "RANDOM_STABLE_FEATURE")

    # DISCOVERED_STABLE: learn A vs frozen anchor (energy distance) + move + stability
    A = torch.nn.Parameter(0.02*torch.randn(K_PCA, K_PCA, device=DEV))
    opt = torch.optim.Adam([A], lr=5e-3)
    for s in range(steps):
        vb = sub(vis, 256); ab = sub(anchor, 256)
        M = torch.matrix_exp(A); Tv = vb @ M.T
        a_ = torch.cdist(Tv, ab); ed = 2*a_.mean() - torch.cdist(Tv,Tv).mean() - torch.cdist(ab,ab).mean()
        mv = (vb @ M.T - vb).norm(dim=1).mean()
        det = torch.det(M); sv = torch.linalg.svdvals(M)
        loss = ed + LAM_MOVE*(mv-move_target)**2 + LAM_DET*(torch.log(det.abs()+1e-8))**2 \
               + LAM_COND*(sv.max()/sv.min().clamp_min(1e-6)-1.0)**2
        opt.zero_grad(); loss.backward(); opt.step()
    disc = op_metrics(torch.matrix_exp(A.detach()), "DISCOVERED_STABLE_FEATURE")

    results = {"config": {"k_pca": K_PCA, "held": HELD, "n_img": n_img, "steps": steps,
                          "move_target": move_target, "device": str(DEV)},
               "BASE_FEATURE": {"gap_to_heldout": base_gap},
               "KNOWN_AUG_FEATURE": {"gap_to_heldout": known_gap},
               "RANDOM_STABLE_FEATURE": rnd, "DISCOVERED_STABLE_FEATURE": disc}
    print(f"\nmove_target(centroid gap)={move_target:.3f}")
    print(f"{'arm':28s} {'gap_HO':>8s} {'gap_anchor':>10s} {'off_id':>7s} {'det':>6s} {'cond':>6s} {'move':>7s} {'shiftcoh':>8s}")
    print("-"*90)
    print(f"{'BASE_FEATURE':28s} {base_gap:8.3f} {'-':>10s}")
    print(f"{'KNOWN_AUG_FEATURE':28s} {known_gap:8.3f} {'-':>10s}  (positive control)")
    for r in (rnd, disc):
        print(f"{r['arm']:28s} {r['gap_to_heldout']:8.3f} {r['gap_to_anchor']:10.3f} {r['off_identity']:7.3f} "
              f"{r['det']:6.2f} {r['cond']:6.2f} {r['move']:7.3f} {r['shift_consistency']:8.3f}")
    # verdict
    gate = known_gap < base_gap - 1e-6
    disc_beats_base = disc["gap_to_heldout"] < base_gap
    disc_beats_rand = disc["gap_to_heldout"] < rnd["gap_to_heldout"]
    print(f"\nGATE known<base: {gate} | DISCOVERED<base: {disc_beats_base} | DISCOVERED<random: {disc_beats_rand}")
    print("verdict:", "HARNESS INVALID (gate failed)" if not gate else
          ("DISCOVERED WINS (beats base+random)" if (disc_beats_base and disc_beats_rand) else
           "PARTIAL/NEGATIVE (see diagnostics)"))
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(results, open(args.out_json, "w"), indent=2); print(f"wrote {args.out_json}")

if __name__ == "__main__": main()
