"""v16 Exp 1 — KNOWN augmentation ceiling. Establish BEST_KNOWN (the residual-policy base to beat).

v15 confirmed crop_pad4 is the translation ceiling. v16 asks a different question: does STACKING standard
mild augs (color / cutout / scale) on top of crop_pad4 raise the ceiling? Whichever wins on a realistic
mixed-corruption validation (mild translate+scale+brightness — a generalization test, not only translation)
becomes BEST_KNOWN. If none beat crop_pad4 beyond noise, BEST_KNOWN = crop_pad4.

Arms: base, crop_pad4, crop_pad4_color, crop_pad4_cutout, crop_pad4_scale, crop_pad4_color_cutout.
3 seeds, fixed budget. Reports clean acc + robust acc (mixed corruption). BEST_KNOWN = argmax robust acc.
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEV = torch.device("cpu")
PXN = 2.0 / 32.0


def warp(x, txty, scale=None):
    B = x.size(0); th = torch.zeros(B, 2, 3, device=x.device)
    s = torch.ones(B, device=x.device) if scale is None else scale
    th[:, 0, 0] = s; th[:, 1, 1] = s
    th[:, 0, 2] = txty[:, 0] * PXN; th[:, 1, 2] = txty[:, 1] * PXN
    return F.grid_sample(x, F.affine_grid(th, list(x.size()), align_corners=False),
                         align_corners=False, padding_mode="reflection")


def color_jitter(x, b=0.15, c=0.15):
    B = x.size(0)
    bri = (torch.rand(B,1,1,1,device=x.device)*2-1)*b
    con = 1 + (torch.rand(B,1,1,1,device=x.device)*2-1)*c
    mu = x.mean(dim=(1,2,3), keepdim=True)
    return ((x - mu)*con + mu + bri).clamp(-1, 1)


def cutout(x, k=8):
    B, _, H, W = x.shape; out = x.clone()
    cy = torch.randint(0, H, (B,)).tolist(); cx = torch.randint(0, W, (B,)).tolist()
    for i in range(B):
        y0 = max(0, cy[i]-k//2); y1 = min(H, cy[i]+k//2); x0 = max(0, cx[i]-k//2); x1 = min(W, cx[i]+k//2)
        out[i, :, y0:y1, x0:x1] = 0.0
    return out


def make_aug(name):
    def crop(x): return warp(x, (torch.rand(x.size(0),2,device=x.device)*2-1)*4.0)
    def f(x):
        if name == "base": return x
        if name == "crop_pad4": return crop(x)
        if name == "crop_pad4_color": return color_jitter(crop(x))
        if name == "crop_pad4_cutout": return cutout(crop(x))
        if name == "crop_pad4_scale":
            s = 0.9 + torch.rand(x.size(0),device=x.device)*0.2
            return warp(x, (torch.rand(x.size(0),2,device=x.device)*2-1)*4.0, s)
        if name == "crop_pad4_color_cutout": return cutout(color_jitter(crop(x)))
        raise ValueError(name)
    return f


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1); self.c2 = nn.Conv2d(32, 64, 3, 1, 1); self.fc = nn.Linear(64*8*8, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def robust_corrupt(x):
    """Realistic mixed test-time shift: mild translate±4 + scale 0.9-1.1 + brightness/contrast."""
    B = x.size(0); s = 0.9 + torch.rand(B, device=x.device)*0.2
    xc = warp(x, (torch.rand(B,2,device=x.device)*2-1)*4.0, s)
    return color_jitter(xc, b=0.1, c=0.1)


def evaluate(net, xv, yv, mode):
    net.eval()
    with torch.no_grad():
        if mode == "clean": acc = (net(xv).argmax(1) == yv).float().mean().item()
        else:
            accs = [ (net(robust_corrupt(xv)).argmax(1) == yv).float().mean().item() for _ in range(4) ]
            acc = sum(accs)/len(accs)
    net.train(); return acc


def train_arm(xtr, ytr, xva, yva, aug, epochs, seed=0, bs=128):
    torch.manual_seed(seed); net = Net().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); n = xtr.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; loss = F.cross_entropy(net(aug(xtr[idx])), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(net, xva, yva, "clean"), evaluate(net, xva, yva, "robust")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_v16_exp1.json")
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    epochs = 3 if a.quick else 8; n_seeds = 1 if a.quick else 3
    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                 if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    idx = torch.randperm(len(ds))[:3500].tolist()
    X = torch.stack([ds[i][0] for i in idx]); Y = torch.tensor([ds[i][1] for i in idx])
    xtr, ytr, xva, yva = X[:2500], Y[:2500], X[2500:], Y[2500:]

    arms = ["base","crop_pad4","crop_pad4_color","crop_pad4_cutout","crop_pad4_scale","crop_pad4_color_cutout"]
    res = {"n_seeds": n_seeds, "arms": {}}
    print(f"{'arm':24s} {'clean':>6s} {'robust':>7s} {'rstd':>6s}  (mean/{n_seeds} seeds)")
    for name in arms:
        aug = make_aug(name); cs, rs = [], []
        for sd in range(n_seeds):
            c, r = train_arm(xtr, ytr, xva, yva, aug, epochs, seed=sd); cs.append(c); rs.append(r)
        clean = sum(cs)/len(cs); rob = sum(rs)/len(rs); rstd = (sum((r-rob)**2 for r in rs)/len(rs))**0.5
        res["arms"][name] = {"clean": clean, "robust": rob, "robust_std": rstd, "seeds": rs}
        print(f"{name:24s} {clean:6.3f} {rob:7.3f} {rstd:6.3f}")

    A = res["arms"]; ranked = sorted(arms, key=lambda k: -A[k]["robust"])
    top = ranked[0]; crop = "crop_pad4"
    margin = A[top]["robust"] - A[crop]["robust"]; noise = max(A[top]["robust_std"], A[crop]["robust_std"])
    best_known = top if (top == crop or margin > noise) else crop
    res["ranked"] = ranked; res["best_known"] = best_known; res["best_known_robust"] = A[best_known]["robust"]
    print(f"\nranked by robust: {ranked}")
    print(f"top={top} ({A[top]['robust']:.3f}) vs crop_pad4 ({A[crop]['robust']:.3f}) margin={margin:.3f} noise~{noise:.3f}")
    print(f"BEST_KNOWN = {best_known} (robust {A[best_known]['robust']:.3f}) -> the base for Exp 2 residual policy.")
    json.dump(res, open(a.out_json,"w"), indent=2); print(f"wrote {a.out_json}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
