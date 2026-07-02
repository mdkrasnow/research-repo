"""v15 (safe-adversarial aug) Rung 1 — KNOWN ceiling. Establish the true baseline-to-beat.

v14 asked "beat crop(pad4)". But if a wider crop / crop+scale / crop+flip already beats pad4, then
"beat crop" meant "beat a sub-optimal known aug" — wrong target. Sweep hand-coded policies, train a
small CNN under each, eval on a BROAD translated val (U[-6,6]^2). The winner = the real ceiling the
learned safe-adversarial policy (rungs 2-4) must clear.

Arms (all KNOWN, no learning): base, crop_pad2/4/6, transl_only6, transl_scale (±4px + scale U[0.9,1.1]),
transl_flip (±4px + hflip). 3 seeds each. Output: ranked table + which known aug is the ceiling.
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEV = torch.device("cpu")
PXN = 2.0 / 32.0


def warp_affine(x, txty, scale=None):
    """Per-image translation (px) + optional isotropic scale. txty:[B,2] px, scale:[B] or None."""
    B = x.size(0); th = torch.zeros(B, 2, 3, device=x.device)
    s = torch.ones(B, device=x.device) if scale is None else scale
    th[:, 0, 0] = s; th[:, 1, 1] = s
    th[:, 0, 2] = txty[:, 0] * PXN; th[:, 1, 2] = txty[:, 1] * PXN
    return F.grid_sample(x, F.affine_grid(th, list(x.size()), align_corners=False),
                         align_corners=False, padding_mode="reflection")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1); self.c2 = nn.Conv2d(32, 64, 3, 1, 1); self.fc = nn.Linear(64*8*8, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def make_aug(name):
    def f(x):
        B = x.size(0)
        if name == "base": return x
        if name == "crop_pad2": return warp_affine(x, (torch.rand(B,2,device=x.device)*2-1)*2.0)
        if name == "crop_pad4": return warp_affine(x, (torch.rand(B,2,device=x.device)*2-1)*4.0)
        if name == "crop_pad6": return warp_affine(x, (torch.rand(B,2,device=x.device)*2-1)*6.0)
        if name == "transl_only6": return warp_affine(x, (torch.rand(B,2,device=x.device)*2-1)*6.0)
        if name == "transl_scale":
            s = 0.9 + torch.rand(B, device=x.device)*0.2
            return warp_affine(x, (torch.rand(B,2,device=x.device)*2-1)*4.0, s)
        if name == "transl_flip":
            xt = warp_affine(x, (torch.rand(B,2,device=x.device)*2-1)*4.0)
            flip = torch.rand(B, device=x.device) < 0.5
            xt[flip] = torch.flip(xt[flip], dims=[3]); return xt
        raise ValueError(name)
    return f


def evaluate(net, xv, yv, rng=0.0):
    net.eval()
    with torch.no_grad():
        x = xv if rng == 0 else warp_affine(xv, (torch.rand(xv.size(0),2)*2-1)*rng)
        acc = (net(x).argmax(1) == yv).float().mean().item()
    net.train(); return acc


def train_arm(xtr, ytr, xva, yva, aug, epochs, seed=0, bs=128):
    torch.manual_seed(seed); net = Net().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); n = xtr.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; loss = F.cross_entropy(net(aug(xtr[idx])), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(net, xva, yva, 0.0), evaluate(net, xva, yva, 6.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_v15_rung1.json")
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

    arms = ["base","crop_pad2","crop_pad4","crop_pad6","transl_only6","transl_scale","transl_flip"]
    res = {"n_seeds": n_seeds, "arms": {}}
    print(f"{'arm':14s} {'clean':>6s} {'transl±6':>9s} {'tstd':>6s}  (mean/{n_seeds} seeds)")
    for name in arms:
        aug = make_aug(name); cs, ts = [], []
        for sd in range(n_seeds):
            c, t = train_arm(xtr, ytr, xva, yva, aug, epochs, seed=sd); cs.append(c); ts.append(t)
        clean = sum(cs)/len(cs); transl = sum(ts)/len(ts); tstd = (sum((t-transl)**2 for t in ts)/len(ts))**0.5
        res["arms"][name] = {"clean": clean, "translated6": transl, "translated_std": tstd, "seeds": ts}
        print(f"{name:14s} {clean:6.3f} {transl:9.3f} {tstd:6.3f}")

    A = res["arms"]
    ranked = sorted(arms, key=lambda k: -A[k]["translated6"])
    ceiling = ranked[0]; crop4 = "crop_pad4"
    margin = A[ceiling]["translated6"] - A[crop4]["translated6"]
    noise = max(A[ceiling]["translated_std"], A[crop4]["translated_std"])
    crop4_is_ceiling = (ceiling == crop4) or (margin <= noise)
    res["ranked"] = ranked; res["ceiling"] = ceiling; res["crop_pad4_is_ceiling"] = crop4_is_ceiling
    print(f"\nranked by transl±6: {ranked}")
    print(f"ceiling={ceiling} ({A[ceiling]['translated6']:.3f}); crop_pad4={A[crop4]['translated6']:.3f}; "
          f"margin={margin:.3f} noise~{noise:.3f}")
    if crop4_is_ceiling:
        print("RUNG 1: crop_pad4 IS (within noise) the known ceiling -> 'beat crop' target VALID. Proceed to Rung 2.")
    else:
        print(f"RUNG 1: REFRAME — {ceiling} BEATS crop_pad4 by {margin:.3f}>noise. The ceiling-to-beat is {ceiling}, "
              f"not crop_pad4. Learned policy must clear {ceiling}.")
    json.dump(res, open(a.out_json,"w"), indent=2); print(f"wrote {a.out_json}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
