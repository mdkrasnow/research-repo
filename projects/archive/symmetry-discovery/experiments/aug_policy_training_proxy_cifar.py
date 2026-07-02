"""v14 (beat-crop) Rung C — utility proxy: can a discovered POLICY beat known crop on robustness?

Train a small CNN on a CIFAR subset under each augmentation; evaluate on a BROAD translated validation
(U[-6,6]^2 — wider than crop's pad4 ±4, so a richer learned policy has room to beat the hand-coded crop).
The discovered policy adds a UTILITY term (make augs the current model finds hard, kept on-manifold by the
anchor + bounded + diverse by the entropy floor) — task-adaptive, unlike a fixed crop.

Arms: BASE, KNOWN_CROP (U[-4,4]^2), RANDOM_POLICY, DISCOVERED_ANCHOR_ONLY, DISCOVERED_ANCHOR_ENTROPY,
DISCOVERED_ANCHOR_ENTROPY_UTILITY. Discovery unsupervised except utility uses TRAINING labels (no held-out).

PASS: DISCOVERED_ANCHOR_ENTROPY_UTILITY beats KNOWN_CROP on broad translated-val acc AND beats RANDOM_POLICY.
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from _se2_discovery import discover_policy, policy_txty, warp_txty

DEV = torch.device("cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1); self.c2 = nn.Conv2d(32, 64, 3, 1, 1); self.fc = nn.Linear(64*8*8, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def make_aug(name, pol, rand_pol):
    def f(x):
        B = x.size(0)
        if name == "base": return x
        if name == "known_crop": return warp_txty(x, (torch.rand(B,2,device=x.device)*2-1)*4.0)
        if name == "random_policy": return warp_txty(x, policy_txty(*rand_pol, B, x.device))
        return warp_txty(x, policy_txty(*pol[name], B, x.device))
    return f


def evaluate(net, xv, yv, translate_range=0.0):
    net.eval()
    with torch.no_grad():
        x = xv if translate_range == 0 else warp_txty(xv, (torch.rand(xv.size(0),2)*2-1)*translate_range)
        acc = (net(x).argmax(1) == yv).float().mean().item()
    net.train(); return acc


def train_arm(xtr, ytr, xva, yva, aug, epochs, seed=0, bs=128):
    torch.manual_seed(seed); net = Net().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); n = xtr.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; loss = F.cross_entropy(net(aug(xtr[idx])), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(net, xva, yva, 0.0), evaluate(net, xva, yva, 6.0)   # clean, BROAD translated (±6px)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_aug_policy.json")
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    epochs = 3 if a.quick else 7; dsteps = 200 if a.quick else 450; n_seeds = 1 if a.quick else 3
    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                 if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    idx = torch.randperm(len(ds))[:3500].tolist()
    X = torch.stack([ds[i][0] for i in idx]); Y = torch.tensor([ds[i][1] for i in idx])
    xtr, ytr, xva, yva = X[:2500], Y[:2500], X[2500:], Y[2500:]

    print("training scorer (for utility term)...")
    scorer = Net().to(DEV); sopt = torch.optim.Adam(scorer.parameters(), lr=1e-3)
    for ep in range(4):
        perm = torch.randperm(2500)
        for i in range(0, 2500, 128):
            ii = perm[i:i+128]; l = F.cross_entropy(scorer(xtr[ii]), ytr[ii]); sopt.zero_grad(); l.backward(); sopt.step()
    for p in scorer.parameters(): p.requires_grad_(False)

    print("discovering policies (unsupervised; utility uses train labels only)...")
    pol = {}
    pol["DISCOVERED_ANCHOR_ONLY"] = discover_policy(xtr.clone(), steps=dsteps, entropy=False)
    pol["DISCOVERED_ANCHOR_ENTROPY"] = discover_policy(xtr.clone(), steps=dsteps, entropy=True)
    pol["DISCOVERED_ANCHOR_ENTROPY_UTILITY"] = discover_policy(xtr.clone(), steps=dsteps, entropy=True,
                                                               scorer=scorer, labels=ytr, lam_util=0.5)
    torch.manual_seed(99)
    rand_pol = (0.15*torch.randn(2,3), 0.15*torch.randn(2,3), torch.tensor([0.0,0.0]))
    for k,(A1,A2,ls) in pol.items():
        import math
        print(f"  {k}: logsig=({float(ls[0]):.2f},{float(ls[1]):.2f}) |A1_t|=({float(A1[0,2]/(2/32)):.1f},{float(A1[1,2]/(2/32)):.1f})px")

    arms = ["base","known_crop","random_policy","DISCOVERED_ANCHOR_ONLY","DISCOVERED_ANCHOR_ENTROPY","DISCOVERED_ANCHOR_ENTROPY_UTILITY"]
    res = {"n_seeds": n_seeds, "arms": {}}
    print(f"\n{'arm':36s} {'clean':>6s} {'transl±6':>8s} {'tstd':>6s}  (mean/{n_seeds} seeds)")
    for name in arms:
        aug = make_aug(name, pol, rand_pol)
        cs, ts = [], []
        for sd in range(n_seeds):
            c, t = train_arm(xtr, ytr, xva, yva, aug, epochs, seed=sd); cs.append(c); ts.append(t)
        clean = sum(cs)/len(cs); transl = sum(ts)/len(ts); tstd = (sum((t-transl)**2 for t in ts)/len(ts))**0.5
        res["arms"][name] = {"clean": clean, "translated6": transl, "translated_std": tstd, "seeds": ts}
        print(f"{name:36s} {clean:6.3f} {transl:8.3f} {tstd:6.3f}")

    A = res["arms"]; base=A["base"]["translated6"]; crop=A["known_crop"]["translated6"]
    util=A["DISCOVERED_ANCHOR_ENTROPY_UTILITY"]["translated6"]; rand=A["random_policy"]["translated6"]
    noise = max(A["base"]["translated_std"], A["known_crop"]["translated_std"], A["DISCOVERED_ANCHOR_ENTROPY_UTILITY"]["translated_std"])
    beats_crop = util > crop + noise; beats_rand = util > rand + 1e-4; crop_helps = crop > base + 1e-4
    print(f"\ncrop_helps:{crop_helps} (base {base:.3f}->crop {crop:.3f}) | util>crop+noise:{beats_crop} (util {util:.3f}, noise {noise:.3f}) | util>random:{beats_rand}")
    ok = crop_helps and beats_crop and beats_rand
    print("RUNG C:", "PASS — discovered anchor+entropy+utility policy BEATS known crop on broad robustness + beats random"
          if ok else "FAIL — discovered policy does not beat crop on broad robustness (interpret)")
    json.dump(res, open(a.out_json,"w"), indent=2); print(f"wrote {a.out_json}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
