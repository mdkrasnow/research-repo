"""v15 Rung 3 — TRAIN a model with the safe-hard policy; does it beat crop on real robustness?

Rung 2 shows the safe-hard policy produces harder-but-on-manifold augs. Rung 3 is the payoff test:
train a small CNN under each augmentation and measure broad translated-val robustness. The discovered
safe-hard policy must BEAT the Rung-1 known ceiling (crop_pad4 / transl_scale) AND beat random, beyond
3-seed noise. If it only ties (like v14), the global policy has no headroom -> escalate to Rung 4
(conditional). Tie != beat (v14 lesson).

Arms: base, crop_pad4, transl_scale (Rung-1 ceiling), random_policy, safe_hard (best lam from Rung 2).
Eval: clean acc + broad translated±6 acc, 3 seeds.
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from _se2_discovery import discover_policy, policy_txty, warp_txty

DEV = torch.device("cpu")
PXN = 2.0 / 32.0


def warp_scale(x, txty, scale):
    B = x.size(0); th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = scale; th[:, 1, 1] = scale
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


def make_aug(name, pol, rand_pol):
    def f(x):
        B = x.size(0)
        if name == "base": return x
        if name == "crop_pad4": return warp_txty(x, (torch.rand(B,2,device=x.device)*2-1)*4.0)
        if name == "crop_pad6": return warp_txty(x, (torch.rand(B,2,device=x.device)*2-1)*6.0)
        if name == "transl_scale":
            s = 0.9 + torch.rand(B, device=x.device)*0.2
            return warp_scale(x, (torch.rand(B,2,device=x.device)*2-1)*4.0, s)
        if name == "random_policy": return warp_txty(x, policy_txty(*rand_pol, B, x.device))
        return warp_txty(x, policy_txty(*pol, B, x.device))
    return f


def evaluate(net, xv, yv, rng=0.0):
    net.eval()
    with torch.no_grad():
        x = xv if rng == 0 else warp_txty(xv, (torch.rand(xv.size(0),2)*2-1)*rng)
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
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_v15_rung3.json")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--lam", type=float, default=1.5)
    a = ap.parse_args()
    epochs = 3 if a.quick else 8; dsteps = 200 if a.quick else 450; n_seeds = 1 if a.quick else 3
    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                 if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    idx = torch.randperm(len(ds))[:3500].tolist()
    X = torch.stack([ds[i][0] for i in idx]); Y = torch.tensor([ds[i][1] for i in idx])
    xtr, ytr, xva, yva = X[:2500], Y[:2500], X[2500:], Y[2500:]

    print("training scorer (frozen) for safe-hard utility...")
    scorer = Net().to(DEV); sopt = torch.optim.Adam(scorer.parameters(), lr=1e-3)
    for _ in range(6):
        perm = torch.randperm(2500)
        for i in range(0, 2500, 128):
            ii = perm[i:i+128]; l = F.cross_entropy(scorer(xtr[ii]), ytr[ii]); sopt.zero_grad(); l.backward(); sopt.step()
    for p in scorer.parameters(): p.requires_grad_(False)

    print(f"discovering safe-hard policy (lam_util={a.lam})...")
    pol = discover_policy(xtr.clone(), steps=dsteps, entropy=True, scorer=scorer, labels=ytr, lam_util=a.lam)
    torch.manual_seed(99); rand_pol = (0.15*torch.randn(2,3), 0.15*torch.randn(2,3), torch.tensor([0.0,0.0]))

    arms = ["base","crop_pad4","crop_pad6","transl_scale","random_policy","safe_hard"]
    res = {"n_seeds": n_seeds, "lam": a.lam, "arms": {}}
    print(f"\n{'arm':14s} {'clean':>6s} {'transl±6':>9s} {'tstd':>6s}")
    for name in arms:
        aug = make_aug(name, pol, rand_pol); cs, ts = [], []
        for sd in range(n_seeds):
            c, t = train_arm(xtr, ytr, xva, yva, aug, epochs, seed=sd); cs.append(c); ts.append(t)
        clean = sum(cs)/len(cs); transl = sum(ts)/len(ts); tstd = (sum((t-transl)**2 for t in ts)/len(ts))**0.5
        res["arms"][name] = {"clean": clean, "translated6": transl, "translated_std": tstd, "seeds": ts}
        print(f"{name:14s} {clean:6.3f} {transl:9.3f} {tstd:6.3f}")

    A = res["arms"]
    ceiling = max(("crop_pad4","crop_pad6","transl_scale"), key=lambda k: A[k]["translated6"])
    sh = A["safe_hard"]["translated6"]; cl = A[ceiling]["translated6"]; rp = A["random_policy"]["translated6"]
    noise = max(A["safe_hard"]["translated_std"], A[ceiling]["translated_std"])
    beats_ceiling = sh > cl + noise; beats_random = sh > rp + 1e-4
    res["ceiling_arm"] = ceiling; res["beats_ceiling"] = beats_ceiling; res["beats_random"] = beats_random
    print(f"\nknown ceiling={ceiling} ({cl:.3f}); safe_hard={sh:.3f} (beats ceiling+noise[{noise:.3f}]:{beats_ceiling}) "
          f"beats_random:{beats_random}")
    ok = beats_ceiling and beats_random
    if ok:
        print("RUNG 3: PASS — global safe-hard policy BEATS the known ceiling + random. Proceed to Rung 5 (EqM-lite).")
    else:
        print("RUNG 3: NO BEAT — global safe-hard ties/loses to ceiling. Escalate to Rung 4 (conditional policy) "
              "OR stop if Rung 4 also ties (v14 lesson: tie != beat).")
    json.dump({**res, "verdict": "pass" if ok else "no_beat"}, open(a.out_json,"w"), indent=2)
    print(f"wrote {a.out_json}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
