"""v14 Rung D — augmentation-training proxy.

Does the unsupervised discovered translation DISTRIBUTION actually help a model become translation-robust,
and does it beat a single discovered operator + a random distribution and close the known-crop gap?

Train a small CNN on a CIFAR subset under each augmentation, evaluate on TRANSLATED validation.
Arms: BASE (no aug), KNOWN_CROP_DISTRIBUTION (per-img U[-4,4]^2 ~ crop pad4),
SINGLE_DISCOVERED_SE2 (one fixed discovered shift every step — v13-style, low diversity),
RANDOM_DISTRIBUTION_SE2 (per-img N(0, sd_rand^2), arbitrary spread),
DISCOVERED_DISTRIBUTION_SE2 (per-img N(discovered mean, std^2)).

Discovery is unsupervised (frozen-anchor mixture, no labels) via _se2_discovery.
PASS: DISCOVERED_DISTRIBUTION translated-val acc > SINGLE and > RANDOM_DISTRIBUTION, and closes a
meaningful fraction of the (KNOWN_CROP - BASE) robustness gap.
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from _se2_discovery import discover_single, discover_distribution, warp_txty

DEV = torch.device("cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1); self.c2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def aug_fn(name, single_txty, disc_mean, disc_std, sd_rand=4.0):
    def f(x):
        B = x.size(0)
        if name == "base":
            return x
        if name == "known_crop":
            tt = (torch.rand(B, 2, device=x.device) * 2 - 1) * 4.0          # U[-4,4]^2
        elif name == "single":
            tt = single_txty.to(x.device).unsqueeze(0).repeat(B, 1)          # one fixed shift
        elif name == "random_dist":
            tt = torch.randn(B, 2, device=x.device) * sd_rand
        else:  # disc_dist
            tt = disc_mean.to(x.device) + torch.randn(B, 2, device=x.device) * disc_std.to(x.device)
        return warp_txty(x, tt)
    return f


def evaluate(net, xv, yv, translate):
    net.eval()
    with torch.no_grad():
        x = xv
        if translate:
            tt = (torch.rand(xv.size(0), 2) * 2 - 1) * 4.0
            x = warp_txty(xv, tt)
        acc = (net(x).argmax(1) == yv).float().mean().item()
    net.train(); return acc


def train_arm(name, xtr, ytr, xva, yva, aug, epochs, bs=128, seed=0):
    torch.manual_seed(seed); net = Net().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = xtr.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; xb = aug(xtr[idx]); yb = ytr[idx]
            loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(net, xva, yva, False), evaluate(net, xva, yva, True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_aug_training.json")
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    epochs = 4 if a.quick else 8; disc_steps = 250 if a.quick else 500
    n_tr = 3000; n_va = 1000
    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                 if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    idx = torch.randperm(len(ds))[:n_tr+n_va].tolist()
    X = torch.stack([ds[i][0] for i in idx]); Y = torch.tensor([ds[i][1] for i in idx])
    xtr, ytr, xva, yva = X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]

    print("discovering operators (unsupervised, no labels)...")
    single = discover_single(xtr.clone(), steps=disc_steps)
    dmean, dstd = discover_distribution(xtr.clone(), steps=disc_steps)
    print(f"  single tx,ty = ({single[0]:.2f},{single[1]:.2f})px")
    print(f"  dist mean=({dmean[0]:.2f},{dmean[1]:.2f}) std=({dstd[0]:.2f},{dstd[1]:.2f})px")

    arms = ["base", "known_crop", "single", "random_dist", "disc_dist"]
    n_seeds = 1 if a.quick else 3
    res = {"single_txty": single.tolist(), "disc_mean": dmean.tolist(), "disc_std": dstd.tolist(),
           "n_seeds": n_seeds, "arms": {}}
    print(f"\n{'arm':14s} {'clean':>7s} {'transl':>7s} {'transl_std':>10s} {'robust_gap':>10s}  (mean over {n_seeds} seeds)")
    for name in arms:
        aug = aug_fn(name, single, dmean, dstd)
        cs, ts = [], []
        for sd in range(n_seeds):
            c, t = train_arm(name, xtr, ytr, xva, yva, aug, epochs, seed=sd)
            cs.append(c); ts.append(t)
        clean = sum(cs)/len(cs); transl = sum(ts)/len(ts)
        tstd = (sum((t-transl)**2 for t in ts)/len(ts))**0.5
        res["arms"][name] = {"clean": clean, "translated": transl, "translated_std": tstd,
                             "robust_gap": clean - transl, "translated_seeds": ts}
        print(f"{name:14s} {clean:7.3f} {transl:7.3f} {tstd:10.3f} {clean-transl:10.3f}")

    A = res["arms"]
    base_t = A["base"]["translated"]; known_t = A["known_crop"]["translated"]
    disc_t = A["disc_dist"]["translated"]; single_t = A["single"]["translated"]; rand_t = A["random_dist"]["translated"]
    gap = known_t - base_t
    closed = (disc_t - base_t) / gap if gap > 1e-4 else 0.0
    beats_single = disc_t > single_t; beats_rand = disc_t > rand_t
    known_helps = known_t > base_t + 0.005
    print(f"\nknown-crop robustness gap (known_t - base_t) = {gap:.3f}")
    print(f"disc closes {closed*100:.0f}% of gap | disc>single: {beats_single} | disc>random_dist: {beats_rand} | known>base: {known_helps}")
    ok = known_helps and beats_single and beats_rand and closed > 0.3
    print("RUNG D:", "PASS — discovered distribution improves translation robustness, beats single+random, closes gap"
          if ok else "FAIL — discovered distribution does not beat controls / closes <30% of gap (interpret)")
    json.dump(res, open(a.out_json, "w"), indent=2); print(f"wrote {a.out_json}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
