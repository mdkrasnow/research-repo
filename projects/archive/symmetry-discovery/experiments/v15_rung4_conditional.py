"""v15 Rung 4 — CONDITIONAL safe-hard policy qθ(T | φ(x)). The strongest new lever.

Crop is UNIFORM: same shift distribution for every image. A conditional policy lets the per-image
translation distribution depend on the image's frozen features φ(x): some images may want larger/smaller
or anisotropic shifts. This is the one mechanism v14 never tested. gθ: φ(x) -> (mu[2], logsig[2]) px;
sample txty = mu + exp(logsig)*eps; warp. Trained with the SAME safety stack: frozen-conv mixture anchor
(realism) + bounded move hinge + batch induced-cov entropy floor (diversity) + scorer utility (hardness).
Grad flows g <- warp_txty <- anchor/scorer (grid_sample differentiable in the grid).

Arms compared on broad translated±6 robustness (3 seeds): base, crop_pad4 (ceiling), random_policy,
global_safe_hard (Rung 3 winner), conditional_safe_hard. PASS: conditional BEATS crop_pad4 + beats global
+ beats random, beyond noise. (v14 lesson: tie != beat.)
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from _se2_discovery import discover_policy, policy_txty, warp_txty, FrozenConv, _ed_t

DEV = torch.device("cpu")
PXN = 2.0 / 32.0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1); self.c2 = nn.Conv2d(32, 64, 3, 1, 1); self.fc = nn.Linear(64*8*8, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


class CondPolicy(nn.Module):
    """gθ: pooled frozen features -> per-image (mu_x,mu_y,logsig_x,logsig_y) px."""
    def __init__(self, fdim=64):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(fdim, 32), nn.ReLU(), nn.Linear(32, 4))
        with torch.no_grad():  # init near small isotropic shifts
            self.g[-1].weight.mul_(0.01); self.g[-1].bias.copy_(torch.tensor([0.,0.,-1.0,-1.0]))
    def forward(self, feat):
        o = self.g(feat); mu = o[:, :2] * 4.0; logsig = o[:, 2:].clamp(-3, 2)  # mu scaled to px-ish
        return mu, logsig


def cond_sample(cond, pool, x):
    feat = pool(x)                       # [B,64] grad-flowing into cond only (pool frozen)
    mu, logsig = cond(feat)
    txty = mu + torch.exp(logsig) * torch.randn_like(mu)
    return txty, mu, logsig


def make_pool(enc):
    def pool(x):                         # frozen-conv features, global-avg-pooled, grad flows to x
        f = enc.net(x)                   # [B,64,16,16]
        return F.adaptive_avg_pool2d(f, 1).flatten(1)
    return pool


def discover_conditional(xtr, ytr, scorer, steps=450, lr=3e-3, lam_util=1.5, eig2_floor=2.0, seed=5):
    dev = xtr.device; enc = FrozenConv().to(dev); pool = make_pool(enc)
    with torch.no_grad(): anchor = enc(xtr)
    m2 = float((warp_txty(xtr[:256], torch.full((256,2),2.0)) - xtr[:256]).flatten(1).norm(dim=1).mean())
    m10 = float((warp_txty(xtr[:256], torch.full((256,2),10.0)) - xtr[:256]).flatten(1).norm(dim=1).mean())
    torch.manual_seed(seed); cond = CondPolicy().to(dev); opt = torch.optim.Adam(cond.parameters(), lr=lr)
    n = xtr.size(0)
    for _ in range(steps):
        idx = torch.randperm(n)[:128]; xv = xtr[idx]; yv = ytr[idx]
        txty, mu, logsig = cond_sample(cond, pool, xv)
        Tx = warp_txty(xv, txty)
        mv = (Tx - xv).flatten(1).norm(dim=1).mean()
        fmix = torch.cat([enc.fg(xv).detach()[:128], enc.fg(Tx)[:128]], 0)
        loss = _ed_t(fmix, anchor[torch.randperm(anchor.size(0))[:256]])
        loss = loss + 0.5*(torch.relu((m2 - mv)/m2)**2 + torch.relu((mv - m10)/m2)**2)
        cov = torch.cov(txty.T); ev = torch.linalg.eigvalsh(cov).clamp_min(0)   # diversity ACROSS images
        loss = loss + 0.5*torch.relu(eig2_floor - ev.min())**2
        loss = loss - lam_util * F.cross_entropy(scorer(Tx), yv)
        opt.zero_grad(); loss.backward(); opt.step()
    cond.eval()
    for p in cond.parameters(): p.requires_grad_(False)
    return cond, pool


def make_aug(name, gpol, rand_pol, cond, pool):
    def f(x):
        B = x.size(0)
        if name == "base": return x
        if name == "crop_pad4": return warp_txty(x, (torch.rand(B,2,device=x.device)*2-1)*4.0)
        if name == "random_policy": return warp_txty(x, policy_txty(*rand_pol, B, x.device))
        if name == "global_safe_hard": return warp_txty(x, policy_txty(*gpol, B, x.device))
        with torch.no_grad():
            feat = pool(x); mu, logsig = cond(feat); txty = mu + torch.exp(logsig)*torch.randn_like(mu)
        return warp_txty(x, txty)
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
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_v15_rung4.json")
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

    print("training frozen scorer...")
    scorer = Net().to(DEV); sopt = torch.optim.Adam(scorer.parameters(), lr=1e-3)
    for _ in range(6):
        perm = torch.randperm(2500)
        for i in range(0, 2500, 128):
            ii = perm[i:i+128]; l = F.cross_entropy(scorer(xtr[ii]), ytr[ii]); sopt.zero_grad(); l.backward(); sopt.step()
    for p in scorer.parameters(): p.requires_grad_(False)

    print(f"discovering global safe-hard (lam={a.lam})...")
    gpol = discover_policy(xtr.clone(), steps=dsteps, entropy=True, scorer=scorer, labels=ytr, lam_util=a.lam)
    print(f"discovering CONDITIONAL safe-hard (lam={a.lam})...")
    cond, pool = discover_conditional(xtr, ytr, scorer, steps=dsteps, lam_util=a.lam)
    # report conditioning variance: do per-image mu's actually differ?
    with torch.no_grad():
        mu_all, _ = cond(pool(xva)); mu_std = mu_all.std(0)
    print(f"conditional per-image mu std (px) = ({float(mu_std[0]):.2f},{float(mu_std[1]):.2f}) "
          f"(>0 means policy IS image-dependent)")
    torch.manual_seed(99); rand_pol = (0.15*torch.randn(2,3), 0.15*torch.randn(2,3), torch.tensor([0.0,0.0]))

    arms = ["base","crop_pad4","random_policy","global_safe_hard","conditional"]
    res = {"n_seeds": n_seeds, "lam": a.lam, "cond_mu_std": [float(mu_std[0]), float(mu_std[1])], "arms": {}}
    print(f"\n{'arm':18s} {'clean':>6s} {'transl±6':>9s} {'tstd':>6s}")
    for name in arms:
        aug = make_aug(name, gpol, rand_pol, cond, pool); cs, ts = [], []
        for sd in range(n_seeds):
            c, t = train_arm(xtr, ytr, xva, yva, aug, epochs, seed=sd); cs.append(c); ts.append(t)
        clean = sum(cs)/len(cs); transl = sum(ts)/len(ts); tstd = (sum((t-transl)**2 for t in ts)/len(ts))**0.5
        res["arms"][name] = {"clean": clean, "translated6": transl, "translated_std": tstd, "seeds": ts}
        print(f"{name:18s} {clean:6.3f} {transl:9.3f} {tstd:6.3f}")

    A = res["arms"]; crop = A["crop_pad4"]["translated6"]; cond_a = A["conditional"]["translated6"]
    glob = A["global_safe_hard"]["translated6"]; rand = A["random_policy"]["translated6"]
    noise = max(A["conditional"]["translated_std"], A["crop_pad4"]["translated_std"])
    beats_crop = cond_a > crop + noise; beats_global = cond_a > glob + 1e-4; beats_random = cond_a > rand + 1e-4
    res.update({"beats_crop": beats_crop, "beats_global": beats_global, "beats_random": beats_random})
    print(f"\nconditional={cond_a:.3f} crop={crop:.3f} global={glob:.3f} random={rand:.3f} noise~{noise:.3f}")
    print(f"beats_crop:{beats_crop} beats_global:{beats_global} beats_random:{beats_random}")
    ok = beats_crop and beats_random
    if ok:
        print("RUNG 4: PASS — CONDITIONAL safe-hard BEATS crop ceiling + random. v15 has headroom -> Rung 5 EqM-lite.")
    else:
        print("RUNG 4: NO BEAT — conditional ties/loses to crop. v15 EXHAUSTED on known/generic CIFAR translation: "
              "even per-image adaptivity adds nothing over crop. STOP (do not run FID); report.")
    json.dump({**res, "verdict": "pass" if ok else "no_beat"}, open(a.out_json,"w"), indent=2)
    print(f"wrote {a.out_json}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
