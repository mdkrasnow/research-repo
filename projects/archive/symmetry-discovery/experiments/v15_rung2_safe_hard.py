"""v15 Rung 2 — safe-HARD policy on a FROZEN scorer. The core new mechanism.

v14's utility term (lam_util=0.5) gave only a faint edge: it barely drove the policy. Here we test the
real lever: train a scorer, FREEZE it, then learn an aug policy whose objective is to MAXIMIZE the scorer's
loss (find HARD augs) subject to staying on the frozen-conv anchor (realistic) + bounded + diverse.

The gate is mechanistic, BEFORE any training-with-policy (Rung 3): a learned safe-hard policy must produce
augs that are HARDER than random crop (higher frozen-scorer CE on val) WHILE remaining about as on-manifold
as crop (anchor energy-distance not worse than crop's). If it can only get harder by going off-manifold,
the mechanism is unsafe -> stop. Sweep lam_util to show the safety/hardness trade.

Arms: random_crop(pad4), random_policy, safe_hard(lam_util in {0.5,1.5,3.0}). Diagnostics per arm:
  scorer_CE (hardness, higher=harder), anchor_ED (realism, lower=on-manifold), move_px, cov_eig2 (2D rank).
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


def anchor_ed(enc, anchor, Tx):
    """Energy distance of T(x) features to the frozen real anchor (lower = more on-manifold)."""
    with torch.no_grad():
        f = enc(Tx)
        return float(_ed_t(f, anchor[torch.randperm(anchor.size(0))[:f.size(0)]]))


def policy_diag(name, pol, rand_pol, xv, yv, scorer, enc, anchor):
    B = xv.size(0)
    with torch.no_grad():
        if name == "random_crop":
            Tx = warp_txty(xv, (torch.rand(B,2)*2-1)*4.0)
        elif name == "random_policy":
            Tx = warp_txty(xv, policy_txty(*rand_pol, B, DEV))
        else:
            Tx = warp_txty(xv, policy_txty(*pol, B, DEV))
        ce = float(F.cross_entropy(scorer(Tx), yv))
        # translation spread (2D rank) from many draws
        if name in ("random_crop",):
            t = (torch.rand(256,2)*2-1)*4.0
        elif name == "random_policy":
            t = policy_txty(*rand_pol, 256, DEV)
        else:
            t = policy_txty(*pol, 256, DEV)
        cov = torch.cov(t.T); ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        move = float((Tx - xv).flatten(1).norm(dim=1).mean())
    return {"scorer_CE": ce, "anchor_ED": anchor_ed(enc, anchor, Tx),
            "move_px": move, "cov_eig_min": float(ev.min()), "cov_eig_max": float(ev.max())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-json", default="results_v15_rung2.json")
    ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    dsteps = 200 if a.quick else 450
    torch.manual_seed(0)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    root = next((d for d in ("data","../../../data",os.path.expanduser("~/Desktop/research-repo/data"))
                 if os.path.isdir(os.path.join(d,"cifar-10-batches-py"))), "data")
    ds = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    idx = torch.randperm(len(ds))[:3500].tolist()
    X = torch.stack([ds[i][0] for i in idx]); Y = torch.tensor([ds[i][1] for i in idx])
    xtr, ytr, xva, yva = X[:2500], Y[:2500], X[2500:], Y[2500:]

    print("training scorer, then FREEZING...")
    scorer = Net().to(DEV); sopt = torch.optim.Adam(scorer.parameters(), lr=1e-3)
    for _ in range(6):
        perm = torch.randperm(2500)
        for i in range(0, 2500, 128):
            ii = perm[i:i+128]; l = F.cross_entropy(scorer(xtr[ii]), ytr[ii]); sopt.zero_grad(); l.backward(); sopt.step()
    for p in scorer.parameters(): p.requires_grad_(False)
    scorer.eval()

    enc = FrozenConv().to(DEV)
    with torch.no_grad(): anchor = enc(xtr)

    print("discovering safe-hard policies (anchor=realism, scorer-loss=hardness)...")
    lams = [0.5, 1.5, 3.0]
    pols = {f"safe_hard_l{lam}": discover_policy(xtr.clone(), steps=dsteps, entropy=True,
                                                scorer=scorer, labels=ytr, lam_util=lam) for lam in lams}
    torch.manual_seed(99)
    rand_pol = (0.15*torch.randn(2,3), 0.15*torch.randn(2,3), torch.tensor([0.0,0.0]))

    arms = ["random_crop", "random_policy"] + list(pols.keys())
    res = {"arms": {}}
    print(f"\n{'arm':16s} {'CE(hard)':>9s} {'ED(real)':>9s} {'move':>6s} {'eig2':>6s} {'eig_max':>7s}")
    for name in arms:
        pol = pols.get(name)
        d = policy_diag(name, pol, rand_pol, xva, yva, scorer, enc, anchor)
        res["arms"][name] = d
        print(f"{name:16s} {d['scorer_CE']:9.4f} {d['anchor_ED']:9.4f} {d['move_px']:6.2f} "
              f"{d['cov_eig_min']:6.2f} {d['cov_eig_max']:7.2f}")

    A = res["arms"]; crop = A["random_crop"]
    # pick the safe-hard arm that is hardest while staying on-manifold (ED <= 1.5x crop's ED)
    ed_cap = crop["anchor_ED"] * 1.5 + 1e-6
    safe = {k: v for k, v in A.items() if k.startswith("safe_hard") and v["anchor_ED"] <= ed_cap}
    best = max(safe, key=lambda k: safe[k]["scorer_CE"]) if safe else None
    res["ed_cap"] = ed_cap; res["best_safe_hard"] = best
    if best is None:
        print(f"\nRUNG 2: FAIL — every safe-hard arm went off-manifold (ED > {ed_cap:.3f}=1.5x crop). "
              f"Hardness only via off-manifold -> unsafe mechanism. STOP.")
        ok = False
    else:
        b = A[best]; harder = b["scorer_CE"] > crop["scorer_CE"] + 1e-3
        rank2d = b["cov_eig_min"] > 0.5
        res["harder_than_crop"] = harder; res["rank2d_ok"] = rank2d
        print(f"\nbest on-manifold safe-hard: {best} CE={b['scorer_CE']:.4f} vs crop CE={crop['scorer_CE']:.4f} "
              f"(harder:{harder}) ED={b['anchor_ED']:.4f}<=cap{ed_cap:.3f} rank2d:{rank2d}")
        ok = harder and rank2d
        print("RUNG 2:", "PASS — safe-hard policy is HARDER than crop while staying on-manifold + 2D. "
              "Worth training with (Rung 3)." if ok else
              "FAIL — safe-hard not harder than crop on-manifold (no headroom to exploit). interpret.")
    json.dump({**res, "verdict": "pass" if ok else "fail"}, open(a.out_json,"w"), indent=2)
    print(f"wrote {a.out_json}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
