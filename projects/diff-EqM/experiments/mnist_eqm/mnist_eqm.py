"""MNIST inpainting rung — Step A: train a small unconditional EqM + a classifier oracle.

Reuses real MNIST (torchvision) + a standard inpainting protocol (RePaint-style clamp).
The EqM is the generator; a tiny CNN classifier is the inpainting ORACLE (does the
inpainted digit still read as the true label?). Both small, CPU, ~minutes.

Faithful EqM target (same c(t)=min(1,5(1-t))*4 as the maze/image EqM). Unconditional
field f(xt, t); RePaint inpainting (Step B) clamps observed pixels each GD step.

Run: python mnist_eqm.py --eqm-epochs 12 --clf-epochs 2 --n-train 20000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maze_eqm"))
from eqm_maze import TimeEmb, ResBlock, c_of_t, eqm_target      # noqa: E402  (faithful EqM target)


class MnistEqM(nn.Module):
    """Unconditional EqM field: 1ch in -> 1ch field. UNet 28->14->7."""
    def __init__(self, C=64, temb=128):
        super().__init__()
        self.temb = TimeEmb(temb)
        self.inp = nn.Conv2d(1, C, 3, padding=1)
        self.d1 = ResBlock(C, temb); self.down1 = nn.Conv2d(C, C, 3, 2, 1)
        self.d2 = ResBlock(C, temb); self.down2 = nn.Conv2d(C, C, 3, 2, 1)
        self.mid = ResBlock(C, temb)
        self.up2 = nn.ConvTranspose2d(C, C, 4, 2, 1); self.u2 = ResBlock(C, temb)
        self.up1 = nn.ConvTranspose2d(C, C, 4, 2, 1); self.u1 = ResBlock(C, temb)
        self.out = nn.Sequential(nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, 1, 3, padding=1))

    def forward(self, xt, t):
        temb = self.temb(t)
        h0 = self.inp(xt)
        h1 = self.d1(h0, temb)
        h2 = self.d2(self.down1(h1), temb)
        hm = self.mid(self.down2(h2), temb)
        u2 = self.up2(hm)
        u2 = F.interpolate(u2, size=h2.shape[-2:]) if u2.shape[-2:] != h2.shape[-2:] else u2
        u2 = self.u2(u2 + h2, temb)
        u1 = self.up1(u2)
        u1 = F.interpolate(u1, size=h1.shape[-2:]) if u1.shape[-2:] != h1.shape[-2:] else u1
        u1 = self.u1(u1 + h1, temb)
        return self.out(u1 + h0)


class SmallCNN(nn.Module):
    """Tiny MNIST classifier = inpainting oracle."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def load_mnist(root, n_train, n_test):
    tr = torchvision.datasets.MNIST(root=root, train=True, download=True)
    te = torchvision.datasets.MNIST(root=root, train=False, download=True)
    def to_t(ds, n):
        X = (ds.data[:n].float() / 255.0) * 2 - 1          # [-1,1], (n,28,28)
        y = ds.targets[:n]
        return X[:, None], y                                # (n,1,28,28)
    return to_t(tr, n_train) + to_t(te, n_test)


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    Xtr, ytr, Xte, yte = load_mnist(args.root, args.n_train, args.n_test)
    Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[mnist] dev={dev} train={len(Xtr)} test={len(Xte)}", flush=True)

    # ---- classifier oracle ----
    clf = SmallCNN().to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for ep in range(args.clf_epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            loss = F.cross_entropy(clf(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        acc = (clf(Xte).argmax(1) == yte).float().mean().item()
    print(f"[clf] test acc={acc:.4f}", flush=True)
    torch.save({"model": clf.state_dict(), "acc": acc}, out / "clf.pt")

    # ---- EqM ----
    model = MnistEqM(C=args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[eqm] params={nparam/1e3:.0f}K", flush=True)
    t0 = time.time(); N = len(Xtr); bs = args.batch
    for ep in range(args.eqm_epochs):
        perm = torch.randperm(N, device=dev); tot = 0.0
        model.train()
        for i in range(0, N, bs):
            idx = perm[i:i + bs]; x1 = Xtr[idx]
            t = torch.rand(len(idx), device=dev)
            xt, target = eqm_target(x1, t)
            loss = F.mse_loss(model(xt, t), target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % args.eval_every == 0 or ep == args.eqm_epochs - 1:
            print(f"  ep{ep+1:3d} loss={tot/N:.4f}  {time.time()-t0:.0f}s", flush=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "eqm.pt")
    print(f"saved -> {out}/eqm.pt , clf.pt", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--eqm-epochs", type=int, default=12)
    ap.add_argument("--clf-epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/mnist")
    args = ap.parse_args()
    main(args)
