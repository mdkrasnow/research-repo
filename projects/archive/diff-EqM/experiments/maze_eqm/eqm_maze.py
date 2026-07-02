"""Maze-EqM Step 2 — train a small conditional EqM to solve grid mazes.

Faithful EqM (target replicated from eqm-upstream/transport/transport.py):
  x0~N(0,I), x1=path in {-1,+1}; linear plan xt=(1-t)x0+t*x1, ut=x1-x0;
  TARGET = ut * c(t),  c(t)=min(1, 5(1-t))*4   (=4 for t<=0.8, ->0 at t=1).
Field f(xt, t, cond) is a small conditional UNet (cond = [wall,start,goal]).
Inference = EqM gradient descent: feed t=0, iterate xt += f(xt,0,cond)*eta (the
diff-EqM gd convention), decode path channel, BFS-validity.

POSITIVE CONTROL (Step-2 gate): trained valid-path-rate >> random floor (~0).

Run (CPU smoke): python eqm_maze.py --train data/maze_c5_train.npz \
    --test data/maze_c5_test.npz --epochs 30 --steps 200 --eta 0.02
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gen_maze_data import path_valid, decode_path


# ----------------------------- faithful EqM target ------------------------- #
def c_of_t(t):
    # c(t)=min(1, 5(1-t))*4 ; t:(B,) -> (B,)
    return torch.clamp(torch.minimum(torch.ones_like(t), 5.0 * (1.0 - t)), min=0.0) * 4.0


def eqm_target(x1, t):
    """x1:(B,1,H,W) data; t:(B,) -> (xt, target)."""
    x0 = torch.randn_like(x1)
    tt = t[:, None, None, None]
    xt = (1 - tt) * x0 + tt * x1
    ut = x1 - x0
    target = ut * c_of_t(t)[:, None, None, None]
    return xt, target


# ----------------------------- model: small conditional UNet --------------- #
class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(a), torch.cos(a)], -1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, c, temb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c); self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c); self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.t = nn.Linear(temb, c)

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class MazeEqM(nn.Module):
    """4ch in (xt + cond[3]) -> 1ch field. UNet 2 down/up for global receptive field."""
    def __init__(self, cond_ch=3, C=64, temb=128):
        super().__init__()
        self.temb = TimeEmb(temb)
        self.inp = nn.Conv2d(1 + cond_ch, C, 3, padding=1)
        self.d1 = ResBlock(C, temb); self.down1 = nn.Conv2d(C, C, 3, 2, 1)
        self.d2 = ResBlock(C, temb); self.down2 = nn.Conv2d(C, C, 3, 2, 1)
        self.mid = ResBlock(C, temb)
        self.up2 = nn.ConvTranspose2d(C, C, 4, 2, 1); self.u2 = ResBlock(C, temb)
        self.up1 = nn.ConvTranspose2d(C, C, 4, 2, 1); self.u1 = ResBlock(C, temb)
        self.out = nn.Sequential(nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, 1, 3, padding=1))

    def forward(self, xt, t, cond):
        temb = self.temb(t)
        h0 = self.inp(torch.cat([xt, cond], 1))
        h1 = self.d1(h0, temb)
        h2 = self.d2(self.down1(h1), temb)
        hm = self.mid(self.down2(h2), temb)
        # up, crop to skip sizes (odd grids -> size mismatch on transpose)
        u2 = self.up2(hm)
        u2 = F.interpolate(u2, size=h2.shape[-2:]) if u2.shape[-2:] != h2.shape[-2:] else u2
        u2 = self.u2(u2 + h2, temb)
        u1 = self.up1(u2)
        u1 = F.interpolate(u1, size=h1.shape[-2:]) if u1.shape[-2:] != h1.shape[-2:] else u1
        u1 = self.u1(u1 + h1, temb)
        return self.out(u1 + h0)


# ----------------------------- data ---------------------------------------- #
def load(npz):
    d = np.load(npz)
    return (torch.tensor(d["cond"], dtype=torch.float32),
            torch.tensor(d["target"], dtype=torch.float32))


# ----------------------------- GD sampler (EqM) ---------------------------- #
@torch.no_grad()
def gd_sample(model, cond, eta, steps, log=False):
    """EqM GD from noise, t=0 fixed. Returns final path channel + optional traj."""
    dev = cond.device
    xt = torch.randn(cond.shape[0], 1, cond.shape[2], cond.shape[3], device=dev)
    t0 = torch.zeros(cond.shape[0], device=dev)
    norms, dots = [], []
    for _ in range(steps):
        f = model(xt, t0, cond)
        if log:
            norms.append(f.flatten(1).norm(dim=1))
            dots.append((f * xt).flatten(1).sum(1))
        xt = xt + f * eta
    if log:
        return xt, torch.stack(norms, 1).cpu().numpy(), torch.stack(dots, 1).cpu().numpy()
    return xt


def valid_rate(model, cond, eta, steps, thr=0.0):
    """Fraction of sampled paths that pass BFS validity."""
    paths = gd_sample(model, cond, eta, steps).cpu().numpy()
    ok = 0
    for i in range(cond.shape[0]):
        wall = cond[i, 0].cpu().numpy().astype(np.int8)
        s = tuple(np.argwhere(cond[i, 1].cpu().numpy() > 0)[0])
        g = tuple(np.argwhere(cond[i, 2].cpu().numpy() > 0)[0])
        ok += path_valid(decode_path(paths[i, 0], thr), wall, s, g)
    return ok / cond.shape[0]


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cond_tr, tgt_tr = load(args.train)
    cond_te, tgt_te = load(args.test)
    print(f"[maze-eqm] dev={dev} train={len(cond_tr)} test={len(cond_te)} "
          f"grid={cond_tr.shape[-1]}", flush=True)
    model = MazeEqM(C=args.width).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cond_tr, tgt_tr = cond_tr.to(dev), tgt_tr.to(dev)
    N = len(cond_tr); bs = args.batch
    print(f"[maze-eqm] params={nparam/1e3:.0f}K", flush=True)

    # negative control: random path valid-rate on test
    rng = np.random.default_rng(0); H = cond_te.shape[-1]; ok_rand = 0
    for i in range(len(cond_te)):
        wall = cond_te[i, 0].numpy().astype(np.int8)
        s = tuple(np.argwhere(cond_te[i, 1].numpy() > 0)[0]); g = tuple(np.argwhere(cond_te[i, 2].numpy() > 0)[0])
        ok_rand += path_valid((rng.random((H, H)) > 0.5).astype(np.int8), wall, s, g)
    rand_rate = ok_rand / len(cond_te)

    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=dev)
        model.train(); tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            x1 = tgt_tr[idx]; cond = cond_tr[idx]
            t = torch.rand(len(idx), device=dev)
            xt, target = eqm_target(x1, t)
            pred = model(xt, t, cond)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            vr = valid_rate(model, cond_te[:args.eval_n].to(dev), args.eta, args.steps)
            print(f"  ep{ep+1:3d}  loss={tot/N:.4f}  test-valid={vr:.3f} "
                  f"(rand {rand_rate:.3f})  {time.time()-t0:.0f}s", flush=True)
    # final gate
    model.eval()
    vr = valid_rate(model, cond_te[:args.eval_n].to(dev), args.eta, args.steps)
    gate = "PASS" if vr >= max(0.30, 5 * max(rand_rate, 0.001)) else "FAIL"
    print(f"\n[STEP-2 GATE {gate}] test-valid={vr:.3f}  random-floor={rand_rate:.3f}  "
          f"(need >=0.30 and >>floor)", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "valid_rate": vr, "rand_rate": rand_rate,
                "args": vars(args)}, out)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/maze_c5_train.npz")
    ap.add_argument("--test", default="data/maze_c5_test.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--out", default="runs/maze_c5/model.pt")
    args = ap.parse_args()
    main(args)
