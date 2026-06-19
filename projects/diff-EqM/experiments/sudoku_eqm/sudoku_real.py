"""EqM on REAL Sudoku (IRED-parity) — Stage 1/2 of REAL_SUDOKU_PLAN.md.

Keeps EqM pure: faithful EqM target (c(t)=min(1,5(1-t))*4, GD sampler t=0) — only the
BACKBONE is upgraded to IRED's Sudoku architecture (full-res 9x9, h=384, ResBlocks +
self-attention), which carries the global row/col/box constraints our toy conv-UNet lost.

Data (IRED parity): SATNet (features.pt/labels.pt) and RRN-hard (train/valid/test.csv).
Encoding: board (9,9,9) one-hot, rescaled {-1,+1}; givens = conditioning; label = solution.
Eval: board accuracy (whole grid valid + givens respected) — IRED's metric.

Run:
  python sudoku_real.py --data satnet --data-dir data_real/sudoku --epochs 60 --width 384
  python sudoku_real.py --data rrn   --data-dir data_real/sudoku-rrn --epochs 120 --width 384
"""
import argparse
import math
import os.path as osp
import time
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


# ---------------- faithful EqM target (same as transport.py / maze) ---------- #
def c_of_t(t):
    return torch.clamp(torch.minimum(torch.ones_like(t), 5.0 * (1.0 - t)), min=0.0) * 4.0


def eqm_target(x1, t):
    x0 = torch.randn_like(x1)
    tt = t[:, None, None, None]
    xt = (1 - tt) * x0 + tt * x1
    target = (x1 - x0) * c_of_t(t)[:, None, None, None]
    return xt, target


# ---------------- IRED backbone blocks (copied; self-contained) -------------- #
def swish(x):
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden = dim_head * heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)
        scale = q.shape[-1] ** -0.5
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class ResBlock(nn.Module):
    def __init__(self, filters=384, time_dim=64):
        super().__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(filters, filters, 5, 1, 2)
        self.conv2 = nn.Conv2d(filters, filters, 5, 1, 2)
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-5)
        self.time_fc2 = nn.Linear(time_dim, 2 * filters)

    def forward(self, x, t):
        l2 = self.time_fc2(t)
        gain2 = l2[:, :self.filters, None, None]; bias2 = l2[:, self.filters:, None, None]
        h = swish(self.conv1(x))
        h = swish((gain2 + 1) * self.conv2(h) + bias2)
        return x + h


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SudokuEqM(nn.Module):
    """Faithful-EqM FIELD model on Sudoku. Input [xt(9) + givens(9)] at full 9x9;
    IRED-style full-res h-channel ResBlocks + attention; outputs a 9ch field."""
    def __init__(self, width=384, time_dim=64, attn=True):
        super().__init__()
        self.D = 9
        fourier = 128
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(fourier), nn.Linear(fourier, time_dim),
                                      nn.GELU(), nn.Linear(time_dim, time_dim))
        self.conv1 = nn.Conv2d(9 + 9, width, 3, padding=1)
        self.res1a = ResBlock(width, time_dim); self.res1b = ResBlock(width, time_dim)
        self.attn1 = Attention(width, dim_head=128) if attn else None
        self.res2a = ResBlock(width, time_dim); self.res2b = ResBlock(width, time_dim)
        self.attn2 = Attention(width, dim_head=128) if attn else None
        self.res3a = ResBlock(width, time_dim); self.res3b = ResBlock(width, time_dim)
        self.attn3 = Attention(width, dim_head=128) if attn else None
        self.conv5 = nn.Conv2d(width, 9, 1)

    def forward(self, xt, t, cond):
        te = self.time_mlp(t)
        h = swish(self.conv1(torch.cat([xt, cond], 1)))
        h = self.res1b(self.res1a(h, te), te)
        if self.attn1 is not None:
            h = h + self.attn1(h)
        h = self.res2b(self.res2a(h, te), te)
        if self.attn2 is not None:
            h = h + self.attn2(h)
        h = self.res3b(self.res3a(h, te), te)
        if self.attn3 is not None:
            h = h + self.attn3(h)
        return self.conv5(h)


# ---------------- data (IRED parity) ---------------------------------------- #
def _rescale(x):
    return (x - 0.5) * 2.0


def load_satnet(data_dir):
    f = torch.load(osp.join(data_dir, "features.pt"))   # (N,9,9,9) one-hot
    l = torch.load(osp.join(data_dir, "labels.pt"))
    return f.float(), l.float()


def _str2onehot(s):
    x = np.array(list(map(int, s)), dtype="int64")
    y = np.zeros((len(x), 9), dtype="float32")
    idx = np.where(x > 0)[0]
    y[idx, x[idx] - 1] = 1
    return y.reshape((9, 9, 9))


def load_rrn(data_dir, split):
    import pandas as pd
    fn = {"train": "train.csv", "val": "valid.csv", "test": "test.csv"}[split]
    df = pd.read_csv(osp.join(data_dir, fn), header=None)
    feats = np.stack([_str2onehot(df.iloc[i][0]) for i in range(len(df))])
    labs = np.stack([_str2onehot(df.iloc[i][1]) for i in range(len(df))])
    return torch.tensor(feats), torch.tensor(labs)


def to_chw(x):
    """(B,9,9,9) one-hot {0,1} -> rescaled {-1,+1} (B,9ch,9,9)."""
    return _rescale(einops.rearrange(x, 'b h w c -> b c h w'))


# ---------------- validity oracle (exact, IRED board accuracy) -------------- #
def board_solved(grid, given_digit, given_mask):
    full = set(range(1, 10))
    if np.any(grid[given_mask] != given_digit[given_mask]):
        return False
    for i in range(9):
        if set(grid[i]) != full or set(grid[:, i]) != full:
            return False
    for br in range(3):
        for bc in range(3):
            if set(grid[br * 3:br * 3 + 3, bc * 3:bc * 3 + 3].flatten()) != full:
                return False
    return True


def board_acc(field_chw, feats_oh):
    """field_chw:(B,9,9,9) -> board-accuracy vs givens-from-feats. feats_oh:(B,9,9,9){0,1}."""
    g = np.argmax(field_chw, axis=1) + 1                          # (B,9,9) decoded digits
    fo = feats_oh.cpu().numpy()
    gm = (fo.sum(-1) == 1)                                        # (B,9,9) given mask
    gd = np.argmax(fo, axis=-1) + 1                               # given digit (valid where gm)
    ok = np.zeros(len(g), bool)
    for i in range(len(g)):
        ok[i] = board_solved(g[i], gd[i], gm[i])
    return ok


@torch.no_grad()
def gd_sample(model, cond, eta, steps, clamp_feats=None, log=False):
    """EqM GD from noise, t=0. cond=(B,9,9,9) givens (rescaled). Optional hard clamp of
    given cells to their one-hot each step (RePaint-style; helps respect givens)."""
    dev = cond.device
    xt = torch.randn(cond.shape[0], 9, 9, 9, device=dev)
    t0 = torch.zeros(cond.shape[0], device=dev)
    norms, dots = [], []
    # given-cell mask = cells with a +1 channel (rescaled one-hot); empty cells = all -1
    obs = (clamp_feats > 0).any(1, keepdim=True).expand_as(clamp_feats) if clamp_feats is not None else None
    for _ in range(steps):
        f = model(xt, t0, cond)
        if log:
            norms.append(f.flatten(1).norm(dim=1)); dots.append((f * xt).flatten(1).sum(1))
        xt = xt + f * eta
        if clamp_feats is not None:
            xt = torch.where(obs, clamp_feats, xt)            # fix given cells to their one-hot
    if log:
        return xt, torch.stack(norms, 1).cpu().numpy(), torch.stack(dots, 1).cpu().numpy()
    return xt


def main(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.data == "satnet":
        f, l = load_satnet(args.data_dir)
        n = len(f); ntr = int(n * 0.9)
        ftr, ltr, fte, lte = f[:ntr], l[:ntr], f[ntr:], l[ntr:]
    else:
        ftr, ltr = load_rrn(args.data_dir, "train")
        fte, lte = load_rrn(args.data_dir, "test")
    if args.max_train:
        ftr, ltr = ftr[:args.max_train], ltr[:args.max_train]
    fte, lte = fte[:args.eval_n], lte[:args.eval_n]
    print(f"[data={args.data}] train={len(ftr)} test={len(fte)} givens/board~{int((ftr[0].sum())) }", flush=True)

    cond_tr = to_chw(ftr).to(dev); x1_tr = to_chw(ltr).to(dev)
    cond_te = to_chw(fte).to(dev)
    clamp_te = to_chw(fte).to(dev) if args.clamp else None

    model = SudokuEqM(width=args.width, attn=not args.no_attn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[sudoku-real-eqm] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"width={args.width} attn={not args.no_attn} clamp={args.clamp}", flush=True)

    M = len(cond_tr); bs = args.batch; t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(M, device=dev); tot = 0.0; model.train()
        for i in range(0, M, bs):
            idx = perm[i:i + bs]
            t = torch.rand(len(idx), device=dev)
            xt, target = eqm_target(x1_tr[idx], t)
            loss = F.mse_loss(model(xt, t, cond_tr[idx]), target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            fld = gd_sample(model, cond_te, args.eta, args.steps, clamp_feats=clamp_te).cpu().numpy()
            acc = board_acc(fld, fte)
            print(f"  ep{ep+1:3d} loss={tot/M:.4f}  board-acc={acc.mean():.3f}  {time.time()-t0:.0f}s", flush=True)
    model.eval()
    fld = gd_sample(model, cond_te, args.eta, args.steps, clamp_feats=clamp_te).cpu().numpy()
    acc = float(board_acc(fld, fte).mean())
    gate = "PASS" if acc >= args.gate else "FAIL"
    print(f"[GATE {gate}] board-acc={acc:.3f} (need >={args.gate})", flush=True)
    out = Path(args.model_out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "board_acc": acc}, out)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["satnet", "rrn"], default="satnet")
    ap.add_argument("--data-dir", default="data_real/sudoku")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--clamp", action="store_true", help="hard-clamp given cells each GD step")
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-out", default="runs/sudoku_real/model.pt")
    main(ap.parse_args())
