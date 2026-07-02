"""Sudoku-EqM — a THIRD task type for trajectory-metacognition: constraint reasoning.

Distinct from image generation (pixels) and maze planning (path): pure CSP. A conditional
EqM maps clues -> a full 9x9 solution; an invalid solve = constraint violation (exact
checker = oracle, no learned label noise). Tests whether the SAME descent-shape probe that
works on generation+planning also catches constraint-reasoning failures — predicted to,
because a constraint-violating fill is an inconsistent local minimum (disturbed descent =
instability), the failure type the probe grips.

Faithful EqM target (replicated from transport.py, same as maze):
  x0~N(0,I); x1 = one-hot solution in {-1,+1}^(9,9,9); xt=(1-t)x0+t*x1, ut=x1-x0;
  TARGET = ut * c(t), c(t)=min(1,5(1-t))*4. Field f(xt,t,cond) = conditional UNet.
  cond = clue one-hot (9ch in {-1,+1}, empty=all -1) + given-mask (1ch). Inference = EqM GD.

Synthetic data: valid solutions via base-pattern + band/stack/digit shuffle (unlimited, no
internet); puzzles by removing cells. No external dataset (ired sudoku not present locally).

POSITIVE CONTROL (gate): trained solve-rate >> random floor (~0).

Run (CPU smoke): python sudoku_eqm.py --gen --n 200 --out data/sudoku_smoke.npz
                 python sudoku_eqm.py --train data/sudoku_smoke.npz --epochs 5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# shared faithful-EqM blocks from the maze module (sibling dir)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maze_eqm"))
from eqm_maze import TimeEmb, ResBlock, eqm_target  # noqa: E402

import math


# --------------------------- sudoku data (synthetic) ----------------------- #
def _full_solution(rng, N=9):
    """A valid NxN grid (digits 1..N) via base pattern + structure-preserving shuffles.
    N must be a perfect square; box size b=sqrt(N)."""
    b = int(round(math.sqrt(N)))
    def pat(r, c):
        return (b * (r % b) + r // b + c) % N
    g = np.array([[pat(r, c) for c in range(N)] for r in range(N)])
    g = rng.permutation(N)[g]                        # relabel digits 0..N-1
    def blocks():
        order = []
        for band in rng.permutation(b):
            for row in rng.permutation(b):
                order.append(band * b + row)
        return np.array(order)
    g = g[np.ix_(blocks(), blocks())]               # shuffle rows-in-bands & cols-in-stacks
    if rng.random() < 0.5:
        g = g.T
    return g + 1                                     # 1..N


def make_puzzle(sol, n_clues, rng):
    N = sol.shape[0]
    mask = np.zeros((N, N), bool)
    idx = rng.permutation(N * N)[:n_clues]
    mask.flat[idx] = True
    clue = np.where(mask, sol, 0)
    return clue, mask


def encode(sol, clue, mask, N=9):
    """sol/clue:(N,N) digits 1..N (clue 0=empty); -> target(N,N,N) onehot{-1,+1},
    cond(N+1,N,N) = clue-onehot{-1,+1} ++ given-mask{-1,+1}."""
    tgt = -np.ones((N, N, N), np.float32)
    for d in range(N):
        tgt[d][sol == d + 1] = 1.0
    cco = -np.ones((N, N, N), np.float32)
    for d in range(N):
        cco[d][clue == d + 1] = 1.0
    gm = np.where(mask, 1.0, -1.0).astype(np.float32)[None]
    cond = np.concatenate([cco, gm], 0)             # (N+1,N,N)
    return tgt, cond


def gen_dataset(n, n_clues, seed, out, N=9):
    rng = np.random.default_rng(seed)
    conds, tgts, sols, masks = [], [], [], []
    for _ in range(n):
        sol = _full_solution(rng, N)
        clue, mask = make_puzzle(sol, n_clues, rng)
        tgt, cond = encode(sol, clue, mask, N)
        conds.append(cond); tgts.append(tgt); sols.append(sol); masks.append(mask)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cond=np.stack(conds), target=np.stack(tgts),
             sol=np.stack(sols), mask=np.stack(masks), n_clues=n_clues, grid=N)
    print(f"gen {n} puzzles N={N} ({n_clues} clues) -> {out}", flush=True)


def load(npz):
    d = np.load(npz)
    return (torch.tensor(d["cond"], dtype=torch.float32),
            torch.tensor(d["target"], dtype=torch.float32),
            d["sol"], d["mask"])


# --------------------------- validity oracle (exact) ----------------------- #
def decode_grid(x):
    """x:(9,9,9) field -> (9,9) digits 1-9 by argmax over digit channels."""
    return np.argmax(x, axis=0) + 1


def solved(grid, clue_sol, mask):
    """Exact: all rows/cols/boxes are permutations of 1..N AND given clues respected."""
    g = grid; Ng = g.shape[0]; b = int(round(math.sqrt(Ng)))
    if mask is not None and clue_sol is not None:
        if np.any(g[mask] != clue_sol[mask]):
            return False
    full = set(range(1, Ng + 1))
    for i in range(Ng):
        if set(g[i]) != full or set(g[:, i]) != full:
            return False
    for br in range(b):
        for bc in range(b):
            if set(g[br * b:br * b + b, bc * b:bc * b + b].flatten()) != full:
                return False
    return True


def valid_mask(fields, sols, masks):
    """fields:(B,9,9,9) -> bool per puzzle (exact-solved, clues respected)."""
    out = np.zeros(len(fields), bool)
    for i in range(len(fields)):
        out[i] = solved(decode_grid(fields[i]), sols[i], masks[i])
    return out


# --------------------------- model: conditional UNet ----------------------- #
class SudokuEqM(nn.Module):
    """D=9 digit channels in/out (+ cond 10ch). UNet 9->5->3 for grid-global receptive field."""
    def __init__(self, cond_ch=10, C=96, temb=128, D=9):
        super().__init__()
        self.D = D
        self.temb = TimeEmb(temb)
        self.inp = nn.Conv2d(D + cond_ch, C, 3, padding=1)
        self.d1 = ResBlock(C, temb); self.down1 = nn.Conv2d(C, C, 3, 2, 1)
        self.d2 = ResBlock(C, temb); self.down2 = nn.Conv2d(C, C, 3, 2, 1)
        self.mid = ResBlock(C, temb)
        self.up2 = nn.ConvTranspose2d(C, C, 4, 2, 1); self.u2 = ResBlock(C, temb)
        self.up1 = nn.ConvTranspose2d(C, C, 4, 2, 1); self.u1 = ResBlock(C, temb)
        self.out = nn.Sequential(nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, D, 3, padding=1))

    def forward(self, xt, t, cond):
        temb = self.temb(t)
        h0 = self.inp(torch.cat([xt, cond], 1))
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


# --------------------------- EqM GD sampler -------------------------------- #
@torch.no_grad()
def gd_sample(model, cond, eta, steps, log=False):
    dev = cond.device
    xt = torch.randn(cond.shape[0], model.D, cond.shape[2], cond.shape[3], device=dev)
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


def solve_rate(model, cond, sols, masks, eta, steps):
    f = gd_sample(model, cond, eta, steps).cpu().numpy()
    return float(valid_mask(f, sols, masks).mean())


# --------------------------- train ----------------------------------------- #
def main(args):
    if args.gen:
        gen_dataset(args.n, args.clues, args.seed, args.out, args.grid)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cond, tgt, sols, masks = load(args.train)
    cond, tgt = cond.to(dev), tgt.to(dev)
    condte, tgtte, solte, maskte = load(args.test) if args.test else (cond, tgt, sols, masks)
    condte = condte.to(dev)
    Nd = tgt.shape[1]                                # digit channels = grid size N
    model = SudokuEqM(cond_ch=cond.shape[1], C=args.width, D=Nd).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[sudoku-eqm] params={sum(p.numel() for p in model.parameters())/1e3:.0f}K "
          f"clues={int(np.load(args.train)['n_clues'])}", flush=True)
    t0 = time.time(); M = len(cond); bs = args.batch
    for ep in range(args.epochs):
        perm = torch.randperm(M, device=dev); tot = 0.0; model.train()
        for i in range(0, M, bs):
            idx = perm[i:i + bs]
            t = torch.rand(len(idx), device=dev)
            xt, target = eqm_target(tgt[idx], t)
            loss = F.mse_loss(model(xt, t, cond[idx]), target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            sr = solve_rate(model, condte[:args.eval_n], solte[:args.eval_n],
                            maskte[:args.eval_n], args.eta, args.steps)
            rf = float(valid_mask(np.random.randn(min(args.eval_n, len(solte)), Nd, Nd, Nd),
                                  solte[:args.eval_n], maskte[:args.eval_n]).mean())
            print(f"  ep{ep+1:3d} loss={tot/M:.4f}  solve={sr:.3f} (rand {rf:.3f})  {time.time()-t0:.0f}s", flush=True)
    sr = solve_rate(model, condte[:args.eval_n], solte[:args.eval_n], maskte[:args.eval_n], args.eta, args.steps)
    gate = "PASS" if sr >= 0.10 else "FAIL"
    print(f"[GATE {gate}] solve-rate={sr:.3f}  random-floor~0", flush=True)
    out = Path(args.model_out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "solve_rate": sr}, out)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--clues", type=int, default=35)
    ap.add_argument("--train", default="")
    ap.add_argument("--test", default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sudoku.npz")
    ap.add_argument("--model-out", default="runs/sudoku/model.pt")
    main(ap.parse_args())
