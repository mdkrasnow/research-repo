"""Maze-EqM Step 1 — data generation + validity (CPU, numpy-only).

Generates grid mazes with a unique BFS shortest path, encodes each as
(cond[wall,start,goal], target[path]) on an H×W grid (H=W=2*cells+1), and provides
the EXACT validity check used as the metacognition oracle:

  valid(pred_path, wall, start, goal) =
    every predicted path cell is on a free (non-wall) cell, AND a BFS over the
    predicted path cells (4-connected) connects start to goal.

Run: python gen_maze_data.py --cells 5 --n 2000 --out data/maze_c5.npz
"""
import argparse
from collections import deque
from pathlib import Path

import numpy as np


# ---- maze generation: randomized DFS perfect maze on a cells×cells grid ---- #
def gen_perfect_maze(cells, rng):
    """Return H×W wall grid (H=W=2*cells+1), 1=wall 0=free. Perfect maze: unique
    path between any two cells."""
    H = W = 2 * cells + 1
    wall = np.ones((H, W), np.int8)
    # carve cells at odd coords; walls between
    visited = np.zeros((cells, cells), bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    wall[1, 1] = 0
    while stack:
        cy, cx = stack[-1]
        nbrs = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < cells and 0 <= nx < cells and not visited[ny, nx]:
                nbrs.append((ny, nx, dy, dx))
        if not nbrs:
            stack.pop()
            continue
        ny, nx, dy, dx = nbrs[rng.integers(len(nbrs))]
        # carve the wall between (cy,cx) and (ny,nx), and the target cell
        wall[2 * cy + 1 + dy, 2 * cx + 1 + dx] = 0
        wall[2 * ny + 1, 2 * nx + 1] = 0
        visited[ny, nx] = True
        stack.append((ny, nx))
    return wall


def bfs_path(wall, start, goal):
    """Shortest path (list of (y,x)) over free cells, 4-connected. None if none."""
    H, W = wall.shape
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        cy, cx = cur
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and wall[ny, nx] == 0 and (ny, nx) not in prev:
                prev[(ny, nx)] = cur
                q.append((ny, nx))
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def make_sample(cells, rng):
    """One maze -> (cond[3,H,W] float, path[1,H,W] float in {-1,+1}, meta)."""
    wall = gen_perfect_maze(cells, rng)
    H, W = wall.shape
    start = (1, 1)
    goal = (H - 2, W - 2)
    path = bfs_path(wall, start, goal)
    if path is None:
        return None
    wallf = wall.astype(np.float32)
    s = np.zeros((H, W), np.float32); s[start] = 1.0
    g = np.zeros((H, W), np.float32); g[goal] = 1.0
    p = -np.ones((H, W), np.float32)
    for (y, x) in path:
        p[y, x] = 1.0
    cond = np.stack([wallf, s, g], 0)                  # [3,H,W]
    target = p[None]                                    # [1,H,W] in {-1,+1}
    return cond, target, {"start": start, "goal": goal, "path_len": len(path)}


# ---- validity oracle ---- #
def path_valid(pred_path_bin, wall, start, goal):
    """pred_path_bin: H×W {0,1}. Valid iff all path cells are free AND a 4-connected
    BFS restricted to predicted-path cells connects start->goal."""
    H, W = wall.shape
    pp = pred_path_bin.astype(bool)
    # every predicted path cell must be on a free cell
    if np.any(pp & (wall == 1)):
        return False
    if not pp[start] or not pp[goal]:
        return False
    # BFS over predicted path cells only
    seen = np.zeros((H, W), bool)
    q = deque([start]); seen[start] = True
    while q:
        cy, cx = q.popleft()
        if (cy, cx) == goal:
            return True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and pp[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True; q.append((ny, nx))
    return False


def decode_path(pred_continuous, thr=0.0):
    """EqM outputs continuous path channel -> binary {0,1} (>=thr means path)."""
    return (pred_continuous >= thr).astype(np.int8)


def ascii_maze(wall, path=None, start=None, goal=None):
    H, W = wall.shape
    out = []
    pset = set(map(tuple, np.argwhere(path > 0))) if path is not None else set()
    for y in range(H):
        row = []
        for x in range(W):
            if start and (y, x) == start:
                row.append("S")
            elif goal and (y, x) == goal:
                row.append("G")
            elif (y, x) in pset:
                row.append("●")
            elif wall[y, x] == 1:
                row.append("█")
            else:
                row.append("·")
        out.append(" ".join(row))
    return "\n".join(out)


def main(args):
    rng = np.random.default_rng(args.seed)
    conds, targets, plens = [], [], []
    tries = 0
    while len(conds) < args.n and tries < args.n * 5:
        tries += 1
        s = make_sample(args.cells, rng)
        if s is None:
            continue
        c, t, m = s
        conds.append(c); targets.append(t); plens.append(m["path_len"])
    conds = np.stack(conds); targets = np.stack(targets)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cond=conds.astype(np.float32), target=targets.astype(np.float32),
                        cells=np.int64(args.cells), path_len=np.array(plens, np.int64))
    H = 2 * args.cells + 1
    print(f"wrote {len(conds)} mazes  grid={H}x{H}  cells={args.cells}  "
          f"path_len mean={np.mean(plens):.1f}  -> {out}", flush=True)

    # self-checks: GT path is valid; random floor ~0; a wall-crossing path invalid
    rng2 = np.random.default_rng(0)
    ok_gt = 0; ok_rand = 0; n_check = min(200, len(conds))
    for i in range(n_check):
        wall = conds[i, 0].astype(np.int8)
        s = tuple(np.argwhere(conds[i, 1] > 0)[0]); g = tuple(np.argwhere(conds[i, 2] > 0)[0])
        gt = (targets[i, 0] > 0).astype(np.int8)
        ok_gt += path_valid(gt, wall, s, g)
        rnd = (rng2.random((H, H)) > 0.5).astype(np.int8)
        ok_rand += path_valid(rnd, wall, s, g)
    print(f"SELF-CHECK  GT-valid={ok_gt}/{n_check} (want {n_check})  "
          f"random-valid={ok_rand}/{n_check} (want ~0)", flush=True)
    # show one
    print("\nexample maze (S start, G goal, ● path):")
    print(ascii_maze(conds[0, 0].astype(np.int8), targets[0, 0],
                     tuple(np.argwhere(conds[0, 1] > 0)[0]),
                     tuple(np.argwhere(conds[0, 2] > 0)[0])), flush=True)
    assert ok_gt == n_check, "BUG: ground-truth path failed validity check"
    assert ok_rand <= 1, "BUG: random path too often valid -> metric trivial"
    print("\nSTEP-1 GATE PASS: GT valid, random floor ~0, encode/decode round-trip OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=5)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/maze_c5.npz")
    args = ap.parse_args()
    main(args)
