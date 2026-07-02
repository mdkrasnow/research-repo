"""Phase 3D — maze planning prototype (CPU, numpy-only).

Tests the GENERAL metacognition claim outside EqM: in an energy-descent planner
that refines a candidate path, does a probe over the REFINEMENT-DYNAMICS predict
failure (path clips a wall = "garbage"), and does risk-guided branching beat
random branching AT EQUAL COMPUTE?

This is the maze analog of the EqM probe-gated restart sampler:
  EqM GD sampling      <-> path energy-descent refinement
  spurious minimum     <-> low-energy path that still clips a wall (invalid)
  descent-shape probe  <-> refinement-trajectory-shape probe
  best-of-R restart     <-> branch to a fresh init when risk is high

Arms (all from the SAME R candidate draws, exactly as the EqM gated sampler):
  vanilla  = R=1, refined for the FULL matched step budget        (neg / floor)
  random   = R short partial refines, pick one at RANDOM, finish  (compute-matched neg)
  probe    = R short partial refines, pick argmin P(invalid),finish(treatment)
  oracle   = pick the candidate whose FULL refine is valid         (pos / ceiling; over-budget, labeled)

Compute (NFE = number of single-path gradient steps):
  vanilla = R*T_partial + T_finish     (one init, that many steps)
  random  = R*T_partial + T_finish     (R partials + 1 finish)  -> EXACTLY matched to vanilla
  probe   = R*T_partial + T_finish     (same)                    -> EXACTLY matched
  oracle  = R*T_full                   (cheats: refines all R fully) -> ceiling, NOT matched

Metric: valid-path rate by difficulty. Success = probe > random at equal compute,
inside the vanilla..oracle band.

Honest-negative discipline: if probe does not beat random, we report it. The probe
is trained ONLY on refinement dynamics (no ground-truth validity at selection time);
oracle uses ground truth and is the ceiling.

Run:
  python maze_planning.py --train-mazes 240 --eval-mazes 300 --seed 0
Outputs under results/capabilities/maze/.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Maze: axis-aligned rectangular wall obstacles between a left start and right
# goal. Difficulty = number of wall rows and gap width. A valid path goes from
# start to goal without any sampled point landing inside a wall rectangle.
# --------------------------------------------------------------------------
class Maze:
    def __init__(self, walls, start, goal):
        self.walls = walls          # list of (x0, y0, x1, y1)
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)

    def inside_any(self, pts):
        """pts: (...,2) -> bool mask inside any wall."""
        pts = np.asarray(pts, float)
        m = np.zeros(pts.shape[:-1], bool)
        for (x0, y0, x1, y1) in self.walls:
            m |= ((pts[..., 0] >= x0) & (pts[..., 0] <= x1) &
                  (pts[..., 1] >= y0) & (pts[..., 1] <= y1))
        return m

    def signed_clear(self, pts):
        """Soft clearance per point: positive outside, negative (penetration
        depth) inside the nearest wall. Smooth-ish; drives the wall energy."""
        pts = np.asarray(pts, float)
        best = np.full(pts.shape[:-1], 1e9)
        for (x0, y0, x1, y1) in self.walls:
            dx = np.maximum(np.maximum(x0 - pts[..., 0], pts[..., 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - pts[..., 1], pts[..., 1] - y1), 0.0)
            outside = np.sqrt(dx * dx + dy * dy)
            # penetration depth when inside
            px = np.minimum(pts[..., 0] - x0, x1 - pts[..., 0])
            py = np.minimum(pts[..., 1] - y0, y1 - pts[..., 1])
            inside_depth = np.minimum(px, py)
            inside = (dx == 0) & (dy == 0)
            d = np.where(inside, -inside_depth, outside)
            best = np.minimum(best, d)
        return best


def make_maze(rng, difficulty):
    """difficulty in {0,1,2,3}: more wall rows + narrower gaps = harder."""
    n_rows = 1 + difficulty            # vertical wall bands to cross
    gap = 0.34 - 0.06 * difficulty     # passage half-height; shrinks with difficulty
    walls = []
    for r in range(n_rows):
        x = (r + 1) / (n_rows + 1)
        wx = 0.04
        gy = rng.uniform(0.3, 0.7)      # gap center, jittered per band
        # wall = full vertical column minus a gap of half-height `gap` at gy
        walls.append((x - wx, 0.0, x + wx, gy - gap))
        walls.append((x - wx, gy + gap, x + wx, 1.0))
    start = (0.03, rng.uniform(0.3, 0.7))
    goal = (0.97, rng.uniform(0.3, 0.7))
    return Maze(walls, start, goal)


# --------------------------------------------------------------------------
# Path = K waypoints; endpoints pinned to start/goal, optimize the interior.
# Energy = tension (smoothness, pulls path straight) + wall penalty (soft barrier
# along densely-sampled segments). Soft + local barrier => spurious minima: a
# path can settle low-gradient while still clipping a wall corner.
# --------------------------------------------------------------------------
K = 12          # waypoints
SEG = 6         # sub-samples per segment for collision energy
W_TENSION = 1.0
W_WALL = 6.0


def seg_points(path):
    """Dense points along the polyline for collision evaluation: (N,2)."""
    a = path[:-1]; b = path[1:]
    ts = (np.arange(SEG) + 0.5) / SEG
    pts = a[:, None, :] + (b - a)[:, None, :] * ts[None, :, None]
    return pts.reshape(-1, 2)


def energy_and_grad(path, maze):
    """Scalar energy + grad wrt interior waypoints (endpoints fixed)."""
    grad = np.zeros_like(path)
    # tension: |w_{i+1} - 2 w_i + w_{i-1}|^2
    lap = path[2:] - 2 * path[1:-1] + path[:-2]
    e_tension = W_TENSION * np.sum(lap ** 2)
    # d/dw of sum lap^2 (discrete laplacian transpose)
    gt = np.zeros_like(path)
    gt[2:] += 2 * lap
    gt[1:-1] += -4 * lap
    gt[:-2] += 2 * lap
    grad += W_TENSION * gt
    # wall penalty along segments: relu(margin - clear)^2 ; margin keeps a buffer
    margin = 0.02
    a = path[:-1]; b = path[1:]
    ts = (np.arange(SEG) + 0.5) / SEG
    pts = a[:, None, :] + (b - a)[:, None, :] * ts[None, :, None]   # (K-1,SEG,2)
    clear = maze.signed_clear(pts)                                   # (K-1,SEG)
    viol = np.maximum(margin - clear, 0.0)
    e_wall = W_WALL * np.sum(viol ** 2)
    # numeric grad of wall term wrt waypoints (cheap: finite diff on interior pts)
    gw = np.zeros_like(path)
    if np.any(viol > 0):
        eps = 1e-3
        for i in range(1, K - 1):
            for d in range(2):
                p2 = path.copy(); p2[i, d] += eps
                a2 = p2[:-1]; b2 = p2[1:]
                pts2 = a2[:, None, :] + (b2 - a2)[:, None, :] * ts[None, :, None]
                clear2 = maze.signed_clear(pts2)
                viol2 = np.maximum(margin - clear2, 0.0)
                e2 = W_WALL * np.sum(viol2 ** 2)
                gw[i, d] = (e2 - e_wall) / eps
    grad += gw
    grad[0] = 0; grad[-1] = 0     # pin endpoints
    return e_tension + e_wall, grad, e_wall


def init_path(rng, maze):
    """Random-ish init: straight line + Gaussian bumps on interior waypoints."""
    t = np.linspace(0, 1, K)[:, None]
    line = maze.start[None] * (1 - t) + maze.goal[None] * t
    line[1:-1] += rng.normal(0, 0.18, (K - 2, 2))
    return np.clip(line, 0.0, 1.0)


def refine(path, maze, n_steps, lr=0.02, log=True):
    """Gradient-descend the path energy. Returns (final_path, traj_dict)."""
    p = path.copy()
    es, ews, gns = [], [], []
    for _ in range(n_steps):
        e, g, ew = energy_and_grad(p, maze)
        gn = float(np.sqrt(np.sum(g ** 2)))
        if log:
            es.append(e); ews.append(ew); gns.append(gn)
        p = p - lr * g
        p = np.clip(p, 0.0, 1.0)
    traj = {"energy": np.array(es), "wall": np.array(ews), "gnorm": np.array(gns)}
    return p, traj


def is_valid(path, maze):
    """Ground truth: densely sample the polyline, none inside a wall."""
    pts = seg_points(path)
    # finer check
    a = path[:-1]; b = path[1:]
    ts = np.linspace(0, 1, 24)
    fine = (a[:, None, :] + (b - a)[:, None, :] * ts[None, :, None]).reshape(-1, 2)
    return not bool(maze.inside_any(fine).any())


# --------------------------------------------------------------------------
# Probe: features from the PARTIAL refinement trajectory (dynamics only — never
# the ground-truth validity). Predict P(invalid).
# --------------------------------------------------------------------------
def traj_features(traj, final_path, maze):
    e = traj["energy"]; w = traj["wall"]; g = traj["gnorm"]
    if len(e) < 4:
        e = np.pad(e, (0, 4 - len(e)), mode="edge") if len(e) else np.zeros(4)
    def feats(x):
        x = np.asarray(x, float)
        if len(x) < 2:
            return [0.0, 0.0, 0.0, 0.0]
        xn = x / (abs(x[0]) + 1e-9)
        slope = (x[-1] - x[0]) / (len(x) * (abs(x[0]) + 1e-9))
        osc = float(np.mean(np.abs(np.diff(np.sign(np.diff(x))))))  # sign flips
        return [float(x[-1]), float(slope), osc, float(np.mean(xn[-max(2, len(x)//4):]))]
    f = []
    f += feats(e)
    f += feats(w)
    f += feats(g)
    # endpoint structural cues available at refine time (NOT ground truth):
    f.append(float(traj["wall"][-1]))                  # residual wall energy
    f.append(float(traj["energy"][-1]))                # residual total energy
    # min clearance along final path (soft, model-internal — not the validity bool)
    f.append(float(maze.signed_clear(seg_points(final_path)).min()))
    return np.array(f, float)


def fit_logistic(X, y, l2=1.0, iters=600, lr=0.3):
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xs @ w))
        g = Xs.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0] / len(y)
        w -= lr * g
    return w, mu, sd


def predict_logistic(w, mu, sd, X):
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    return 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos = y == 1; npos = pos.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def build_probe_dataset(rng, n_mazes, T_partial):
    """Generate partial-refinement trajectories + validity labels for training."""
    X, y, diffs = [], [], []
    for _ in range(n_mazes):
        d = int(rng.integers(0, 4))
        mz = make_maze(rng, d)
        p0 = init_path(rng, mz)
        p_partial, traj = refine(p0, mz, T_partial, log=True)
        # label = will the FULL refine of THIS init be invalid? (teacher signal)
        p_full, _ = refine(p0, mz, T_partial + T_FINISH, log=False)
        X.append(traj_features(traj, p_partial, mz))
        y.append(0 if is_valid(p_full, mz) else 1)
        diffs.append(d)
    return np.array(X), np.array(y, float), np.array(diffs)


T_PARTIAL = 40
T_FINISH = 120
R = 4


def run_arms(rng, maze, probe):
    """One maze: produce a valid/invalid outcome for each arm. NFE-matched."""
    w, mu, sd = probe
    budget = R * T_PARTIAL + T_FINISH      # matched step budget for vanilla/random/probe

    # R candidate inits, each partially refined (shared draws across arms)
    cands = []
    for _ in range(R):
        p0 = init_path(rng, maze)
        p_part, traj = refine(p0, maze, T_PARTIAL, log=True)
        risk = float(predict_logistic(w, mu, sd, traj_features(traj, p_part, maze)[None])[0])
        cands.append({"p0": p0, "p_part": p_part, "risk": risk})

    # vanilla: single init (first draw), refined the FULL matched budget
    pv, _ = refine(cands[0]["p0"], maze, budget, log=False)
    valid_vanilla = is_valid(pv, maze)

    # random: pick a candidate uniformly, finish it
    ri = int(rng.integers(0, R))
    pr, _ = refine(cands[ri]["p_part"], maze, T_FINISH, log=False)
    valid_random = is_valid(pr, maze)

    # probe: pick argmin predicted risk, finish it
    pi = int(np.argmin([c["risk"] for c in cands]))
    pp, _ = refine(cands[pi]["p_part"], maze, T_FINISH, log=False)
    valid_probe = is_valid(pp, maze)

    # oracle (ceiling, over-budget): finish ALL candidates, valid if ANY valid
    valid_oracle = False
    for c in cands:
        pf, _ = refine(c["p_part"], maze, T_FINISH, log=False)
        if is_valid(pf, maze):
            valid_oracle = True
            break
    return dict(vanilla=valid_vanilla, random=valid_random,
                probe=valid_probe, oracle=valid_oracle,
                nfe_vanilla=budget, nfe_random=budget, nfe_probe=budget,
                nfe_oracle=R * (T_PARTIAL + T_FINISH))


def main(args):
    out = Path(args.out) if args.out else Path(__file__).parent / "results" / "capabilities" / "maze"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1) train probe on partial-trajectory dynamics
    Xtr, ytr, dtr = build_probe_dataset(rng, args.train_mazes, T_PARTIAL)
    # held-out AUC via simple split
    n = len(Xtr); cut = int(0.7 * n)
    w, mu, sd = fit_logistic(Xtr[:cut], ytr[:cut])
    auc_ho = auc(ytr[cut:], predict_logistic(w, mu, sd, Xtr[cut:]))
    # refit on all for deployment
    w, mu, sd = fit_logistic(Xtr, ytr)
    probe = (w, mu, sd)
    print(f"probe: train n={n} invalid-rate={ytr.mean():.2f} held-out AUC={auc_ho:.3f}", flush=True)

    # 2) evaluate arms across difficulties
    per_diff = {d: {"vanilla": [], "random": [], "probe": [], "oracle": [], "n": 0}
                for d in range(4)}
    rows = []
    for _ in range(args.eval_mazes):
        d = int(rng.integers(0, 4))
        mz = make_maze(rng, d)
        res = run_arms(rng, mz, probe)
        for arm in ["vanilla", "random", "probe", "oracle"]:
            per_diff[d][arm].append(1.0 if res[arm] else 0.0)
        per_diff[d]["n"] += 1

    # aggregate
    def rate(arm, d=None):
        if d is None:
            vals = [v for dd in range(4) for v in per_diff[dd][arm]]
        else:
            vals = per_diff[d][arm]
        return float(np.mean(vals)) if vals else float("nan")

    arms = ["vanilla", "random", "probe", "oracle"]
    overall = {a: rate(a) for a in arms}
    with open(out / "maze_table.csv", "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["difficulty", "n", *arms, "nfe_matched", "nfe_oracle"])
        for d in range(4):
            wtr.writerow([d, per_diff[d]["n"], *[f"{rate(a, d):.4f}" for a in arms],
                          R * T_PARTIAL + T_FINISH, R * (T_PARTIAL + T_FINISH)])
        wtr.writerow(["ALL", args.eval_mazes, *[f"{overall[a]:.4f}" for a in arms],
                      R * T_PARTIAL + T_FINISH, R * (T_PARTIAL + T_FINISH)])

    # verdict: probe must beat random at equal compute, inside vanilla..oracle band
    gap_pr = overall["probe"] - overall["random"]
    gap_or = overall["oracle"] - overall["random"]
    recovered = gap_pr / gap_or if gap_or > 1e-6 else float("nan")
    if gap_pr > 0.02 and overall["probe"] <= overall["oracle"] + 1e-6:
        verdict = "PROBE>RANDOM (equal compute) — general mechanism transfers to maze planning"
    elif abs(gap_pr) <= 0.02:
        verdict = "PROBE≈RANDOM — no transfer; refinement-dynamics probe not actionable here"
    else:
        verdict = "PROBE<RANDOM — probe anti-correlated; mechanism does NOT transfer"

    summary = {
        "probe_heldout_auc": round(float(auc_ho), 4),
        "overall": {a: round(overall[a], 4) for a in arms},
        "probe_minus_random": round(float(gap_pr), 4),
        "oracle_minus_random": round(float(gap_or), 4),
        "fraction_of_oracle_recovered": None if not np.isfinite(recovered) else round(float(recovered), 3),
        "nfe_matched_arms": R * T_PARTIAL + T_FINISH,
        "nfe_oracle": R * (T_PARTIAL + T_FINISH),
        "R": R, "T_partial": T_PARTIAL, "T_finish": T_FINISH,
        "eval_mazes": args.eval_mazes, "train_mazes": args.train_mazes, "seed": args.seed,
        "verdict": verdict,
    }
    (out / "maze_summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# Maze planning prototype — results", "",
          f"Probe held-out AUC (predict invalid from partial-refinement dynamics): **{auc_ho:.3f}**",
          "",
          "Valid-path rate by arm (all arms below the line are EXACTLY compute-matched; "
          "oracle is the over-budget ceiling):", "",
          "| difficulty | n | vanilla | random | probe | oracle |",
          "|---|---|---|---|---|---|"]
    for d in range(4):
        md.append(f"| {d} | {per_diff[d]['n']} | {rate('vanilla', d):.3f} | "
                  f"{rate('random', d):.3f} | {rate('probe', d):.3f} | {rate('oracle', d):.3f} |")
    md.append(f"| **ALL** | {args.eval_mazes} | {overall['vanilla']:.3f} | "
              f"{overall['random']:.3f} | **{overall['probe']:.3f}** | {overall['oracle']:.3f} |")
    md += ["",
           f"- NFE matched (vanilla/random/probe) = **{R*T_PARTIAL+T_FINISH}** path-grad steps; "
           f"oracle = {R*(T_PARTIAL+T_FINISH)} (cheats, labeled).",
           f"- probe − random = **{gap_pr:+.3f}** ; oracle − random = {gap_or:+.3f} ; "
           f"fraction of oracle gain recovered = "
           f"{'n/a' if not np.isfinite(recovered) else f'{recovered:.0%}'}.",
           "",
           f"## VERDICT: {verdict}",
           "",
           "Controls: vanilla = single-init floor (neg), random = compute-matched "
           "random-branch (neg), oracle = any-of-R valid (pos ceiling). Probe read "
           "ONLY refinement dynamics — never ground-truth validity — at selection time."]
    (out / "MAZE_SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-mazes", type=int, default=240)
    ap.add_argument("--eval-mazes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    main(args)
