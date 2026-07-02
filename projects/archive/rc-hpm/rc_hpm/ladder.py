"""D2 H-A' difficulty ladder: knobbed generator, pre-flight metrics, gamma
probe, arm training. Implements documentation/preregistration-d2.md exactly;
reuses rc_hpm/core.py (validated) and the E0.2' student/probe protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import core, losses
from .toy import aug_view, train_gate

TAU = 0.5
ALPHA0 = 0.10
DELTA_R = 0.05
N_BATCH = 64
K_PLUS, K_MINUS = 2, 8
STEPS = 1500
GRID_BP = (0.55, 0.65, 0.75, 0.85, 0.92, 0.97)
GRID_BM = (0.03, 0.08, 0.15, 0.25, 0.35, 0.45)
LAM_GRID = [(bp, bm) for bp in GRID_BP for bm in GRID_BM if bm < bp]


# ----------------------------------------------------------------------------
# Knobbed generator (prereg-d2: K, sigma, imbalance exponent a)
# ----------------------------------------------------------------------------

@dataclass
class Rung:
    K: int
    sigma: float
    a: float                      # imbalance exponent (weights ~ rank^-a)
    family_seed: int = 0          # population seed shared across arms/seeds
    dim: int = 16
    means: np.ndarray = field(default=None, repr=False)
    weights: np.ndarray = field(default=None, repr=False)
    W: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.family_seed)
        self.means = rng.normal(0, 2.0, (self.K, self.dim))
        w = np.arange(1, self.K + 1, dtype=float) ** (-self.a)
        self.weights = w / w.sum()
        self.W = rng.normal(0, 1.0 / np.sqrt(self.dim), (self.dim, self.dim))

    def tag(self):
        return f"K{self.K}_s{self.sigma}_a{self.a}"


def draw(rung: Rung, n: int, rng: np.random.Generator):
    labels = rng.choice(rung.K, n, p=rung.weights)
    return rung.means[labels] + rng.normal(0, rung.sigma,
                                           (n, rung.dim)), labels


def teacher_embed(rung: Rung, x: np.ndarray) -> np.ndarray:
    e = x @ rung.W
    return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-12)


# ----------------------------------------------------------------------------
# Pre-flight metrics (prereg-d2)
# ----------------------------------------------------------------------------

def rho_tail(rung: Rung, n_pairs: int = 200_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x, lab = draw(rung, 20_000, rng)
    e = teacher_embed(rung, x)
    i = rng.integers(0, len(x), n_pairs)
    j = rng.integers(0, len(x), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    s = (e[i] * e[j]).sum(1)
    top = s >= np.quantile(s, 0.9)
    return float((lab[i] == lab[j])[top].mean())


def calibrate_rung(rung: Rung, seed: int = 0, alpha: float = ALPHA0):
    """Standard gate + LTT on this rung. Returns (q_fn, calib)."""
    rng = np.random.default_rng(seed + 4000)
    xg, lg = draw(rung, 2000, rng)
    q_fn = train_gate(teacher_embed(rung, xg), lg, seed)
    xc, lc = draw(rung, 32_000, rng)
    ec = teacher_embed(rung, xc)
    ac = teacher_embed(rung, aug_view(xc, rng, 0.25))
    calib = core.calibrate_ltt(ec, lc, ac, q_fn, LAM_GRID, alpha, alpha,
                               DELTA_R, N_BATCH, 250, rng, K_PLUS, K_MINUS,
                               m_fit=40)
    return q_fn, calib


def supply_S(rung: Rung, q_fn, calib, n_batches: int = 40,
             seed: int = 1) -> float:
    """Certified-negative fraction of the top similarity decile at lam*."""
    if calib.aborted:
        return 0.0
    rng = np.random.default_rng(seed)
    S_all, CN_all = [], []
    for _ in range(n_batches):
        xb, lb = draw(rung, N_BATCH, rng)
        eb = teacher_embed(rung, xb)
        qb = core.q_matrix(q_fn, eb, lb)
        s = eb @ eb.T
        np.fill_diagonal(s, np.nan)
        m = ~np.isnan(s)
        S_all.append(s[m])
        CN_all.append((qb <= calib.lam[1])[m])
    S = np.concatenate(S_all)
    CN = np.concatenate(CN_all)
    top = S >= np.quantile(S, 0.9)
    return float(CN[top].mean())


# ----------------------------------------------------------------------------
# gamma probe (kNN symmetric normalized Laplacian; covariate only)
# ----------------------------------------------------------------------------

def gamma_probe(emb: np.ndarray, true_K: int, knn: int = 15):
    """Returns (K_hat, gamma_at_true_K, first eigenvalues)."""
    n = emb.shape[0]
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    nbr = np.argpartition(-sim, knn, axis=1)[:, :knn]
    A = np.zeros((n, n))
    A[np.repeat(np.arange(n), knn), nbr.ravel()] = 1.0
    A = np.maximum(A, A.T)                 # symmetrize
    d = A.sum(1).clip(1e-12)
    Dm = 1.0 / np.sqrt(d)
    L = np.eye(n) - (Dm[:, None] * A * Dm[None, :])
    vals = np.sort(np.linalg.eigvalsh(L))  # dense fine at n<=1500
    gaps = np.diff(vals[:50])
    K_hat = int(np.argmax(gaps) + 1)
    gamma = float(vals[true_K] - vals[true_K - 1]) if true_K < len(vals) else None
    return K_hat, gamma, [float(v) for v in vals[:12]]


# ----------------------------------------------------------------------------
# Arms (E0.2' protocol; new: rc_neg_only, naive_positive, view-noise foil)
# ----------------------------------------------------------------------------

class Student(torch.nn.Module):
    def __init__(self, d_in=16, h=64, d_out=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, h), torch.nn.ReLU(),
            torch.nn.Linear(h, d_out))

    def forward(self, x):
        z = self.net(x)
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def linear_probe(student, rung: Rung, rng, n_train=1000, n_test=2000,
                 steps=300):
    xtr, ytr = draw(rung, n_train, rng)
    xte, yte = draw(rung, n_test, rng)
    with torch.no_grad():
        ztr = student(torch.tensor(xtr, dtype=torch.float32))
        zte = student(torch.tensor(xte, dtype=torch.float32))
    Wp = torch.zeros(ztr.shape[1], rung.K, requires_grad=True)
    bp = torch.zeros(rung.K, requires_grad=True)
    opt = torch.optim.Adam([Wp, bp], lr=0.05)
    Ytr = torch.tensor(ytr)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(ztr @ Wp + bp, Ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float(((zte @ Wp + bp).argmax(1).numpy() == yte).mean())


def rc_neg_only_loss(z1, z2, mined: core.MinedBatch, rho_hat, rho_amb,
                     tau=TAU):
    """H-B': positive set = aug view ONLY; negative side identical to RC."""
    n = z1.shape[0]
    dev = z1.device
    v_neg = torch.tensor((1.0 - rho_hat(mined.s)) * mined.neg_mask,
                         dtype=torch.float32, device=dev)
    omega = torch.tensor((1.0 - rho_amb(mined.q)) * core.PINNED["c_amb"] *
                         mined.amb_mask, dtype=torch.float32, device=dev)
    sim = (z1 @ z2.T) / tau
    exps = torch.exp(sim - sim.max().detach())
    D = (v_neg * exps).sum(1) + (omega * exps).sum(1)
    log_frac = -torch.log1p(D / (exps.diagonal() + 1e-12))
    return -log_frac.mean()


def naive_positive_loss(z1, z2, s_teacher: np.ndarray, k: int = 2, tau=TAU):
    """FP-pull probe: top-k teacher-similar non-self as UNCERTIFIED positives
    (multi-positive InfoNCE over in-batch view-2 embeddings)."""
    n = z1.shape[0]
    s = s_teacher.copy()
    np.fill_diagonal(s, -np.inf)
    topk = np.argsort(-s, axis=1)[:, :k]
    sim = (z1 @ z2.T) / tau
    eye = torch.eye(n, dtype=torch.bool, device=z1.device)
    exps = torch.exp(sim - sim.max().detach())
    D = (exps * (~eye)).sum(1)
    log_frac = -torch.log1p((D[:, None] - exps) / (exps + 1e-12))
    pos_mask = torch.zeros(n, n, dtype=torch.bool)
    pos_mask[np.repeat(np.arange(n), k), topk.ravel()] = True
    terms = (log_frac * pos_mask).sum(1) / k + log_frac.diagonal()
    return -(terms / 2).mean()


def make_view_noise(rung: Rung, rate: float, mode: str):
    """Boundary-crossing view noise (prereg-d2 RINCE foil). Returns
    fn(x, lab, rng) -> (view2, crossed_mask)."""
    d_means = rung.means

    def crossing_prob(x):
        d = np.sqrt(((x[:, None, :] - d_means[None]) ** 2).sum(-1))
        d_sorted = np.sort(d, axis=1)
        margin = (d_sorted[:, 1] - d_sorted[:, 0]) / max(rung.sigma, 1e-9)
        if mode == "diffuse":
            return np.full(len(x), rate)
        raw = 1.0 / (1.0 + np.exp(margin))          # concentrated at boundary
        return raw * (rate / max(raw.mean(), 1e-9))

    def fn(x, lab, rng):
        p = np.clip(crossing_prob(x), 0, 1)
        cross = rng.random(len(x)) < p
        view = aug_view(x, rng, 0.25)
        if cross.any():
            d = np.sqrt(((x[cross][:, None, :] - d_means[None]) ** 2).sum(-1))
            d[np.arange(cross.sum()), lab[cross]] = np.inf
            other = d.argmin(1)
            view[cross] = (rung.means[other] +
                           rng.normal(0, rung.sigma, (cross.sum(), rung.dim)))
        return view, cross
    return fn


def train_arm(rung: Rung, arm: str, seed: int, alpha: float = ALPHA0,
              q_fn=None, calib=None, view_noise=None):
    """One arm, one seed. q_fn/calib passed in for RC arms (shared per rung).
    Returns dict with probe acc + telemetry. P7: RC abort -> un-mined."""
    torch.manual_seed(seed * 37 + 5)
    rng = np.random.default_rng(seed + 4000)
    eff = arm
    if arm in ("rc_hpm", "rc_neg_only", "cert_random_k") and \
            (calib is None or calib.aborted):
        eff = "no_mine"
    student = Student(d_in=rung.dim)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    abst = []
    import time as _t
    t0 = _t.time()
    for step in range(STEPS):
        xb, lb = draw(rung, N_BATCH, rng)
        x1 = aug_view(xb, rng, 0.25)
        if view_noise is None:
            x2 = aug_view(xb, rng, 0.25)
        else:
            x2, _ = view_noise(xb, lb, rng)
        z1 = student(torch.tensor(x1, dtype=torch.float32))
        z2 = student(torch.tensor(x2, dtype=torch.float32))
        eb = teacher_embed(rung, xb)
        if eff == "no_mine":
            loss = losses.plain_infonce(z1, z2, TAU)
        elif eff == "naive_neg":
            loss = losses.naive_hardmine_infonce(z1, z2, eb @ eb.T,
                                                 k=K_MINUS, tau=TAU)
        elif eff == "naive_pos":
            loss = naive_positive_loss(z1, z2, eb @ eb.T, k=K_PLUS, tau=TAU)
        elif eff == "rince":
            loss = losses.rince(z1, z2, q_exp=0.5, lam=0.025, tau=TAU)
        elif eff == "supcon":
            loss = losses.supcon(z1, z2, lb, TAU)
        else:
            qb = core.q_matrix(q_fn, eb, lb)
            ab = teacher_embed(rung, aug_view(xb, rng, 0.25))
            s_aug = (eb * ab).sum(1)
            mined = core.mine_batch(eb, qb, calib.lam[0], calib.lam[1],
                                    calib.rho_hat, calib.rho_plus,
                                    K_PLUS, K_MINUS)
            abst.append(mined.abstention)
            if eff == "rc_hpm":
                loss = losses.rc_hpm_loss(z1, z2, mined, calib.rho_hat,
                                          calib.rho_plus, calib.rho_amb,
                                          s_aug, TAU)
            elif eff == "rc_neg_only":
                loss = rc_neg_only_loss(z1, z2, mined, calib.rho_hat,
                                        calib.rho_amb, TAU)
            else:                                    # cert_random_k
                loss = losses.certified_random_k(
                    z1, z2, qb, calib.lam[0], calib.lam[1], calib.rho_hat,
                    calib.rho_plus, calib.rho_amb, eb @ eb.T, s_aug,
                    K_PLUS, K_MINUS, rng, TAU)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = linear_probe(student, rung, rng)
    return dict(arm=arm, arm_effective=eff, seed=seed, alpha=alpha,
                probe_acc=acc, abstention=float(np.mean(abst)) if abst else None,
                wall=round(_t.time() - t0, 1))
