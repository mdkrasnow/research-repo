"""2D EqM toy (paper Fig-1 setting) for E1.1 (ARM A disp-pair) and
E1.2 (ARM B endpoint-cert).

EqM objective exactly as transport.py: target = (x1 - x0) * c(t),
c = truncated decay (interp 0.8, start 1.0, x4) — get_ct port. Field f is
noise-unconditional; sampling x <- x + eta * f(x).

P4: gate and rho are gamma-conditional (3 bins); mined pairs restricted to
same-bin (cross-bin -> ambiguous by construction); calibration batches
replicate the exact training (x1, eps, t) law.
P5: ARM B mines via PGD against the FROZEN teacher field only; certification
descends the frozen teacher field from the training input x_t and checks the
attractor's analytic (Voronoi) basin against the source class. eta=0 labeler.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import core

# Amendment A3 (pre-run, 2026-06-12): the original 4-equal-mode layout was
# saturated (vanilla purity 1.0 at every radius/step count) — a positive
# control could not move the metric, violating the controls rule. Harder
# population: 8 modes on the r=1.5 circle with UNEQUAL weights (rare-mode
# coverage is the EqM-relevant failure axis); ARM B primary switched to the
# continuous mean nearest-mode distance (no ceiling).
_ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
MEANS = 1.5 * np.stack([np.cos(_ang), np.sin(_ang)], 1)
WEIGHTS = np.array([0.40, 0.20, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02])
SIGMA = 0.10
GAMMA_BINS = [(0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0001)]
# A4 (final): contrastive mining/loss/risk windowed to bins where the frozen
# teacher separates classes (measured: bin2 q|same=.82 q|diff=.05; bin1
# .36/.19 -> ambiguous omega-mass floor 0.19-0.22 > alpha at every lambda).
MINE_BINS = [2]


def get_ct(t: np.ndarray | torch.Tensor):
    """Exact port of transport.py get_ct (truncated decay, a=0.8, lambda=4)."""
    if isinstance(t, torch.Tensor):
        return torch.minimum(torch.ones_like(t), 5.0 - 5.0 * t) * 4.0
    return np.minimum(1.0, 5.0 - 5.0 * t) * 4.0


def draw_data(n: int, rng: np.random.Generator):
    labels = rng.choice(len(MEANS), n, p=WEIGHTS)
    return MEANS[labels] + rng.normal(0, SIGMA, (n, 2)), labels


def voronoi_basin(x: np.ndarray) -> np.ndarray:
    """Analytic basin label (equal isotropic Gaussians -> Voronoi). Exact."""
    d = ((x[:, None, :] - MEANS[None]) ** 2).sum(-1)
    return d.argmin(1)


def make_triplet(n: int, rng: np.random.Generator):
    """One training draw: (x1, label, eps, t, xt, target)."""
    x1, lab = draw_data(n, rng)
    eps = rng.normal(0, 1, (n, 2))
    t = rng.uniform(0, 1, n)
    xt = t[:, None] * x1 + (1 - t[:, None]) * eps
    target = (x1 - eps) * get_ct(t)[:, None]
    return x1, lab, eps, t, xt, target


# ----------------------------------------------------------------------------
# Field model
# ----------------------------------------------------------------------------

class Field(torch.nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.l1 = torch.nn.Linear(2, h)
        self.l2 = torch.nn.Linear(h, h)
        self.out = torch.nn.Linear(h, 2)

    def forward(self, x, return_act=False):
        a = torch.relu(self.l1(x))
        a = torch.relu(self.l2(a))
        y = self.out(a)
        return (y, a) if return_act else y


def gd_sample(field: Field, n: int, rng: np.random.Generator,
              eta=0.05, steps=400):
    x = torch.tensor(rng.normal(0, 1, (n, 2)), dtype=torch.float32)
    with torch.no_grad():
        for _ in range(steps):
            x = x + eta * field(x)
    return x.numpy()


def attractor_purity(samples: np.ndarray, radius=0.3) -> float:
    d = np.sqrt(((samples[:, None, :] - MEANS[None]) ** 2).sum(-1)).min(1)
    return float((d <= radius).mean())


def mean_mode_distance(samples: np.ndarray) -> float:
    """Mean distance to nearest true mode (precision axis; descriptive)."""
    d = np.sqrt(((samples[:, None, :] - MEANS[None]) ** 2).sum(-1)).min(1)
    return float(d.mean())


def mode_recall_distance(samples: np.ndarray) -> float:
    """ARM B PRIMARY (Amendment A3): weight-summed distance from each true
    mode to its nearest sample. Captures rare-mode dropping continuously;
    lower is better; no ceiling."""
    d = np.sqrt(((samples[:, None, :] - MEANS[None]) ** 2).sum(-1))  # (n, K)
    return float((WEIGHTS * d.min(0)).sum())


def mode_coverage(samples: np.ndarray, radius=0.3) -> float:
    d = np.sqrt(((samples[:, None, :] - MEANS[None]) ** 2).sum(-1))
    hit = (d.min(0) <= radius)
    return float(hit.mean())


# ----------------------------------------------------------------------------
# MC reference field (population-level, fixed seed) — primary metric ARM A
# ----------------------------------------------------------------------------

def reference_field(n_draws=2_000_000, lo=-3.0, hi=3.0, cells=40, seed=123):
    """f*(x) = E[target | x_t = x] pooled over the (x1, eps, t) law, estimated
    by cell-averaging (cell width = bandwidth 0.15). Returns (grid_xy, f_star,
    counts). Cells with <20 draws are masked (NaN)."""
    rng = np.random.default_rng(seed)
    _, _, _, _, xt, target = make_triplet(n_draws, rng)
    w = (hi - lo) / cells
    ix = np.floor((xt[:, 0] - lo) / w).astype(int)
    iy = np.floor((xt[:, 1] - lo) / w).astype(int)
    ok = (ix >= 0) & (ix < cells) & (iy >= 0) & (iy < cells)
    flat = ix[ok] * cells + iy[ok]
    cnt = np.bincount(flat, minlength=cells * cells)
    fx = np.bincount(flat, weights=target[ok, 0], minlength=cells * cells)
    fy = np.bincount(flat, weights=target[ok, 1], minlength=cells * cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_star = np.stack([fx / cnt, fy / cnt], 1)
    f_star[cnt < 20] = np.nan
    centers = lo + w * (np.arange(cells) + 0.5)
    gx, gy = np.meshgrid(centers, centers, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    return grid, f_star, cnt


def field_mse(field: Field, grid: np.ndarray, f_star: np.ndarray) -> float:
    with torch.no_grad():
        pred = field(torch.tensor(grid, dtype=torch.float32)).numpy()
    mask = np.isfinite(f_star[:, 0])
    return float(((pred[mask] - f_star[mask]) ** 2).mean())


# ----------------------------------------------------------------------------
# ARM A: gamma-binned teacher features + gate + per-bin rho + mining
# ----------------------------------------------------------------------------

class RFFTeacher:
    """Frozen random-Fourier-feature embedding of x_t (class-informative at
    high gamma, noise-dominated at low gamma — the P4 structure)."""

    def __init__(self, seed=7, d_out=16, scale=3.0):
        # scale=3.0 from the pre-treatment bandwidth sweep (A4 measurement:
        # bin-2 risk floor 0.072 at 3.0 vs 0.091 at 1.5)
        rng = np.random.default_rng(seed)
        self.Om = rng.normal(0, scale, (2, d_out))
        self.b = rng.uniform(0, 2 * np.pi, d_out)

    def __call__(self, xt: np.ndarray) -> np.ndarray:
        e = np.cos(xt @ self.Om + self.b)
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-12)


def gamma_bin(t: np.ndarray) -> np.ndarray:
    out = np.zeros(len(t), int)
    for b, (lo, hi) in enumerate(GAMMA_BINS):
        out[(t >= lo) & (t < hi)] = b
    return out


def train_binned_gate(teacher: RFFTeacher, seed: int, rng: np.random.Generator,
                      n=4000, n_pairs=8000, steps=400):
    """Per-bin symmetric bilinear gates on teacher features."""
    _, lab, _, t, xt, _ = make_triplet(n, rng)
    emb = teacher(xt)
    bins = gamma_bin(t)
    gates = []
    for b in range(len(GAMMA_BINS)):
        idx = np.where(bins == b)[0]
        sub_emb, sub_lab = emb[idx], lab[idx]
        torch.manual_seed(seed * 31 + b)
        prng = np.random.default_rng(seed * 17 + b)
        i = prng.integers(0, len(idx), n_pairs)
        j = prng.integers(0, len(idx), n_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        E1 = torch.tensor(sub_emb[i], dtype=torch.float32)
        E2 = torch.tensor(sub_emb[j], dtype=torch.float32)
        Y = torch.tensor((sub_lab[i] == sub_lab[j]).astype(np.float32))
        d = sub_emb.shape[1]
        A = torch.zeros(d, d, requires_grad=True)
        c = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([A, c], lr=0.05)
        for _ in range(steps):
            lo = ((E1 @ A) * E2).sum(1) * 0.5 + ((E2 @ A) * E1).sum(1) * 0.5 + c
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, Y)
            opt.zero_grad(); loss.backward(); opt.step()
        An = A.detach().numpy(); An = 0.5 * (An + An.T)
        gates.append((An, float(c.detach())))

    def q_fn_binned(emb_b: np.ndarray, b: int) -> np.ndarray:
        An, cn = gates[b]
        n_ = emb_b.shape[0]
        ii, jj = np.meshgrid(np.arange(n_), np.arange(n_), indexing="ij")
        logit = np.einsum("nd,de,ne->n", emb_b[ii.ravel()], An,
                          emb_b[jj.ravel()]) + cn
        return (1.0 / (1.0 + np.exp(-logit))).reshape(n_, n_)
    return q_fn_binned


@dataclass
class BinnedCalib:
    aborted: bool
    lam: tuple | None = None
    rhos: list | None = None          # per bin: (rho_hat, rho_plus, rho_amb)
    diag: dict | None = None


def fit_bin_rhos(teacher, q_fn_binned, rng, n=6000):
    _, lab, _, t, xt, _ = make_triplet(n, rng)
    emb, bins = teacher(xt), gamma_bin(t)
    rhos = []
    for b in range(len(GAMMA_BINS)):
        idx = np.where(bins == b)[0]
        e, l_ = emb[idx], lab[idx]
        prng = np.random.default_rng(int(rng.integers(1 << 30)))
        i = prng.integers(0, len(idx), 8000)
        j = prng.integers(0, len(idx), 8000)
        keep = i != j
        i, j = i[keep], j[keep]
        s = (e[i] * e[j]).sum(1)
        y = (l_[i] == l_[j]).astype(float)
        iso_same = core.Isotonic(True).fit(s, y)
        iso_diff = core.Isotonic(False).fit(s, 1 - y)
        qm = _pair_q(q_fn_binned, e, i, j, b)   # rho_amb on gate scores
        iso_amb = core.Isotonic(True).fit(qm, y)
        rhos.append((
            lambda x, iso=iso_same: np.clip(iso.predict(np.asarray(x)), 0, 1),
            lambda x, iso=iso_diff: np.clip(iso.predict(np.asarray(x)), 0, 1),
            lambda x, iso=iso_amb: np.clip(iso.predict(np.asarray(x)), 0, 1)))
    return rhos


def _pair_q(q_fn_binned, emb, i, j, b):
    """Pair gate scores without building the full matrix."""
    # build tiny matrices in chunks of pairs via 2-row trick
    out = np.empty(len(i))
    chunk = 2048
    for s0 in range(0, len(i), chunk):
        ii, jj = i[s0:s0 + chunk], j[s0:s0 + chunk]
        # 2-point matrices are wasteful; use direct bilinear via stored gates —
        # q_fn_binned only exposes the matrix form, so evaluate on the pair set
        # by stacking unique points per mini-chunk
        pts = np.unique(np.concatenate([ii, jj]))
        remap = {p: k for k, p in enumerate(pts)}
        Q = q_fn_binned(emb[pts], b)
        out[s0:s0 + chunk] = Q[[remap[p] for p in ii], [remap[p] for p in jj]]
    return out


def mine_binned(emb, t, lab_or_none, q_fn_binned, lam, rhos, k_plus, k_minus):
    """Per-bin mining; returns merged masks aligned to the full batch plus
    per-pair (s, q, bin) matrices. Cross-bin pairs are ambiguous-by-construction
    but EXCLUDED from the loss (omega=0 cross-bin) to keep per-bin calibration
    clean — documented restriction."""
    n = emb.shape[0]
    bins = gamma_bin(t)
    s_full = emb @ emb.T
    q_full = np.zeros((n, n))
    neg = np.zeros((n, n), bool)
    pos = np.zeros((n, n), bool)
    amb = np.zeros((n, n), bool)
    sel_means = []
    for b in MINE_BINS:                   # A4: gamma-window — gate separates
        idx = np.where(bins == b)[0]      # only near the manifold
        if len(idx) < 3:
            continue
        e_b = emb[idx]
        q_b = q_fn_binned(e_b, b)
        rho_hat, rho_plus, _ = rhos[b]
        mined = core.mine_batch(e_b, q_b, lam[0], lam[1], rho_hat, rho_plus,
                                k_plus, k_minus)
        ix = np.ix_(idx, idx)
        q_full[ix] = q_b
        neg[ix] = mined.neg_mask
        pos[ix] = mined.pos_mask
        amb[ix] = mined.amb_mask
        if mined.sel_scores.size:
            sel_means.append(float(mined.sel_scores.mean()))
    return s_full, q_full, neg, pos, amb, bins, sel_means


def binned_batch_risks(s, q, neg, pos, amb, bins, y_same, rhos, s_aug):
    """L-, L+ pooled across bins (marginal over the gamma law, P4)."""
    num_m = den_m = num_p = den_p = 0.0
    w = core.w_repulsive(s)
    wp = core.w_attractive(s)
    for b in MINE_BINS:                   # A4: gamma-window matches mining
        rho_hat, rho_plus, rho_amb = rhos[b]
        in_b = (bins == b)
        blk = np.ix_(np.where(in_b)[0], np.where(in_b)[0])
        v_neg = 1.0 - rho_hat(s[blk])
        omega = (1.0 - rho_amb(q[blk])) * core.PINNED["c_amb"]
        v_pos = 1.0 - rho_plus(s[blk])
        nb, pb, ab_ = neg[blk], pos[blk], amb[blk]
        ys = y_same[blk]
        num_m += (v_neg * w[blk] * ys)[nb].sum() + (omega * w[blk] * ys)[ab_].sum()
        den_m += (v_neg * w[blk])[nb].sum() + (omega * w[blk])[ab_].sum()
        num_p += (v_pos * wp[blk] * (1 - ys))[pb].sum()
        den_p += (v_pos * wp[blk])[pb].sum()
        # aug views: same-bin by construction
        rp_aug = 1.0 - rho_plus(s_aug[in_b])
        den_p += (rp_aug * core.w_attractive(s_aug[in_b])).sum()
    Lm = num_m / den_m if den_m > 0 else 0.0
    Lp = num_p / den_p if den_p > 0 else 0.0
    return float(Lm), float(Lp)


def calibrate_arm_a(teacher, q_fn_binned, grid, alpha, delta_r, rng,
                    n_batch=128, m=250, m_fit=40, k_plus=2, k_minus=8):
    """LTT for ARM A: calibration batches replicate the exact training law."""
    rhos = fit_bin_rhos(teacher, q_fn_binned, rng)        # D_fit role
    def losses_for(lam, m_):
        Lm, Lp = [], []
        for _ in range(m_):
            x1, lab, eps, t, xt, _ = make_triplet(n_batch, rng)
            emb = teacher(xt)
            s, q, neg, pos, amb, bins, _ = mine_binned(
                emb, t, lab, q_fn_binned, lam, rhos, k_plus, k_minus)
            y_same = (lab[:, None] == lab[None, :]).astype(float)
            # aug view: same x1 and t, fresh eps
            eps2 = rng.normal(0, 1, (n_batch, 2))
            xt2 = t[:, None] * x1 + (1 - t[:, None]) * eps2
            s_aug = (emb * teacher(xt2)).sum(1)
            lm, lp = binned_batch_risks(s, q, neg, pos, amb, bins, y_same,
                                        rhos, s_aug)
            Lm.append(lm); Lp.append(lp)
        return np.array(Lm), np.array(Lp)

    fit_stats = {}
    for lam in grid:
        Lm, Lp = losses_for(lam, m_fit)
        fit_stats[lam] = max(Lm.mean() / alpha, Lp.mean() / alpha)
    path = sorted(grid, key=lambda l: fit_stats[l])
    valid = []
    for lam in path:
        Lm, Lp = losses_for(lam, m)
        p = max(core.hb_pvalue(Lm.mean(), m, alpha),
                core.hb_pvalue(Lp.mean(), m, alpha))
        if p <= delta_r:
            valid.append(lam)
        else:
            break
    if not valid:
        return BinnedCalib(aborted=True, rhos=rhos)
    lam = valid[-1] if len(valid) else None   # most lenient certified on path
    # throughput proxy: prefer the LAST certified on the fit-ordered path
    return BinnedCalib(aborted=False, lam=lam, rhos=rhos,
                       diag={"n_valid": len(valid)})


# ----------------------------------------------------------------------------
# ARM B: frozen-teacher PGD endpoint mining + basin certification (P5)
# ----------------------------------------------------------------------------

def pgd_mine(teacher_field: Field, x1, eps, t, steps=3, rel_step=0.15,
             eps_ball=0.5, live_field: Field | None = None):
    """PGD ascent on the EqM residual w.r.t. the endpoint eps, against the
    FROZEN teacher field (P5). live_field overrides (the FORBIDDEN theta-
    dependent variant — negative-control arm only)."""
    net = live_field if live_field is not None else teacher_field
    x1_t = torch.tensor(x1, dtype=torch.float32)
    t_t = torch.tensor(t, dtype=torch.float32)[:, None]
    ct = get_ct(t_t)
    e0 = torch.tensor(eps, dtype=torch.float32)
    norm0 = e0.norm(dim=1, keepdim=True).clamp_min(1e-12)
    delta = torch.zeros_like(e0)
    for _ in range(steps):
        ep = (e0 + delta).detach().requires_grad_(True)
        xt = t_t * x1_t + (1 - t_t) * ep
        target = (x1_t - ep) * ct
        res = ((net(xt) - target) ** 2).sum()
        g = torch.autograd.grad(res, ep)[0]
        g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        delta = (delta + rel_step * norm0 * g).detach()
        dn = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
        delta = delta * (eps_ball * norm0 / dn).clamp(max=1.0)
    return (e0 + delta).detach().numpy()


def basin_certify(teacher_field: Field, x1, lab, eps_adv, t,
                  eta=0.05, steps=200):
    """Descend the FROZEN teacher field from the training input x_t; certified
    iff the attractor's Voronoi basin == source class (analytic labeler)."""
    xt = t[:, None] * x1 + (1 - t[:, None]) * eps_adv
    x = torch.tensor(xt, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(steps):
            x = x + eta * teacher_field(x)
    return voronoi_basin(x.numpy()) == lab


def calibrate_arm_b(teacher_field: Field, eps_grid, alpha, delta_r, rng,
                    n_batch=128, m=250, m_fit=30, pgd_steps=3, rel_step=0.15):
    """LTT over eps_ball: risk = wrong-basin fraction among ACCEPTED mined
    pairs (w=1 pinned). Throughput = mean accepted displacement."""
    def losses_for(eb, m_):
        # A4 (final): FLIP risk — fraction of accepted mined pairs (t >= 2/3,
        # manifold-adjacent) whose UN-MINED counterpart sat in the correct
        # Voronoi cell but whose mined x_t does not. Pure mining-induced
        # damage: vanilla floor is 0 by construction; absolute wrong-basin
        # rate has a geometry-driven floor (8 modes spaced 1.15 apart) that
        # mining cannot control.
        L, disp = [], []
        for _ in range(m_):
            x1, lab, eps, t, _, _ = make_triplet(n_batch, rng)
            adv = pgd_mine(teacher_field, x1, eps, t, pgd_steps, rel_step, eb)
            cert = basin_certify(teacher_field, x1, lab, adv, t)
            sel = cert & (t >= 2 / 3)
            if sel.sum() == 0:
                L.append(0.0)                      # degenerate := 0
                disp.append(0.0)
                continue
            xt_adv = t[:, None] * x1 + (1 - t[:, None]) * adv
            xt_orig = t[:, None] * x1 + (1 - t[:, None]) * eps
            flip = (voronoi_basin(xt_adv) != lab) & (voronoi_basin(xt_orig) == lab)
            L.append(float(flip[sel].mean()))
            disp.append(float(np.linalg.norm(adv - eps, axis=1).mean()))
        return np.array(L), float(np.mean(disp))

    fit = {eb: losses_for(eb, m_fit) for eb in eps_grid}
    path = sorted(eps_grid, key=lambda e: fit[e][0].mean() / alpha)
    valid = []
    for eb in path:
        L, _ = losses_for(eb, m)
        if core.hb_pvalue(L.mean(), m, alpha) <= delta_r:
            valid.append(eb)
        else:
            break
    if not valid:
        return None
    return max(valid, key=lambda e: fit[e][1])      # max displacement throughput
