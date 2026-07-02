"""RC-HPM statistical core: HB p-values, isotonic PAV, mining, risk
functionals, Learn-then-Test calibration, drift monitor, validity checkers.

All selection/calibration math is numpy and teacher-driven (theta-free).
Pinned functions per documentation/preregistration.md — changing any pinned
function invalidates downstream calibrations (hash-checked).

Conventions:
  s    teacher cosine similarity in [-1, 1] (embeddings unit-norm)
  q    gate score in (0, 1), symmetric in (i, j)
  y    pair label, 1 = same class ("false pair" for a negative,
       "true pair" for a positive)
  Risk L- : fraction of repulsive surrogate-gradient budget on same-class pairs
  Risk L+ : fraction of attractive surrogate-gradient budget on cross-class pairs
  Degenerate batch (zero budget): L := 0 (spec v1.4 convention)
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------
# Hoeffding-Bentkus p-values (LTT, Angelopoulos et al.)
# ----------------------------------------------------------------------------

def _kl_bernoulli(p: float, q: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    q = min(max(q, 1e-12), 1 - 1e-12)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def hb_pvalue(mu_hat: float, m: int, alpha: float, use_bentkus: bool = True) -> float:
    """p-value for H0: E[L] > alpha given mean mu_hat of m iid losses in [0,1].

    use_bentkus=False is the partition/finite-pool mode (Hoeffding-only);
    the checker enforces that pairing.
    """
    if not np.isfinite(mu_hat):
        raise ValueError("NaN/inf loss reached hb_pvalue — degenerate-batch "
                         "convention violated upstream")
    if mu_hat >= alpha:
        return 1.0
    p_h = math.exp(-m * _kl_bernoulli(min(mu_hat, alpha), alpha))
    if not use_bentkus:
        return min(1.0, p_h)
    p_b = math.e * stats.binom.cdf(math.ceil(m * mu_hat), m, alpha)
    return min(1.0, p_h, p_b)


# ----------------------------------------------------------------------------
# Isotonic regression (PAV) — own implementation (sklearn broken in env)
# ----------------------------------------------------------------------------

class Isotonic:
    """Monotone fit of E[y|x] via pool-adjacent-violators. increasing=True/False."""

    def __init__(self, increasing: bool = True):
        self.increasing = increasing
        self._x = None
        self._y = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Isotonic":
        order = np.argsort(x)
        xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
        if not self.increasing:
            ys = ys[::-1]
        # PAV: blocks of (sum, weight)
        vals, wts, lens = [], [], []
        for v in ys:
            vals.append(v); wts.append(1.0); lens.append(1)
            while len(vals) > 1 and (vals[-2] / wts[-2]) > (vals[-1] / wts[-1]):
                v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
                vals[-1] += v2; wts[-1] += w2; lens[-1] += l2
        fitted = np.concatenate([np.full(l, v / w) for v, w, l in zip(vals, wts, lens)])
        if not self.increasing:
            fitted = fitted[::-1]
        self._x = xs
        self._y = fitted
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self._x, np.asarray(x, float), side="right") - 1
        idx = np.clip(idx, 0, len(self._y) - 1)
        return self._y[idx]


# ----------------------------------------------------------------------------
# Pinned surrogate weights (preregistration.md — never tune after calibration)
# ----------------------------------------------------------------------------

PINNED = dict(tau=0.5, tau_w=0.5, c_amb=0.5, hardness_weight="1")


def pinned_hash() -> str:
    return hashlib.sha256(repr(sorted(PINNED.items())).encode()).hexdigest()[:16]


def w_repulsive(s: np.ndarray, tau: float = PINNED["tau"]) -> np.ndarray:
    return np.exp(s / tau)


def w_attractive(s: np.ndarray, tau_w: float = PINNED["tau_w"]) -> np.ndarray:
    return 1.0 - 1.0 / (1.0 + np.exp(-s / tau_w))   # 1 - sigmoid(s/tau_w)


# ----------------------------------------------------------------------------
# MINE_BATCH — pure, teacher-driven (theta appears nowhere)
# ----------------------------------------------------------------------------

@dataclass
class MinedBatch:
    s: np.ndarray          # (n, n) teacher similarity
    q: np.ndarray          # (n, n) gate score
    neg_mask: np.ndarray   # (n, n) bool — mined certified negatives N_i
    pos_mask: np.ndarray   # (n, n) bool — mined certified positives P_i
    amb_mask: np.ndarray   # (n, n) bool — ambiguous (soft, never hard)
    abstention: float      # |Amb| / (n^2 - n)
    sel_scores: np.ndarray # teacher similarities of selected pairs (monitor)
    throughput: float      # sum of selection utility over mined sets


def mine_batch(emb: np.ndarray, q: np.ndarray, beta_plus: float, beta_minus: float,
               rho_hat, rho_plus, k_plus: int, k_minus: int,
               tau: float = PINNED["tau"]) -> MinedBatch:
    n = emb.shape[0]
    s = emb @ emb.T
    np.fill_diagonal(s, -np.inf)  # exclude self
    cert_pos = (q >= beta_plus) & np.isfinite(s)
    cert_neg = (q <= beta_minus) & np.isfinite(s)
    amb = (~cert_pos) & (~cert_neg) & np.isfinite(s)

    s_safe = np.where(np.isfinite(s), s, 0.0)
    u_neg = np.exp(s_safe / tau) * (1.0 - rho_hat(s_safe))
    u_pos = np.exp(-s_safe / tau) * (1.0 - rho_plus(s_safe))

    def _topk_mask(util: np.ndarray, cert: np.ndarray, k: int) -> np.ndarray:
        u = np.where(cert, util, -np.inf)
        if k >= n:
            return cert.copy()
        thresh_idx = np.argpartition(-u, k - 1, axis=1)[:, :k]
        mask = np.zeros((n, n), bool)
        rows = np.repeat(np.arange(n), k)
        mask[rows, thresh_idx.ravel()] = True
        return mask & cert  # -inf placeholders never selected

    neg_mask = _topk_mask(u_neg, cert_neg, k_minus)
    pos_mask = _topk_mask(u_pos, cert_pos, k_plus)

    sel = neg_mask | pos_mask
    sel_scores = s_safe[sel]
    sel_q = q[sel]
    throughput = float(u_neg[neg_mask].sum() + u_pos[pos_mask].sum())
    abstention = float(amb.sum()) / max(n * n - n, 1)
    mb = MinedBatch(s_safe, q, neg_mask, pos_mask, amb, abstention,
                    sel_scores, throughput)
    mb.stats = dict(
        s_mean=float(sel_scores.mean()) if sel_scores.size else np.nan,
        q_mean=float(sel_q.mean()) if sel_q.size else np.nan,
        count=float(sel.sum()))
    return mb


# ----------------------------------------------------------------------------
# Risk functionals L-, L+ — single source of truth for calibration,
# realized-risk evaluation, AND the loss weights (spec v1.3/v1.4)
# ----------------------------------------------------------------------------

def batch_risks(mined: MinedBatch, y_same: np.ndarray,
                rho_hat, rho_plus, rho_amb,
                s_aug: np.ndarray | None = None) -> tuple[float, float]:
    """(L-, L+) for one batch given ground-truth pair labels y_same (n,n).

    s_aug: (n,) teacher similarity anchor<->own aug view; enters the L+
    denominator with numerator 0 (true positives by construction).
    """
    s, q = mined.s, mined.q
    w = w_repulsive(s)
    v_neg = 1.0 - rho_hat(s)                       # pinned: hardness_weight = 1
    omega = (1.0 - rho_amb(q)) * PINNED["c_amb"]

    num_m = (v_neg * w * y_same)[mined.neg_mask].sum() + \
            (omega * w * y_same)[mined.amb_mask].sum()
    den_m = (v_neg * w)[mined.neg_mask].sum() + (omega * w)[mined.amb_mask].sum()
    L_minus = float(num_m / den_m) if den_m > 0 else 0.0   # degenerate := 0

    wp = w_attractive(s)
    v_pos = 1.0 - rho_plus(s)
    num_p = (v_pos * wp * (1 - y_same))[mined.pos_mask].sum()
    den_p = (v_pos * wp)[mined.pos_mask].sum()
    if s_aug is not None:
        den_p += ((1.0 - rho_plus(s_aug)) * w_attractive(s_aug)).sum()
    L_plus = float(num_p / den_p) if den_p > 0 else 0.0
    return L_minus, L_plus


# ----------------------------------------------------------------------------
# Validity checkers (always on in clean runs; E0.0 verifies they fire)
# ----------------------------------------------------------------------------

class ValidityError(RuntimeError):
    pass


def check_disjoint_batches(batch_ids: list[np.ndarray], pool_mode: bool):
    if pool_mode:
        return  # P1(b): guarantee w.r.t. pool empirical dist, declared
    seen = set()
    for ids in batch_ids:
        ids_set = set(int(i) for i in ids)
        if seen & ids_set:
            raise ValidityError(
                "overlapping calibration batches while declared i.i.d. "
                "(P1 violation) — use pool_mode or enlarge the labeled pool")
        seen |= ids_set


def check_bentkus_mode(batch_mode: str, use_bentkus: bool):
    if batch_mode == "partition_finite_pool" and use_bentkus:
        raise ValidityError("Bentkus invalid under finite-pool partitioning — "
                            "drop Bentkus (Hoeffding-Serfling only)")


def check_pinned(deploy_hash: str, calib_hash: str):
    if deploy_hash != calib_hash:
        raise ValidityError("pinned surrogate-weight functions changed between "
                            "calibration and deployment (w' mismatch)")


def check_score_symmetry(calib_stats, train_stats, level: float = 1e-3):
    """A1' check: teacher must treat calibration and training draws
    symmetrically — KS per label-free batch statistic (batch means are the
    i.i.d. unit; pooled pair scores are dependent). Accepts dicts of
    per-batch arrays or bare arrays (legacy)."""
    if isinstance(calib_stats, np.ndarray):
        calib_stats = {"s_mean": calib_stats}
    if isinstance(train_stats, np.ndarray):
        train_stats = {"s_mean": train_stats}
    for k, ref in calib_stats.items():
        w = train_stats.get(k)
        if w is None or len(ref) < 30 or len(w) < 10:
            continue
        p = stats.ks_2samp(ref, w).pvalue
        if p < level / max(len(calib_stats), 1):
            raise ValidityError(f"A1' violated: calibration vs deployment "
                                f"'{k}' distributions differ (KS p={p:.2e})")


def check_gamma_law(calib_gamma: np.ndarray, train_gamma: np.ndarray,
                    level: float = 1e-3):
    if len(calib_gamma) < 50 or len(train_gamma) < 50:
        return
    p = stats.ks_2samp(calib_gamma, train_gamma).pvalue
    if p < level:
        raise ValidityError(f"gamma-sampling law changed between calibration and "
                            f"training (P4 violation, KS p={p:.2e})")


# ----------------------------------------------------------------------------
# CALIBRATE_LTT
# ----------------------------------------------------------------------------

@dataclass
class CalibResult:
    aborted: bool
    lam: tuple[float, float] | None       # (beta_plus, beta_minus)
    rho_hat: object = None
    rho_plus: object = None
    rho_amb: object = None
    F_ref: dict = field(default_factory=dict)  # per-batch monitor stats
    a_ref: float = 1.0
    n_valid: int = 0
    pinned_hash: str = ""
    diag: dict = field(default_factory=dict)


def fit_rhos(emb: np.ndarray, labels: np.ndarray, q_fn, rng: np.random.Generator,
             n_pairs: int = 20000):
    """Fit rho_hat (P(same|s), increasing), rho_plus (P(diff|s), decreasing),
    rho_amb (P(same|q), increasing) on D_fit ONLY."""
    n = emb.shape[0]
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    s = (emb[i] * emb[j]).sum(1)
    y = (labels[i] == labels[j]).astype(float)
    q = q_fn(emb[i], emb[j], labels[i], labels[j])
    iso_same = Isotonic(increasing=True).fit(s, y)
    rho_hat = lambda x: np.clip(iso_same.predict(np.asarray(x)), 0, 1)
    iso_diff = Isotonic(increasing=False).fit(s, 1 - y)
    rho_plus = lambda x: np.clip(iso_diff.predict(np.asarray(x)), 0, 1)
    iso_amb = Isotonic(increasing=True).fit(q, y)
    rho_amb = lambda x: np.clip(iso_amb.predict(np.asarray(x)), 0, 1)
    return rho_hat, rho_plus, rho_amb


def make_batches(n_total: int, n_batch: int, m: int, rng: np.random.Generator,
                 mode: str = "disjoint") -> list[np.ndarray]:
    """disjoint: i.i.d.-justified blocks (population sampling).
    pool: with-replacement across batches (P1(b) fallback)."""
    if mode == "disjoint":
        if m * n_batch > n_total:
            raise ValidityError(f"labeled pool too small for {m} disjoint "
                                f"batches of {n_batch} (P1 arithmetic)")
        perm = rng.permutation(n_total)
        return [perm[t * n_batch:(t + 1) * n_batch] for t in range(m)]
    return [rng.choice(n_total, n_batch, replace=False) for _ in range(m)]


def calibrate_ltt(emb: np.ndarray, labels: np.ndarray, aug_emb: np.ndarray,
                  q_fn, grid: list[tuple[float, float]],
                  alpha_plus: float, alpha_minus: float, delta_r: float,
                  n_batch: int, m: int, rng: np.random.Generator,
                  k_plus: int = 2, k_minus: int = 8,
                  batch_mode: str = "disjoint", use_bentkus: bool = True,
                  inject_fold_reuse: bool = False, m_fit: int = 40,
                  n_pairs: int = 20000) -> CalibResult:
    """Full LTT calibration on one fold. emb/labels/aug_emb = the fold's data.

    inject_fold_reuse (E0.0 bug #1): fit rhos + path on D_test instead of D_fit.
    """
    check_bentkus_mode(batch_mode, use_bentkus)
    n = emb.shape[0]
    perm = rng.permutation(n)
    fit_idx, test_idx = perm[: n // 2], perm[n // 2:]

    rho_src = test_idx if inject_fold_reuse else fit_idx
    # Data-flow guard: rho/path fitting must never touch the LTT test split.
    # (1-D isotonic fold reuse has near-zero realized-risk signature at toy
    # scale, so the structural violation is checked structurally.)
    if np.intersect1d(rho_src, test_idx).size > 0:
        raise ValidityError("fold reuse: rho/path fit data overlaps the LTT "
                            "test split — p-values invalid (spec v1.2 bug)")
    rho_hat, rho_plus, rho_amb = fit_rhos(emb[rho_src], labels[rho_src], q_fn,
                                          rng, n_pairs=n_pairs)

    def batch_losses(idx_pool_emb, idx_pool_lab, idx_pool_aug, batches, lam):
        bp, bm = lam
        Lm, Lp, sel_means, abst = [], [], [], []
        thr = []
        for ids in batches:
            e, lab, ae = idx_pool_emb[ids], idx_pool_lab[ids], idx_pool_aug[ids]
            q = q_matrix(q_fn, e, lab)
            mined = mine_batch(e, q, bp, bm, rho_hat, rho_plus, k_plus, k_minus)
            y_same = (lab[:, None] == lab[None, :]).astype(float)
            s_aug = (e * ae).sum(1)
            lm, lp = batch_risks(mined, y_same, rho_hat, rho_plus, rho_amb, s_aug)
            Lm.append(lm); Lp.append(lp)
            # per-batch monitor statistics: the i.i.d. unit is the BATCH
            # (pair-level scores are within-batch dependent -> KS invalid)
            sel_means.append(mined.stats)
            abst.append(mined.abstention)
            thr.append(mined.throughput)
        ref = {k: np.array([d[k] for d in sel_means
                            if np.isfinite(d[k])])
               for k in ("s_mean", "q_mean", "count")}
        return (np.array(Lm), np.array(Lp), ref,
                float(np.mean(abst)), float(np.mean(thr)))

    # --- path + throughput on D_fit (or injected D_test) ---
    path_src = test_idx if inject_fold_reuse else fit_idx
    fit_batches = make_batches(len(path_src), n_batch, m_fit, rng, "pool")
    fit_stats = {}
    for lam in grid:
        Lm, Lp, _, _, thr = batch_losses(emb[path_src], labels[path_src],
                                         aug_emb[path_src], fit_batches, lam)
        score = max(Lm.mean() / max(alpha_minus, 1e-9),
                    Lp.mean() / max(alpha_plus, 1e-9))
        fit_stats[lam] = (score, thr)
    path = sorted(grid, key=lambda l: fit_stats[l][0])

    # --- fixed-sequence test on D_test ---
    test_batches = make_batches(len(test_idx), n_batch, m, rng, batch_mode)
    check_disjoint_batches(test_batches, pool_mode=(batch_mode == "pool"))
    valid = []
    test_cache = {}
    for lam in path:
        Lm, Lp, sel, abst, _ = batch_losses(emb[test_idx], labels[test_idx],
                                            aug_emb[test_idx], test_batches, lam)
        p = max(hb_pvalue(Lm.mean(), m, alpha_minus, use_bentkus),
                hb_pvalue(Lp.mean(), m, alpha_plus, use_bentkus))
        if p <= delta_r:
            valid.append(lam)
            test_cache[lam] = (sel, abst)
        else:
            break  # fixed-sequence: stop at first failure

    if not valid:
        return CalibResult(aborted=True, lam=None, rho_hat=rho_hat,
                           rho_plus=rho_plus, rho_amb=rho_amb,
                           pinned_hash=pinned_hash(),
                           diag={"n_valid": 0, "path_head": path[:3]})
    lam_star = max(valid, key=lambda l: fit_stats[l][1])  # throughput on D_fit
    sel, abst = test_cache[lam_star]
    return CalibResult(aborted=False, lam=lam_star, rho_hat=rho_hat,
                       rho_plus=rho_plus, rho_amb=rho_amb, F_ref=sel,
                       a_ref=abst, n_valid=len(valid), pinned_hash=pinned_hash(),
                       diag={"throughput": fit_stats[lam_star][1]})


def q_matrix(q_fn, emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = emb.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return q_fn(emb[ii.ravel()], emb[jj.ravel()], labels[ii.ravel()],
                labels[jj.ravel()]).reshape(n, n)


# ----------------------------------------------------------------------------
# Drift monitor (Bonferroni across checks; per gamma-bin for EqM arms)
# ----------------------------------------------------------------------------

class DriftMonitor:
    """KS per label-free batch statistic (s_mean, q_mean, count) + relative
    abstention tripwire. Bonferroni across checks AND statistics."""

    def __init__(self, F_ref, a_ref: float, n_checks_planned: int,
                 delta_mon: float = 0.05, kappa_a: float = 0.05,
                 kappa_a_rel: float = 0.5, bonferroni: bool = True):
        if isinstance(F_ref, np.ndarray):                # legacy single-stat
            F_ref = {"s_mean": F_ref}
        self.F_ref, self.a_ref = F_ref, a_ref
        n_tests = max(n_checks_planned, 1) * max(len(F_ref), 1)
        self.level = (delta_mon / n_tests) if bonferroni else delta_mon
        self.kappa_a = kappa_a
        self.kappa_a_rel = kappa_a_rel
        self.alarms = 0
        self.checks = 0

    def check(self, window_stats, window_abstention: float) -> bool:
        # window_stats: dict of per-batch statistic arrays (i.i.d. batch units)
        if isinstance(window_stats, np.ndarray):
            window_stats = {"s_mean": window_stats}
        self.checks += 1
        alarm = False
        for k, ref in self.F_ref.items():
            w = window_stats.get(k)
            if w is None or len(ref) < 30 or len(w) < 8:
                continue
            if stats.ks_2samp(ref, w).pvalue < self.level:
                alarm = True
        gap = abs(window_abstention - self.a_ref)
        if gap > max(self.kappa_a, self.kappa_a_rel * self.a_ref):
            alarm = True
        self.alarms += int(alarm)
        return alarm
