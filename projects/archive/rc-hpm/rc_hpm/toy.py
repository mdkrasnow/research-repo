"""Synthetic Gaussian-mixture contrastive toy: population, teacher, gates,
end-to-end pipeline with E0.0 injection hooks, true-risk evaluation.

Population access (synthetic) means realized risk is evaluated on FRESH draws
from the true law — the cleanest instantiation of the guarantee's "fresh batch".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import core


# ----------------------------------------------------------------------------
# Population + teacher
# ----------------------------------------------------------------------------

@dataclass
class ToyConfig:
    K: int = 10
    dim: int = 16
    mean_scale: float = 2.0
    sigma_cluster: float = 1.0
    sigma_aug: float = 0.25
    n_gate: int = 2000
    n_fold: int = 32000         # per calibration fold (fit+test halves), A1
    n_batch: int = 64
    m: int = 250                # disjoint test batches (Amendment A1)
    m_fit: int = 40             # pool batches for path/throughput on D_fit
    k_plus: int = 2
    k_minus: int = 8
    n_pairs: int = 20000        # rho-fit pairs (small in adversarial scenarios)
    delta_r: float = 0.05
    eval_batches: int = 200
    grid_beta_plus: tuple = (0.55, 0.65, 0.75, 0.85, 0.92, 0.97)
    grid_beta_minus: tuple = (0.03, 0.08, 0.15, 0.25, 0.35, 0.45)


def lam_grid(cfg: ToyConfig) -> list[tuple[float, float]]:
    return [(bp, bm) for bp in cfg.grid_beta_plus for bm in cfg.grid_beta_minus
            if bm < bp]


@dataclass
class Population:
    means: np.ndarray
    sigma: float
    W: np.ndarray               # frozen teacher linear map
    rng: np.random.Generator = field(repr=False, default=None)


def make_population(seed: int, cfg: ToyConfig) -> Population:
    rng = np.random.default_rng(seed)
    means = rng.normal(0, cfg.mean_scale, (cfg.K, cfg.dim))
    W = rng.normal(0, 1.0 / np.sqrt(cfg.dim), (cfg.dim, cfg.dim))
    return Population(means, cfg.sigma_cluster, W, rng)


def draw(pop: Population, n: int, rng: np.random.Generator):
    labels = rng.integers(0, pop.means.shape[0], n)
    x = pop.means[labels] + rng.normal(0, pop.sigma, (n, pop.means.shape[1]))
    return x, labels


def teacher_embed(pop: Population, x: np.ndarray, noise: float = 0.0,
                  rng: np.random.Generator | None = None,
                  drift: np.ndarray | None = None) -> np.ndarray:
    e = x @ pop.W
    if noise > 0:
        e = e + rng.normal(0, noise, e.shape)
    if drift is not None:        # live-student injection: non-isometric drift
        e = e + e @ drift
    return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-12)


def aug_view(x: np.ndarray, rng: np.random.Generator,
             sigma_aug: float) -> np.ndarray:
    return x + rng.normal(0, sigma_aug, x.shape)


# ----------------------------------------------------------------------------
# Gates: learned bilinear / oracle / random  — q_fn(e1, e2, l1, l2)
# ----------------------------------------------------------------------------

def train_gate(emb: np.ndarray, labels: np.ndarray, seed: int,
               n_pairs: int = 20000, steps: int = 400):
    """Bilinear logistic gate, symmetrized: q = sigmoid((e A e' + e' A e)/2 + b)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    n, d = emb.shape
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    E1 = torch.tensor(emb[i], dtype=torch.float32)
    E2 = torch.tensor(emb[j], dtype=torch.float32)
    Y = torch.tensor((labels[i] == labels[j]).astype(np.float32))
    A = torch.zeros(d, d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([A, b], lr=0.05)
    for _ in range(steps):
        logits = ((E1 @ A) * E2).sum(1) * 0.5 + ((E2 @ A) * E1).sum(1) * 0.5 + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    A_np = A.detach().numpy()
    A_sym = 0.5 * (A_np + A_np.T)
    b_np = float(b.detach())

    def q_fn(e1, e2, l1=None, l2=None):
        logit = np.einsum("nd,de,ne->n", np.atleast_2d(e1), A_sym,
                          np.atleast_2d(e2)) + b_np
        return 1.0 / (1.0 + np.exp(-logit))
    return q_fn


def oracle_q_fn(e1, e2, l1, l2):
    return np.where(np.asarray(l1) == np.asarray(l2), 0.98, 0.02)


def make_random_q_fn(seed: int, dim: int):
    r = np.random.default_rng(seed + 7).normal(0, 1, dim)

    def q_fn(e1, e2, l1=None, l2=None):
        # symmetric, deterministic, label-uninformative
        z = (np.atleast_2d(e1) + np.atleast_2d(e2)) @ r
        return 1.0 / (1.0 + np.exp(-3.0 * z))
    return q_fn


# ----------------------------------------------------------------------------
# End-to-end pipeline (one seed) with E0.0 injection hooks
# ----------------------------------------------------------------------------

@dataclass
class Injections:
    fold_reuse: bool = False
    a1prime_noise: float = 0.0       # extra teacher noise at deployment only
    partition_bentkus: bool = False  # finite-pool partition but keep Bentkus
    wprime_mismatch: bool = False    # change pinned weights post-calibration
    skip_degenerate: bool = False    # remove L:=0 convention (NaN path)
    live_student_drift: float = 0.0  # non-isometric drift strength at deploy
    gamma_pooled: bool = False       # calibrate gamma-law A, deploy law B
    overlapping_batches: bool = False


@dataclass
class SeedResult:
    aborted: bool
    abstention: float = 1.0
    realized_minus: float = 0.0
    realized_plus: float = 0.0
    se_minus: float = 0.0
    se_plus: float = 0.0
    mined_any: bool = False
    checker_error: str | None = None
    monitor_alarms: int = 0
    lam: tuple | None = None
    true_risk_minus: float = 0.0     # high-precision (for E0.0 true-risk tests)
    true_risk_plus: float = 0.0


def run_seed(seed: int, cfg: ToyConfig, alpha: float, gate_kind: str = "learned",
             inj: Injections | None = None,
             eval_batches: int | None = None,
             pop_seed: int | None = None) -> SeedResult:
    inj = inj or Injections()
    rng = np.random.default_rng(seed + 1000)
    pop = make_population(seed if pop_seed is None else pop_seed, cfg)

    # ---- gate ----
    xg, lg = draw(pop, cfg.n_gate, rng)
    eg = teacher_embed(pop, xg)
    if gate_kind == "learned":
        q_fn = train_gate(eg, lg, seed)
    elif gate_kind == "oracle":
        q_fn = oracle_q_fn
    elif gate_kind == "random":
        q_fn = make_random_q_fn(seed, cfg.dim)
    else:
        raise ValueError(gate_kind)

    # ---- calibration fold ----
    xc, lc = draw(pop, cfg.n_fold, rng)
    gamma_calib = None
    if inj.gamma_pooled:   # gamma-law A at calibration: mostly clean
        gamma_calib = rng.beta(8, 2, cfg.n_fold)
        eps = rng.normal(0, 1, xc.shape)
        xc = gamma_calib[:, None] * xc + (1 - gamma_calib[:, None]) * eps
    ec = teacher_embed(pop, xc)
    ac = teacher_embed(pop, aug_view(xc, rng, cfg.sigma_aug))

    batch_mode = "partition_finite_pool" if inj.partition_bentkus else "disjoint"
    try:
        calib = core.calibrate_ltt(
            ec, lc, ac, q_fn, lam_grid(cfg), alpha, alpha, cfg.delta_r,
            cfg.n_batch, cfg.m, rng, cfg.k_plus, cfg.k_minus,
            batch_mode=batch_mode, use_bentkus=True,
            inject_fold_reuse=inj.fold_reuse, m_fit=cfg.m_fit,
            n_pairs=cfg.n_pairs)
    except core.ValidityError as e:
        return SeedResult(aborted=True, checker_error=str(e))

    if calib.aborted:
        return SeedResult(aborted=True, lam=None)

    # ---- deployment / realized risk on fresh population draws ----
    if inj.wprime_mismatch:
        deploy_hash = "tampered" + core.pinned_hash()[:8]
    else:
        deploy_hash = core.pinned_hash()
    try:
        core.check_pinned(deploy_hash, calib.pinned_hash)
    except core.ValidityError as e:
        return SeedResult(aborted=True, checker_error=str(e), lam=calib.lam)

    E = eval_batches or cfg.eval_batches
    drift_mat = None
    monitor = core.DriftMonitor(calib.F_ref, calib.a_ref, n_checks_planned=10)
    Lm_all, Lp_all, sel_all, abst_all = [], [], [], []
    window_scores, window_abst = [], []
    gamma_train = None
    if inj.gamma_pooled:
        gamma_train = []
    for t in range(E):
        xb, lb = draw(pop, cfg.n_batch, rng)
        if inj.gamma_pooled:   # gamma-law B at deployment: mostly noisy
            g = rng.beta(2, 8, cfg.n_batch)
            eps = rng.normal(0, 1, xb.shape)
            xb = g[:, None] * xb + (1 - g[:, None]) * eps
            gamma_train.append(g)
        if inj.live_student_drift > 0:
            # drift grows over deployment — theta-dependent mining simulation
            P = np.random.default_rng(seed + 5).normal(
                0, 1, (cfg.dim, cfg.dim)) / np.sqrt(cfg.dim)
            drift_mat = P * (inj.live_student_drift * (t + 1) / E)
        eb = teacher_embed(pop, xb, noise=inj.a1prime_noise, rng=rng,
                           drift=drift_mat)
        ab = teacher_embed(pop, aug_view(xb, rng, cfg.sigma_aug),
                           noise=inj.a1prime_noise, rng=rng, drift=drift_mat)
        qb = core.q_matrix(q_fn, eb, lb)
        mined = core.mine_batch(eb, qb, calib.lam[0], calib.lam[1],
                                calib.rho_hat, calib.rho_plus,
                                cfg.k_plus, cfg.k_minus)
        y_same = (lb[:, None] == lb[None, :]).astype(float)
        s_aug = (eb * ab).sum(1)
        lm, lp = core.batch_risks(mined, y_same, calib.rho_hat, calib.rho_plus,
                                  calib.rho_amb, s_aug)
        if inj.skip_degenerate:
            # bug: 0/0 treated as nan instead of the L:=0 convention
            den_zero = (mined.neg_mask.sum() == 0 and mined.amb_mask.sum() == 0)
            if den_zero:
                lm = float("nan")
        Lm_all.append(lm); Lp_all.append(lp)
        if np.isfinite(mined.stats["s_mean"]):
            sel_all.append(mined.stats); window_scores.append(mined.stats)
        abst_all.append(mined.abstention); window_abst.append(mined.abstention)
        if (t + 1) % max(E // 8, 1) == 0:
            wdict = {k: np.array([d[k] for d in window_scores
                                  if np.isfinite(d[k])])
                     for k in ("s_mean", "q_mean", "count")}
            monitor.check(wdict, float(np.mean(window_abst)))
            window_scores, window_abst = [], []

    checker_error = None
    try:
        sdict = {k: np.array([d[k] for d in sel_all if np.isfinite(d[k])])
                 for k in ("s_mean", "q_mean", "count")}
        core.check_score_symmetry(calib.F_ref, sdict)
        if inj.gamma_pooled:
            core.check_gamma_law(gamma_calib, np.concatenate(gamma_train))
    except core.ValidityError as e:
        checker_error = str(e)

    Lm = np.array(Lm_all, dtype=float)
    Lp = np.array(Lp_all, dtype=float)
    if np.isnan(Lm).any() or np.isnan(Lp).any():
        # NaN guard — loud failure expected under skip_degenerate injection
        return SeedResult(aborted=True, checker_error="NaN in realized risk "
                          "(degenerate-batch convention violated)",
                          lam=calib.lam)
    mined_any = bool(len(sel_all) > 0)
    return SeedResult(
        aborted=False, abstention=float(np.mean(abst_all)),
        realized_minus=float(Lm.mean()), realized_plus=float(Lp.mean()),
        se_minus=float(Lm.std(ddof=1) / np.sqrt(len(Lm))),
        se_plus=float(Lp.std(ddof=1) / np.sqrt(len(Lp))),
        mined_any=mined_any, checker_error=checker_error,
        monitor_alarms=monitor.alarms, lam=calib.lam,
        true_risk_minus=float(Lm.mean()), true_risk_plus=float(Lp.mean()))
