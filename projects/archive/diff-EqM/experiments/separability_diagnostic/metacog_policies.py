"""Metacognition POLICIES for EqM B/2 inference-time method improvement.

Pure-numpy policy logic (CPU-testable; no torch). The GPU engine
(run_metacog_policy_sweep.py) calls these to decide, at trajectory read points,
which of R draws to keep (selection engine) or what mid-descent action to take
(segmented engine). NEVER touches image pixels and NEVER uses test-set quality
labels at test time — the only labels used are the cached good/garbage DEV pool
used to fit the (frozen) probe + stacked ranker offline.

Two engine families (see README_METACOG.md):
  * selection : run R full draws (250 steps each) + read partial traj features;
                keep argmin policy.score(...). NFE/img = R*steps EXACTLY (paired,
                identical to pareto baselines by construction).
  * segmented : run lanes in segments; at reads apply per-lane actions
                {continue, restart, churn, eta_scale, heun, drop}. Engine counts
                every model() eval -> exact measured NFE/img (may differ from the
                750 target -> flagged, not silently matched).

Feature vector available at a read step k (per draw/lane), all from (norm,dot)
curves of the descent — probe@k risk, energy_path Σ‖f‖, gradnorm ‖f‖_end, norm
slope, dot slope, oscillation, decay, step. Magnitude features are kept SEPARATE
from the shape probe so the de-confounding story (shape>magnitude) is auditable.
"""
import json
from pathlib import Path

import numpy as np

from probe_validate import feature_groups  # shared shape-feature builder


# ----------------------------------------------------------------------------- #
# feature extraction from a partial descent (norm[:, :k], dot[:, :k])
# ----------------------------------------------------------------------------- #
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def probe_risk(norm_k, dot_k, art):
    """Frozen shape probe -> P(garbage) in [0,1]. art = {w,b,mu,sd} for this k."""
    X = feature_groups(norm_k, dot_k)["ALL-shape"]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return _sigmoid(((X - art["mu"]) / art["sd"]) @ art["w"] + float(art["b"]))


def scalar_features(norm_k, dot_k):
    """Cheap per-draw scalars at decision step k (magnitude + shape summaries)."""
    eps = 1e-8
    k = norm_k.shape[1]
    energy_path = norm_k.sum(1)                      # Σ‖f‖  (magnitude)
    gradnorm = norm_k[:, -1]                         # ‖f‖ at k (magnitude)
    nn = norm_k / (norm_k[:, :1] + eps)              # magnitude-normalized shape
    steps = np.arange(k, dtype=np.float64)
    logn = np.log(norm_k + eps)
    sl = ((steps - steps.mean()) @ (logn - logn.mean(1, keepdims=True)).T) \
        / max(eps, ((steps - steps.mean()) ** 2).sum())
    dnorm = np.diff(norm_k, axis=1)
    osc = (np.sign(dnorm[:, 1:]) != np.sign(dnorm[:, :-1])).mean(1) if k > 2 else np.zeros(len(norm_k))
    q = max(1, k // 4)
    decay = nn[:, :q].mean(1) - nn[:, -q:].mean(1)   # how much normalized norm fell
    ddot = np.diff(dot_k, axis=1)
    dot_slope = ddot.mean(1) if k > 1 else np.zeros(len(norm_k))
    return {"energy_path": energy_path, "gradnorm": gradnorm, "norm_slope": sl,
            "dot_slope": dot_slope, "oscillation": osc, "decay": decay,
            "step": np.full(len(norm_k), float(k))}


# ----------------------------------------------------------------------------- #
# offline trainer for the stacked ranker (DEV labels only, frozen at test)
# ----------------------------------------------------------------------------- #
def build_stacked_artifact(folder, k=50, l2=1.0):
    """Fit a logistic ranker over [probe_risk@k, log energy_path, log gradnorm,
    norm_slope, oscillation, decay] on the cached good/garbage DEV pool. Saves
    stacked_artifact_k{k}.npz. Returns the path. DEV-ONLY — frozen at test."""
    from probe_validate import load
    from learned_probe import fit_logreg
    folder = Path(folder)
    norm, dot, y = load(folder)                      # full-length cached pool
    nk, dk = norm[:, :k], dot[:, :k]
    pp = folder / "results" / "partial_probe" / f"partial_probe_k{k}.npz"
    d = np.load(pp, allow_pickle=True)
    art = {kk: d[kk] for kk in ["w", "b", "mu", "sd"]}
    pr = probe_risk(nk, dk, art)
    sf = scalar_features(nk, dk)
    feats = np.stack([pr, np.log(sf["energy_path"] + 1e-8), np.log(sf["gradnorm"] + 1e-8),
                      sf["norm_slope"], sf["oscillation"], sf["decay"]], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    mu = feats.mean(0); sd = feats.std(0) + 1e-8
    w, b = fit_logreg((feats - mu) / sd, y, l2=l2)
    out = folder / "results" / "partial_probe" / f"stacked_artifact_k{k}.npz"
    np.savez(out, w=w, b=np.float64(b), mu=mu, sd=sd, k_dec=np.int64(k),
             feature_spec=json.dumps(["probe_risk", "log_energy_path", "log_gradnorm",
                                      "norm_slope", "oscillation", "decay"]))
    return out


def stacked_risk(norm_k, dot_k, probe_art, stacked_art):
    pr = probe_risk(norm_k, dot_k, probe_art)
    sf = scalar_features(norm_k, dot_k)
    feats = np.stack([pr, np.log(sf["energy_path"] + 1e-8), np.log(sf["gradnorm"] + 1e-8),
                      sf["norm_slope"], sf["oscillation"], sf["decay"]], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    a = stacked_art
    return _sigmoid(((feats - a["mu"]) / a["sd"]) @ a["w"] + float(a["b"]))


# ----------------------------------------------------------------------------- #
# SELECTION policies — keep argmin score over R draws; NFE/img = R*steps EXACT.
# Each .score(reads, ctx) -> np.array[R, B]; lower = keep. reads is a dict
# {k: {"norm":(R,B,k),"dot":(R,B,k)}} for the k's the policy requested.
# ----------------------------------------------------------------------------- #
class Selection:
    engine = "selection"
    reads = [50]            # which partial-traj read steps this policy needs

    def __init__(self, **kw):
        self.cfg = kw

    def score(self, reads, ctx):
        raise NotImplementedError


class Vanilla(Selection):           # keep draw 0 (no selection; control)
    name = "vanilla"; reads = []
    def score(self, reads, ctx):
        R, B = ctx["R"], ctx["B"]
        s = np.full((R, B), 1.0); s[0] = 0.0; return s


class Random(Selection):            # keep a random draw (null control)
    name = "random"; reads = []
    def score(self, reads, ctx):
        rng = ctx["rng"]; R, B = ctx["R"], ctx["B"]
        return rng.random((R, B))


class EnergyPath(Selection):        # argmax Σ‖f‖ (best trivial selector)
    name = "energy_path"; reads = [250]
    def score(self, reads, ctx):
        nm = reads[max(reads)]["norm"]      # (R,B,k)
        return -nm.sum(2)


class ProbeK(Selection):            # the locked early metacog selector
    name = "probe_k"
    def __init__(self, k=50, **kw):
        super().__init__(**kw); self.k = k; self.reads = [k]; self.name = f"probe_k{k}"
    def score(self, reads, ctx):
        a = ctx["probe_art"][self.k]; r = reads[self.k]
        R, B, kk = r["norm"].shape
        flat_n = r["norm"].reshape(R * B, kk); flat_d = r["dot"].reshape(R * B, kk)
        return probe_risk(flat_n, flat_d, a).reshape(R, B)


class Stacked(Selection):           # policy 1: calibrated ranker over probe+mag+shape
    name = "stacked_selector"
    def __init__(self, k=50, **kw):
        super().__init__(**kw); self.k = k; self.reads = [k]
    def score(self, reads, ctx):
        pa = ctx["probe_art"][self.k]; sa = ctx["stacked_art"][self.k]; r = reads[self.k]
        R, B, kk = r["norm"].shape
        fn = r["norm"].reshape(R * B, kk); fd = r["dot"].reshape(R * B, kk)
        return stacked_risk(fn, fd, pa, sa).reshape(R, B)


class SMC(Selection):               # policy 2: risk-weighted stochastic selection
    name = "smc_metacog"
    def __init__(self, k=50, beta=8.0, **kw):
        super().__init__(**kw); self.k = k; self.reads = [k]; self.beta = beta
    def score(self, reads, ctx):
        # softmax(-beta*risk) weights over R particles; sample one (paired rng).
        # keep-1 + no extra NFE => SMC reduces to risk-weighted selection (README).
        a = ctx["probe_art"][self.k]; r = reads[self.k]; rng = ctx["rng"]
        R, B, kk = r["norm"].shape
        risk = probe_risk(r["norm"].reshape(R * B, kk), r["dot"].reshape(R * B, kk), a).reshape(R, B)
        w = np.exp(-self.beta * (risk - risk.min(0, keepdims=True)))
        w /= w.sum(0, keepdims=True)
        u = rng.random(B)
        cum = np.cumsum(w, 0)
        pick = (cum < u[None, :]).sum(0).clip(0, R - 1)     # inverse-CDF sample per col
        s = np.ones((R, B)); s[pick, np.arange(B)] = 0.0
        return s


class MultiReadTriage(Selection):   # policy 8: drop obvious-bad early, keep best late
    name = "multiread_triage"; reads = [50, 75, 100]
    def __init__(self, hi=0.85, **kw):
        super().__init__(**kw); self.hi = hi
    def score(self, reads, ctx):
        pa = ctx["probe_art"]; R = ctx["R"]; B = ctx["B"]
        def risk_at(k):
            r = reads[k]; kk = r["norm"].shape[2]
            return probe_risk(r["norm"].reshape(R * B, kk), r["dot"].reshape(R * B, kk),
                              pa[k]).reshape(R, B)
        r50, r100 = risk_at(50), risk_at(100)
        elig = r50 <= self.hi
        s = r100.copy()
        s[~elig] += 10.0                                    # demote early-flagged lanes
        # if a column has no eligible lane, the +10 cancels out (all demoted) -> argmin r100
        none_elig = (~elig).all(0)
        s[:, none_elig] = r100[:, none_elig]
        return s


# ----------------------------------------------------------------------------- #
# SEGMENTED policies — mid-descent actions; engine counts exact NFE. Each maps a
# read (k, per-lane features) to per-lane actions. Declared nfe_target is the
# DESIGN budget; engine reports MEASURED nfe and the aggregator flags mismatch.
# Action codes: 0 continue, 1 restart(fresh noise, reset step), 2 churn(+sigma noise),
#               3 eta_scale, 4 heun (2-eval corrector this segment), 5 drop(freeze).
# ----------------------------------------------------------------------------- #
class Segmented:
    engine = "segmented"
    reads = [50]
    nfe_target = 750

    def __init__(self, **kw):
        self.cfg = kw

    def act(self, k, risk, feats, ctx):
        """Return dict per lane: {'action':code(R,B), 'param':float}. Default continue."""
        R, B = ctx["R"], ctx["B"]
        return {"action": np.zeros((R, B), int), "param": 0.0}


class ChurnRescue(Segmented):       # policy 6: high-risk lanes get noise kick, continue
    name = "churn_rescue"; reads = [50]
    def __init__(self, hi=0.7, sigma=0.3, **kw):
        super().__init__(**kw); self.hi = hi; self.sigma = sigma
    def act(self, k, risk, feats, ctx):
        a = (risk > self.hi).astype(int) * 2
        return {"action": a, "param": self.sigma}


class HeunCorrector(Segmented):     # policy 7: high-risk lanes use 2-eval Heun steps
    name = "heun_corrector"; reads = [50]
    def __init__(self, hi=0.7, **kw):
        super().__init__(**kw); self.hi = hi
    def act(self, k, risk, feats, ctx):
        a = (risk > self.hi).astype(int) * 4
        return {"action": a, "param": 0.0}


class OptimizerSwitch(Segmented):   # policy 5: failure-type -> action
    name = "optimizer_switch"; reads = [50]
    def __init__(self, hi=0.7, **kw):
        super().__init__(**kw); self.hi = hi
    def act(self, k, risk, feats, ctx):
        osc = feats["oscillation"]; gradn = feats["gradnorm"]
        a = np.zeros_like(risk, int)
        risky = risk > self.hi
        # oscillatory -> eta_down(3); high-mag -> heun(4); else restart(1)
        gmed = np.median(gradn)
        a[risky & (osc > np.median(osc))] = 3
        a[risky & (osc <= np.median(osc)) & (gradn > gmed)] = 4
        a[risky & (osc <= np.median(osc)) & (gradn <= gmed)] = 1
        return {"action": a, "param": 0.5}


class RiskAllocator(Segmented):     # policy 4: worst-fraction get rescue draws, budget-balanced
    name = "risk_compute_allocator"; reads = [50]
    def __init__(self, frac=0.33, **kw):
        super().__init__(**kw); self.frac = frac
    def act(self, k, risk, feats, ctx):
        # restart the worst `frac` of lanes (per column) — engine balances budget.
        thr = np.quantile(risk, 1 - self.frac, axis=0, keepdims=True)
        a = (risk >= thr).astype(int)                       # 1 = restart
        return {"action": a, "param": 0.0}


# selection-engine registry (matched-NFE screen — safe to launch overnight)
SELECTION = {p.name if hasattr(p, "name") else p.__name__: p for p in []}
def make_selection(name, **kw):
    table = {"vanilla": Vanilla, "random": Random, "energy_path": EnergyPath,
             "probe_k": ProbeK, "stacked_selector": Stacked, "smc_metacog": SMC,
             "multiread_triage": MultiReadTriage}
    return table[name](**kw)


def make_segmented(name, **kw):
    table = {"churn_rescue": ChurnRescue, "heun_corrector": HeunCorrector,
             "optimizer_switch": OptimizerSwitch, "risk_compute_allocator": RiskAllocator}
    return table[name](**kw)
