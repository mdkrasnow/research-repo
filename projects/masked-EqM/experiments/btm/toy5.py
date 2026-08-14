"""Experiment I: the decisive five-atom BTM benchmark.

Reproduces the unequal-weight atomic experiment of arXiv:2608.01692v2 Section 3
(mu0 = N(0, I_2); mu1 = sum_j p_j delta_{x_j} with p = (.30,.30,.15,.15,.10)).
The paper does not publish the atom coordinates, so this uses a documented
fixed set of five well-separated points on a circle of radius 3 -- a faithful
benchmark VARIANT, not an exact reproduction.  The essential property, the
deliberately unequal weights, is preserved exactly.

Reference numbers from the paper for this benchmark:
    BTM (corrected)  mass MAE = 0.005
    EqM (legacy)     mass MAE = 0.102   with c_t = (1-t)^0.8

Primary metric: MAE_mass = (1/5) sum_j |p_hat_j - p_j| where p_hat_j is the
basin mass, i.e. the fraction of fresh x0 ~ mu0 whose autonomous trajectory
terminates at atom j.  For every scalar arm the EVALUATION drift is the exact
b = grad phi from autograd -- finite differences are a training mechanism only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch

from .fd import assert_no_double_backward, exact_gradient
from .interpolant import EqMLinearTarget, build_interpolant
from .models import build_model, count_params
from .objectives import FD_ARMS, compute_loss, FDConfig

ATOM_WEIGHTS = (0.30, 0.30, 0.15, 0.15, 0.10)


# Two atom geometries.  "ring" is 5 equally spaced points on a radius-3 circle.
# It is clean and well separated, but it is SYMMETRIC, and symmetry turns out to
# mask most of the legacy-EqM mass-allocation bias: on the ring the legacy
# control lands near MAE ~0.015 rather than the 0.102 the BTM paper reports,
# because with equal spacing the c_t-induced distortion is nearly the same for
# every basin and largely cancels in the MAE.  "asym" deliberately breaks that:
# unequal radii and unequal angular gaps, so basins differ in size a priori and
# a schedule-induced reweighting cannot cancel.  The paper does not publish its
# coordinates; both of ours are documented benchmark VARIANTS, and the unequal
# WEIGHTS -- the essential property -- are identical in each.
ATOM_GEOMETRIES = {
    "ring": [(3.0, 0.0), (3.0, 72.0), (3.0, 144.0), (3.0, 216.0), (3.0, 288.0)],
    "asym": [(3.4, 0.0), (2.6, 55.0), (3.8, 150.0), (2.2, 205.0), (3.0, 300.0)],
}


def atoms(device="cpu", dtype=torch.float32, geometry="ring") -> torch.Tensor:
    spec = ATOM_GEOMETRIES[geometry]
    r = torch.tensor([s[0] for s in spec], dtype=torch.float64)
    a = torch.tensor([s[1] for s in spec], dtype=torch.float64) * math.pi / 180.0
    pts = torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=1)
    return pts.to(device=device, dtype=dtype)


def sample_mu1(n, device, generator=None, dtype=torch.float32, geometry="ring"):
    w = torch.tensor(ATOM_WEIGHTS, device=device, dtype=torch.float64)
    idx = torch.multinomial(w, n, replacement=True, generator=generator)
    return atoms(device, dtype, geometry)[idx], idx


def sample_mu0(n, device, generator=None, dtype=torch.float32):
    return torch.randn(n, 2, device=device, generator=generator, dtype=dtype)


# --------------------------------------------------------------------------
@dataclass
class ToyConfig:
    arm: str = "btm_vector"
    seed: int = 0
    steps: int = 6000
    batch: int = 1024
    lr: float = 1e-3
    width: int = 256
    depth: int = 3
    tc: float = 0.8
    kappa: float = 0.8          # legacy-EqM schedule exponent
    eps_fd: float = 1e-3
    K: int = 1
    fp32_subtract: bool = True
    device: str = "cpu"
    eval_n: int = 100_000
    eval_T: float = 30.0
    eval_dt: float = 0.01
    freeze_tol: float = 0.05    # |b| below this -> particle declared frozen
    resolve_tol: float = 0.25   # distance to nearest atom for "resolved"
    grad_clip: float = 0.0      # 0 disables; kept for parity with image stage
    geometry: str = "ring"      # ring | asym  (see ATOM_GEOMETRIES)


def make_batch(cfg: ToyConfig, interp, n, device, generator):
    x0 = sample_mu0(n, device, generator)
    x1, _ = sample_mu1(n, device, generator, geometry=cfg.geometry)
    t = torch.rand(n, device=device, generator=generator)
    z, zdot = interp.interpolate(t, x0, x1)
    return {"z": z, "zdot": zdot, "x0": x0, "x1": x1, "t": t}


def field_fn(arm, model):
    """Evaluation-time autonomous drift b(x).  Exact gradient for scalar arms."""
    if arm in ("btm_vector", "eqm_legacy_vector"):
        return lambda x: model(x)
    return lambda x: exact_gradient(model, x, create_graph=False)


# --------------------------------------------------------------------------
def train(cfg: ToyConfig, log_every: int = 500, logger=None):
    device = torch.device(cfg.device)
    interp_name = ("eqm_legacy" if cfg.arm.startswith("eqm_legacy")
                   else "self_stopping")
    interp = build_interpolant(interp_name, tc=cfg.tc, kappa=cfg.kappa)

    model = build_model(cfg.arm, d=2, width=cfg.width, depth=cfg.depth,
                        seed=cfg.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    fdcfg = FDConfig(eps_fd=cfg.eps_fd, K=cfg.K,
                     fp32_subtract=cfg.fp32_subtract)

    # Data-order generator is seeded from the run seed but SEPARATELY from the
    # weight init, so that shared-init paired arms still see identical data.
    gen = torch.Generator(device=device)
    gen.manual_seed(10_000 + cfg.seed)

    history = []
    nonfinite = 0
    t_start = time.time()
    for step in range(cfg.steps):
        batch = make_batch(cfg, interp, cfg.batch, device, gen)
        loss = compute_loss(cfg.arm, model, batch, fdcfg, generator=gen)
        opt.zero_grad(set_to_none=True)
        if cfg.arm in FD_ARMS:
            with assert_no_double_backward():
                loss.backward()
        else:
            loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
        if not torch.isfinite(gnorm):
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        if step % log_every == 0 or step == cfg.steps - 1:
            rec = {"step": step, "loss": float(loss.detach()),
                   "grad_norm": float(gnorm)}
            history.append(rec)
            if logger:
                logger(f"[{cfg.arm} s{cfg.seed}] {rec}")
    return model, {"history": history, "nonfinite_steps": nonfinite,
                   "train_seconds": time.time() - t_start,
                   "n_params": count_params(model)}


# --------------------------------------------------------------------------
@torch.no_grad()
def _nearest_atom(x, A):
    d = torch.cdist(x, A)
    dist, idx = d.min(dim=1)
    return idx, dist


def evaluate_transport(cfg: ToyConfig, model):
    """Integrate xdot = b(x) to convergence; report basin masses."""
    device = torch.device(cfg.device)
    A = atoms(device, geometry=cfg.geometry)
    b = field_fn(cfg.arm, model)

    gen = torch.Generator(device=device)
    gen.manual_seed(777_000 + cfg.seed)
    x = sample_mu0(cfg.eval_n, device, gen)

    frozen = torch.zeros(cfg.eval_n, dtype=torch.bool, device=device)
    freeze_time = torch.full((cfg.eval_n,), float("nan"), device=device)
    n_steps = int(cfg.eval_T / cfg.eval_dt)
    dt = cfg.eval_dt

    for k in range(n_steps):
        act = ~frozen
        if not act.any():
            break
        xa = x[act]
        k1 = b(xa).detach()
        k2 = b(xa + dt * k1).detach()          # Heun
        x[act] = (xa + 0.5 * dt * (k1 + k2)).detach()
        with torch.no_grad():
            speed = k1.norm(dim=1)
            newly = speed < cfg.freeze_tol
            if newly.any():
                idx = act.nonzero(as_tuple=True)[0][newly]
                frozen[idx] = True
                freeze_time[idx] = (k + 1) * dt
        if not torch.isfinite(x).all():
            break

    with torch.no_grad():
        finite = torch.isfinite(x).all(dim=1)
        idx, dist = _nearest_atom(torch.nan_to_num(x, nan=1e6), A)
        resolved = finite & (dist < cfg.resolve_tol)
        counts = torch.bincount(idx[resolved], minlength=5).double()
        p_hat = (counts / max(int(resolved.sum()), 1)).cpu()
        p = torch.tensor(ATOM_WEIGHTS, dtype=torch.float64)
        mae = float((p_hat - p).abs().mean())
        # mass share computed over ALL particles (unresolved counted as lost)
        p_hat_all = (counts / cfg.eval_n).cpu()
        mae_all = float((p_hat_all - p).abs().mean())

    return {
        "mass_mae": mae,
        "mass_mae_over_all": mae_all,
        "p_hat": [float(v) for v in p_hat],
        "unresolved_frac": float(1.0 - resolved.double().mean()),
        "nonfinite_frac": float(1.0 - finite.double().mean()),
        "frozen_frac": float(frozen.double().mean()),
        "median_freeze_time": float(
            torch.nanmedian(freeze_time)) if frozen.any() else float("nan"),
        "median_final_dist": float(dist[finite].median()) if finite.any()
        else float("nan"),
        "mean_final_dist": float(dist[finite].mean()) if finite.any()
        else float("nan"),
    }


# --------------------------------------------------------------------------
def weak_conservation_residual(cfg: ToyConfig, model, n=200_000, n_probe=32):
    """R_psi = E_nu[grad psi . grad phi] - (E_mu1 psi - E_mu0 psi).

    Sign derivation: div(nu b) = mu0 - mu1.  Multiply by psi and integrate by
    parts:  -E_nu[grad psi . b] = E_mu0 psi - E_mu1 psi, i.e.
             E_nu[grad psi . b] = E_mu1 psi - E_mu0 psi.
    With b = grad phi this is the boxed residual.  It is also exactly the first
    variation of J_A in the direction psi, which is an independent check that
    the objective and the divergence equation agree in sign.

    Test-function bank: linear, quadratic, Gaussian RBF, low-frequency sinusoid.
    Residuals are normalized by the RMS of the two terms so they are comparable
    across probes of different scale.
    """
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(555_000 + cfg.seed)
    interp = build_interpolant(
        "linear" if cfg.arm.startswith("eqm_legacy") else "self_stopping",
        tc=cfg.tc)
    if isinstance(interp, EqMLinearTarget):  # pragma: no cover - defensive
        raise AssertionError("nu must come from a true interpolant")

    x0 = sample_mu0(n, device, gen)
    x1, _ = sample_mu1(n, device, gen, geometry=cfg.geometry)
    t = torch.rand(n, device=device, generator=gen)
    z, _ = interp.interpolate(t, x0, x1)

    b = field_fn(cfg.arm, model)
    with torch.enable_grad():
        gphi = b(z).detach()

    # probe bank
    torch.manual_seed(31_337 + cfg.seed)
    A = torch.randn(n_probe, 2, device=device)
    A = A / A.norm(dim=1, keepdim=True)
    centers = torch.randn(n_probe, 2, device=device) * 2.0
    freqs = torch.randn(n_probe, 2, device=device) * 0.6

    def eval_bank(x):
        lin = x @ A.T                                  # [N, P]
        quad = 0.5 * lin ** 2
        r2 = ((x.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
        rbf = torch.exp(-0.5 * r2)
        sin = torch.sin(x @ freqs.T)
        return lin, quad, rbf, sin

    def grad_bank(x):
        lin = x @ A.T
        g_lin = A.unsqueeze(0).expand(x.shape[0], -1, -1)          # [N,P,2]
        g_quad = lin.unsqueeze(-1) * A.unsqueeze(0)
        diff = x.unsqueeze(1) - centers.unsqueeze(0)
        rbf = torch.exp(-0.5 * (diff ** 2).sum(-1))
        g_rbf = -diff * rbf.unsqueeze(-1)
        g_sin = torch.cos(x @ freqs.T).unsqueeze(-1) * freqs.unsqueeze(0)
        return g_lin, g_quad, g_rbf, g_sin

    out = {}
    names = ("linear", "quadratic", "rbf", "sinusoid")
    with torch.no_grad():
        gz = grad_bank(z)
        p0 = eval_bank(x0)
        p1 = eval_bank(x1)
        for name, gpsi, v0, v1 in zip(names, gz, p0, p1):
            lhs = torch.einsum("npd,nd->np", gpsi, gphi).mean(0)   # [P]
            rhs = v1.mean(0) - v0.mean(0)
            scale = 0.5 * (lhs.abs() + rhs.abs()) + 1e-8
            rel = ((lhs - rhs).abs() / scale)
            out[f"R_{name}_median_rel"] = float(rel.median())
            out[f"R_{name}_mean_abs"] = float((lhs - rhs).abs().mean())
    out["R_overall_median_rel"] = float(
        sum(out[f"R_{n}_median_rel"] for n in names) / len(names))
    return out


# --------------------------------------------------------------------------
def run(cfg: ToyConfig, out_dir=None, verbose=True):
    logger = (lambda m: print(m, flush=True)) if verbose else None
    model, tinfo = train(cfg, logger=logger)
    ev = evaluate_transport(cfg, model)
    wc = weak_conservation_residual(cfg, model)
    rec = {"config": asdict(cfg), **tinfo, **ev, **wc}
    rec["stable"] = (tinfo["nonfinite_steps"] == 0
                     and ev["nonfinite_frac"] < 0.01
                     and math.isfinite(tinfo["history"][-1]["loss"]))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tag = (f"{cfg.arm}_{cfg.geometry}_tc{cfg.tc}_K{cfg.K}"
               f"_eps{cfg.eps_fd}_seed{cfg.seed}")
        with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}.pt"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    for k, v in asdict(ToyConfig()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_','-')}",
                            type=lambda s: s.lower() in ("1", "true", "yes"),
                            default=v)
        else:
            ap.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    cfg = ToyConfig(**{k: getattr(args, k) for k in asdict(ToyConfig())})
    rec = run(cfg, out_dir=args.out_dir)
    print(json.dumps({k: v for k, v in rec.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
