"""
Two verifiers. The DIFFERENCE between them IS the harness (H) update.

verifier_v0 (prediction-only)  -- the weak harness. Sees only clean MSE.
                                  Cannot detect shortcut cheating. This is the trap.
verifier_v1 (structural)       -- the patched harness. Adds negative-control, shortcut
                                  sensitivity, composition, equivariance. Detects fake wins.

Metrics convention:
  clean_mse            lower = better prediction
  neg_control_mse      HIGHER = better (honest learner must FAIL the broken-symmetry task)
  shortcut_sensitivity HIGHER = more reliance on shortcut channel (bad)
  composition_error    lower = better (A_d1 A_d2 ~ A_(d1+d2))
  equivariance_error   lower = better (A_d enc(x) ~ enc(rotated x))

verdict heuristic (v1): shortcut win if clean low AND neg_control low.
"""

import torch
import torch.nn.functional as F

from data import make_batch


@torch.no_grad()
def _mse(model, batch):
    pred = model(batch["input"], batch["delta"])
    return F.mse_loss(pred, batch["y"]).item()


@torch.no_grad()
def composition_error(model, n=512, seed=7):
    g = torch.Generator(); g.manual_seed(seed)
    d1 = (torch.rand(n, generator=g) * 2 - 1) * torch.pi
    d2 = (torch.rand(n, generator=g) * 2 - 1) * torch.pi
    A1 = model.action_matrix(d1)
    A2 = model.action_matrix(d2)
    A12 = model.action_matrix(d1 + d2)
    composed = torch.bmm(A2, A1)
    return F.mse_loss(composed, A12).item()


@torch.no_grad()
def identity_error(model, n=512, seed=13):
    """Group axiom: the action at delta=0 must be the identity. ||A(0) - I||^2 (mean)."""
    d0 = torch.zeros(n)
    A0 = model.action_matrix(d0)
    I = torch.eye(model.latent_dim).expand_as(A0)
    return F.mse_loss(A0, I).item()


@torch.no_grad()
def inverse_error(model, n=512, seed=15):
    """Group axiom: A(-delta) must invert A(delta). ||A(-d) A(d) - I||^2 (mean)."""
    g = torch.Generator(); g.manual_seed(seed)
    d = (torch.rand(n, generator=g) * 2 - 1) * torch.pi
    Ad = model.action_matrix(d)
    Aninv = model.action_matrix(-d)
    prod = torch.bmm(Aninv, Ad)
    I = torch.eye(model.latent_dim).expand_as(prod)
    return F.mse_loss(prod, I).item()


def verifier_v0(model, seed=0):
    """Weak harness: prediction-only. Blind to shortcut."""
    clean = make_batch(2048, mode="clean", seed=seed)
    return {"clean_mse": _mse(model, clean)}


def verifier_v1(model, seed=0, leak_alpha=1.0):
    """Patched harness: structural battery. Detects fake wins.

    leak_alpha matches the model's training distribution so partial-leak models are evaluated
    on the same shortcut strength they saw.
    """
    clean = make_batch(2048, mode="clean", seed=seed, leak_alpha=leak_alpha)
    neg = make_batch(2048, mode="neg_control", seed=seed + 1, leak_alpha=leak_alpha)
    rand = make_batch(2048, mode="shortcut_rand", seed=seed + 2)

    clean_mse = _mse(model, clean)
    neg_mse = _mse(model, neg)
    rand_mse = _mse(model, rand)

    metrics = {
        "clean_mse": clean_mse,
        "neg_control_mse": neg_mse,
        "shortcut_sensitivity": rand_mse - clean_mse,
        "composition_error": composition_error(model),
        "identity_error": identity_error(model),
        "inverse_error": inverse_error(model),
    }
    # verdict: shortcut win if predicts clean well AND ALSO solves broken control
    metrics["verdict"] = (
        "shortcut_win"
        if (clean_mse < 0.05 and neg_mse < 0.10)
        else ("clean_win" if clean_mse < 0.05 else "unsolved")
    )
    return metrics
