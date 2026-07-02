"""
WEAK HARNESS (starting state for the agentic-H step).

This verifier checks ONLY prediction error on clean rotated examples. It cannot tell a model
that learned the real rotation from one that copies a leaked shortcut channel — both score ~0.

A model under test (harness/cheater.pt) scores clean_mse ~0.0002 here and looks SOLVED. But it
is cheating: it also "solves" a broken-symmetry task that has no real symmetry, by reading the
shortcut. This harness is blind to that.

TASK (agentic-H): extend this verifier so it DETECTS the shortcut cheat.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import torch
import torch.nn.functional as F

from data import make_batch
from model import SymmetryLearner


@torch.no_grad()
def _mse(model, batch):
    pred = model(batch["input"], batch["delta"])
    return F.mse_loss(pred, batch["y"]).item()


def verify(model, seed=0):
    """Weak harness: prediction-only."""
    clean = make_batch(2048, mode="clean", seed=seed)
    return {"clean_mse": _mse(model, clean)}


def _load(path):
    m = SymmetryLearner()
    m.load_state_dict(torch.load(path))
    m.eval()
    return m


if __name__ == "__main__":
    for name in ["cheater.pt", "honest.pt"]:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        print(name, verify(_load(p)))
