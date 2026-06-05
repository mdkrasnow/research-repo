"""
Training objectives.

prediction_only  -- the naive objective. Minimizes clean MSE only. A model trained this
                    way will happily read the shortcut channel => fake win.
structural       -- the H->W objective. Adds composition + shortcut-invariance penalties so
                    the model must learn the real action and stop leaning on the shortcut.

Both train the SAME SymmetryLearner. Difference is loss => this is the WEIGHT (W) update.
"""

import torch
import torch.nn.functional as F

from data import make_batch
from model import SymmetryLearner


def prediction_loss(model, batch):
    pred = model(batch["input"], batch["delta"])
    return F.mse_loss(pred, batch["y"])


def shortcut_invariance_loss(model, batch_clean, batch_rand):
    """Output should not change when shortcut channel is randomized."""
    y1 = model(batch_clean["input"], batch_clean["delta"])
    y2 = model(batch_rand["input"], batch_rand["delta"])
    return F.mse_loss(y1, y2)


def composition_penalty(model, n=256, seed=None):
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    d1 = (torch.rand(n, generator=g) * 2 - 1) * torch.pi
    d2 = (torch.rand(n, generator=g) * 2 - 1) * torch.pi
    A1 = model.action_matrix(d1)
    A2 = model.action_matrix(d2)
    A12 = model.action_matrix(d1 + d2)
    return F.mse_loss(torch.bmm(A2, A1), A12)


def train(model=None, steps=2000, bs=512, lr=1e-3, objective="prediction_only",
          lam_short=1.0, lam_comp=1.0, seed=0, log_every=500, leak_alpha=1.0):
    if model is None:
        torch.manual_seed(seed)
        model = SymmetryLearner()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        clean = make_batch(bs, mode="clean", leak_alpha=leak_alpha)
        loss = prediction_loss(model, clean)

        if objective == "structural":
            rand = make_batch(bs, mode="shortcut_rand")
            loss = loss + lam_short * shortcut_invariance_loss(model, clean, rand)
            loss = loss + lam_comp * composition_penalty(model)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if log_every and step % log_every == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}  ({objective})")

    return model
