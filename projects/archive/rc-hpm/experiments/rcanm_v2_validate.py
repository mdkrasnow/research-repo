"""RC-ANM v2 proxy-validation gate (preregistration-rc-anm-v2.md): with a
step-2500 teacher, does the short-rollout proxy basin agree with the analytic
oracle on >= 65% balanced acc? If yes -> proceed to v2 ladder; if no -> R2'
(concept exists but not instrumentable at toy scale, escalate to CIFAR).
"""
import os
import sys
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm.eqm2d import Field, make_triplet, pgd_mine                # noqa: E402
from rc_hpm import rc_anm                                             # noqa: E402


def train_teacher(steps, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    f = Field(); opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    ema = deepcopy(f)
    for p in ema.parameters():
        p.requires_grad_(False)
    for s in range(steps):
        x1, lab, eps, t, xt, tg = make_triplet(128, rng)
        out = f(torch.tensor(xt, dtype=torch.float32))
        loss = ((out - torch.tensor(tg, dtype=torch.float32)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), f.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema, rng


def main():
    for snap in (1000, 2500):
        teacher, rng = train_teacher(snap)
        accs, agrees = [], []
        for _ in range(20):
            x1, lab, eps, t, _, _ = make_triplet(128, rng)
            sel = t >= 2 / 3
            if sel.sum() < 4:
                continue
            adv = pgd_mine(teacher, x1, eps, t, eps_ball=0.8)
            m = rc_anm.proxy_oracle_agreement(teacher, x1[sel], lab[sel],
                                              adv[sel], t[sel])
            accs.append(m["balanced_acc"]); agrees.append(m["agree"])
        print(f"teacher@{snap}: proxy-oracle balanced_acc="
              f"{np.mean(accs):.3f} agree={np.mean(agrees):.3f} "
              f"-> {'PASS' if np.mean(accs) >= 0.65 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
