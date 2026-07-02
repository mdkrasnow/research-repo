"""RC-ANM confound check: is oracle_safe's apparent success basin-safety or
just lower mining DOSE? Dose-matched random-reject (42% accept) vs oracle
(true-basin, ~42% accept). If random42 ~= oracle42, the toy benefit is dose,
not certification -> strengthens the R4 postmortem.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm.eqm2d import (Field, get_ct, make_triplet, pgd_mine, gd_sample,
                          field_mse, mode_coverage, reference_field,
                          voronoi_basin)

_REF = None


def run(args):
    arm, seed = args
    global _REF
    if _REF is None:
        _REF = reference_field()
    torch.manual_seed(seed * 101 + 3)
    rng = np.random.default_rng(seed + 9000)
    f = Field(); opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    ema = deepcopy(f)
    for p in ema.parameters():
        p.requires_grad_(False)
    teacher = None
    for step in range(4000):
        x1, lab, eps, t, xt, tg = make_triplet(128, rng)
        if step >= 1000:
            if teacher is None:
                teacher = deepcopy(ema)
                for p in teacher.parameters():
                    p.requires_grad_(False)
            adv = pgd_mine(teacher, x1, eps, t, eps_ball=0.8)
            if arm == "oracle42":
                xt_adv = t[:, None] * x1 + (1 - t[:, None]) * adv
                acc = voronoi_basin(xt_adv) == lab
            else:                                   # random42 (dose-matched)
                acc = rng.random(128) < 0.42
            eps_used = np.where(acc[:, None], adv, eps)
            xt = t[:, None] * x1 + (1 - t[:, None]) * eps_used
            tg = (x1 - eps_used) * get_ct(t)[:, None]
        out = f(torch.tensor(xt, dtype=torch.float32))
        loss = ((out - torch.tensor(tg, dtype=torch.float32)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), f.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
    g, fs, _ = _REF
    s = gd_sample(f, 2048, np.random.default_rng(seed + 12345))
    return dict(arm=arm, seed=seed, mse=field_mse(f, g, fs),
                cov=mode_coverage(s))


def main():
    jobs = [(a, s) for a in ("oracle42", "random42") for s in range(5)]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(run, jobs))
    for a in ("oracle42", "random42"):
        mse = [r["mse"] for r in rows if r["arm"] == a]
        cov = [r["cov"] for r in rows if r["arm"] == a]
        print(f"{a}: mse={np.mean(mse):.3f}+/-{np.std(mse, ddof=1):.3f} "
              f"cov={np.mean(cov):.3f}+/-{np.std(cov, ddof=1):.3f}")


if __name__ == "__main__":
    main()
