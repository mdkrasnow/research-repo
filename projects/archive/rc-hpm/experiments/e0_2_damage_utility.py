"""E0.2' — damage/utility probe with robust-loss + decomposition arms. Gate G1.

Concentration ladder sigma_cluster in {0.6, 1.2, 1.8} (ALL levels run, no
escalation retune). Arms (matched compute: same steps/bs/optimizer):
  no_mine / naive_hard / rc_hpm / rince / cert_random_k / supcon (+)

Primary metric: linear-probe accuracy (P3a). 5 seeds per arm x level.
Margin = 2 x seed-SD of the no_mine arm at that level (recorded from the
no_mine runs BEFORE treatment arms are read).

G1 at HIGH concentration: (K2) naive < no_mine - margin;
(K3) rc_hpm > no_mine + margin; (e) rc_hpm > rince + margin at matched
compute; (f) cert_random_k recorded (descriptive).

Writes results/e0_2_results.json + results/e0_2_verdict.json.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm import core, losses                                   # noqa: E402
from rc_hpm.toy import (ToyConfig, make_population, draw, teacher_embed,
                        aug_view, train_gate, lam_grid)           # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

ALPHA = 0.10
LEVELS = {"low": 0.6, "med": 1.2, "high": 1.8}
ARMS = ["no_mine", "naive_hard", "rc_hpm", "rince", "cert_random_k", "supcon"]
N_SEEDS = 5
STEPS = 1500
TAU = 0.5


class Student(torch.nn.Module):
    def __init__(self, d_in=16, h=64, d_out=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, h), torch.nn.ReLU(), torch.nn.Linear(h, d_out))

    def forward(self, x):
        z = self.net(x)
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def linear_probe(student, pop, rng, n_train=1000, n_test=2000, steps=300):
    xtr, ytr = draw(pop, n_train, rng)
    xte, yte = draw(pop, n_test, rng)
    with torch.no_grad():
        ztr = student(torch.tensor(xtr, dtype=torch.float32))
        zte = student(torch.tensor(xte, dtype=torch.float32))
    Wp = torch.zeros(ztr.shape[1], pop.means.shape[0], requires_grad=True)
    bp = torch.zeros(pop.means.shape[0], requires_grad=True)
    opt = torch.optim.Adam([Wp, bp], lr=0.05)
    Ytr = torch.tensor(ytr)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(ztr @ Wp + bp, Ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((zte @ Wp + bp).argmax(1).numpy() == yte).mean()
    return float(acc)


def run_arm(args):
    arm, level_name, sigma, seed = args
    torch.manual_seed(seed * 37 + 5)
    cfg = ToyConfig(sigma_cluster=sigma)
    rng = np.random.default_rng(seed + 4000)
    pop = make_population(seed, cfg)

    calib = None
    q_fn = None
    risk_log = []
    if arm in ("rc_hpm", "cert_random_k"):
        xg, lg = draw(pop, cfg.n_gate, rng)
        q_fn = train_gate(teacher_embed(pop, xg), lg, seed)
        xc, lc = draw(pop, cfg.n_fold, rng)
        ec = teacher_embed(pop, xc)
        ac = teacher_embed(pop, aug_view(xc, rng, cfg.sigma_aug))
        calib = core.calibrate_ltt(ec, lc, ac, q_fn, lam_grid(cfg), ALPHA, ALPHA,
                                   cfg.delta_r, cfg.n_batch, cfg.m, rng,
                                   cfg.k_plus, cfg.k_minus, m_fit=cfg.m_fit)
        if calib.aborted:   # P7: train un-mined, record throughput 0
            arm_effective = "no_mine"
        else:
            arm_effective = arm
    else:
        arm_effective = arm

    student = Student()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    for step in range(STEPS):
        xb, lb = draw(pop, cfg.n_batch, rng)
        x1 = aug_view(xb, rng, cfg.sigma_aug)
        x2 = aug_view(xb, rng, cfg.sigma_aug)
        z1 = student(torch.tensor(x1, dtype=torch.float32))
        z2 = student(torch.tensor(x2, dtype=torch.float32))
        eb = teacher_embed(pop, xb)
        if arm_effective == "no_mine":
            loss = losses.plain_infonce(z1, z2, TAU)
        elif arm_effective == "naive_hard":
            loss = losses.naive_hardmine_infonce(z1, z2, eb @ eb.T,
                                                 k=cfg.k_minus, tau=TAU)
        elif arm_effective == "rince":
            loss = losses.rince(z1, z2, q_exp=0.5, lam=0.025, tau=TAU)
        elif arm_effective == "supcon":
            loss = losses.supcon(z1, z2, lb, TAU)
        elif arm_effective in ("rc_hpm", "cert_random_k"):
            qb = core.q_matrix(q_fn, eb, lb)
            ab = teacher_embed(pop, aug_view(xb, rng, cfg.sigma_aug))
            s_aug = (eb * ab).sum(1)
            if arm_effective == "rc_hpm":
                mined = core.mine_batch(eb, qb, calib.lam[0], calib.lam[1],
                                        calib.rho_hat, calib.rho_plus,
                                        cfg.k_plus, cfg.k_minus)
                loss = losses.rc_hpm_loss(z1, z2, mined, calib.rho_hat,
                                          calib.rho_plus, calib.rho_amb,
                                          s_aug, TAU)
            else:
                loss = losses.certified_random_k(
                    z1, z2, qb, calib.lam[0], calib.lam[1], calib.rho_hat,
                    calib.rho_plus, calib.rho_amb, eb @ eb.T, s_aug,
                    cfg.k_plus, cfg.k_minus, rng, TAU)
            if arm_effective == "rc_hpm" and step % 100 == 0:
                mined_r = core.mine_batch(eb, qb, calib.lam[0], calib.lam[1],
                                          calib.rho_hat, calib.rho_plus,
                                          cfg.k_plus, cfg.k_minus)
                y_same = (lb[:, None] == lb[None, :]).astype(float)
                risk_log.append(core.batch_risks(mined_r, y_same, calib.rho_hat,
                                                 calib.rho_plus, calib.rho_amb,
                                                 s_aug))
        opt.zero_grad(); loss.backward(); opt.step()

    acc = linear_probe(student, pop, rng)
    out = dict(arm=arm, arm_effective=arm_effective, level=level_name,
               sigma=sigma, seed=seed, probe_acc=acc)
    if risk_log:
        rl = np.array(risk_log)
        out.update(train_risk_minus=float(rl[:, 0].mean()),
                   train_risk_plus=float(rl[:, 1].mean()),
                   train_risk_minus_max=float(rl[:, 0].max()))
    if calib is not None:
        out.update(calib_aborted=bool(calib.aborted),
                   lam=list(calib.lam) if calib.lam else None)
    return out


def main():
    t0 = time.time()
    jobs = [(arm, ln, sg, seed) for ln, sg in LEVELS.items()
            for arm in ARMS for seed in range(N_SEEDS)]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(run_arm, jobs))
    with open(os.path.join(RESULTS, "e0_2_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    def accs(arm, level):
        return np.array([r["probe_acc"] for r in rows
                         if r["arm"] == arm and r["level"] == level])

    verdict = {"gate": "G1", "levels": {}}
    for ln in LEVELS:
        base = accs("no_mine", ln)
        margin = 2 * base.std(ddof=1)
        lv = {"margin": float(margin), "no_mine": float(base.mean())}
        for arm in ARMS[1:]:
            lv[arm] = float(accs(arm, ln).mean())
        lv["K2_damage"] = bool(lv["naive_hard"] < lv["no_mine"] - margin)
        lv["K3_utility"] = bool(lv["rc_hpm"] > lv["no_mine"] + margin)
        lv["e_beats_rince"] = bool(lv["rc_hpm"] > lv["rince"] + margin)
        lv["f_cert_vs_hardness_gap"] = float(lv["rc_hpm"] - lv["cert_random_k"])
        verdict["levels"][ln] = lv

    hi = verdict["levels"]["high"]
    verdict["passed_full"] = bool(hi["K2_damage"] and hi["K3_utility"]
                                  and hi["e_beats_rince"])
    verdict["branch"] = (
        "STRONGEST premise -> Stage 1" if verdict["passed_full"] else
        "K2 only -> harm-bounding framing" if hi["K2_damage"] else
        "K3 only / neither -> see tree branching")
    verdict["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "e0_2_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
