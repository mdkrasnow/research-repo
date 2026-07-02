"""CPU smoke for the metacog suite — no GPU, no real ckpt. Validates:
  1. every selection policy returns (R,B) scores + a valid keep index;
  2. selection engine NFE/img == R*steps EXACT;
  3. segmented engine runs + reports MEASURED NFE within tol of the design budget;
  4. base z/y are paired (identical per (slot,draw) seed) across calls;
  5. no test-time oracle/quality-label use in the run path (static check).
Run: python test_metacog_local.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metacog_policies as MP
import run_metacog_policy_sweep as RUN


def _fake_artifacts(tmp, ks):
    pp = tmp / "results" / "partial_probe"; pp.mkdir(parents=True)
    src = HERE / "runs" / "b2_vanilla" / "results" / "partial_probe" / "partial_probe_k100.npz"
    d = np.load(src, allow_pickle=True)
    for k in ks:
        np.savez(pp / f"partial_probe_k{k}.npz", w=d["w"], b=d["b"], mu=d["mu"], sd=d["sd"])
        # fabricate a stacked artifact (6 feats) — shape-only test
        np.savez(pp / f"stacked_artifact_k{k}.npz", w=np.zeros(6), b=np.float64(0.0),
                 mu=np.zeros(6), sd=np.ones(6))
    (tmp / "probe_artifact.npz").write_bytes(b"")
    return tmp / "probe_artifact.npz"


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(4, 4, 1)  # float32 bias -> catches dtype-mismatch bugs (eta float64)
    def forward(self, xt, t, y):
        # smooth contractive field so descent is well-behaved + finite
        return -0.05 * xt + 0.01 * torch.sin(self.c(xt))


class FakeVAE:
    def decode(self, z):
        b = z.shape[0]
        return SimpleNamespace(sample=torch.zeros(b, 3, 32, 32) + z.mean())


def fake_incep(dev):
    def f(im):
        b = im.shape[0]
        return (im.flatten(1)[:, :1].repeat(1, 2048) * 0.0 + torch.randn(b, 2048) * 0.0 + 1.0,)
    return f


def main():
    fails = []
    # ---- 1. policy score shapes + keep validity (real k100 artifact) ---------
    R, B, k = 3, 8, 100
    rng = np.random.default_rng(0)
    norm = np.abs(rng.normal(1, 0.2, (R, B, k))).cumsum(2) / np.arange(1, k + 1)
    dot = rng.normal(0, 1, (R, B, k))
    art = {kk: np.load(HERE / "runs/b2_vanilla/results/partial_probe/partial_probe_k100.npz",
                       allow_pickle=True)[kk] for kk in ["w", "b", "mu", "sd"]}
    reads = {50: {"norm": norm[:, :, :50], "dot": dot[:, :, :50]},
             75: {"norm": norm[:, :, :75], "dot": dot[:, :, :75]},
             100: {"norm": norm, "dot": dot}, 250: {"norm": norm, "dot": dot}}
    ctx = {"R": R, "B": B, "rng": np.random.default_rng(1),
           "probe_art": {50: art, 75: art, 100: art}, "stacked_art": {50: {"w": np.zeros(6),
           "b": 0.0, "mu": np.zeros(6), "sd": np.ones(6)}}}
    for name in ["vanilla", "random", "energy_path", "probe_k", "stacked_selector",
                 "smc_metacog", "multiread_triage"]:
        pol = MP.make_selection(name)
        sc = pol.score(reads, ctx)
        if sc.shape != (R, B):
            fails.append(f"{name}: score shape {sc.shape} != {(R,B)}")
        pick = np.argmin(sc, 0)
        if not ((pick >= 0).all() and (pick < R).all()):
            fails.append(f"{name}: invalid pick {pick}")
    print(f"[1] selection policy shapes OK ({7} policies)")

    # ---- 4. paired seeds: same (slot,draw) seed -> identical noise -----------
    z1 = RUN._z("cpu", 4, RUN._seed_draw(5, 7, 3, 1))
    z2 = RUN._z("cpu", 4, RUN._seed_draw(5, 7, 3, 1))
    if not torch.equal(z1, z2):
        fails.append("paired seeds: identical seed produced different noise")
    z3 = RUN._z("cpu", 4, RUN._seed_draw(5, 7, 3, 2))
    if torch.equal(z1, z3):
        fails.append("paired seeds: different draw produced identical noise")
    print("[2] paired z/y determinism OK")

    # ---- 2+3. run both engines with fakes; check NFE ------------------------
    tmp = Path(tempfile.mkdtemp())
    probe = _fake_artifacts(tmp, [50, 75, 100])
    model, vae, incep = FakeModel(), FakeVAE(), fake_incep("cpu")
    common = dict(model="EqM-B/2", ckpt="x", probe_artifact=str(probe), R=3, steps=60,
                  num_slots=6, stepsize=0.003, batch_size=4, image_size=32, num_classes=10, 
                  vae="ema", seed_offset=0, policy_kw="{}")

    a = SimpleNamespace(engine="selection", policy="probe_k", out=str(tmp / "sel"), **common)
    s1, s2, n, nfe, _ = RUN.run_selection(a, model, vae, incep, "cpu", 4, 0, 1)
    if abs(nfe - a.R * a.steps) > 1e-9:
        fails.append(f"selection NFE {nfe} != {a.R*a.steps}")
    if n != a.num_slots:
        fails.append(f"selection n {n} != {a.num_slots}")
    print(f"[3] selection engine ran: n={n} nfe/img={nfe} (exact {a.R*a.steps})")

    a = SimpleNamespace(engine="segmented", policy="churn_rescue", out=str(tmp / "seg"),
                        **{**common, "policy_kw": json.dumps({"hi": 0.0})})  # force churn all
    s1, s2, n, nfe, _ = RUN.run_segmented(a, model, vae, incep, "cpu", 4, 0, 1)
    design = a.R * (a.steps - 1)
    if abs(nfe - design) / design > 0.05:
        fails.append(f"segmented churn NFE {nfe} not within 5% of design {design}")
    print(f"[4] segmented churn ran: n={n} nfe/img={nfe:.1f} (design {design})")

    a = SimpleNamespace(engine="segmented", policy="heun_corrector", out=str(tmp / "seg2"),
                        **{**common, "policy_kw": json.dumps({"hi": 0.0})})  # force heun all
    _, _, n, nfe_h, _ = RUN.run_segmented(a, model, vae, incep, "cpu", 4, 0, 1)
    if nfe_h <= design:
        fails.append(f"heun NFE {nfe_h:.1f} should EXCEED budget-neutral {design} (extra evals)")
    print(f"[5] segmented heun ran: nfe/img={nfe_h:.1f} > {design} (off-budget as expected)")

    # ---- 5. no test-time oracle/quality labels in run path -----------------
    src = (HERE / "run_metacog_policy_sweep.py").read_text()
    for bad in ["labels.csv", "compute_quality", "oracle", "y_true"]:
        if bad in src:
            fails.append(f"run path references forbidden test-time token: {bad}")
    print("[6] no-oracle static check OK")

    shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print("\nFAILURES:"); [print("  ✘", f) for f in fails]; sys.exit(1)
    print("\nALL LOCAL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
