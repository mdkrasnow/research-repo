"""E1.0 — target-domain premise (spec E1 node). Gate G1.5.

Estimates rho(s) = P(cross/same-class | teacher similarity s) and the
gradient-weighted damage density w(s)*rho_same(s) on CIFAR-10 with a frozen
resnet18 (ImageNet) teacher — Deviation D2 (no EqM EMA exists at CPU stage).

G1.5 (v2.1 density form): mean damage DENSITY in the top hardness (similarity)
decile >= 3x the mean density in the bottom half, gamma=1 bin. Pass -> Stage 1
with motivation figure. Fail (flat rho) -> harm-bounding framing only. No retune.

Also computes the same curves per gamma-bin {0.9, 0.6, 0.3} (P4 preview).
Writes results/e1_0_verdict.json + figures/e1_0_premise.png.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rc_hpm.core import Isotonic, w_repulsive          # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
DATADIR = os.path.join(ROOT, "data")
for d in (RESULTS, FIGDIR, DATADIR):
    os.makedirs(d, exist_ok=True)

N_SUB = 10000
N_PAIRS = 200000
GAMMAS = [1.0, 0.9, 0.6, 0.3]
TAU = 0.5


def main():
    t0 = time.time()
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = torchvision.datasets.CIFAR10(DATADIR, train=True, download=True,
                                      transform=tf)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), N_SUB, replace=False)

    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity()
    net.eval()
    torch.set_num_threads(os.cpu_count() or 8)

    out = {}
    labels = np.array([ds.targets[i] for i in idx])
    for gamma in GAMMAS:
        feats = []
        with torch.no_grad():
            for start in range(0, N_SUB, 256):
                batch = torch.stack([ds[int(i)][0]
                                     for i in idx[start:start + 256]])
                if gamma < 1.0:
                    eps = torch.randn_like(batch)
                    batch = gamma * batch + (1 - gamma) * eps
                feats.append(net(batch))
        emb = torch.cat(feats).numpy()
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)

        i = rng.integers(0, N_SUB, N_PAIRS)
        j = rng.integers(0, N_SUB, N_PAIRS)
        keep = i != j
        i, j = i[keep], j[keep]
        s = (emb[i] * emb[j]).sum(1)
        y_same = (labels[i] == labels[j]).astype(float)

        iso = Isotonic(increasing=True).fit(s, y_same)
        rho_same = np.clip(iso.predict(s), 0, 1)
        damage_density = w_repulsive(s, TAU) * rho_same   # per-pair damage

        top_decile = s >= np.quantile(s, 0.9)
        bottom_half = s <= np.quantile(s, 0.5)
        dens_top = float(damage_density[top_decile].mean())
        dens_bot = float(damage_density[bottom_half].mean())
        out[f"gamma={gamma}"] = dict(
            density_top_decile=dens_top, density_bottom_half=dens_bot,
            ratio=dens_top / max(dens_bot, 1e-12),
            p_same_overall=float(y_same.mean()),
            rho_same_top_decile=float(rho_same[top_decile].mean()),
            rho_same_bottom_half=float(rho_same[bottom_half].mean()),
            s_deciles=[float(np.quantile(s, q))
                       for q in (0.1, 0.5, 0.9, 0.99)])
        print(f"gamma={gamma}: ratio={out[f'gamma={gamma}']['ratio']:.2f} "
              f"(top {dens_top:.4f} / bot {dens_bot:.4f})", flush=True)

        if gamma == 1.0:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                order = np.argsort(s)
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].plot(s[order], rho_same[order])
                ax[0].set(xlabel="teacher similarity s",
                          ylabel="rho_same(s) = P(same class | s)",
                          title="error concentration (CIFAR-10, rn18 teacher)")
                ax[1].plot(s[order], damage_density[order])
                ax[1].set(xlabel="s", ylabel="w(s) * rho_same(s)",
                          title="gradient-weighted damage density")
                fig.tight_layout()
                fig.savefig(os.path.join(FIGDIR, "e1_0_premise.png"), dpi=120)
            except Exception as e:                        # noqa: BLE001
                print("figure failed:", e)

    ratio_clean = out["gamma=1.0"]["ratio"]
    verdict = dict(gate="G1.5", passed=bool(ratio_clean >= 3.0),
                   primary_ratio_gamma1=ratio_clean,
                   threshold=3.0, bins=out,
                   teacher="resnet18 IMAGENET1K_V1 (Deviation D2)",
                   branch=("premise holds -> Stage 1 with motivation figure"
                           if ratio_clean >= 3.0 else
                           "flat rho -> harm-bounding/guarantee framing only"),
                   wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(RESULTS, "e1_0_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps({k: v for k, v in verdict.items() if k != "bins"}, indent=2))


if __name__ == "__main__":
    main()
