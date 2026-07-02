"""v17 CIFAR validity-detector study — crack a natural-image-robust label-free anchor.

Problem (bridge postmortem): on natural CIFAR no single cheap label-free anchor separates VALID morphisms
from DECOYS. random-conv ED accepts big_shear; a plain AE accepts crop_erase/color_collapse; combined
rejects valid hue. Root idea of the fix: the detector must model the VALID-NUISANCE manifold, not exact
CIFAR. Train an AE on CIFAR + MILD valid augmentations (flip, small crop, color jitter, small rotate/scale,
brightness) -> valid morphisms reconstruct well (low error), only EXTREME destructive decoys fall
off-manifold (high error).

Detectors compared (per-family separability; valid SHOULD score below ALL decoys, sep>0 = WORKS):
  D1 plain_AE            (baseline, known leak)
  D2 robust_AE           (AE trained on valid-nuisance-augmented CIFAR)  <- the candidate fix
  D3 robust_AE + conv-ED (combined max)

Label-free: augmentations are generic nuisances, NOT labels and NOT a pretrained semantic encoder.
CPU-runnable. Run: python v17_cifar_detector.py
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "..", "diff-EqM", "experiments"))
from dganm_variants import _multi_morphism as MM  # noqa: E402

DEV = torch.device("cpu")


def load_cifar(n):
    tf = TT.Compose([TT.ToTensor(), TT.Normalize([0.5] * 3, [0.5] * 3)])
    ds = torchvision.datasets.CIFAR10(os.path.expanduser("~/data"), train=True, download=False, transform=tf)
    return torch.stack([ds[i][0] for i in range(n)])


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                               nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 16, 64))
        self.d = nn.Sequential(nn.Linear(64, 64 * 16), nn.ReLU(), nn.Unflatten(1, (64, 4, 4)),
                               nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),
                               nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                               nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        return self.d(self.e(x))


def valid_nuisance_aug(x, gen):
    """MILD valid nuisances only (flip/crop/colorjitter/small rotate+scale/brightness). NO shear/erase/
    collapse. Defines the manifold the robust AE should accept."""
    B = x.size(0)
    # hflip half
    flip = torch.rand(B, generator=gen) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), torch.flip(x, dims=[3]), x)
    # small translate (crop-pad style) + small scale + small rotate via one affine
    ang = (torch.rand(B, generator=gen) * 2 - 1) * math.radians(15)
    sc = torch.exp((torch.rand(B, generator=gen) * 2 - 1) * 0.15)
    tx = (torch.rand(B, generator=gen) * 2 - 1) * 0.18
    ty = (torch.rand(B, generator=gen) * 2 - 1) * 0.18
    c, s = torch.cos(ang), torch.sin(ang)
    th = torch.zeros(B, 2, 3)
    th[:, 0, 0] = c / sc; th[:, 0, 1] = -s / sc; th[:, 1, 0] = s / sc; th[:, 1, 1] = c / sc
    th[:, 0, 2] = tx; th[:, 1, 2] = ty
    grid = F.affine_grid(th, list(x.size()), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
    # photometric: brightness + hue
    x = MM.m_bright(x, (torch.rand(B, generator=gen) * 2 - 1) * 0.5)
    x = MM.m_hue(x, (torch.rand(B, generator=gen) * 2 - 1) * 0.5)
    return x


def train_ae(real, steps, augment, seed=0):
    torch.manual_seed(seed)
    ae = AE(); opt = torch.optim.Adam(ae.parameters(), 1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        xb = real[torch.randint(0, real.size(0), (128,), generator=g)]
        if augment:
            xb = valid_nuisance_aug(xb, g)
        loss = F.mse_loss(ae(xb), xb)
        opt.zero_grad(); loss.backward(); opt.step()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


@torch.no_grad()
def per_family(scorer_fn, vis, n=512, seed=5):
    g = torch.Generator().manual_seed(seed)
    res = {}
    for fam in MM.ALL_FAMILIES:
        m = (torch.rand(n, generator=g) * 0.5 + 0.5) * (torch.randint(0, 2, (n,), generator=g) * 2 - 1)
        res[fam] = float(scorer_fn(MM.apply_family(fam, vis[:n], m)))
    return res


def report(name, res):
    vmax = max(res[f] for f in MM.VALID_FAMILIES); dmin = min(res[f] for f in MM.DECOY_FAMILIES)
    sep = dmin - vmax
    print("\n=== %s : sep=%.3f %s ===" % (name, sep, "WORKS (valid<all decoys)" if sep > 0 else "LEAK"))
    for f in sorted(res, key=res.get):
        print("  %s %-14s %.4f" % ("V" if f in MM.VALID_FAMILIES else "D", f, res[f]))
    return sep


def main():
    real = load_cifar(4096).to(DEV)
    zoom = 1.5; cc = int(round(32 / zoom)); off = (32 - cc) // 2
    vis = F.interpolate(real[:, :, off:off + cc, off:off + cc], size=32, mode="bilinear", align_corners=False)
    sc = MM.AnchorScorer(real, seed=777)

    def recon_fn(ae):
        return lambda x: ((ae(x) - x) ** 2).flatten(1).mean(1).mean()

    seps = {}
    ae_plain = train_ae(real, 1000, augment=False)
    seps["D1_plain_AE"] = report("D1 plain_AE", per_family(recon_fn(ae_plain), vis))

    ae_rob = train_ae(real, 1500, augment=True)
    res_rob = per_family(recon_fn(ae_rob), vis)
    seps["D2_robust_AE"] = report("D2 robust_AE (valid-nuisance trained)", res_rob)

    # D3 combined: standardized max(robust-AE recon, conv-ED)
    res_conv = per_family(lambda x: sc.ed(x), vis)
    import statistics as S

    def zc(d):
        vs = list(d.values()); mu = S.mean(vs); sd = S.pstdev(vs) + 1e-9
        return {k: (v - mu) / sd for k, v in d.items()}
    za, zb = zc(res_rob), zc(res_conv)
    res_comb = {f: max(za[f], zb[f]) for f in MM.ALL_FAMILIES}
    seps["D3_robustAE_plus_convED"] = report("D3 robust_AE + conv-ED (max-z)", res_comb)

    print("\n[SUMMARY] sep>0 = detector separates valid from all decoys on CIFAR:")
    for k, v in seps.items():
        print("  %-26s sep=%.3f %s" % (k, v, "PASS" if v > 0 else "leak"))
    best = max(seps, key=seps.get)
    print("BEST: %s (sep=%.3f) -> %s" % (best, seps[best],
          "bridge UNBLOCKABLE on this detector" if seps[best] > 0 else "still leaking, needs more"))


if __name__ == "__main__":
    main()
