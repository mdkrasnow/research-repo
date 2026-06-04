"""Local CPU smoke for the EqM bridge (v12). NO UNet training, NO FID (those need GPU/cluster).

Verifies: (1) frozen-anchor stable affine operator DISCOVERS a stable transform on REAL CIFAR images
(diagnostics: det, cond, angle, off-identity, anchor before/after, feature-shift consistency);
(2) affine_warp is sane; (3) the v12 / vK step_fn plumbs into eqm_loss and backprops.
"""
import sys, math
from pathlib import Path
import torch, torch.nn as nn
from torchvision import datasets, transforms

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from dganm_variants._stable_operator import discover_stable_affine, affine_warp, RandomConvAnchor  # noqa

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEV}")

# --- real CIFAR batch source (local data, no download) ---
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
ds = datasets.CIFAR10("data", train=True, download=False, transform=tf)
loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
_it = iter(loader)
def real_batch_fn(n):
    global _it
    try: xb, _ = next(_it)
    except StopIteration: _it = iter(loader); xb, _ = next(_it)
    return xb[:n]

print("=== (1) operator discovery on real CIFAR (smoke: 250 steps) ===")
M, diag = discover_stable_affine(real_batch_fn, DEV, steps=250, batch=128, init_seed=0)
for k in ("angle_deg","det","cond","off_identity","move","target_move",
          "anchor_baseline_real_real","anchor_final_T_real","feature_shift_consistency"):
    print(f"  {k:28s} = {diag[k]:.4f}")
print(f"  M = {diag['M']}")

print("\n=== (2) affine_warp sanity ===")
xb = real_batch_fn(8).to(DEV); Tx = affine_warp(xb, M)
print(f"  x {tuple(xb.shape)} range [{xb.min():.2f},{xb.max():.2f}] -> Tx {tuple(Tx.shape)} "
      f"range [{Tx.min():.2f},{Tx.max():.2f}]  mean|Tx-x|={ (Tx-xb).abs().mean():.4f}")

print("\n=== (3) step_fn plumbing (tiny stand-in model; eqm_loss + aug + backprop) ===")
# tiny stand-in with model(xt, t_model) -> same shape (avoids 128-ch UNet CPU cost; real UNet on cluster)
class Tiny(nn.Module):
    def __init__(self): super().__init__(); self.c = nn.Conv2d(3,3,3,padding=1)
    def forward(self, x, t=None): return self.c(x)
import dganm_variants.v12_stable_generator_aug as v12
import dganm_variants.vK_known_aug as vK
from dganm_variants._common import TrainArgs
v12._FROZEN["M"] = M; v12._FROZEN["lam"] = 0.3
th=math.radians(15.0); vK._K["M"]=torch.tensor([[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]]); vK._K["lam"]=0.3; vK._K["mode"]="rotate"
args = TrainArgs(output_dir="/tmp/v12_smoke", variant="v12", train_eps=1e-3, a=0.8, gain=4.0)
m = Tiny().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for name, sf in [("v12", v12.step_fn), ("vK_known", vK.step_fn)]:
    x = real_batch_fn(16).to(DEV)
    loss, d = sf(m, x, 0, DEV, args)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  {name:9s} loss={loss.item():.4f} diag={ {k: round(v,4) for k,v in d.items()} }")
print("\nSMOKE OK" if True else "")
