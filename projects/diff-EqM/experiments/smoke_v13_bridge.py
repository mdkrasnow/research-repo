"""Local CPU smoke for the v13 SE(2) EqM bridge. NO UNet training, NO FID (cluster/GPU only).

Verifies: (1) SE(2) stable-affine operator DISCOVERS a transform on REAL CIFAR (diagnostics: tx/ty px,
det, cond, lin-off-identity, anchor before/after, shift consistency) — expect a translation-leaning op
with det~1; (2) affine_warp3 sane; (3) v13 / vK(translate_crop) step_fn plumb into eqm_loss + backprop.
"""
import sys
from pathlib import Path
import torch, torch.nn as nn
from torchvision import datasets, transforms

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from dganm_variants._stable_operator_se2 import discover_stable_se2, affine_warp3, _build_M  # noqa

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEV}")

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
data_root = next((d for d in ("data", str(THIS.parent.parent.parent / "data"))
                  if (Path(d) / "cifar-10-batches-py").is_dir()), "data")
ds = datasets.CIFAR10(data_root, train=True, download=False, transform=tf)
loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
_it = iter(loader)
def real_batch_fn(n):
    global _it
    try: xb, _ = next(_it)
    except StopIteration: _it = iter(loader); xb, _ = next(_it)
    return xb[:n]

print("=== (1) SE2 operator discovery on real CIFAR (smoke: 250 steps) ===")
M, diag = discover_stable_se2(real_batch_fn, DEV, steps=250, batch=128, init_seed=0)
for k in ("tx_px","ty_px","det","cond","lin_off_identity","move","move_floor","move_cap",
          "anchor_baseline_real_real","anchor_final_T_real","feature_shift_consistency"):
    print(f"  {k:28s} = {diag[k]:.4f}")
assert diag["anchor_final_T_real"] < diag["anchor_baseline_real_real"] * 1.5, "anchor blew up"

print("\n=== (2) affine_warp3 sanity ===")
xb = real_batch_fn(8).to(DEV); Tx = affine_warp3(xb, M)
print(f"  x {tuple(xb.shape)} -> Tx {tuple(Tx.shape)} mean|Tx-x|={(Tx-xb).abs().mean():.4f}")

print("\n=== (3) step_fn plumbing (tiny stand-in model; eqm_loss + aug + backprop) ===")
class Tiny(nn.Module):
    def __init__(self): super().__init__(); self.c = nn.Conv2d(3,3,3,padding=1)
    def forward(self, x, t=None): return self.c(x)
import dganm_variants.v13_stable_se2_aug as v13
import dganm_variants.vK_known_aug as vK
from dganm_variants._common import TrainArgs
v13._FROZEN["A2"] = torch.tensor(diag["A_gen"], device=DEV); v13._FROZEN["lam"] = 0.3; v13._FROZEN["aug_mode"] = "orbit"
vK._K["lam"] = 0.3; vK._K["mode"] = "translate_crop"; vK._K["crop_pad"] = 4
args = TrainArgs(output_dir="/tmp/v13_smoke", variant="v13", train_eps=1e-3, a=0.8, gain=4.0)
m = Tiny().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for name, sf in [("v13", v13.step_fn), ("vK_translate_crop", vK.step_fn)]:
    x = real_batch_fn(16).to(DEV)
    loss, d = sf(m, x, 0, DEV, args)
    opt.zero_grad(); loss.backward(); opt.step()
    assert d["aug"] > 0, f"{name} aug did not fire"
    print(f"  {name:18s} loss={loss.item():.4f} diag={ {k: round(v,4) for k,v in d.items()} }")
print("\nSMOKE OK")
