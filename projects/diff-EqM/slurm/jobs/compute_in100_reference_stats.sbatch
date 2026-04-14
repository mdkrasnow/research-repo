#!/bin/bash
#SBATCH --job-name=eqm-in100-refstats
#SBATCH --output=slurm/logs/eqm-in100-refstats_%A.out
#SBATCH --error=slurm/logs/eqm-in100-refstats_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1

# One-time: compute Inception V3 activations mu/sigma for IN-100 reference images.
# Samples 10K real images from ~/in100_subset/ (100 classes), writes .npz to results/.

set -euo pipefail

module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01

nvidia-smi || exit 1

REPO_ROOT="/n/home03/mkrasnow/research-repo"
IN100_DATA="/n/home03/mkrasnow/in100_subset"
OUT="$REPO_ROOT/projects/diff-EqM/results/in100_reference_stats.npz"

pip install --quiet pytorch-fid torchvision 2>&1 | tail -3

python - "$IN100_DATA" "$OUT" <<'PY'
import sys, numpy as np, torch, pathlib, random
from pytorch_fid.inception import InceptionV3
from torchvision import transforms
from PIL import Image

data_dir, out = sys.argv[1], sys.argv[2]
device = torch.device('cuda')
block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block]).to(device).eval()
transform = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor()])

# Sample 10K images uniformly across 100 classes (100 per class)
class_dirs = sorted(pathlib.Path(data_dir).iterdir())
print(f"Found {len(class_dirs)} classes", flush=True)
files = []
for cd in class_dirs:
    cls_files = sorted(cd.iterdir())
    random.Random(0).shuffle(cls_files)
    files.extend(cls_files[:100])
print(f"Total reference images: {len(files)}", flush=True)

acts = []
bs = 64
for i in range(0, len(files), bs):
    imgs = []
    for f in files[i:i+bs]:
        try: imgs.append(transform(Image.open(f).convert('RGB')))
        except Exception: pass
    if not imgs: continue
    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        p = model(batch)[0]
    if p.size(2) != 1: p = torch.nn.functional.adaptive_avg_pool2d(p, (1,1))
    acts.append(p.squeeze(-1).squeeze(-1).cpu().numpy())
    if (i // bs) % 20 == 0:
        print(f"  processed {min(i+bs, len(files))}/{len(files)}", flush=True)

act = np.concatenate(acts, 0)
mu, sigma = np.mean(act, 0), np.cov(act, rowvar=False)
np.savez(out, mu=mu, sigma=sigma, num_images=len(files))
print(f"Saved reference stats to {out}")
print(f"  mu shape: {mu.shape}, sigma shape: {sigma.shape}")
PY

echo "Done"
