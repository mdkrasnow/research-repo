#!/bin/bash
#SBATCH --job-name=eqm-direct-ddp-smoke
#SBATCH --output=projects/masked-EqM/slurm/logs/direct-ddp-smoke-%j.out
#SBATCH --error=projects/masked-EqM/slurm/logs/direct-ddp-smoke-%j.err
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:20:00
set -euo pipefail
module load python/3.10.13-fasrc01 2>/dev/null || true
module load cuda/11.8.0-fasrc01 2>/dev/null || true
nvidia-smi >/dev/null

WORK_DIR="/tmp/masked-eqm-ddp-smoke-${SLURM_JOB_ID}"
GIT_URL="${GIT_URL:-https://github.com/mdkrasnow/research-repo.git}"
GIT_SHA="${GIT_SHA:?set GIT_SHA}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git init .
git remote add origin "$GIT_URL"
git fetch --depth=50 origin main
git checkout "$GIT_SHA" || { git fetch --unshallow origin; git checkout "$GIT_SHA"; }
cd projects/masked-EqM

python - <<'PY'
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models import EqM

dist.init_process_group(
    "nccl",
    init_method="tcp://127.0.0.1:29501",
    rank=0,
    world_size=1,
)
device = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(device)
model = EqM(input_size=8, patch_size=2, in_channels=4, hidden_size=64,
            depth=2, num_heads=4, mlp_ratio=2.0, num_classes=10,
            learn_sigma=False, uncond=True, ebm="direct").to(device)
model = DDP(model, device_ids=[device], find_unused_parameters=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
for step in range(2):
    x = torch.randn(2, 4, 8, 8, device=device)
    t = torch.rand(2, device=device)
    y = torch.randint(0, 10, (2,), device=device)
    field = model(x, t, y, train=True)
    loss = field.square().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    print(f"step={step} loss={loss.item():.6f}", flush=True)
dist.destroy_process_group()
PY
