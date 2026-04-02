#!/bin/bash
# Comprehensive adversarial tests for the ImageNet EqM pipeline
# Run this BEFORE submitting any SLURM jobs to catch issues early
#
# Usage (on cluster):
#   module load python/3.10.13-fasrc01 cuda/11.8.0-fasrc01
#   bash projects/diff-EqM/tests/test_imagenet_pipeline.sh
#
# Usage (via SSH from local):
#   scripts/cluster/ssh.sh "cd /n/home03/mkrasnow/research-repo && module load python/3.10.13-fasrc01 cuda/11.8.0-fasrc01 && bash projects/diff-EqM/tests/test_imagenet_pipeline.sh"

set -euo pipefail

PASS=0
FAIL=0
SKIP=0

pass() { echo "  ✓ PASS: $1"; ((PASS++)); }
fail() { echo "  ✗ FAIL: $1"; ((FAIL++)); }
skip() { echo "  - SKIP: $1"; ((SKIP++)); }

echo "============================================================"
echo "ImageNet EqM Pipeline Tests"
echo "============================================================"

# ============================================================
echo ""
echo "--- 1. DEPENDENCY TESTS ---"
# ============================================================

echo "1.1 Python imports"
python -c "import torch; print(f'  torch {torch.__version__}')" 2>/dev/null && pass "torch" || fail "torch not importable"
python -c "import torchvision; print(f'  torchvision {torchvision.__version__}')" 2>/dev/null && pass "torchvision" || fail "torchvision not importable"
python -c "import timm; print(f'  timm {timm.__version__}')" 2>/dev/null && pass "timm" || fail "timm not importable"
python -c "import diffusers; print(f'  diffusers {diffusers.__version__}')" 2>/dev/null && pass "diffusers" || fail "diffusers not importable"
python -c "import scipy; print(f'  scipy {scipy.__version__}')" 2>/dev/null && pass "scipy" || fail "scipy not importable"
python -c "import pytorch_fid; print('  pytorch_fid OK')" 2>/dev/null && pass "pytorch_fid" || fail "pytorch_fid not importable (pip install pytorch-fid)"
python -c "
try:
    import wandb; print(f'  wandb {wandb.__version__}')
except ImportError:
    print('  wandb not installed (OK - optional)')
" 2>/dev/null && pass "wandb check" || fail "wandb check"

echo ""
echo "1.2 Upstream EqM imports"
python -c "
import sys; sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from models import EqM_models
print(f'  Available models: {list(EqM_models.keys())}')
assert 'EqM-B/2' in EqM_models, 'EqM-B/2 not found'
" 2>/dev/null && pass "EqM_models importable with EqM-B/2" || fail "Cannot import EqM_models"

python -c "
import sys; sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from transport import create_transport, Sampler
print('  transport OK')
" 2>/dev/null && pass "transport importable" || fail "Cannot import transport"

python -c "
import sys; sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from download import find_model
print('  download OK')
" 2>/dev/null && pass "download importable" || fail "Cannot import download"

echo ""
echo "1.3 train_imagenet.py imports"
python -c "
import sys; sys.path.insert(0, 'projects/diff-EqM/experiments')
# Just check the file parses without import errors
import importlib.util
spec = importlib.util.spec_from_file_location('train_imagenet', 'projects/diff-EqM/experiments/train_imagenet.py')
# Don't actually import (needs DDP), just check syntax
import ast
with open('projects/diff-EqM/experiments/train_imagenet.py') as f:
    ast.parse(f.read())
print('  Syntax OK')
" 2>/dev/null && pass "train_imagenet.py syntax valid" || fail "train_imagenet.py has syntax errors"

# ============================================================
echo ""
echo "--- 2. DATA TESTS ---"
# ============================================================

IMAGENET_PATH="/n/home03/mkrasnow/imagenet100/train"

echo "2.1 ImageNet-100 data exists"
if [ -d "$IMAGENET_PATH" ]; then
    pass "ImageNet-100 train dir exists"
else
    fail "ImageNet-100 train dir missing at $IMAGENET_PATH"
fi

echo "2.2 Class count"
NUM_CLASSES=$(ls -d "$IMAGENET_PATH"/*/ 2>/dev/null | wc -l)
if [ "$NUM_CLASSES" -eq 100 ]; then
    pass "100 classes found"
else
    fail "Expected 100 classes, found $NUM_CLASSES"
fi

echo "2.3 Images exist in classes"
SAMPLE_CLASS=$(ls -d "$IMAGENET_PATH"/*/ 2>/dev/null | head -1)
if [ -n "$SAMPLE_CLASS" ]; then
    IMG_COUNT=$(ls "$SAMPLE_CLASS"/*.jpg 2>/dev/null | wc -l)
    if [ "$IMG_COUNT" -gt 0 ]; then
        pass "Found $IMG_COUNT images in $(basename $SAMPLE_CLASS)"
    else
        fail "No .jpg images in $SAMPLE_CLASS"
    fi
else
    fail "No class directories found"
fi

echo "2.4 ImageFolder compatibility"
python -c "
from torchvision.datasets import ImageFolder
from torchvision import transforms
t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
ds = ImageFolder('$IMAGENET_PATH', transform=t)
print(f'  Dataset: {len(ds)} images, {len(ds.classes)} classes')
assert len(ds.classes) == 100, f'Expected 100 classes, got {len(ds.classes)}'
img, label = ds[0]
assert img.shape == (3, 256, 256), f'Expected (3,256,256), got {img.shape}'
print(f'  Sample image shape: {img.shape}, label: {label}')
" 2>/dev/null && pass "ImageFolder loads correctly" || fail "ImageFolder cannot load dataset"

echo "2.5 Image dimensions"
python -c "
from PIL import Image
import os
path = '$IMAGENET_PATH'
first_class = sorted(os.listdir(path))[0]
first_img = sorted(os.listdir(os.path.join(path, first_class)))[0]
img = Image.open(os.path.join(path, first_class, first_img))
w, h = img.size
print(f'  Sample image: {first_img}, size: {w}x{h}')
assert min(w, h) >= 256, f'Image too small: {w}x{h}, need >= 256 on shortest side'
" 2>/dev/null && pass "Images large enough for 256x256 crop" || fail "Images too small"

# ============================================================
echo ""
echo "--- 3. MODEL TESTS ---"
# ============================================================

echo "3.1 EqM-B/2 model creation"
python -c "
import sys; sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from models import EqM_models
model = EqM_models['EqM-B/2'](input_size=32, num_classes=100, uncond=True, ebm='none')
params = sum(p.numel() for p in model.parameters())
print(f'  EqM-B/2 parameters: {params:,}')
assert params > 100_000_000, f'Expected >100M params, got {params:,}'
" 2>/dev/null && pass "EqM-B/2 model creates successfully (100 classes)" || fail "Cannot create EqM-B/2"

echo "3.2 Model forward pass"
python -c "
import torch, sys
sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from models import EqM_models
model = EqM_models['EqM-B/2'](input_size=32, num_classes=100, uncond=True, ebm='none')
model.eval()
x = torch.randn(2, 4, 32, 32)
t = torch.ones(2)
y = torch.randint(0, 100, (2,))
with torch.no_grad():
    out = model(x, t, y)
print(f'  Input: {x.shape}, Output: {out.shape}')
assert out.shape[0] == 2, 'Batch size mismatch'
assert out.shape[2] == 32 and out.shape[3] == 32, f'Spatial size mismatch: {out.shape}'
" 2>/dev/null && pass "Forward pass works" || fail "Forward pass fails"

echo "3.3 Model return_act"
python -c "
import torch, sys
sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from models import EqM_models
model = EqM_models['EqM-B/2'](input_size=32, num_classes=100, uncond=True, ebm='none')
model.eval()
x = torch.randn(2, 4, 32, 32)
t = torch.ones(2)
y = torch.randint(0, 100, (2,))
with torch.no_grad():
    out, acts = model(x, t, y, return_act=True)
print(f'  Activations: {len(acts)} layers, last shape: {acts[-1].shape}')
features = acts[-1].mean(dim=1)
print(f'  Features shape: {features.shape}')
assert features.shape == (2, 768), f'Expected (2, 768), got {features.shape}'
" 2>/dev/null && pass "return_act works, features shape correct" || fail "return_act fails"

echo "3.4 VAE encode/decode"
python -c "
import torch
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema')
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
    decoded = vae.decode(latent / 0.18215).sample
print(f'  Image: {x.shape} -> Latent: {latent.shape} -> Decoded: {decoded.shape}')
assert latent.shape == (1, 4, 32, 32), f'Latent shape wrong: {latent.shape}'
assert decoded.shape == (1, 3, 256, 256), f'Decoded shape wrong: {decoded.shape}'
" 2>/dev/null && pass "VAE encode/decode works" || fail "VAE fails"

echo "3.5 Transport loss computation"
python -c "
import torch, sys
sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from models import EqM_models
from transport import create_transport
model = EqM_models['EqM-B/2'](input_size=32, num_classes=100, uncond=True, ebm='none')
transport = create_transport('Linear', 'velocity', None, None, None)
x = torch.randn(2, 4, 32, 32)
y = torch.randint(0, 100, (2,))
model_kwargs = dict(y=y, train=True)
loss_dict = transport.training_losses(model, x, model_kwargs)
loss = loss_dict['loss'].mean()
print(f'  Loss: {loss.item():.4f}')
assert not torch.isnan(loss), 'Loss is NaN'
assert not torch.isinf(loss), 'Loss is Inf'
" 2>/dev/null && pass "Transport loss computes" || fail "Transport loss fails"

# ============================================================
echo ""
echo "--- 4. DG-ANM MINING TESTS ---"
# ============================================================

echo "4.1 Geometry estimation"
python -c "
import torch, sys
sys.path.insert(0, 'projects/diff-EqM/experiments')
from train_imagenet import estimate_local_geometry
features = torch.randn(16, 768)
P_T, P_N = estimate_local_geometry(features, k=10)
print(f'  P_T: {P_T.shape}, P_N: {P_N.shape}')
assert P_T.shape == (16, 768, 768)
assert P_N.shape == (16, 768, 768)
# Verify projectors: P_T + P_N should be identity
I = torch.eye(768).unsqueeze(0).expand(16, -1, -1)
diff = (P_T + P_N - I).abs().max().item()
print(f'  P_T + P_N - I max deviation: {diff:.6f}')
assert diff < 1e-4, f'Projector sum deviates from identity: {diff}'
" 2>/dev/null && pass "Geometry estimation correct" || fail "Geometry estimation fails"

echo "4.2 Mine negatives"
python -c "
import torch, sys
sys.path.insert(0, 'projects/diff-EqM/experiments')
sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from train_imagenet import estimate_local_geometry, mine_negatives
from models import EqM_models
model = EqM_models['EqM-B/2'](input_size=32, num_classes=100, uncond=True, ebm='none')
model.eval()
B = 4
x = torch.randn(B, 4, 32, 32)
y = torch.randint(0, 100, (B,))
t_ones = torch.ones(B)
with torch.no_grad():
    _, acts = model(x, t_ones, y, return_act=True)
    features = acts[-1].mean(dim=1)
P_T, P_N = estimate_local_geometry(features, k=min(B-1, 3))
x_neg, info = mine_negatives(model, x, y, features, P_N, P_T, epsilon=0.1, mining_steps=2, mining_lr=0.01)
print(f'  x_neg shape: {x_neg.shape}')
print(f'  Mining info: {info}')
assert x_neg.shape == x.shape, f'Shape mismatch: {x_neg.shape} vs {x.shape}'
# Verify perturbation is within epsilon ball
delta = (x_neg - x).flatten(1).norm(dim=1)
print(f'  Perturbation norms: {delta.tolist()}')
assert (delta <= 0.1 + 1e-5).all(), f'Perturbation exceeds epsilon: {delta.max():.4f}'
" 2>/dev/null && pass "Mine negatives works within epsilon" || fail "Mine negatives fails"

echo "4.3 Negative loss"
python -c "
import torch, sys
sys.path.insert(0, 'projects/diff-EqM/experiments')
sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from train_imagenet import dganm_negative_loss
from models import EqM_models
model = EqM_models['EqM-B/2'](input_size=32, num_classes=100, uncond=True, ebm='none')
model.train()
x = torch.randn(4, 4, 32, 32)
x_neg = x + torch.randn_like(x) * 0.1
y = torch.randint(0, 100, (4,))
loss = dganm_negative_loss(model, x, x_neg.detach(), y, margin=5.0, rho=0.0)
print(f'  Negative loss: {loss.item():.4f}')
assert not torch.isnan(loss), 'Loss is NaN'
loss.backward()
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
print(f'  Gradients flow: {has_grad}')
assert has_grad, 'No gradients from negative loss'
" 2>/dev/null && pass "Negative loss computes with gradients" || fail "Negative loss fails"

# ============================================================
echo ""
echo "--- 5. CHECKPOINT TESTS ---"
# ============================================================

echo "5.1 torch.load with weights_only=False"
python -c "
import torch, sys, tempfile, argparse
sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from models import EqM_models
model = EqM_models['EqM-B/2'](input_size=32, num_classes=100, uncond=True, ebm='none')
# Save checkpoint with argparse.Namespace (like training does)
ckpt = {'model': model.state_dict(), 'ema': model.state_dict(), 'args': argparse.Namespace(test=True)}
with tempfile.NamedTemporaryFile(suffix='.pt') as f:
    torch.save(ckpt, f.name)
    # Load with weights_only=False (our fix)
    loaded = torch.load(f.name, map_location='cpu', weights_only=False)
    assert 'model' in loaded and 'ema' in loaded and 'args' in loaded
    print('  Checkpoint save/load roundtrip OK')
" 2>/dev/null && pass "Checkpoint roundtrip with argparse.Namespace" || fail "Checkpoint roundtrip fails"

echo "5.2 download.py find_model with weights_only=False"
python -c "
import torch, sys, tempfile, argparse
sys.path.insert(0, 'projects/diff-EqM/eqm-upstream')
from download import find_model
from models import EqM_models
model = EqM_models['EqM-B/2'](input_size=32, num_classes=100, uncond=True, ebm='none')
ckpt = {'model': model.state_dict(), 'ema': model.state_dict(), 'args': argparse.Namespace(test=True)}
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    torch.save(ckpt, f.name)
    loaded = find_model(f.name)
    assert 'model' in loaded
    print('  find_model works with argparse checkpoint')
import os; os.unlink(f.name)
" 2>/dev/null && pass "find_model handles argparse checkpoints" || fail "find_model fails"

# ============================================================
echo ""
echo "--- 6. SBATCH SCRIPT TESTS ---"
# ============================================================

echo "6.1 imagenet_train.sbatch syntax"
bash -n projects/diff-EqM/slurm/jobs/imagenet_train.sbatch 2>/dev/null && pass "imagenet_train.sbatch syntax OK" || fail "imagenet_train.sbatch syntax error"

echo "6.2 imagenet_fid.sbatch syntax"
bash -n projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch 2>/dev/null && pass "imagenet_fid.sbatch syntax OK" || fail "imagenet_fid.sbatch syntax error"

echo "6.3 imagenet_smoke_test.sbatch syntax"
bash -n projects/diff-EqM/slurm/jobs/imagenet_smoke_test.sbatch 2>/dev/null && pass "imagenet_smoke_test.sbatch syntax OK" || fail "imagenet_smoke_test.sbatch syntax error"

echo "6.4 IMAGENET_PATH defaults to imagenet100"
grep -q 'imagenet100' projects/diff-EqM/slurm/jobs/imagenet_train.sbatch && pass "train uses imagenet100 default" || fail "train missing imagenet100 default"
grep -q 'imagenet100' projects/diff-EqM/slurm/jobs/imagenet_smoke_test.sbatch && pass "smoke test uses imagenet100 default" || fail "smoke test missing imagenet100 default"

echo "6.5 NUM_CLASSES passed to training"
grep -q 'num-classes.*NUM_CLASSES' projects/diff-EqM/slurm/jobs/imagenet_train.sbatch && pass "train passes NUM_CLASSES" || fail "train missing NUM_CLASSES"
grep -q 'num-classes.*NUM_CLASSES' projects/diff-EqM/slurm/jobs/imagenet_smoke_test.sbatch && pass "smoke test passes NUM_CLASSES" || fail "smoke test missing NUM_CLASSES"
grep -q 'num-classes.*NUM_CLASSES' projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch && pass "FID passes NUM_CLASSES" || fail "FID missing NUM_CLASSES"

echo "6.6 MASTER_PORT randomized"
grep -q 'MASTER_PORT.*RANDOM' projects/diff-EqM/slurm/jobs/imagenet_train.sbatch && pass "train randomizes port" || fail "train missing random port"
grep -q 'MASTER_PORT.*RANDOM' projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch && pass "FID randomizes port" || fail "FID missing random port"

echo "6.7 python -m torch.distributed.run (not torchrun)"
grep -q 'python -m torch.distributed.run' projects/diff-EqM/slurm/jobs/imagenet_train.sbatch && pass "train uses python -m" || fail "train uses torchrun"
grep -q 'python -m torch.distributed.run' projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch && pass "FID uses python -m" || fail "FID uses torchrun"

echo "6.8 FID uses pytorch-fid (not ADM evaluator)"
grep -q 'pytorch_fid' projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch && pass "FID uses pytorch_fid" || fail "FID still uses ADM evaluator"

echo "6.9 wandb import is conditional in train_imagenet.py"
grep -q 'except ImportError' projects/diff-EqM/experiments/train_imagenet.py && pass "wandb import guarded" || fail "wandb import not guarded"

echo "6.10 download.py uses weights_only=False"
grep -q 'weights_only=False' projects/diff-EqM/eqm-upstream/download.py && pass "download.py fixed" || fail "download.py missing weights_only=False"

echo "6.11 sample_gd.py uses args.num_fid_samples"
grep -q 'args.num_fid_samples' projects/diff-EqM/eqm-upstream/sample_gd.py && pass "sample_gd.py uses dynamic sample count" || fail "sample_gd.py hardcodes 50000"

# ============================================================
echo ""
echo "--- 7. DISK SPACE TEST ---"
# ============================================================

echo "7.1 Home directory space"
AVAIL=$(df -BG /n/home03/mkrasnow 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
if [ -n "$AVAIL" ] && [ "$AVAIL" -gt 10 ]; then
    pass "Home has ${AVAIL}GB free (>10GB needed)"
else
    fail "Home has ${AVAIL:-unknown}GB free (need >10GB)"
fi

# ============================================================
echo ""
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    echo "FIX THE FAILURES BEFORE SUBMITTING JOBS"
    exit 1
fi
echo "ALL TESTS PASSED - safe to submit"
exit 0
