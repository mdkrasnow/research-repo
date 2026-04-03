#!/bin/bash
# Targeted adversarial tests for the FID evaluation pipeline
# Tests the specific failure modes we've encountered
#
# Usage (on cluster):
#   module load python/3.10.13-fasrc01 cuda/11.8.0-fasrc01
#   bash projects/diff-EqM/tests/test_fid_pipeline.sh

set +e
PASS=0
FAIL=0

pass() { echo "  ✓ PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  ✗ FAIL: $1"; FAIL=$((FAIL+1)); }

echo "============================================================"
echo "FID Pipeline Adversarial Tests"
echo "============================================================"

IMAGENET_REF="/n/home03/mkrasnow/imagenet100/train"
REPO_ROOT="/n/home03/mkrasnow/research-repo"

# ============================================================
echo ""
echo "--- 1. pytorch-fid TESTS ---"
# ============================================================

echo "1.1 pytorch-fid importable"
python -c "from pytorch_fid import fid_score; print('OK')" 2>/dev/null && pass "pytorch_fid imports" || fail "pytorch_fid not importable"

echo "1.2 pytorch-fid glob behavior (THE BUG)"
python -c "
from pytorch_fid.fid_score import IMAGE_EXTENSIONS
import pathlib

# Test: does pytorch-fid find files in nested directories?
path = pathlib.Path('$IMAGENET_REF')
files_flat = sorted([f for ext in IMAGE_EXTENSIONS for f in path.glob('*.{}'.format(ext))])
files_recursive = sorted([f for ext in IMAGE_EXTENSIONS for f in path.rglob('*.{}'.format(ext))])
print(f'  Flat glob (*.ext): {len(files_flat)} files')
print(f'  Recursive glob (**/*.ext): {len(files_recursive)} files')
if len(files_flat) == 0 and len(files_recursive) > 0:
    print('  CONFIRMED: pytorch-fid cannot handle nested dirs!')
    print('  Must flatten reference images before passing to pytorch-fid')
" 2>/dev/null && pass "Verified pytorch-fid glob behavior" || fail "Could not check glob behavior"

echo "1.3 Flat symlink directory creation"
TEST_FLAT="/tmp/test_flat_ref_$$"
mkdir -p "$TEST_FLAT"
find "$IMAGENET_REF" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | head -100 | while read f; do
    CLASSDIR=$(basename "$(dirname "$f")")
    FNAME=$(basename "$f")
    ln -s "$f" "$TEST_FLAT/${CLASSDIR}_${FNAME}" 2>/dev/null
done
FLAT_COUNT=$(ls "$TEST_FLAT" | wc -l)
if [ "$FLAT_COUNT" -gt 0 ]; then
    pass "Created $FLAT_COUNT symlinks in flat dir"
else
    fail "Failed to create flat symlinks"
fi

echo "1.4 pytorch-fid finds files in flat symlink dir"
python -c "
from pytorch_fid.fid_score import IMAGE_EXTENSIONS
import pathlib
path = pathlib.Path('$TEST_FLAT')
files = sorted([f for ext in IMAGE_EXTENSIONS for f in path.glob('*.{}'.format(ext))])
print(f'  Found {len(files)} files in flat dir')
assert len(files) > 0, 'No files found in flat dir!'
" 2>/dev/null && pass "pytorch-fid finds files in flat dir" || fail "pytorch-fid can't find files in flat dir"
rm -rf "$TEST_FLAT"

echo "1.5 pytorch-fid end-to-end with tiny sample"
python -c "
import torch, tempfile, os
from PIL import Image
import numpy as np

# Create 2 tiny dirs with random 256x256 images
dir1 = tempfile.mkdtemp()
dir2 = tempfile.mkdtemp()
for i in range(10):
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(os.path.join(dir1, f'{i:06d}.png'))
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(os.path.join(dir2, f'{i:06d}.png'))

# Run pytorch-fid
from pytorch_fid.fid_score import calculate_fid_given_paths
fid = calculate_fid_given_paths([dir1, dir2], batch_size=10, device='cpu', dims=2048)
print(f'  FID between random images: {fid:.1f}')
assert not np.isnan(fid), 'FID is NaN'
assert fid > 0, 'FID should be positive for random images'

import shutil
shutil.rmtree(dir1)
shutil.rmtree(dir2)
" 2>/dev/null && pass "pytorch-fid end-to-end works" || fail "pytorch-fid end-to-end fails"

# ============================================================
echo ""
echo "--- 2. SAMPLE GENERATION TESTS ---"
# ============================================================

echo "2.1 sample_gd.py syntax"
python -c "
import ast
with open('$REPO_ROOT/projects/diff-EqM/eqm-upstream/sample_gd.py') as f:
    ast.parse(f.read())
print('  Syntax OK')
" 2>/dev/null && pass "sample_gd.py syntax" || fail "sample_gd.py syntax error"

echo "2.2 sample_gd.py uses args.num_fid_samples (not hardcoded 50000)"
grep -q 'args.num_fid_samples' "$REPO_ROOT/projects/diff-EqM/eqm-upstream/sample_gd.py" && pass "Dynamic sample count" || fail "Hardcoded 50000"

echo "2.3 sample_gd.py create_npz_from_sample_folder callable"
python -c "
import sys
sys.path.insert(0, '$REPO_ROOT/projects/diff-EqM/eqm-upstream')
from sample_gd import create_npz_from_sample_folder
import inspect
sig = inspect.signature(create_npz_from_sample_folder)
print(f'  Signature: {sig}')
assert 'num' in sig.parameters, 'Missing num parameter'
" 2>/dev/null && pass "create_npz_from_sample_folder signature OK" || fail "create_npz_from_sample_folder broken"

echo "2.4 download.py uses weights_only=False"
grep -q 'weights_only=False' "$REPO_ROOT/projects/diff-EqM/eqm-upstream/download.py" && pass "weights_only fix present" || fail "weights_only fix missing"

# ============================================================
echo ""
echo "--- 3. CHECKPOINT COMPATIBILITY ---"
# ============================================================

echo "3.1 1-epoch DG-ANM checkpoint exists"
DGANM_CKPT="$REPO_ROOT/projects/diff-EqM/results/imagenet_smoke_test/000-EqM-B-2-Linear-velocity-None-dganm/checkpoints/final.pt"
[ -f "$DGANM_CKPT" ] && pass "DG-ANM 1-epoch checkpoint exists ($(du -h "$DGANM_CKPT" | cut -f1))" || fail "DG-ANM 1-epoch checkpoint missing"

echo "3.2 1-epoch vanilla checkpoint exists"
VANILLA_CKPT="$REPO_ROOT/projects/diff-EqM/results/imagenet_smoke_test/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/final.pt"
[ -f "$VANILLA_CKPT" ] && pass "Vanilla 1-epoch checkpoint exists ($(du -h "$VANILLA_CKPT" | cut -f1))" || fail "Vanilla 1-epoch checkpoint missing"

echo "3.3 80-epoch vanilla checkpoint exists"
VANILLA80_CKPT="$REPO_ROOT/projects/diff-EqM/results/imagenet/000-EqM-B-2-Linear-velocity-None-vanilla/checkpoints/final.pt"
[ -f "$VANILLA80_CKPT" ] && pass "Vanilla 80-epoch checkpoint exists ($(du -h "$VANILLA80_CKPT" | cut -f1))" || fail "Vanilla 80-epoch checkpoint missing"

echo "3.4 Checkpoint loadable with find_model"
if [ -f "$VANILLA_CKPT" ]; then
    python -c "
import sys
sys.path.insert(0, '$REPO_ROOT/projects/diff-EqM/eqm-upstream')
from download import find_model
ckpt = find_model('$VANILLA_CKPT')
print(f'  Keys: {list(ckpt.keys())}')
assert 'model' in ckpt or 'ema' in ckpt, 'Missing model/ema keys'
" 2>/dev/null && pass "Checkpoint loads with find_model" || fail "Checkpoint won't load"
else
    fail "No checkpoint to test"
fi

echo "3.5 Checkpoint num_classes matches model"
if [ -f "$VANILLA_CKPT" ]; then
    python -c "
import sys, torch
sys.path.insert(0, '$REPO_ROOT/projects/diff-EqM/eqm-upstream')
from download import find_model
from models import EqM_models
ckpt = find_model('$VANILLA_CKPT')
state = ckpt.get('ema', ckpt.get('model', ckpt))
embed_shape = state['y_embedder.embedding_table.weight'].shape
num_classes_in_ckpt = embed_shape[0] - 1  # -1 for null class
print(f'  Checkpoint num_classes: {num_classes_in_ckpt} (embedding shape: {embed_shape})')
model = EqM_models['EqM-B/2'](input_size=32, num_classes=num_classes_in_ckpt, uncond=True, ebm='none')
model.load_state_dict(state)
print(f'  Model loads successfully with num_classes={num_classes_in_ckpt}')
" 2>/dev/null && pass "Checkpoint/model num_classes match" || fail "num_classes mismatch"
else
    fail "No checkpoint to test"
fi

# ============================================================
echo ""
echo "--- 4. SBATCH SCRIPT VALIDATION ---"
# ============================================================

echo "4.1 FID sbatch syntax"
bash -n "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" 2>/dev/null && pass "FID sbatch syntax OK" || fail "FID sbatch syntax error"

echo "4.2 FID sbatch uses pytorch-fid (not evaluator.py)"
grep -q 'pytorch_fid' "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" && pass "Uses pytorch_fid" || fail "Still uses evaluator.py"

echo "4.3 FID sbatch flattens reference directory"
grep -q 'flat' "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" && pass "Flattens ref dir" || fail "No ref flattening"

echo "4.4 FID sbatch passes NUM_CLASSES"
grep -q 'NUM_CLASSES' "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" && pass "NUM_CLASSES passed" || fail "NUM_CLASSES missing"

echo "4.5 FID sbatch passes NUM_FID_SAMPLES"
grep -q 'NUM_FID_SAMPLES' "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" && pass "NUM_FID_SAMPLES passed" || fail "NUM_FID_SAMPLES missing"

echo "4.6 FID sbatch uses MASTER_PORT randomization"
grep -q 'MASTER_PORT.*RANDOM' "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" && pass "Random MASTER_PORT" || fail "No random port"

echo "4.7 FID sbatch uses python -m torch.distributed.run"
grep -q 'python -m torch.distributed.run' "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" && pass "Uses python -m" || fail "Uses torchrun"

echo "4.8 FID sbatch verifies sample count after generation"
grep -q 'SAMPLE_COUNT' "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" && pass "Verifies sample count" || fail "No sample count verification"

echo "4.9 FID sbatch does NOT download VIRTUAL_imagenet (no TF needed)"
! grep -q 'VIRTUAL_imagenet' "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" && pass "No VIRTUAL_imagenet download" || fail "Still downloads unused VIRTUAL_imagenet"

echo "4.10 FID sbatch does NOT download evaluator.py"
! grep -q 'evaluator.py' "$REPO_ROOT/projects/diff-EqM/slurm/jobs/imagenet_fid.sbatch" && pass "No evaluator.py download" || fail "Still downloads unused evaluator.py"

# ============================================================
echo ""
echo "============================================================"
echo "RESULTS: $PASS passed, $FAIL failed"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    echo "FIX THE FAILURES BEFORE SUBMITTING FID JOBS"
    exit 1
fi
echo "ALL TESTS PASSED - FID pipeline ready"
exit 0
