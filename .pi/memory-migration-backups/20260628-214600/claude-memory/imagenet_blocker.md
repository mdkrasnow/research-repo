---
name: ImageNet data not on cluster
description: ImageNet directories on FASRC cluster are empty (0 bytes) - blocks ImageNet-scale EqM training
type: project
---

ImageNet data is NOT available on the Harvard FASRC cluster.

**Locations checked (all empty):**
- `/n/netscratch/ydu_lab/Lab/jslee/imagenet/` — 1000 class dirs but 0 files
- `/n/netscratch/hlakkaraju_lab/Everyone/imagenet/` — train/val dirs but 0 files
- No other copies found via broad search

**Why:** Blocks the ImageNet 256x256 EqM-B/2 experiments needed to match the paper's FID 32.85.

**How to apply:** Before submitting ImageNet training jobs, user must either:
1. Download ImageNet-1K (~150GB) to the cluster (requires image-net.org credentials)
2. Ask ydu_lab if they have a working copy on a different filesystem
3. Use HuggingFace datasets (`ILSVRC/imagenet-1k`) with a HF access token
4. Update `IMAGENET_PATH` in `projects/diff-EqM/slurm/jobs/imagenet_*.sbatch`

**Status as of 2026-04-02:** All training scripts, SLURM jobs, and FID evaluation pipeline are implemented and ready. Only missing the data.
