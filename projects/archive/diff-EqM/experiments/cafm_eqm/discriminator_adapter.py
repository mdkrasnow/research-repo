"""Reuse Lin et al.'s CAFM discriminator with EqM γ in place of flow-matching t.

Why no new discriminator class:
- Lin's `models/cafm/sit/discriminator.Discriminator` accepts (x, y, t) where t ∈
  [0,1]. EqM's γ shares the same convention (γ=0 → image, γ=1 → noise, matching
  CAFM's `(1-t)·x + t·ε`).
- Lin's `models/cafm/jvp/discriminator.DiscriminatorJVP` wraps the above with
  `torch.func.jvp` over (x, t). Passing γ as t and `dt = ones_like(γ)` for the
  time-tangent gives a γ-directional JVP, which is the correct EqM analog.
- The `t = 1.0 - t` flip in Lin's Discriminator.forward is a SiT pretraining
  convention compensator. EqM uses the SAME convention as CAFM training already
  (x_γ = (1-γ)·x + γ·ε with x0=image, x1=noise), so the flip is consistent for
  our use too. No code change needed.

This module provides a path-based import helper so we don't have to bake Lin's
repo as a pip dependency. The Lin code lives in
`projects/diff-EqM/external/Adversarial-Flow-Models/` (gitignored).

Usage:
    from cafm_eqm.discriminator_adapter import load_cafm_discriminator_classes
    Discriminator, DiscriminatorJVP = load_cafm_discriminator_classes()
    dis = Discriminator(**dit_b2_config)
    dis_jvp = DiscriminatorJVP(dis)
    # In training step:
    out, out_jvp = dis_jvp(x=x_gamma, y=labels, t=gamma, dx=tangent_x, dt=tangent_t)
"""
from __future__ import annotations

import sys
from pathlib import Path


# cafm_eqm/discriminator_adapter.py → parents: [cafm_eqm, experiments, diff-EqM, projects, repo]
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # projects/diff-EqM/
EXTERNAL_AFM_ROOT = PROJECT_ROOT / "external" / "Adversarial-Flow-Models"


def _ensure_external_path():
    """Add the cloned AFM repo to sys.path so its `models.cafm.*` imports work."""
    if not EXTERNAL_AFM_ROOT.exists():
        raise RuntimeError(
            f"External AFM repo not found at {EXTERNAL_AFM_ROOT}. "
            "Clone it via:\n"
            "  cd projects/diff-EqM/external/\n"
            "  git clone https://github.com/ByteDance-Seed/Adversarial-Flow-Models.git\n"
        )
    p = str(EXTERNAL_AFM_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_cafm_discriminator_classes():
    """Return (Discriminator, DiscriminatorJVP) classes from the cloned AFM repo.

    Approach: Lin's `models/cafm/` is a namespace package (no __init__.py).
    EqM upstream has a top-level `models.py` that shadows it irreversibly via
    Python's import finders. Rather than fight Python's namespace resolution,
    we copy Lin's `models/cafm/` directory to a uniquely-named local staging
    directory, add a stub __init__.py, and import via that distinct name.

    Lin's internal relative imports (e.g. `from .sit_mod import ...`) work
    because they are relative within the cafm sub-package.
    """
    _ensure_external_path()
    src_cafm = EXTERNAL_AFM_ROOT / "models" / "cafm"
    sit_path = src_cafm / "sit" / "sit.py"
    if not sit_path.exists():
        raise RuntimeError(
            f"Missing {sit_path}. Per Lin's CAFM README:\n"
            "  Download sit.py from https://github.com/willisma/SiT/blob/main/models.py\n"
            f"  and place it at {sit_path}\n"
        )

    # Stage to a unique top-level package name to avoid `models` collision.
    # Sbatch pre-stages this (single-threaded, before DDP fork). If missing,
    # fall back to runtime copy with a lock for DDP-safety.
    staging = PROJECT_ROOT / "external" / "cafm_pkg"
    cafm_dest = staging / "cafm"
    if not cafm_dest.exists():
        # Runtime fallback. Use file lock to serialize across DDP ranks.
        import fcntl
        import shutil
        staging.mkdir(parents=True, exist_ok=True)
        lock_path = staging / ".stage.lock"
        with open(lock_path, "w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not cafm_dest.exists():
                (staging / "__init__.py").touch(exist_ok=True)
                shutil.copytree(src_cafm, cafm_dest)
                for d in [cafm_dest, *(p for p in cafm_dest.rglob("*") if p.is_dir())]:
                    (d / "__init__.py").touch(exist_ok=True)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    parent = str(staging.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from cafm_pkg.cafm.sit.discriminator import Discriminator  # noqa: E402
    from cafm_pkg.cafm.jvp.discriminator import DiscriminatorJVP  # noqa: E402
    return Discriminator, DiscriminatorJVP


__all__ = ["EXTERNAL_AFM_ROOT", "PROJECT_ROOT", "load_cafm_discriminator_classes"]
