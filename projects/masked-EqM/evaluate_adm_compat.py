"""Run the pinned ADM evaluator on modern NumPy without changing its FID math."""

from __future__ import annotations

import runpy
import sys

import numpy as np


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: evaluate_adm_compat.py EVALUATOR.py REFERENCE.npz SAMPLES.npz")
    # The historical ADM evaluator calls np.bool only in its post-FID
    # precision/recall calculation.  NumPy 1.24 removed that alias; restoring
    # it for this subprocess is equivalent to the evaluator's original bool
    # dtype and leaves its FID calculation unchanged.
    if "bool" not in np.__dict__:
        np.bool = bool
    evaluator, reference, samples = sys.argv[1:]
    sys.argv = [evaluator, reference, samples]
    runpy.run_path(evaluator, run_name="__main__")


if __name__ == "__main__":
    main()
