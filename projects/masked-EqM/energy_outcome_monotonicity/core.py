"""Statistics for the preregistered energy-to-outcome pilot."""
from __future__ import annotations

import numpy as np


def within_task_kendall(lower_score: np.ndarray, lower_error: np.ndarray) -> np.ndarray:
    """Kendall-style concordance per task; positive means score ranks outcome."""
    score = np.asarray(lower_score, dtype=np.float64)
    error = np.asarray(lower_error, dtype=np.float64)
    if score.shape != error.shape or score.ndim != 2 or score.shape[1] < 2:
        raise ValueError("scores/errors must be [tasks, candidates>=2]")
    left, right = np.triu_indices(score.shape[1], 1)
    product = (score[:, left] - score[:, right]) * (error[:, left] - error[:, right])
    return np.sign(product).mean(axis=1)


def paired_bootstrap_difference(direct: np.ndarray, dot: np.ndarray, *, replicates: int,
                                seed: int) -> tuple[float, tuple[float, float], np.ndarray]:
    """Image/task-cluster bootstrap for direct-minus-dot concordance."""
    direct = np.asarray(direct, dtype=np.float64)
    dot = np.asarray(dot, dtype=np.float64)
    if direct.shape != dot.shape or direct.ndim != 1 or len(direct) == 0:
        raise ValueError("direct and dot must be same nonempty task vectors")
    delta = direct - dot
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(replicates, len(delta)))
    samples = delta[draws].mean(1)
    return float(delta.mean()), tuple(float(x) for x in np.quantile(samples, [.025, .975])), samples


def shuffled_control(scores: np.ndarray, errors: np.ndarray, *, seed: int) -> np.ndarray:
    """Negative control: independently shuffle candidate scores within every task."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(scores, copy=True)
    for row in shuffled:
        rng.shuffle(row)
    return within_task_kendall(shuffled, errors)
