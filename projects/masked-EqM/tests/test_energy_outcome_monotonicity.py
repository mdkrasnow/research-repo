from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_outcome_monotonicity.core import (paired_bootstrap_difference,
                                              shuffled_control, within_task_kendall)


def test_kendall_is_positive_for_perfect_energy_ranking():
    score = np.array([[1., 2., 3.], [4., 5., 6.]])
    assert np.allclose(within_task_kendall(score, score), 1.)


def test_kendall_is_negative_for_reversed_ranking():
    score = np.array([[1., 2., 3.]])
    assert np.allclose(within_task_kendall(score, score[:, ::-1]), -1.)


def test_bootstrap_retains_positive_paired_effect():
    direct = np.array([.8, .6, .4, .2])
    dot = np.zeros(4)
    mean, ci, _ = paired_bootstrap_difference(direct, dot, replicates=500, seed=3)
    assert mean > 0 and ci[0] > 0


def test_shuffled_control_preserves_shape_without_mutating_input():
    scores = np.arange(20.).reshape(5, 4)
    errors = scores.copy()
    original = scores.copy()
    result = shuffled_control(scores, errors, seed=4)
    assert result.shape == (5,)
    assert np.array_equal(scores, original)
