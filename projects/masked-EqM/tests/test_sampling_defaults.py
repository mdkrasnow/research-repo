import pytest

from sampling_defaults import resolve_gd_step_size


def test_paper_step_size_for_b2():
    assert resolve_gd_step_size("EqM-B/2", None) == pytest.approx(0.003)


def test_paper_step_size_for_xl2():
    assert resolve_gd_step_size("EqM-XL/2", None) == pytest.approx(0.0017)


def test_explicit_step_size_override_is_preserved():
    assert resolve_gd_step_size("EqM-B/2", 0.0025) == pytest.approx(0.0025)


def test_unknown_model_requires_explicit_step_size():
    with pytest.raises(ValueError, match="pass --stepsize explicitly"):
        resolve_gd_step_size("EqM-S/2", None)
