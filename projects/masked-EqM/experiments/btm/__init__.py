"""Corrected-BTM scalar-potential training campaign (2026-08-13).

Hypothesis under test: the late-training pathology of explicit scalar EqM is
caused substantially by the mixed input-parameter derivative required by direct
gradient matching, NOT by the nonexistence of a useful scalar autonomous
transport potential.  See documentation/btm-fd-scalar-plan-2026-08-13.md.
"""

from .interpolant import (  # noqa: F401
    EqMLinearTarget,
    LinearInterpolant,
    SelfStoppingInterpolant,
    build_interpolant,
)
from .fd import (  # noqa: F401
    DoubleBackwardViolation,
    assert_no_double_backward,
    directional_fd,
    exact_directional_derivative,
    exact_gradient,
    rademacher_directions,
)
from .objectives import ARMS, FDConfig, compute_loss  # noqa: F401
