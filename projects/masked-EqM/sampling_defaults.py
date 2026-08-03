"""Paper-matched sampling defaults by EqM model scale."""

PAPER_GD_STEP_SIZE = {
    "EqM-B/2": 0.003,
    "EqM-XL/2": 0.0017,
}


def resolve_gd_step_size(model: str, requested: float | None) -> float:
    """Return an explicit override or the paper's GD step size for ``model``."""
    if requested is not None:
        return requested
    try:
        return PAPER_GD_STEP_SIZE[model]
    except KeyError as exc:
        raise ValueError(
            f"No paper GD step size registered for {model!r}; pass --stepsize explicitly."
        ) from exc
