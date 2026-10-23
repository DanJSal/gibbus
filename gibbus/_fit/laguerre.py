"""Compatibility entry point for the generalized polynomial coordinate planner."""

from .coordinates import _polynomial_coordinates


def _laguerre_model(model, stats, start):
    """Return half-line recurrence coordinates when observations are available."""
    if stats.points is None or model.polynomial_coordinates is not None:
        return None
    numerical = _polynomial_coordinates(model, stats.points, stats.weights, start)
    plan = numerical.polynomial_coordinates
    return numerical if plan is not None and plan.family == "laguerre" else None
