"""Typed errors raised by the public API layer.

These live in their own module so the natural mixture machinery and fitting controller
can share typed control-flow failures without creating cross-module cycles.
"""


class _PointMixtureEstimabilityError(RuntimeError):
    """Structural point-mixture singularity, not a recoverable seed failure."""
