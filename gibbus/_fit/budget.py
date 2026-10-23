"""Cooperative wall-time budgets shared by every stage of one fit."""

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Real


class FitTimeoutError(TimeoutError):
    """A fitting request exhausted its wall-time budget.

    Attributes
    ----------
    max_fit_seconds : float
        Requested budget in seconds.
    elapsed_seconds : float
        Elapsed wall time when a fitting checkpoint detected exhaustion.
    stage : str
        Fitting stage at that checkpoint.
    """

    def __init__(self, max_fit_seconds, elapsed_seconds, stage):
        self.max_fit_seconds = float(max_fit_seconds)
        self.elapsed_seconds = float(elapsed_seconds)
        self.stage = str(stage)
        super().__init__(
            f"fit exceeded max_fit_seconds={self.max_fit_seconds:g} "
            f"({self.elapsed_seconds:.3f} seconds elapsed; {self.stage})"
        )

    def __reduce__(self):
        return (type(self), (self.max_fit_seconds, self.elapsed_seconds, self.stage))


@dataclass(frozen=True)
class _Deadline:
    started: float
    seconds: float


_deadline = ContextVar("gibbus_fit_deadline", default=None)


def _check_fit_time(stage="fitting"):
    """Raise at a checkpoint when the active request's budget has expired."""
    deadline = _deadline.get()
    if deadline is not None:
        elapsed = time.monotonic() - deadline.started
        if elapsed >= deadline.seconds:
            raise FitTimeoutError(deadline.seconds, elapsed, stage)


@contextmanager
def _fit_time_budget(seconds):
    """Share the earliest deadline with nested work and restore it on exit."""
    current = _deadline.get()
    token = None
    if seconds is not None:
        if isinstance(seconds, bool) or not isinstance(seconds, Real):
            raise TypeError("max_fit_seconds must be a positive finite number or None")
        try:
            seconds = float(seconds)
        except OverflowError:
            raise ValueError(
                "max_fit_seconds must be a positive finite number or None"
            ) from None
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("max_fit_seconds must be a positive finite number or None")
        started = time.monotonic()
        if current is None or started + seconds < current.started + current.seconds:
            token = _deadline.set(_Deadline(started, seconds))
    try:
        _check_fit_time("fit preparation")
        yield
        _check_fit_time("fit completion")
    finally:
        if token is not None:
            _deadline.reset(token)
