"""Projection solve for homogeneous scalar-inequality Newton models.

For ``H = L L.T``, the coordinates ``x = L.T theta`` turn the model into
Euclidean projection of its unconstrained optimum ``u`` onto ``C x >= 0``.
Moreau decomposition reduces its dual to ``nnls(C.T, -u)``. The active span
is small even when the curvature exchange has retained many similar rows.
The returned gap is checked against the original model, not the Cholesky
factor's nearby model. A direction may be useful even when that bound cannot
be resolved; an infinite bound never licenses convergence.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

from .._defaults import CONIC_QP_GAP_TOL, NUMERIC_FAILURES, _reraise_if_debug

_WIDE = np.longdouble
"""Accumulation type of the certificate: extended precision on x86 Linux, but
plain binary64 on Windows and macOS arm64, so the certificate must hold in
both."""


@dataclass(frozen=True)
class _LinearQPResult:
    params: np.ndarray
    dual: np.ndarray
    model_value: float
    gap: float


def _strict_direction(rows):
    """Projection solution of a small homogeneous max-margin problem.

    Project ``(0, 1)`` onto ``rows @ v >= t >= 0``. A positive ``t``
    supplies a strict direction without a separate linear-program solver or
    process-global scheduler state.

    Parameters
    ----------
    rows : numpy.ndarray
        Normalized constraint rows, shape ``(m, n)``.

    Returns
    -------
    numpy.ndarray or None
        A strict direction, or ``None`` when the margin is not resolved.
    """
    n, m = rows.shape[1], rows.shape[0]
    augmented = np.vstack([np.c_[rows, -np.ones(m)], np.eye(n + 1)[-1]])
    target = np.eye(n + 1)[-1]
    dual, _ = nnls(augmented.T, -target, maxiter=10 * max(augmented.shape))
    point = target + augmented.T @ dual
    direction, margin = point[:-1], point[-1]
    if margin <= 0.0 or np.min(rows @ direction) < 0.5 * margin:
        return None
    return direction


def _linear_subproblem(hessian, gradient, params, rows, /):
    """Solve a positive-definite quadratic over ``rows @ theta >= 0``.

    All returned bounds concern the coefficient-space quadratic. A lower
    eigenvalue estimate includes a conservative matrix/eigensolver rounding
    allowance. When it is not positive the bound is infinite, even if NNLS
    finds a useful direction. Residuals and model values are accumulated in
    extended precision where available, with an arithmetic error allowance.

    Parameters
    ----------
    hessian : numpy.ndarray
        Model metric, shape ``(n, n)``.
    gradient, params : numpy.ndarray
        Model gradient and current point, shape ``(n,)``.
    rows : numpy.ndarray
        Homogeneous constraint rows, shape ``(m, n)``.

    Returns
    -------
    _LinearQPResult or None
        The endpoint, or ``None`` when the projection is unresolved (a
        non-finite model, a non-positive metric diagonal, a null or
        non-finite row, an unresolved interior or endpoint): the caller
        defers to the interior-point kernel.
    """
    h = np.asarray(hessian, dtype=float)
    g = np.asarray(gradient, dtype=float)
    theta = np.asarray(params, dtype=float)
    b = np.asarray(rows, dtype=float)
    diagonal = np.diag(h)
    finite = all(np.all(np.isfinite(v)) for v in (h, g, theta, b))
    if not finite or np.any(diagonal <= 0.0):
        return None
    d = 1.0 / np.sqrt(diagonal)
    hs = d[:, None] * (0.5 * (h + h.T)) * d[None, :]
    gs = d * g
    t = theta / d
    bs = b * d
    magnitude = np.linalg.norm(bs, axis=1)
    if np.any(magnitude <= 0.0) or not np.all(np.isfinite(magnitude)):
        return None
    bs = bs / magnitude[:, None]
    factor = np.linalg.cholesky(hs)
    a = solve_triangular(factor, bs.T, lower=True)
    u = factor.T @ t - solve_triangular(factor, gs, lower=True)
    z, _ = nnls(a, -u, maxiter=10 * max(a.shape))
    active = z > 0.0
    if np.any(active):
        vectors, singular, _ = np.linalg.svd(a[:, active], full_matrices=True)
        cut = np.finfo(float).eps * max(a.shape) * singular[0]
        rank = int(np.count_nonzero(singular > cut))
        null = vectors[:, rank:]
        x = null @ (null.T @ u)
    else:
        x = u
    p = solve_triangular(factor.T, x, lower=False)

    # A floating projection can miss active inequalities. Restore the QP
    # endpoint itself to a slack margin above the rounding of re-forming its
    # slacks from the returned endpoint (without extended precision, an
    # active row restored only to zero slack re-forms slightly negative);
    # its new objective and residual are certified below.
    slack = bs @ p
    floor = 16.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(bs) @ np.abs(p))
    direction = _strict_direction(bs) if np.min(slack - floor) < 0.0 else None
    if direction is not None and np.min(bs @ direction) > 0.0:
        margin = bs @ direction
        p = p + max(0.0, float(np.max((floor - slack) / margin))) * direction
    elif np.min(slack) < 0.0:
        return None

    endpoint = d * p
    wide = _WIDE
    dw = np.asarray(d, dtype=wide)
    hw = np.asarray(h, dtype=wide)
    hw = 0.5 * (hw + hw.T) * dw[:, None] * dw[None, :]
    gw = np.asarray(g, dtype=wide) * dw
    bw = np.asarray(b, dtype=wide) * dw / np.asarray(magnitude, dtype=wide)[:, None]
    zw = np.asarray(z, dtype=wide)
    pw = np.asarray(endpoint, dtype=wide) / dw
    delta = (np.asarray(endpoint, dtype=wide) - np.asarray(theta, dtype=wide)) / dw
    slacks = bw @ pw
    stationarity = hw @ delta + gw - bw.T @ zw
    eps = np.finfo(wide).eps
    error = (
        16.0
        * max(b.shape)
        * eps
        * (np.abs(hw) @ np.abs(delta) + np.abs(gw) + np.abs(bw).T @ np.abs(zw))
    )
    lower = float(np.linalg.eigvalsh(hs)[0]) - (
        32.0 * t.size * np.finfo(float).eps * np.linalg.norm(hs, ord=np.inf)
    )
    gap = np.inf
    if lower > 0.0 and np.min(slacks) >= 0.0:
        pairing = zw @ slacks
        pairing_error = (
            16.0 * max(b.shape) * eps * (np.abs(zw) @ (np.abs(bw) @ np.abs(pw)))
        )
        gap = float(
            pairing
            + pairing_error
            + 0.5 * np.sum((np.abs(stationarity) + error) ** 2) / lower
        )
    model = float(gw @ delta + 0.5 * delta @ hw @ delta)
    if np.isfinite(gap):
        gap += float(
            16.0
            * t.size
            * eps
            * (
                np.abs(gw) @ np.abs(delta)
                + 0.5 * np.abs(delta) @ np.abs(hw) @ np.abs(delta)
            )
            + np.finfo(float).eps * abs(model)
        )
    if not np.all(np.isfinite(endpoint)) or not np.isfinite(model):
        return None
    return _LinearQPResult(endpoint, -z / magnitude, model, gap)


def _projection_candidate(hessian, gradient, params, rows, /):
    """Return a resolved scalar QP endpoint, or defer to the conic kernel.

    Kernel eligibility uses the common quadratic gap target in the original
    model. An unresolved projection defers to the interior-point solver.
    This decision has no knowledge of basis families.

    Parameters
    ----------
    hessian : numpy.ndarray
        Model metric, shape ``(n, n)``.
    gradient, params : numpy.ndarray
        Model gradient and current point, shape ``(n,)``.
    rows : numpy.ndarray
        Homogeneous constraint rows, shape ``(m, n)``.

    Returns
    -------
    _LinearQPResult or None
        A resolved endpoint, or ``None`` to defer to the interior-point kernel.
    """
    try:
        result = _linear_subproblem(hessian, gradient, params, rows)
    except NUMERIC_FAILURES as exc:
        # Routine: NumPy/SciPy could not factor or solve this model (an
        # indefinite metric, an NNLS iteration limit); the interior-point
        # kernel solves the same subproblem.
        _reraise_if_debug(exc, "scalar QP projection", routine=True)
        return None
    if result is None:
        return None
    limit = CONIC_QP_GAP_TOL * max(1.0, abs(result.model_value))
    if not np.isfinite(result.gap) or result.gap > limit:
        return None
    bound = max(0.0, -result.model_value) + max(0.0, result.gap)
    slope = float(gradient @ (result.params - params))
    if not np.isfinite(slope) or (bound > limit and slope >= 0.0):
        return None
    return result
