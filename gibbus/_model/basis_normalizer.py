"""Normalized candidate state of a G3 basis density ``p = exp(-q) / Z``.

G1 owner replaced: ``_model/natural_state.py`` (``_NaturalCoreState``),
``_model/numerics.py``, ``_model/moments.py`` and the normalization/Fisher
traversal of ``_model/_state_kernels.pyx``.  As in G1, one state object binds
a coefficient vector to its normalization and model moments and is what the
objectives, solver and (Stage 5) postfit consume.  Replaced: G1's
mode-bracketed QUADPACK windows, which assume a convex polynomial potential;
G3 trial potentials need not be convex and are not polynomial.

For a coefficient vector ``theta`` this module returns ``log Z``,
``E[Phi]`` and ``Cov[Phi]`` (the point-likelihood gradient and exact Fisher
matrix) from one shared set of quadrature nodes, with an error estimate.
There is no fallback path: a configuration the algorithm cannot resolve to
the requested accuracy raises :class:`_QuadratureFailure` (G3-030).

1. **Properness** (``basis_tails``): at each infinite end the leading term of
   ``dq/dzeta`` must be positive and non-decaying.  If not, ``Z = inf``
   exactly and the result says ``improper``.  No convexity is assumed.
2. **Tail truncation**: beyond the growth cutoff ``dq/dzeta >= s(zeta)``,
   nondecreasing, so ``q(x) >= q(x_T) + s_x |x - x_T|`` and
   ``int_{x_T}^inf e^{-q} <= e^{-q(x_T)} / s_x``.  The moment tails use the
   same bound times a growth envelope ``|Phi_c| <= C zeta^b e^{r zeta}``:
   ``int_T^inf zeta^b e^{-k zeta} <= 2 T^b e^{-k T} / k`` for
   ``T >= max(0, 2 b / k)``.  ``x_T`` moves outward until these bounds are
   negligible.
3. **Cells**: interval enclosures ``q(X) in [q_lo, q_hi]`` bound each cell's
   mass between ``|X| e^{-q_hi}`` and ``|X| e^{-q_lo}`` (times column
   magnitudes for the moments).  Cells whose upper bound is negligible against
   the guaranteed lower bound of ``Z`` are dropped (their bounds join the
   error budget); cells where ``q`` may vary by more than ``V`` are split, so
   no region carrying mass can be missed, however narrow or multimodal.
4. **Integration**, all kept cells batched; each cell's rule is fixed by the
   integrand's analytic structure on it (G3-050).  On a cell where
   ``e^{-q}`` is analytic, the Gauss--Kronrod 10/21 rule: the Kronrod value,
   settled when its contributions to ``Z``, ``E[Phi]`` and ``E[Phi^2]``
   differ from the embedded Gauss rule's by less than the cell's share of
   the tolerance (that difference is its error).  On a cell ending at an
   enabled endpoint log, where ``e^{-q}`` behaves as ``d^gamma``, nested
   tanh-sinh, halving the step until consecutive levels agree within the
   same share.  Cells that do not settle are split.

Values are shifted by a reference potential before exponentiation, and the
covariance is computed in a second, centred pass.  Truncation also requires
the moment tails to be negligible against the column scales (the moments of
an exponential column under a flat potential live far beyond the mass), and
where ``e^{-(q - q_ref)}`` is subnormal the moment products are formed in
logs (G3-043).

**Interval rows** (``_interval_state``, G3-043).  The mass ``P(B)`` and
conditional moments ``E[Phi | B]``, ``Cov(Phi | B)`` of each observation row
``B`` are resolved to ``rel_tol`` relative to ``P(B)`` itself, so small
masses far in a tail keep their relative accuracy.  A row containing settled
cells of the whole-support integration sums them (prefix/suffix sums) and
integrates only the two cut end pieces, when the settled cells' error budget
is within half the tolerance relative to the row; otherwise the row is its
own domain of the same integrator (reference, truncation, cells,
integration).  Point rows are log densities and whole-support rows the model.
The weighted aggregates ``sum w log P``, ``sum w E[Phi | B]`` and
``sum w Cov(Phi | B)`` are reduced in C.
"""

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from .._defaults import (
    NORMALIZER_DROP_TOL,
    NORMALIZER_INITIAL_CELLS,
    NORMALIZER_MAX_CELLS,
    NORMALIZER_MAX_LEVEL,
    NORMALIZER_REL_TOL,
    NORMALIZER_VARIATION,
)
from ._normalizer_kernels import (
    STATUS_ACCURACY as _NK_ACCURACY,
)
from ._normalizer_kernels import (
    STATUS_CELLS_INTEGRATE as _NK_CELLS_INTEGRATE,
)
from ._normalizer_kernels import (
    STATUS_CELLS_RESOLVE as _NK_CELLS_RESOLVE,
)
from ._normalizer_kernels import (
    STATUS_IMPROPER as _NK_IMPROPER,
)
from ._normalizer_kernels import (
    STATUS_NO_REFERENCE as _NK_NO_REFERENCE,
)
from ._normalizer_kernels import (
    STATUS_PROPER as _NK_PROPER,
)
from ._normalizer_kernels import (
    STATUS_ROW_POINT as _NK_ROW_POINT,
)
from ._normalizer_kernels import (
    STATUS_TAIL_CAP as _NK_TAIL_CAP,
)
from ._normalizer_kernels import (
    STATUS_TAIL_NOCONV as _NK_TAIL_NOCONV,
)
from ._normalizer_kernels import (
    STATUS_TAIL_NONFINITE as _NK_TAIL_NONFINITE,
)
from ._normalizer_kernels import (
    STATUS_UNDECIDED as _NK_UNDECIDED,
)
from ._normalizer_kernels import (
    GrowthKernel,
    basis_state,
    interval_states,
)
from .basis import _LOWER
from .basis_eval import _evaluate_basis
from .basis_tails import _build_tail_model, _growth_verdict, _tail_on_face


class _ImproperPotential(RuntimeError):
    """Typed signal that a candidate has ``Z = inf`` (exactly decided).

    G1 objectives raise ``RuntimeError`` for non-normalizable candidates;
    this subclass lets solvers reject the step without matching messages.
    """


class _QuadratureFailure(RuntimeError):
    """The normalizer could not reach its accuracy target (fails loudly).

    Parameters
    ----------
    message : str
        What failed.
    diagnostics : dict
        Cells, levels, error budget and the stage that failed.
    """

    def __init__(self, message, diagnostics, /):
        """Store the message and the diagnostics."""
        super().__init__(f"{message}: {diagnostics}")
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class _NormalizerOptions:
    """Normalizer settings (G3-030 defaults).

    Parameters
    ----------
    rel_tol : float
        Target relative error of ``Z`` and of the moments.
    initial_cells : int
        Asinh-spaced initial cells.
    max_cells : int
        Cell budget (resolution plus splits of unsettled cells).
    max_level : int
        Finest tanh-sinh level (step ``2**-level``) of a cell ending at an
        endpoint log before it is split.
    variation : float
        Largest enclosed variation of ``q`` over a cell before it is split.
    drop_tol : float
        Total relative mass that dropped cells may carry.
    zeta_cap : float
        Largest growth cutoff tried, in units of ``h``.
    """

    rel_tol: float = NORMALIZER_REL_TOL
    initial_cells: int = NORMALIZER_INITIAL_CELLS
    max_cells: int = NORMALIZER_MAX_CELLS
    max_level: int = NORMALIZER_MAX_LEVEL
    variation: float = NORMALIZER_VARIATION
    drop_tol: float = NORMALIZER_DROP_TOL
    zeta_cap: float = 1e15


@dataclass(frozen=True)
class _BasisState:
    """Normalized candidate state for one coefficient vector.

    The G3 counterpart of G1's ``_NaturalCoreState``.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    theta : numpy.ndarray
        Coefficients, shape ``(d,)`` (a private copy).
    status : str
        ``"proper"`` or ``"improper"`` (``log_z = +inf``, no moments).
    log_z : float
        ``log Z`` in user units.
    mean : numpy.ndarray or None
        ``E[Phi]``, shape ``(d,)``.
    cov : numpy.ndarray or None
        ``Cov[Phi]``, shape ``(d, d)``.
    rel_error : float
        Estimated relative error bound of ``Z`` (quadrature, dropped cells,
        tails).
    cond_error : float
        Relative effect on ``Z`` of the binary64 rounding of ``q`` itself
        (first order, ``(d + 4) u sum |theta_c Phi_c|`` per node): the floor
        no quadrature of the floating-point potential can beat.  Reported,
        not counted against the target.
    nodes : numpy.ndarray or None
        Quadrature nodes (user units).
    weights : numpy.ndarray or None
        Probability weights at the nodes (sum to one).
    cells : int
        Cells integrated.
    """

    model: object
    theta: np.ndarray
    status: str
    log_z: float
    mean: np.ndarray | None
    cov: np.ndarray | None
    rel_error: float
    cond_error: float
    nodes: np.ndarray | None
    weights: np.ndarray | None
    cells: int


@dataclass
class _NormalizerContext:
    """Theta-independent data for one compiled basis (order-1 tail atoms)."""

    model: object
    tails: dict = field(default_factory=dict)
    growth: dict = field(default_factory=dict)
    column_scale: np.ndarray | None = None
    growth_kernels: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate the fields."""
        for side in ("lower", "upper"):
            if not math.isfinite(getattr(self.model.request, side)):
                self.tails[side] = _build_tail_model(self.model, side, order=1)
                self.growth[side] = _column_growth(self.model, self.tails[side])
                self.growth_kernels[side] = GrowthKernel(self.growth[side])
        # Natural magnitude of each column where the data live (c +- 4h):
        # moment accuracy is measured against it, so a column that vanishes
        # where a density happens to sit is not held to a meaningless
        # relative standard.
        req, g = self.model.request, self.model.geometry
        x = g.center + g.scale * np.linspace(-4.0, 4.0, 33)
        x = x[(x > req.lower) & (x < req.upper)]
        phi = _evaluate_basis(self.model, x, max_order=0).phi
        mag = np.max(np.where(np.isfinite(phi), np.abs(phi), 0.0), axis=0)
        self.column_scale = np.ascontiguousarray(np.where(mag > 0.0, mag, 1.0))

    def on_face(self, basis, /):
        """Context for points of a face ``theta = N phi`` (exact-zero atoms dropped).

        Parameters
        ----------
        basis : numpy.ndarray
            Orthonormal face basis ``N``.
        """
        out = copy.copy(self)
        out.tails = {s: _tail_on_face(t, basis) for s, t in self.tails.items()}
        return out


def _column_growth(model, tail, /):
    """Per column, atoms ``(C, b, r)`` with ``|Phi_c| <= sum C zeta^b e^{r zeta}``.

    Valid for ``zeta >= tail.zeta_min`` (raised to keep log arguments >= 1).
    Uses ``|G| <= |alpha_0| + |alpha_1| + |alpha_2|``.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    tail : _TailGeometry
        One infinite side's tail geometry.
    """
    geometry, request = model.geometry, model.request
    h, c = geometry.scale, geometry.center
    sgn = tail.sign
    out = []
    for column in model.columns:
        atoms = []
        for j, gate in column.terms:
            fn = model.functions[j]
            gmag = float(abs(gate[0]) + abs(gate[1]) + abs(gate[2]))
            if fn.kind == "monomial":
                atoms.append((gmag, float(fn.degree), 0.0))
            elif fn.kind == "soft_power":
                eps = fn.delta / h**2 / tail.zeta_min**2
                atoms.append((gmag * (1.0 + eps) ** (0.5 * fn.power), fn.power, 0.0))
            elif fn.kind == "exponential":
                lam = fn.rate * h
                zloc = (fn.location - c) / h
                if fn.side == "upper":
                    r_f, log_k = sgn * lam, -lam * zloc
                else:
                    r_f, log_k = -sgn * lam, lam * zloc
                atoms.append((gmag * math.exp(log_k), 0.0, r_f))
            else:  # log of the opposite endpoint: |log(h (zeta + c0))|
                c0 = (
                    -(request.lower - c) / h
                    if fn.side == _LOWER
                    else (request.upper - c) / h
                )
                # log(zeta + c0) <= zeta + |c0| for zeta + c0 >= 1.
                atoms.append((gmag * (abs(math.log(h)) + abs(c0)), 0.0, 0.0))
                atoms.append((gmag, 1.0, 0.0))
        out.append(atoms)
    return out


_FAILURES = {
    _NK_UNDECIDED: ("properness undecided", "properness"),
    _NK_TAIL_CAP: ("tail truncation beyond the cutoff cap", "tails"),
    _NK_TAIL_NOCONV: ("tail truncation did not converge", "tails"),
    _NK_TAIL_NONFINITE: ("potential not finite at the tail truncation point", "tails"),
    _NK_CELLS_RESOLVE: ("cell budget exhausted while resolving", "cells"),
    _NK_CELLS_INTEGRATE: ("cell budget exhausted while integrating", "integration"),
    _NK_ACCURACY: ("accuracy target not met", "budget"),
    _NK_NO_REFERENCE: ("no finite potential on the reference grid", "tails"),
    _NK_ROW_POINT: ("potential not finite at a point observation", "rows"),
}


def _failure(context, theta, opts, status, side, diag, /):
    """The ``_QuadratureFailure`` of a compiled normalizer status.

    Parameters
    ----------
    context : _NormalizerContext
        Normalizer of the basis.
    theta : numpy.ndarray
        Coefficients.
    opts : _NormalizerOptions
        Normalizer policy.
    status : int
        Compiled normalizer status.
    side : int
        Side code.
    diag : float
        Diagnostic value.
    """
    message, stage = _FAILURES[status]
    info = {"stage": stage}
    name = ("lower", "upper")[side] if side >= 0 else None
    if status in (_NK_UNDECIDED, _NK_TAIL_CAP, _NK_TAIL_NOCONV, _NK_TAIL_NONFINITE):
        info["side"] = name
    if status in (_NK_TAIL_CAP, _NK_TAIL_NONFINITE):
        info["zeta"] = diag[0]
    if status in (_NK_UNDECIDED, _NK_TAIL_CAP):
        info["slope"] = _growth_verdict(
            context.tails[name], context.model, theta, zeta_cap=opts.zeta_cap
        ).slope
    if status == _NK_CELLS_RESOLVE:
        info.update(cells=int(diag[0]), open=int(diag[1]))
    if status == _NK_CELLS_INTEGRATE:
        info.update(cells=int(diag[0]), unsettled=int(diag[1]))
    if status == _NK_ACCURACY:
        info.update(
            rel_error=diag[0],
            moment_rel_error=diag[1],
            quadrature=diag[2],
            dropped=diag[3],
            tails=diag[4],
        )
    return _QuadratureFailure(message, info)


def _basis_state(context, theta, /, options=None):
    """Compute ``log Z``, ``E[Phi]`` and ``Cov[Phi]`` for ``theta``.

    The computation runs in ``_normalizer_kernels.basis_state_c`` (method in
    the module docstring).

    Parameters
    ----------
    context : _NormalizerContext
        Compiled-basis data.
    theta : numpy.ndarray
        Coefficients, shape ``(d,)``; structurally nonnegative log amplitudes
        must be ``>= 0``.
    options : _NormalizerOptions or None
        Settings.

    Returns
    -------
    _BasisState

    Raises
    ------
    ValueError
        If a log amplitude is negative (outside the model).
    _QuadratureFailure
        If properness is undecidable within the cap, or the cell budget is
        exhausted before the accuracy target is met.
    """
    opts = options or _NormalizerOptions()
    model = context.model
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    if any(col.nonnegative and theta[i] < 0.0 for i, col in enumerate(model.columns)):
        raise ValueError("log amplitudes must be nonnegative")
    tl, tu = context.tails.get("lower"), context.tails.get("upper")
    gl, gu = context.growth_kernels.get("lower"), context.growth_kernels.get("upper")
    (
        status,
        _,
        side,
        log_z,
        mean,
        cov,
        rel_error,
        cond_error,
        nodes,
        weights,
        cells,
        diag,
    ) = basis_state(
        model.kernel,
        theta,
        None if tl is None else tl.kernel,
        None if tu is None else tu.kernel,
        gl,
        gu,
        context.column_scale,
        opts,
    )
    if status == _NK_IMPROPER:
        return _BasisState(
            model,
            theta.copy(),
            "improper",
            math.inf,
            None,
            None,
            0.0,
            0.0,
            None,
            None,
            0,
        )
    if status != _NK_PROPER:
        raise _failure(context, theta, opts, status, side, diag)
    return _BasisState(
        model,
        theta.copy(),
        "proper",
        log_z,
        mean,
        cov,
        rel_error,
        cond_error,
        nodes,
        weights,
        cells,
    )


@dataclass(frozen=True)
class _IntervalState:
    """Normalized state plus the weighted conditional moments of interval rows.

    The interval rows are G1's canonical observation rows (clipped to the
    support, duplicates merged, normalized weights; ``_observations``).  A
    zero-width row is a point observation, a row covering the support the
    whole model, any other row ``B`` an integral over ``B`` resolved to the
    normalizer's ``rel_tol`` relative to its own mass (G3-043).

    Parameters
    ----------
    state : _BasisState
        Normalized state (``nodes``/``weights`` not kept).
    log_p : float
        ``sum_i w_i log P(B_i)`` (a point row contributes its log density).
    mean : numpy.ndarray
        ``sum_i w_i E[Phi | B_i]`` (``Phi(x_i)`` for a point row).
    cov : numpy.ndarray
        ``sum_i w_i Cov(Phi | B_i)`` (zero for a point row).
    row_log_p, row_mean, row_cov : numpy.ndarray or None
        Per-row values (only when requested).
    """

    state: object
    log_p: float
    mean: np.ndarray
    cov: np.ndarray
    row_log_p: np.ndarray | None
    row_mean: np.ndarray | None
    row_cov: np.ndarray | None


def _interval_state(
    context, theta, lower, upper, weights, /, options=None, per_row=False
):
    """Normalize ``theta`` and reduce interval rows (``interval_states_c``).

    Each positive-width row is one domain of the normalizer's cell
    integrator with its own reference potential, tail truncation and
    tolerance; the whole-support normalization is the same integrator on one
    domain (see the module docstring and ``_normalizer_kernels``).

    Parameters
    ----------
    context : _NormalizerContext
        Compiled-basis data.
    theta : numpy.ndarray
        Coefficients, shape ``(d,)``.
    lower, upper : numpy.ndarray
        Row endpoints inside the support, ``lower <= upper`` (``+-inf`` at an
        infinite support end), shape ``(R,)``.
    weights : numpy.ndarray
        Row weights, shape ``(R,)``.
    options : _NormalizerOptions or None
        Settings.
    per_row : bool
        Also return the per-row values.

    Returns
    -------
    _IntervalState

    Raises
    ------
    ValueError
        If a log amplitude is negative or the rows are malformed.
    _ImproperPotential
        If ``Z = inf``.
    _QuadratureFailure
        If the normalization or a row cannot be resolved (the failing row is
        named in the diagnostics).
    """
    opts = options or _NormalizerOptions()
    model = context.model
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    if any(col.nonnegative and theta[i] < 0.0 for i, col in enumerate(model.columns)):
        raise ValueError("log amplitudes must be nonnegative")
    lower = np.ascontiguousarray(lower, dtype=np.float64).reshape(-1)
    upper = np.ascontiguousarray(upper, dtype=np.float64).reshape(-1)
    weights = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
    req = model.request
    if (
        lower.shape != upper.shape
        or lower.shape != weights.shape
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
        or np.any(lower < req.lower)
        or np.any(upper > req.upper)
        or np.any(lower == np.inf)
        or np.any(upper == -np.inf)
    ):
        raise ValueError("interval rows must be ordered and inside the support")
    tl, tu = context.tails.get("lower"), context.tails.get("upper")
    gl, gu = context.growth_kernels.get("lower"), context.growth_kernels.get("upper")
    (
        status,
        row,
        row_status,
        side,
        row_side,
        log_z,
        mean,
        cov,
        agg_logp,
        agg_mean,
        agg_cov,
        row_logp,
        row_mean,
        row_cov,
        diag,
        row_diag,
    ) = interval_states(
        model.kernel,
        theta,
        None if tl is None else tl.kernel,
        None if tu is None else tu.kernel,
        gl,
        gu,
        context.column_scale,
        opts,
        lower,
        upper,
        weights,
        per_row,
    )
    if status == _NK_IMPROPER:
        raise _ImproperPotential("candidate state is not normalizable")
    if status != _NK_PROPER:
        raise _failure(context, theta, opts, status, side, diag)
    if row >= 0:
        info = _failure(context, theta, opts, row_status, row_side, row_diag)
        info.diagnostics["row"] = int(row)
        info.diagnostics["interval"] = (float(lower[row]), float(upper[row]))
        message = _FAILURES[row_status][0]
        raise _QuadratureFailure(f"interval row {row}: {message}", info.diagnostics)
    state = _BasisState(
        model, theta.copy(), "proper", log_z, mean, cov, 0.0, 0.0, None, None, 0
    )
    return _IntervalState(
        state, agg_logp, agg_mean, agg_cov, row_logp, row_mean, row_cov
    )
