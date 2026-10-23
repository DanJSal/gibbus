"""Curvature separation for the G3 basis: find where ``q''`` may fall below 0.

G1 owner replaced: ``_fit/separation.py`` (exact separator) and the
post-solve ``_certify`` call of ``_fit/conic_newton.py``.  G1's exact
Markov-Lukacs cone needs no cuts; G3 enforces ``q'' >= 0`` by permanent
coefficient-linear cut rows, and the post-solve verdict maps onto G1's
``uncertified`` status.  ``separation.py`` is deleted when the Stage 4 fitter
switches over.

For a coefficient vector ``theta`` the search returns every local minimum of
``q''`` that may lie below the cut threshold ``tau = rho / h**2``, the
coefficient-linear cut rows ``Phi''(x*) . theta >= 0`` for them, and tail
cuts where a leading asymptotic curvature term is negative.  The constraint
itself is always ``q'' >= 0`` (G3-026); ``tau`` only selects points.

1. **Compact domain** (``basis_tails``): each infinite end contributes a
   cutoff beyond which ``q'' > 0`` is proven from the asymptotic atoms (or a
   tail violation / inconclusive verdict); each finite end with a positive
   log amplitude contributes an excluded neighbourhood where
   ``q'' >= tau``; other finite ends are ordinary closed endpoints.
2. **Cells** (``basis_interval``): asinh-spaced cells plus gate locations
   and the coordinate centre.  Per cell, with ``m`` the midpoint and
   ``r`` the half-width, ``q''(x) >= A + B (x - m) + C/2 (x - m)**2`` from
   point enclosures ``A ∋ q''(m)``, ``B ∋ q'''(m)`` and the cell enclosure
   ``C ∋ q''''(X)``.  A cell is

   * *discarded* when that lower bound (or the direct enclosure of ``q''``)
     exceeds ``tau``;
   * *monotone* when ``q'''`` provably has one sign: its minimum is at the
     descending end, which a neighbour covers, unless it is a real support
     endpoint (then a candidate) or a proven boundary (then covered);
   * *convex* when ``q'''' >= 0`` on it: ``q'''`` is nondecreasing, so a
     safeguarded Newton/bisection on ``q'''`` finds the minimum (possibly
     at a cell end, which is then recorded too);
   * otherwise *bisected*, down to a minimum width and within a cell
     budget.  Cells left over are *unresolved*: the verdict is then
     ``inconclusive`` -- never ``nonnegative`` (fail closed).

3. **Cuts**: every minimum whose ``q''`` enclosure reaches below ``tau`` is a
   cut candidate; the lowest ``max_cuts`` are returned.

Bounds are returned as binary64 intervals (numerical evidence, G3-028).
Resolved polynomial contact points use extended-precision Horner arithmetic;
ambiguous ungated polynomial/log points can use exact rational evaluation.
The cell search runs in the compiled ``_curvature_kernels``.
"""

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from .._defaults import (
    CURVATURE_CUT_RHO,
    CURVATURE_FALLBACK_ZETA,
    CURVATURE_FEASIBILITY_TOL,
    CURVATURE_INITIAL_CELLS,
    CURVATURE_MAX_CELLS,
    CURVATURE_MAX_CUTS,
    CURVATURE_MIN_WIDTH,
    CURVATURE_NEWTON_MAX,
    CURVATURE_NEWTON_TOL,
    CURVATURE_ZETA_CAP,
)
from .._model.basis_tails import _build_tail_model, _tail_on_face, _TailResult
from ._curvature_kernels import separate


@dataclass(frozen=True)
class _CurvatureOptions:
    """Search settings (Stage 2 defaults, G3-026/G3-029).

    Parameters
    ----------
    series_candidates : bool
        Enable the experimental cached Taylor candidate search in Newton trial
        gates. The full curvature acceptance check remains in force. Default
        False because cache construction can cost more than it saves on short fits.
    series_order, series_samples, series_steps : int
        Taylor order, scan subdivisions per panel, and refinement step limit.
    series_tolerance : float
        Candidate root-bracket tolerance in fitting-scale units. This does not
        change the feasibility tolerance of the full search.
    series_violation_tol : float
        Fast rejection margin in fitting-coordinate curvature units. Smaller
        violations go to the unchanged full search so approximate contact
        locations cannot keep adding poorly positioned near-active rows.
    series_max_panels : int
        Cache panel budget; unsupported geometry/range or budget exhaustion
        uses the direct candidate search.
    rho : float
        Cut threshold ``tau = rho / h**2`` (dimensionless ``rho``).
    initial_cells : int
        Number of asinh-spaced initial cells.
    max_cells : int
        Budget of processed cells per search.
    min_width : float
        Smallest cell width, in units of ``h``.
    newton_tol : float
        Refinement bracket width, in units of ``h``.
    newton_max : int
        Maximum refinement iterations.
    zeta_cap : float
        Largest tail cutoff tried, in units of ``h``.
    fallback_zeta : float
        Half-width (units of ``h``) of the finite search on an end whose tail
        is violated or numerically zero; an end whose positive leader cannot
        be proven dominant before ``zeta_cap`` is searched out to
        ``zeta_cap`` instead (G3-037).
    max_cuts : int
        Most finite cuts returned per search (lowest minima first).
    feasibility_tol : float
        Shape-verdict tolerance in fitting-coordinate units (G1 ``_certify``):
        a minimum is violated only when its enclosure lies below
        ``-feasibility_tol / h**2`` (G3-033).
    """

    # Approximate candidate generation only; acceptance still uses separate_c.
    series_candidates: bool = False
    series_order: int = 16
    series_samples: int = 8
    series_steps: int = 16
    series_tolerance: float = 1e-5
    series_max_panels: int = 128
    series_violation_tol: float = 1e-10

    rho: float = CURVATURE_CUT_RHO
    initial_cells: int = CURVATURE_INITIAL_CELLS
    max_cells: int = CURVATURE_MAX_CELLS
    min_width: float = CURVATURE_MIN_WIDTH
    newton_tol: float = CURVATURE_NEWTON_TOL
    newton_max: int = CURVATURE_NEWTON_MAX
    zeta_cap: float = CURVATURE_ZETA_CAP
    fallback_zeta: float = CURVATURE_FALLBACK_ZETA
    max_cuts: int = CURVATURE_MAX_CUTS
    feasibility_tol: float = CURVATURE_FEASIBILITY_TOL


@dataclass(frozen=True)
class _CurvatureMinimum:
    """One local minimum (or unresolved cell) of ``q''``.

    Parameters
    ----------
    x : float
        Location (user units).
    value : float
        Floating-point ``q''(x)``.
    lower, upper : float
        Enclosure of ``q''(x)``.
    kind : str
        ``"interior"``, ``"endpoint"`` (closed support endpoint) or
        ``"affine"`` (lower end of a cell where ``q''`` is affine within
        negligible error: a candidate, not a stationary point) or
        ``"unresolved"`` (midpoint of a cell the budget left open).
    """

    x: float
    value: float
    lower: float
    upper: float
    kind: str


@dataclass(frozen=True)
class _CurvatureSearchResult:
    """Outcome of one curvature separation pass.

    Parameters
    ----------
    status : str
        ``"nonnegative"`` (every cell resolved, tails proven, all minima
        enclosed at or above ``-feasibility_tol / h**2``), ``"violated"`` (a
        minimum enclosed below it or a negative tail leader) or
        ``"inconclusive"``.
    threshold : float
        Cut threshold ``tau`` (user curvature units).
    domain : tuple of float
        Compact search interval.
    minima : tuple of _CurvatureMinimum
        All recorded minima, ascending ``x``.
    tails : tuple
        ``_TailResult`` per infinite end.
    cut_points : numpy.ndarray
        Finite cut locations, shape ``(k,)``.
    cut_rows : numpy.ndarray
        Cut rows, shape ``(k + t, d)``: ``Phi''(x*)`` for the finite cuts,
        then the leading functional of each violated tail.
    cells : int
        Cells processed.
    unresolved : int
        Cells left unresolved.
    """

    status: str
    threshold: float
    domain: tuple
    minima: tuple
    tails: tuple
    cut_points: np.ndarray
    cut_rows: np.ndarray
    cells: int
    unresolved: int

    @property
    def feasible(self):
        """Whether ``q'' >= 0`` was established on the whole support."""
        return self.status == "nonnegative"


@dataclass
class _CurvatureContext:
    """Theta-independent search data for one compiled basis (tail atoms)."""

    model: object
    tails: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate the fields."""
        for side in ("lower", "upper"):
            if not math.isfinite(getattr(self.model.request, side)):
                self.tails[side] = _build_tail_model(self.model, side)

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


def _separate_curvature(context, theta, /, options=None):
    """Run the curvature separation for one coefficient vector.

    The search runs in ``_curvature_kernels.separate_c`` (method in the
    module docstring).

    Parameters
    ----------
    context : _CurvatureContext
        Compiled-basis search data.
    theta : numpy.ndarray
        Coefficients, shape ``(d,)``.
    options : _CurvatureOptions or None
        Search settings.

    Returns
    -------
    _CurvatureSearchResult
    """
    options = options or _CurvatureOptions()
    model = context.model
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    d = model.n_coefficients
    tl, tu = context.tails.get("lower"), context.tails.get("upper")
    (
        status,
        tau,
        domain,
        bound,
        minima,
        tails,
        cut_points,
        cut_rows,
        cells,
        unresolved,
    ) = separate(
        model.kernel,
        theta,
        None if tl is None else tl.kernel,
        None if tu is None else tu.kernel,
        options,
    )
    if bound:
        # gamma < 0 sends q'' to -inf at that endpoint: report the bound rows.
        return _CurvatureSearchResult(
            status="violated",
            threshold=tau,
            domain=domain,
            minima=(),
            tails=(),
            cut_points=np.zeros(0),
            cut_rows=np.eye(d)[bound],
            cells=0,
            unresolved=0,
        )
    if (
        status == "inconclusive"
        and unresolved == 0
        and all(t[1] == "nonnegative" for t in tails)
    ):
        floor = -options.feasibility_tol / model.geometry.scale**2
        ambiguous = [i for i, m in enumerate(minima) if m[2] < floor <= m[3]]
        if ambiguous:
            from .._model.basis_interval import _exact_ungated_curvature

            exact = _exact_ungated_curvature(
                model, theta, [minima[i][0] for i in ambiguous]
            )
            if exact is not None:
                minima = list(minima)
                for i, (value, lower, upper) in zip(ambiguous, exact):
                    m = minima[i]
                    lower, upper = max(m[2], lower), min(m[3], upper)
                    if lower > upper:
                        raise ArithmeticError(
                            "curvature interval excludes its exact point value"
                        )
                    minima[i] = (m[0], value, lower, upper, m[4])
                if any(m[3] < floor for m in minima):
                    status = "violated"
                elif all(m[2] >= floor for m in minima):
                    status = "nonnegative"
    results = []
    tail_rows = []
    for side, st, cutoff, zeta, lead, power, rate, margin, atom in tails:
        cut = None
        if st == "violated":
            a = context.tails[side].atoms[atom]
            cut = (a.row if a.cut is None else a.cut).copy()
            tail_rows.append(cut)
        results.append(
            _TailResult(
                side,
                st,
                cutoff,
                zeta,
                lead,
                cut,
                lead_power=power,
                lead_rate=rate,
                margin=margin,
            )
        )
    rows = cut_rows
    if tail_rows:
        rows = np.vstack([rows, np.array(tail_rows)])
    return _CurvatureSearchResult(
        status=status,
        threshold=tau,
        domain=domain,
        minima=tuple(_CurvatureMinimum(*m) for m in minima),
        tails=tuple(results),
        cut_points=cut_points,
        cut_rows=rows,
        cells=cells,
        unresolved=unresolved,
    )
