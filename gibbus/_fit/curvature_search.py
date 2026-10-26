"""Curvature separation: find every region where ``q''`` falls below zero.

``q'' >= 0`` is enforced by permanent coefficient-linear cut rows; this module
finds them.  For a coefficient vector ``theta`` the search returns every
negative interval of the scaled curvature ``H = q'' / ||pieces||`` below the
threshold ``-tau`` (``H`` is bounded, smooth and has the exact sign of
``q''``), cut rows ``R(z) . theta >= 0`` at each interval's minimum and a few
Chebyshev points across it, and tail cut rows where a leading asymptotic
curvature term is negative.  The constraint itself is always ``q'' >= 0``;
``tau`` only selects points and sets what counts as verified.

1. **Domain** (``basis_tails``): each infinite end is searched out to the
   cutoff beyond which ``q'' > 0`` is proven from the asymptotic atoms (or to
   a cap when the tail is violated or unproven); each finite end with a
   positive log amplitude has an excluded neighborhood where ``q'' >= 0`` is
   enclosed.
2. **Panels** (``_curvature_kernels``): a fixed hierarchy per basis whose
   theta-free per-atom rows are cached at Chebyshev nodes, so a new ``theta``
   costs a matrix-vector product and a DCT per panel; panels are bisected
   until resolved to ``tau``, and candidate panels are sampled for roots and
   critical points.
3. **Verdict**: ``"verified"`` (within the threshold: no interval below
   ``-tau``, tails proven, every panel resolved), ``"violated"`` (an interval,
   a tail or a log amplitude is negative) or ``"unverified"`` (fail closed).

The search runs in the compiled ``_curvature_kernels``.
"""

import copy
from dataclasses import dataclass, field

import numpy as np

from .._defaults import CURVATURE_SEARCH_OPTIONS
from .._model.basis_tails import _build_tail_model, _tail_on_face, _TailResult
from ._curvature_kernels import CurvatureSearch, separate


@dataclass(frozen=True)
class _CurvatureMinimum:
    """The minimum of one negative interval of the scaled curvature.

    Parameters
    ----------
    x : float
        Location (user units).
    z : float
        Location in the canonical coordinate.
    value : float
        Scaled curvature ``H`` there (in ``[-1, 0)`` for a single leading
        piece; dimensionless).
    kind : str
        ``"interior"``, ``"endpoint"`` (the interval reaches a finite support
        endpoint) or ``"tail"`` (it extends to an infinite end).
    """

    x: float
    z: float
    value: float
    kind: str


@dataclass(frozen=True)
class _CurvatureSearchResult:
    """Outcome of one curvature separation pass.

    Parameters
    ----------
    status : str
        ``"verified"``, ``"violated"`` or ``"unverified"``.
    threshold : float
        Threshold ``tau`` on the scaled curvature.
    domain : tuple of float
        Searched interval (user units).
    intervals : tuple
        Negative intervals ``(z_lo, z_hi, z_min, h_min, to_inf, touches_end)``
        in canonical units, ascending.
    minima : tuple of _CurvatureMinimum
        The minimum of each negative interval, ascending ``x``.
    tails : tuple
        ``_TailResult`` per infinite end.
    cut_points : numpy.ndarray
        Finite cut locations (user units), shape ``(k,)``.
    cut_rows : numpy.ndarray
        Cut rows, shape ``(k + t, d)``: scaled column curvatures at the finite
        cuts (``q''`` up to a positive factor), then the leading functional of
        each violated tail.
    panels : int
        Panels in the search hierarchy.
    unresolved : int
        Panels or endpoint scans left unresolved.
    """

    status: str
    threshold: float
    domain: tuple
    intervals: tuple
    minima: tuple
    tails: tuple
    cut_points: np.ndarray
    cut_rows: np.ndarray
    panels: int
    unresolved: int

    @property
    def feasible(self):
        """Whether ``q'' >= 0`` was verified within the threshold."""
        return self.status == "verified"


@dataclass
class _CurvatureContext:
    """Theta-independent search data for one compiled basis.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    options : _CurvatureSearchOptions
        Search policy (its degree and ``jmin`` fix the panel hierarchy).
    """

    model: object
    options: object = CURVATURE_SEARCH_OPTIONS
    tails: dict = field(default_factory=dict)
    search: object = None

    def __post_init__(self):
        """Build the tail atoms and the panel hierarchy."""
        for entry in self.model.asymptotics:
            self.tails[entry.side] = _build_tail_model(self.model, entry.side)
        self.search = CurvatureSearch(
            self.model.kernel, int(self.options.degree), int(self.options.jmin)
        )

    def on_face(self, basis, /):
        """Context for points of a face ``theta = N phi`` (exact-zero atoms dropped).

        The panel rows do not depend on the face and are shared.

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

    Parameters
    ----------
    context : _CurvatureContext
        Compiled-basis search data.
    theta : numpy.ndarray
        Coefficients, shape ``(d,)``.
    options : _CurvatureSearchOptions or None
        Search policy; ``None`` uses the context's.  Its degree and ``jmin``
        must match the context's.

    Returns
    -------
    _CurvatureSearchResult
    """
    options = context.options if options is None else options
    model = context.model
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    d = model.n_coefficients
    tl, tu = context.tails.get("lower"), context.tails.get("upper")
    (
        status,
        tau,
        domain,
        bound,
        intervals,
        minima,
        tails,
        cut_points,
        cut_rows,
        panels,
        unresolved,
    ) = separate(
        context.search,
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
            intervals=(),
            minima=(),
            tails=(),
            cut_points=np.zeros(0),
            cut_rows=np.eye(d)[bound],
            panels=0,
            unresolved=0,
        )
    results = []
    tail_rows = []
    for side, st, cutoff, zeta, lead, power, rate, kappa, const, margin, atom in tails:
        cut = None
        if st == "violated":
            a = context.tails[side].atoms[atom]
            cut = (a.row if a.cut is None else a.cut).copy()
            tail_rows.append(cut / np.max(np.abs(cut)))
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
                lead_kappa=kappa,
                lead_const=const,
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
        intervals=tuple(intervals),
        minima=tuple(_CurvatureMinimum(*m) for m in minima),
        tails=tuple(results),
        cut_points=cut_points,
        cut_rows=rows,
        panels=panels,
        unresolved=unresolved,
    )
