"""Explicit immutable construction policy for fitted spectral representations."""

from dataclasses import dataclass

from .._defaults import (
    SPECTRAL_CDF_ABS_TOL,
    SPECTRAL_CDF_COEFF_TOL,
    SPECTRAL_CDF_MAX_DEPTH,
    SPECTRAL_CDF_MAX_PANELS,
    SPECTRAL_CDF_REL_TOL,
    SPECTRAL_DEGREE_OPTIONS,
    SPECTRAL_PPF_CERTIFY_SUBDIVIDE,
    SPECTRAL_PPF_COEFF_TOL,
    SPECTRAL_PPF_FIT_TOL,
    SPECTRAL_PPF_LOGIT_TOL,
    SPECTRAL_PPF_MAX_DEPTH,
    SPECTRAL_PPF_MAX_PANELS,
    SPECTRAL_PPF_PROB_TOL,
)


@dataclass(frozen=True)
class _SpectralCDFOptions:
    """Forward construction controls, resolved by the representation owner.

    Parameters
    ----------
    degree_options : tuple of int
        Candidate panel degrees before subdivision.
    rel_tol, abs_tol, coeff_tol : float
        Relative/absolute panel errors and coefficient-tail tolerance.
    max_depth, max_panels : int
        Subdivision depth and global leaf-panel budgets. Unresolved forward
        panels are retained and exposed through diagnostics on exhaustion.
    """

    degree_options: tuple[int, ...] = SPECTRAL_DEGREE_OPTIONS
    rel_tol: float = SPECTRAL_CDF_REL_TOL
    abs_tol: float = SPECTRAL_CDF_ABS_TOL
    coeff_tol: float = SPECTRAL_CDF_COEFF_TOL
    max_depth: int = SPECTRAL_CDF_MAX_DEPTH
    max_panels: int = SPECTRAL_CDF_MAX_PANELS


@dataclass(frozen=True)
class _SpectralPPFOptions:
    """Inverse construction controls, resolved by the representation owner.

    Parameters
    ----------
    degree_options : tuple of int
        Candidate inverse panel degrees.
    fit_tol, logit_tol, prob_tol, coeff_tol : float
        Compact fit, log-odds, probability and coefficient-tail tolerances.
    max_depth, max_panels : int
        Subdivision depth and global leaf-panel budgets.
    certify_subdivide : int
        Bernstein subdivision depth for monotonicity certification.
    """

    degree_options: tuple[int, ...] = SPECTRAL_DEGREE_OPTIONS
    fit_tol: float = SPECTRAL_PPF_FIT_TOL
    logit_tol: float = SPECTRAL_PPF_LOGIT_TOL
    prob_tol: float = SPECTRAL_PPF_PROB_TOL
    coeff_tol: float = SPECTRAL_PPF_COEFF_TOL
    max_depth: int = SPECTRAL_PPF_MAX_DEPTH
    max_panels: int = SPECTRAL_PPF_MAX_PANELS
    certify_subdivide: int = SPECTRAL_PPF_CERTIFY_SUBDIVIDE
