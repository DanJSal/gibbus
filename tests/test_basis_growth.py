"""First-derivative tail bounds and exact properness.

Checked against the independent high-precision oracle of ``_basis_oracle``:
a ``proper`` end has ``dq/dzeta >= bound(zeta)`` beyond its cutoff (so ``q``
grows at least linearly there), and an ``improper`` end has eventually
non-increasing ``q``.
"""

import math

import mpmath as mp
import numpy as np
import pytest
from _basis_oracle import BasisOracle
from test_basis_construction import _compile_quiet
from test_basis_tails import TAIL_SPECS, _thetas

from gibbus import BasisSpec, Exponential, Polynomial
from gibbus._model.basis_tails import _build_tail_model, _growth_verdict, _tail_verdict

_ORACLES = {}


def _oracle(model, theta, x):
    """High-precision ``(q, q')`` in user units at the current precision."""
    key = id(model)
    if key not in _ORACLES:
        _ORACLES[key] = (model, BasisOracle(model, max_order=1))
    oracle = _ORACLES[key][1]
    return (
        oracle.potential(theta, x, 0, dps=mp.mp.dps)[0],
        oracle.potential(theta, x, 1, dps=mp.mp.dps)[0],
    )


def _dps(rate, zeta):
    return 60 + int(abs(rate or 0.0) * zeta / 2.0)


@pytest.mark.parametrize("spec", TAIL_SPECS, ids=lambda s: "")
def test_slope_bounds_and_properness_agree_with_high_precision(spec):
    model = _compile_quiet(spec)
    g = model.geometry
    for side in ("lower", "upper"):
        if not math.isinf(getattr(model.request, side)):
            continue
        tail = _build_tail_model(model, side, order=1)
        for theta in _thetas(model, 2000 + TAIL_SPECS.index(spec)):
            res = _growth_verdict(tail, model, theta)
            assert res.status in ("proper", "improper")
            slope = res.slope
            if res.status == "proper":
                r0 = slope.zeta
                prev = None
                for mult in (1.0, 1.5, 3.0, 10.0, 40.0):
                    zeta = r0 * mult
                    if slope.lead_power * math.log(zeta) + slope.lead_rate * zeta > 600:
                        break
                    with mp.workdps(_dps(slope.lead_rate, zeta) + 40):
                        q, q1 = _oracle(
                            model, theta, g.center + tail.sign * zeta * g.scale
                        )
                        dqdzeta = tail.sign * q1 * mp.mpf(g.scale)
                        assert dqdzeta >= mp.mpf(slope.bound(zeta)) * (
                            1 - mp.mpf(1e-9)
                        ), (
                            side,
                            mult,
                            float(dqdzeta),
                            slope.bound(zeta),
                        )
                        if prev is not None:
                            # q grows at least linearly: q(z2) - q(z1) >= bound(R) (z2 - z1).
                            assert q - prev[0] >= mp.mpf(slope.bound(r0)) * (
                                zeta - prev[1]
                            ) * (1 - mp.mpf(1e-9))
                        prev = (q, zeta)
            else:
                # Eventually non-increasing q, or q tending to a finite limit.
                lead = slope.lead_value
                found = False
                # Competing terms can delay this enormously (e.g. 1e-7 zeta**0.5 vs 8).
                for zeta in (1e2, 1e3, 1e4, 1e6, 1e8, 1e10, 1e13, 1e16, 1e20, 1e25):
                    with mp.workdps(80):
                        qa, _ = _oracle(
                            model, theta, g.center + tail.sign * zeta * g.scale
                        )
                        qb, _ = _oracle(
                            model, theta, g.center + tail.sign * 2 * zeta * g.scale
                        )
                    if (qb - qa) / zeta <= mp.mpf(1e-6) * (abs(lead) + 1):
                        found = True
                        break
                assert found, (side, slope)


def test_linear_tail_with_negative_ripple_is_still_proper():
    # bulk 3 + linear upper tail: q'' may be negative far out, but q' > 0.
    model = _compile_quiet(BasisSpec(bulk_degree=3, upper_tail=Polynomial(1)))
    tail2 = _build_tail_model(model, "upper", order=2)
    tail1 = _build_tail_model(model, "upper", order=1)
    rng = np.random.default_rng(7)
    seen = 0
    for _ in range(200):
        theta = rng.normal(size=model.n_coefficients)
        curv = _tail_verdict(tail2, model, theta)
        grow = _growth_verdict(tail1, model, theta)
        if curv.status == "violated" and grow.status == "proper":
            seen += 1
    assert seen > 0


def test_exponential_tail_with_negative_amplitude_is_improper():
    model = _compile_quiet(BasisSpec(bulk_degree=2, lower_tail=Exponential(0.7)))
    labels = [c.label for c in model.columns]
    theta = np.zeros(model.n_coefficients)
    # expL (-1 + tL) = -2 expL W_L: a positive coefficient is a negative amplitude.
    theta[labels.index("expL*(-1+tL)")] = 1.0
    theta[labels.index("z^2")] = 1.0
    tail = _build_tail_model(model, "lower", order=1)
    assert _growth_verdict(tail, model, theta).status == "improper"
    theta[labels.index("expL*(-1+tL)")] = -1.0
    assert _growth_verdict(tail, model, theta).status == "proper"


def test_order_one_model_rejected_for_curvature_and_vice_versa():
    model = _compile_quiet(BasisSpec(bulk_degree=2))
    with pytest.raises(ValueError):
        _growth_verdict(_build_tail_model(model, "upper", order=2), model, np.ones(2))
    with pytest.raises(ValueError):
        _build_tail_model(model, "upper", order=3)
