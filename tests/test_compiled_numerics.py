"""Independent parity/oracle tests for compiled numerical hot paths."""

from math import factorial

import numpy as np
import pytest
from gibbus._postfit._mix_kernels import neg_log_mix_derivs_batch, neg_logsumexp_batch
from scipy.special import logsumexp

from gibbus import Distribution


def _python_neg_log_mix_derivs(ell_jets, max_order):
    """Reference Taylor recurrence written only with NumPy/SciPy."""
    jets = np.asarray(ell_jets, dtype=np.float64)
    order = int(max_order)
    fact = np.array([factorial(m) for m in range(order + 1)], dtype=np.float64)
    ell_tc = jets / fact[None, :, None]
    ell0 = ell_tc[:, 0, :]
    offset = logsumexp(ell0, axis=0)

    a = np.zeros_like(ell_tc)
    a[:, 0, :] = np.exp(ell0 - offset[None, :])
    for n in range(1, order + 1):
        for m in range(1, n + 1):
            a[:, n, :] += m * ell_tc[:, m, :] * a[:, n - m, :]
        a[:, n, :] /= n

    summed = a.sum(axis=0)
    h = np.zeros((order + 1, jets.shape[2]), dtype=np.float64)
    h[0] = np.log(summed[0]) + offset
    for n in range(1, order + 1):
        acc = np.zeros(jets.shape[2], dtype=np.float64)
        for m in range(1, n):
            acc += m * h[m] * summed[n - m]
        h[n] = (summed[n] - acc / n) / summed[0]
    return -(h * fact[:, None])


def test_compiled_mixture_order_zero_matches_scipy_logsumexp():
    rng = np.random.default_rng(1201)
    ell0 = np.ascontiguousarray(rng.normal(size=(5, 19)) * 30.0)
    expected = -logsumexp(ell0, axis=0)
    got = neg_logsumexp_batch(ell0)
    np.testing.assert_allclose(got, expected, rtol=3e-15, atol=3e-15)


def test_compiled_mixture_derivative_recurrence_matches_python_reference():
    rng = np.random.default_rng(1202)
    jets = np.ascontiguousarray(rng.normal(size=(4, 6, 13)))
    jets[:, 0, :] *= 20.0
    expected = _python_neg_log_mix_derivs(jets, 5)
    got = neg_log_mix_derivs_batch(jets, 5)
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)


def test_public_mixture_tail_is_weighted_component_tail_identity():
    rng = np.random.default_rng(1203)
    data = np.concatenate(
        [
            rng.normal(-2.0, 0.55, 180),
            rng.normal(2.0, 0.65, 180),
        ]
    )
    model = Distribution().fit(
        data, n_components=2, poly_degree=2, support=(-np.inf, np.inf), rng=0
    )
    x = float(max(component.base.ppf(1.0 - 1e-7) for component in model.components))
    component_logs = np.array(
        [component.base.logsf(x) for component in model.components]
    )
    expected = float(logsumexp(np.log(model.weights) + component_logs))
    assert model.logsf(x) == pytest.approx(expected, rel=0.0, abs=2e-10)
