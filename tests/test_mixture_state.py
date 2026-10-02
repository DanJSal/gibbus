"""Natural mixture fitted-state tests."""

import numpy as np

from gibbus import Distribution


def test_point_mixture_components_finalize_to_natural_state():
    rng = np.random.default_rng(2)
    x = np.ascontiguousarray(
        np.concatenate(
            [
                rng.normal(-2.5, 0.6, 300),
                rng.normal(2.5, 0.6, 300),
            ]
        )
    )
    fitted = Distribution().fit(
        x,
        n_components=2,
        support=(-np.inf, np.inf),
        poly_degree=4,
        progressive=False,
        rng=0,
        em_max_iter=12,
    )
    assert fitted.n_components == 2
    assert all(c.is_fitted for c in fitted.components)


def test_point_mixture_auto_degree_uses_natural_component_selector():
    rng = np.random.default_rng(3)
    x = np.ascontiguousarray(
        np.concatenate(
            [
                rng.normal(-2.0, 0.7, 250),
                rng.normal(2.0, 0.7, 250),
            ]
        )
    )
    fitted = Distribution().fit(
        x,
        n_components=2,
        support=(-np.inf, np.inf),
        poly_degree="auto",
        progressive=False,
        rng=0,
        em_max_iter=8,
    )
    assert fitted.n_components == 2
    assert all(c.is_fitted for c in fitted.components)


def test_finite_interval_mixture_components_finalize_to_natural_state():
    rng = np.random.default_rng(4)
    x = np.concatenate(
        [
            rng.normal(-2.0, 0.65, 220),
            rng.normal(2.0, 0.7, 220),
        ]
    )
    intervals = np.column_stack([x - 0.08, x + 0.08])
    fitted = Distribution().fit(
        intervals,
        n_components=2,
        support=(-np.inf, np.inf),
        poly_degree=2,
        progressive=False,
        rng=0,
        em_max_iter=8,
    )
    assert fitted.n_components == 2
    assert all(c.is_fitted for c in fitted.components)


def test_infinite_censored_mixture_components_finalize_to_natural_state():
    rng = np.random.default_rng(5)
    x = np.concatenate(
        [
            rng.normal(-2.0, 0.65, 220),
            rng.normal(2.0, 0.7, 220),
        ]
    )
    rows = []
    for value in x:
        if value < -2.8:
            rows.append((-np.inf, -2.8))
        elif value > 2.8:
            rows.append((2.8, np.inf))
        else:
            rows.append((value - 0.08, value + 0.08))
    fitted = Distribution().fit(
        np.asarray(rows, dtype=float),
        n_components=2,
        support=(-np.inf, np.inf),
        poly_degree=2,
        progressive=False,
        rng=0,
        em_max_iter=8,
    )
    assert fitted.n_components == 2
    assert all(c.is_fitted for c in fitted.components)
    means = np.asarray([c.mean for c in fitted.components])
    assert means[0] < -1.5
    assert means[1] > 1.5
