"""Natural-coordinate automatic component-count selection."""

import numpy as np

from gibbus._api.selection import select_n_components
from gibbus._fit.mixture import _count_modes_kde

_REAL_LINE = (-np.inf, np.inf)


def _select(x, k_modes, /, *, k_max=6):
    """Run the natural BIC sweep on point data without subsampling."""
    data = np.ascontiguousarray(x, dtype=np.float64)
    return select_n_components(
        S=data[:, None],
        samples_1d=data,
        supp=_REAL_LINE,
        k_modes=int(k_modes),
        effective_k_max=int(k_max),
        gen=np.random.default_rng(0),
        obs_w=None,
        verb=0,
        subsample=False,
    )


def test_natural_bic_selects_two_separated_components():
    """The natural screening family resolves a clear two-component sample."""
    rng = np.random.default_rng(13)
    x = np.concatenate([
        rng.normal(-3.0, 0.7, 250),
        rng.normal(3.0, 0.7, 250),
    ])
    k, resp, weights, diagnostics = _select(x, 2, k_max=4)

    assert k == 2
    assert resp.shape == (x.size, 2)
    assert weights.shape == (2,)
    assert diagnostics["method"] == "log_concave_bic"
    assert diagnostics["selected_n_components"] == 2
    assert diagnostics["scores"]


def test_natural_bic_selects_one_for_unimodal_data():
    """BIC rejects an unnecessary second component on Gaussian data."""
    x = np.random.default_rng(14).normal(size=500)
    k, resp, weights, diagnostics = _select(x, 1, k_max=3)

    assert k == 1
    assert resp is None
    assert weights is None
    assert diagnostics["selected_n_components"] == 1


def test_spurious_fixed_k_cluster_does_not_change_bic_selection():
    """The known narrow fixed-K local maximum is penalized by BIC."""
    rng = np.random.default_rng(270901)
    n = 200
    split = round(0.6 * n)
    x = np.concatenate([
        rng.normal(0.0, 1.0, split),
        rng.normal(1.2, 0.6, n - split),
    ])
    k_modes = _count_modes_kde(np.ascontiguousarray(x))
    assert k_modes == 1

    k, _, _, diagnostics = _select(x, k_modes, k_max=6)

    assert k == 1
    one = next(s for s in diagnostics["scores"] if s["n_components"] == 1)
    two = next(s for s in diagnostics["scores"] if s["n_components"] == 2)
    assert one["status"] == two["status"] == "ok"
    assert one["bic"] < two["bic"]


def test_natural_bic_scores_interval_rows_directly():
    """Separated binned observations are selected using interval likelihoods."""
    rng = np.random.default_rng(15)
    x = np.concatenate([
        rng.normal(-2.5, 0.55, 180),
        rng.normal(2.5, 0.55, 180),
    ])
    lower = np.floor(x / 0.4) * 0.4
    rows = np.ascontiguousarray(np.column_stack([lower, lower + 0.4]))
    representatives = rows.mean(axis=1)
    k, resp, weights, diagnostics = select_n_components(
        S=rows,
        samples_1d=representatives,
        supp=_REAL_LINE,
        k_modes=2,
        effective_k_max=3,
        gen=np.random.default_rng(0),
        obs_w=None,
        verb=0,
        subsample=False,
    )

    assert k == 2
    assert resp.shape == (rows.shape[0], 2)
    assert weights.shape == (2,)
    assert diagnostics["selected_n_components"] == 2

def test_natural_bic_subsample_returns_full_data_initializer():
    """Subsample scoring keeps the winning initializer on the full dataset."""
    rng = np.random.default_rng(16)
    x = np.concatenate([
        rng.normal(-3.0, 0.65, 1000),
        rng.normal(3.0, 0.65, 1000),
    ])
    data = np.ascontiguousarray(x, dtype=np.float64)
    k, resp, weights, diagnostics = select_n_components(
        S=data[:, None],
        samples_1d=data,
        supp=_REAL_LINE,
        k_modes=2,
        effective_k_max=4,
        gen=np.random.default_rng(1),
        obs_w=None,
        verb=0,
        subsample=400,
    )

    assert k == 2
    assert resp.shape == (data.size, 2)
    assert weights.shape == (2,)
    assert diagnostics["subsampled"] is True
    assert diagnostics["subsample_size"] == 400
    assert diagnostics["full_sample_size"] == data.size

