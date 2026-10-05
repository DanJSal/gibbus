"""Natural-coordinate automatic component-count selection."""

import numpy as np
import pytest

from gibbus._api.selection import select_n_components
from gibbus._fit.degree import _DegreeSelectionConfig
from gibbus._fit.inputs import (
    _normalize_sample_weight_1d,
    _prepare_component_selection_policy,
)
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
        gen=np.random.default_rng(0),
        obs_w=None,
        verb=0,
        policy=_prepare_component_selection_policy(
            len(data), _REAL_LINE, k_modes, k_max, False, "auto", False, False
        ),
        refine_candidate=None,
        degree_config=_DegreeSelectionConfig(),
    )


def test_natural_bic_selects_two_separated_components():
    """The natural screening family resolves a clear two-component sample."""
    rng = np.random.default_rng(13)
    x = np.concatenate(
        [
            rng.normal(-3.0, 0.7, 250),
            rng.normal(3.0, 0.7, 250),
        ]
    )
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
    x = np.concatenate(
        [
            rng.normal(0.0, 1.0, split),
            rng.normal(1.2, 0.6, n - split),
        ]
    )
    k_modes = _count_modes_kde(
        np.ascontiguousarray(x), verbose=0, rng=np.random.default_rng(0)
    )
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
    x = np.concatenate(
        [
            rng.normal(-2.5, 0.55, 180),
            rng.normal(2.5, 0.55, 180),
        ]
    )
    lower = np.floor(x / 0.4) * 0.4
    rows = np.ascontiguousarray(np.column_stack([lower, lower + 0.4]))
    representatives = rows.mean(axis=1)
    k, resp, weights, diagnostics = select_n_components(
        S=rows,
        samples_1d=representatives,
        supp=_REAL_LINE,
        k_modes=2,
        gen=np.random.default_rng(0),
        obs_w=None,
        verb=0,
        policy=_prepare_component_selection_policy(
            len(rows), _REAL_LINE, 2, 3, False, "auto", False, False
        ),
        refine_candidate=None,
        degree_config=_DegreeSelectionConfig(),
    )

    assert k == 2
    assert resp.shape == (rows.shape[0], 2)
    assert weights.shape == (2,)
    assert diagnostics["selected_n_components"] == 2


def test_natural_bic_subsample_returns_full_data_initializer():
    """Subsample scoring keeps the winning initializer on the full dataset."""
    rng = np.random.default_rng(16)
    x = np.concatenate(
        [
            rng.normal(-3.0, 0.65, 1000),
            rng.normal(3.0, 0.65, 1000),
        ]
    )
    data = np.ascontiguousarray(x, dtype=np.float64)
    k, resp, weights, diagnostics = select_n_components(
        S=data[:, None],
        samples_1d=data,
        supp=_REAL_LINE,
        k_modes=2,
        gen=np.random.default_rng(1),
        obs_w=None,
        verb=0,
        policy=_prepare_component_selection_policy(
            len(data), _REAL_LINE, 2, 4, 400, "auto", False, False
        ),
        refine_candidate=None,
        degree_config=_DegreeSelectionConfig(),
    )

    assert k == 2
    assert resp.shape == (data.size, 2)
    assert weights.shape == (2,)
    assert diagnostics["subsampled"] is True
    assert diagnostics["subsample_size"] == 400
    assert diagnostics["full_sample_size"] == data.size


def _mock_shared_screens(monkeypatch, likelihood, *, degree_config=None):
    from types import SimpleNamespace

    from gibbus._api import selection

    calls = []

    def fit(count, rows, degree, lower, upper):
        calls.append((count, degree, lower, upper, rows))
        return SimpleNamespace(
            components=tuple(
                SimpleNamespace(spec=SimpleNamespace(requested_poly_degree=degree))
                for _ in range(count)
            ),
            weights=np.full(count, 1.0 / count),
            log_likelihood=float(likelihood(count, degree)),
            n_face_parameters=2 * count + (count - 1) + int(lower) + int(upper),
            responsibilities=np.full((len(rows), count), 1.0 / count),
        )

    def single(support, rows, degree, lower, upper, weights, config):
        assert isinstance(config, _DegreeSelectionConfig)
        if degree_config is not None:
            assert config is degree_config
        return fit(1, rows, degree, lower, upper)

    def mixture(support, rows, degree, lower, upper, weights, resp, **kwargs):
        assert isinstance(kwargs["degree_config"], _DegreeSelectionConfig)
        if degree_config is not None:
            assert kwargs["degree_config"] is degree_config
        assert degree == (degree[0],) * resp.shape[1]
        return fit(resp.shape[1], rows, degree[0], lower, upper)

    def initial(x, count, rng, weights=None):
        resp = np.full((len(x), count), 1.0 / count)
        return [("mock", resp, np.full(count, 1.0 / count))]

    monkeypatch.setattr(selection, "_single_selection_fit", single)
    monkeypatch.setattr(selection, "_run_natural_em", mixture)
    monkeypatch.setattr(selection, "_initial_responsibility_candidates", initial)
    return selection, fit, calls


def test_staged_shortlist_refines_policy_and_expands_open_winning_edge(monkeypatch):
    selection, fit, _ = _mock_shared_screens(
        monkeypatch, lambda k, d: -float((k - (2 if d == 4 else 3)) ** 2)
    )
    seen = []

    def refine(candidate):
        k = candidate.n_components
        seen.append(k)
        fitted = fit(k, candidate.rows, 2, True, False)
        fitted.log_likelihood = -float((k - 5) ** 2)
        return selection._selection_payload(fitted)

    x = np.linspace(0.1, 0.9, 600)
    k, _, _, diagnostics = selection.select_n_components(
        S=x[:, None],
        samples_1d=x,
        supp=(0.0, 1.0),
        k_modes=3,
        gen=np.random.default_rng(5),
        obs_w=None,
        verb=0,
        policy=_prepare_component_selection_policy(
            len(x), (0.0, 1.0), 3, 6, False, "auto", "auto", False
        ),
        refine_candidate=refine,
        degree_config=_DegreeSelectionConfig(),
    )
    assert k == 5
    assert set(seen) == {1, 2, 3, 4, 5, 6}
    assert dict(diagnostics["admissions"])[5] == "open_refined_winner_edge"
    assert diagnostics["pruning"] == "heuristic_not_globally_certified"
    assert diagnostics["reuse_selected_fit"]
    assert all(
        record["degree"] == (2,) * record["n_components"]
        for record in diagnostics["scores"]
    )
    assert diagnostics["selected_bic"] == min(
        record["bic"] for record in diagnostics["scores"]
    )


def test_explicit_degree_and_full_data_endpoint_exclusion_survive_subsampling(
    monkeypatch,
):
    selection, fit, calls = _mock_shared_screens(monkeypatch, lambda k, d: -float(k))
    monkeypatch.setattr(
        selection, "_stratified_subsample", lambda x, m, rng: np.arange(1, m + 1)
    )
    x = np.linspace(0.0, 0.99, 400)
    refined_rows = []

    def refine(candidate):
        refined_rows.append(candidate.rows)
        return selection._selection_payload(
            fit(candidate.n_components, candidate.rows, 8, False, True)
        )

    _, _, _, diagnostics = selection.select_n_components(
        S=x[:, None],
        samples_1d=x,
        supp=(0.0, 1.0),
        k_modes=1,
        gen=np.random.default_rng(5),
        obs_w=None,
        verb=0,
        policy=_prepare_component_selection_policy(
            len(x), (0.0, 1.0), 1, 2, 100, 8, False, "auto"
        ),
        refine_candidate=refine,
        degree_config=_DegreeSelectionConfig(),
    )
    assert calls
    assert all(
        degree == 8 and lower is False and upper is True
        for _, degree, lower, upper, _ in calls
    )
    assert all(np.min(rows) > 0.0 for rows in refined_rows)
    assert diagnostics["screen_degrees"] == (8,)
    assert diagnostics["screen_boundary_enabled"] == (False, True)
    assert not diagnostics["reuse_selected_fit"]


def test_weighted_bic_uses_same_effective_size_for_likelihood_and_penalty(monkeypatch):
    selection, fit, _ = _mock_shared_screens(monkeypatch, lambda k, d: -float(k))
    x = np.linspace(-1.0, 1.0, 400)
    weights = _normalize_sample_weight_1d(len(x), np.linspace(1.0, 3.0, len(x)))
    seen_weights = []

    def refine(candidate):
        seen_weights.append(candidate.observation_weights)
        return selection._selection_payload(
            fit(candidate.n_components, candidate.rows, 2, False, False)
        )

    _, _, _, diagnostics = selection.select_n_components(
        S=x[:, None],
        samples_1d=x,
        supp=_REAL_LINE,
        k_modes=1,
        gen=np.random.default_rng(5),
        obs_w=weights,
        verb=0,
        policy=_prepare_component_selection_policy(
            len(x), _REAL_LINE, 1, 2, 120, "auto", False, False
        ),
        refine_candidate=refine,
        degree_config=_DegreeSelectionConfig(),
    )
    effective_n = weights.sum() ** 2 / np.dot(weights, weights)
    assert diagnostics["score_effective_n"] == pytest.approx(effective_n)
    assert "extrapolated" in diagnostics["likelihood_scale"]
    for record in diagnostics["scores"]:
        expected = -2 * record["mean_log_likelihood"] * effective_n + record[
            "n_parameters"
        ] * np.log(effective_n)
        assert record["bic"] == pytest.approx(expected)
    assert all(w.sum() == pytest.approx(1.0) for w in seen_weights)
    np.testing.assert_array_equal(seen_weights[0], seen_weights[1])


def test_selection_does_not_swallow_callback_value_error(monkeypatch):
    selection, _, _ = _mock_shared_screens(monkeypatch, lambda k, d: -float(k))
    x = np.linspace(-1.0, 1.0, 200)

    def refine(candidate):
        raise ValueError("invalid refinement callback contract")

    with pytest.raises(ValueError, match="invalid refinement callback contract"):
        selection.select_n_components(
            S=x[:, None],
            samples_1d=x,
            supp=_REAL_LINE,
            k_modes=1,
            gen=np.random.default_rng(5),
            obs_w=None,
            verb=0,
            policy=_prepare_component_selection_policy(
                len(x), _REAL_LINE, 1, 2, False, "auto", False, False
            ),
            refine_candidate=refine,
            degree_config=_DegreeSelectionConfig(),
        )


def test_all_failed_shared_refinements_raise_with_diagnostics(monkeypatch):
    selection, _, _ = _mock_shared_screens(monkeypatch, lambda k, d: -float(k))
    x = np.linspace(-1.0, 1.0, 200)

    def refine(candidate):
        raise RuntimeError("shared fit could not be certified")

    with pytest.raises(selection._ComponentSelectionError) as error:
        selection.select_n_components(
            S=x[:, None],
            samples_1d=x,
            supp=_REAL_LINE,
            k_modes=1,
            gen=np.random.default_rng(5),
            obs_w=None,
            verb=0,
            policy=_prepare_component_selection_policy(
                len(x), _REAL_LINE, 1, 2, False, "auto", False, False
            ),
            refine_candidate=refine,
            degree_config=_DegreeSelectionConfig(),
        )
    assert error.value.diagnostics["termination"] == "all_refinements_failed"
    assert error.value.diagnostics["method"] != "kde_fallback"
    assert all(
        "shared fit" in record["reason"] for record in error.value.diagnostics["scores"]
    )


def test_censored_identifiability_receives_shared_face_dimension(monkeypatch):
    selection, fit, _ = _mock_shared_screens(monkeypatch, lambda k, d: -float(k))
    dimensions = []

    def identifiable(
        rows, components, likelihood, support, *, obs_weights, n_parameters
    ):
        dimensions.append((len(components), n_parameters))

    monkeypatch.setattr(selection, "_interval_identifiability_diagnostic", identifiable)
    x = np.linspace(0.1, 0.9, 200)
    rows = np.column_stack((x - 0.01, x + 0.01))
    selection.select_n_components(
        S=rows,
        samples_1d=x,
        supp=(0.0, 1.0),
        k_modes=1,
        gen=np.random.default_rng(5),
        obs_w=None,
        verb=0,
        policy=_prepare_component_selection_policy(
            len(x), (0.0, 1.0), 1, 2, False, "auto", True, True
        ),
        refine_candidate=lambda c: selection._selection_payload(
            fit(c.n_components, c.rows, 2, True, True)
        ),
        degree_config=_DegreeSelectionConfig(),
    )
    assert dimensions
    assert all(dimension == 2 * k + k - 1 + 2 for k, dimension in dimensions)


@pytest.mark.parametrize(
    ("subsample", "n_rows", "expected"),
    [
        (False, 400, 400),
        (0, 400, 400),
        (120, 400, 120),
        (1000, 400, 400),
        ("auto", 400, 400),
    ],
)
def test_selection_policy_resolves_sample_size_at_boundary(subsample, n_rows, expected):
    policy = _prepare_component_selection_policy(
        n_rows, _REAL_LINE, 1, 4, subsample, None, False, False
    )
    assert policy.subsample_size == expected
    assert policy.degree_policy == "auto"
    assert policy.initial_hi <= policy.ceiling


def test_automatic_selection_sample_budget_depends_on_initial_sweep():
    from gibbus._defaults import (
        AUTO_LC_SUBSAMPLE_MIN_N,
        AUTO_LC_SUBSAMPLE_PER_K,
        AUTO_LC_SUBSAMPLE_SIZE,
    )

    n_rows = AUTO_LC_SUBSAMPLE_MIN_N + 10000
    policy = _prepare_component_selection_policy(
        n_rows, _REAL_LINE, 4, 8, "auto", 4, False, False
    )
    assert policy.subsample_size == min(
        n_rows,
        max(AUTO_LC_SUBSAMPLE_SIZE, policy.initial_hi * AUTO_LC_SUBSAMPLE_PER_K),
    )
    assert policy.degrees == (4,)


@pytest.mark.parametrize(
    ("subsample", "degree", "message"),
    [
        ("invalid", "auto", "subsample must"),
        (1, "auto", "subsample must be at least 2"),
        (False, 3, "not admissible"),
    ],
)
def test_selection_policy_rejects_invalid_controls_at_boundary(
    subsample, degree, message
):
    with pytest.raises(ValueError, match=message):
        _prepare_component_selection_policy(
            400, _REAL_LINE, 1, 2, subsample, degree, False, False
        )


def test_full_data_selection_reuses_canonical_inputs_by_identity(monkeypatch):
    config = _DegreeSelectionConfig(alpha=0.01, min_participation=16.0)
    selection, fit, _ = _mock_shared_screens(
        monkeypatch, lambda k, d: -float(k), degree_config=config
    )
    x = np.linspace(-1.0, 1.0, 200)
    rows = np.ascontiguousarray(x[:, None])
    weights = _normalize_sample_weight_1d(len(x), np.linspace(1.0, 2.0, len(x)))
    candidates = []

    def refine(candidate):
        candidates.append(candidate)
        assert candidate.rows is rows
        assert candidate.samples_1d is x
        assert candidate.observation_weights is weights
        return selection._selection_payload(
            fit(candidate.n_components, candidate.rows, 2, False, False)
        )

    selection.select_n_components(
        S=rows,
        samples_1d=x,
        supp=_REAL_LINE,
        k_modes=1,
        gen=np.random.default_rng(5),
        obs_w=weights,
        verb=0,
        policy=_prepare_component_selection_policy(
            len(x), _REAL_LINE, 1, 2, False, 2, False, False
        ),
        refine_candidate=refine,
        degree_config=config,
    )
    assert candidates
