"""Runner contract tests: replay, journaling, numerical metrics and isolation."""

import json

import numpy as np
import pytest

from tests.casegen import (
    CaseSpec,
    FitSpec,
    Named,
    ObservationSpec,
    Sum,
    manifest,
    runner,
)


@pytest.fixture
def normal():
    return CaseSpec(
        Named("normal", mu=0, sigma=1),
        ObservationSpec("points"),
        FitSpec(degree=2),
        100,
    )


@pytest.fixture
def cases(normal):
    censored = CaseSpec(
        Named("gamma", shape=2, scale=1),
        ObservationSpec("right", fraction=0.2),
        FitSpec(degree=2),
        100,
    )
    convolution = CaseSpec(
        Sum(Named("normal", mu=0, sigma=1), Named("logistic", mu=0, scale=1)),
        ObservationSpec("points"),
        FitSpec(degree=2),
        100,
    )
    return [normal, censored, convolution]


def _write(path, case_specs):
    path.write_text(json.dumps(manifest(case_specs, 42)))
    return path


def test_manifest_rejects_ids_and_duplicates(tmp_path, cases):
    doc = manifest(cases, 42)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc))
    assert [c.case_id for c in runner.load_manifest(path)] == [c.case_id for c in cases]
    doc["cases"][0]["case_id"] = "FAKE"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="inconsistent IDs"):
        runner.load_manifest(path)
    doc = manifest([cases[0], cases[0]], 42)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="duplicate case ID"):
        runner.load_manifest(path)


def test_seed_and_sample_replay(cases):
    for c in cases:
        assert runner._seed_for(c, 42, "fit") == runner._seed_for(c, 42, "fit")
        assert runner._seed_for(c, 42, "holdout") != runner._seed_for(c, 42, "fit")
    assert runner._seed_for(cases[0], 43, "fit") != runner._seed_for(
        cases[0], 42, "fit"
    )


def test_run_case_without_gibbus_for_sampling_failures(normal, monkeypatch):
    def fail(*a, **kw):
        raise ValueError("deliberate sampling error")

    monkeypatch.setattr(runner, "materialize", fail)
    result = runner.run_case(normal)
    assert result["status"] == "sampling_error"
    assert result["fit_completed"] is False
    assert result["error"]["type"] == "ValueError"
    assert result["fit_converged"] is None


def test_oracle_unavailable_is_not_zero_error(cases):
    class Fitted:
        def logpdf(self, x):
            return np.zeros_like(x, dtype=float)

        def pdf(self, x):
            return np.ones_like(x, dtype=float)

        def cdf(self, x):
            return np.clip(np.asarray(x, dtype=float) * 0.1 + 0.5, 0, 1)

        def ppf(self, p):
            return (np.asarray(p) - 0.5) * 10

    r = runner._validation_metrics(Fitted(), cases[2], 42, 32)
    assert r["oracle"] == "unavailable_convolution"
    assert r["oracle_cdf_max_abs_error"] is None
    assert r["holdout_logscore_gap_vs_truth"] is None


def test_censored_likelihood_uses_probability_mass_not_latent(cases):
    from tests.casegen import materialize

    generated = materialize(cases[1])

    class Model:
        def logpdf(self, x):
            return np.zeros_like(x, dtype=float)

        def cdf(self, x):
            x = np.asarray(x, dtype=float)
            return np.where(
                np.isneginf(x), 0, np.where(np.isposinf(x), 1, np.clip(x / 10, 0, 1))
            )

        def sf(self, x):
            return 1 - self.cdf(x)

    r = runner._likelihood_metrics(Model(), generated)
    assert r["realized_censored_fraction"] > 0
    assert r["unresolved_observation_mass_count"] == 0
    assert r["observed_mean_log_likelihood"] is not None


def test_summary_preserves_separate_success_and_convergence(normal):
    rows = [
        {
            "case_id": "a",
            "status": "ok",
            "fit_completed": True,
            "fit_converged": False,
            "strata": normal.strata(),
        },
        {
            "case_id": "b",
            "status": "fit_error",
            "fit_completed": False,
            "fit_converged": None,
            "strata": normal.strata(),
        },
    ]
    s = runner.summarize(rows, manifest_total=100)
    assert s["overall"]["fit_completed"] == 1
    assert s["overall"]["fit_converged"] == 0
    assert s["overall"]["status_counts"] == {"fit_error": 1, "ok": 1}


def test_batch_resume_and_run_metadata(tmp_path, cases, monkeypatch):
    manifest_path = _write(tmp_path / "cases.json", cases)
    target = tmp_path / "output"
    seen = []

    def fake_execute(spec, **kw):
        seen.append(spec.case_id)
        return {
            "result_schema": runner.RESULT_SCHEMA,
            "case_id": spec.case_id,
            "scenario_id": spec.scenario_id,
            "seed": kw["seed"],
            "strata": spec.strata(),
            "status": "ok",
            "fit_completed": True,
            "fit_converged": True,
        }

    monkeypatch.setattr(runner, "_execute", fake_execute)
    args = [
        "--manifest",
        str(manifest_path),
        "--output",
        str(target),
        "--limit",
        "2",
        "--jobs",
        "2",
    ]
    assert runner.main(args) == 0
    assert len(seen) == 2
    assert runner.main(args) == 0  # Resume skips prior attempts.
    assert len(seen) == 2
    assert runner.main(args + ["--rerun-failed"]) == 0
    assert len(seen) == 2


def test_batch_blocks_incompatible_resume(tmp_path, cases, monkeypatch):
    path = _write(tmp_path / "cases.json", cases)
    out = tmp_path / "res"
    assert (
        runner.main(
            ["--manifest", str(path), "--output", str(out), "--limit", "1", "--dry-run"]
        )
        == 0
    )
    # Dry-run leaves its provenance manifest written, but no worker processes.
    with pytest.raises(ValueError, match="cannot resume"):
        runner.main(
            ["--manifest", str(path), "--output", str(out), "--seed", "99", "--dry-run"]
        )


def test_canonical_json_nulls():
    assert runner._safe_json(
        {"nan": float("nan"), "inf": float("inf"), "array": np.array([1, np.nan])}
    ) == {"nan": None, "inf": None, "array": [1.0, None]}


def test_real_compiled_fit_smoke(normal):
    try:
        from gibbus import Distribution  # noqa: F401
    except ImportError:
        pytest.skip("Gibbus compiled extensions not installed")
    result = runner.run_case(normal, holdout_size=32)
    assert result["status"] == "ok", result["error"]
    assert result["fit_completed"] is True
    assert result["fit_converged"] is True
    assert result["validation"]["oracle"] == "independent_analytic_or_scipy"
    assert result["validation"]["cdf_ppf_max_abs_error"] < 1e-5


def test_subprocess_timeout_is_recorded(tmp_path, normal):
    manifest_path = _write(tmp_path / "cases.json", [normal])
    result = runner._execute(
        normal,
        manifest_path=manifest_path,
        index=0,
        seed=42,
        holdout_size=32,
        timeout_s=0.001,
    )
    assert result["status"] == "timeout"
    assert result["fit_completed"] is False
    assert result["error"]["type"] == "TimeoutExpired"


def test_rerun_failed_replaces_rows(tmp_path, cases, monkeypatch):
    manifest_path = _write(tmp_path / "cases.json", cases)
    output = tmp_path / "out"
    failed_once = set()

    def fake(spec, **kw):
        first = spec.case_id not in failed_once
        failed_once.add(spec.case_id)
        return {
            "result_schema": runner.RESULT_SCHEMA,
            "case_id": spec.case_id,
            "scenario_id": spec.scenario_id,
            "seed": kw["seed"],
            "strata": spec.strata(),
            "status": "fit_error" if first else "ok",
            "fit_completed": not first,
            "fit_converged": not first,
        }

    monkeypatch.setattr(runner, "_execute", fake)
    args = ["--manifest", str(manifest_path), "--output", str(output), "--limit", "2"]
    runner.main(args)
    assert [
        r["status"] for r in runner._load_existing(output / "results.jsonl").values()
    ] == ["fit_error"] * 2
    runner.main(args + ["--rerun-failed"])
    assert all(
        r["status"] == "ok"
        for r in runner._load_existing(output / "results.jsonl").values()
    )
    assert len((output / "results.jsonl").read_text().splitlines()) == 2
