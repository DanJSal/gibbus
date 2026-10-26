"""Reproducible, isolated fitting benchmarks over case manifests.

A runner records measurements rather than asserting arbitrary quality thresholds.
Subprocess isolation gives each fit a wall-time limit, keeps a crashed numerical
extension from taking down the batch, and limits effects from earlier fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .cases import SCHEMA_VERSION, CaseSpec, case_from_dict, materialize
from .distributions import cdf as target_cdf
from .distributions import pdf as target_pdf
from .distributions import sample as target_sample

RESULT_SCHEMA = 1
SUCCESS = "ok"
FAILURES = (
    "sampling_error",
    "fit_error",
    "diagnostics_error",
    "worker_error",
    "timeout",
)


def _float(value):
    """JSON-safe scalar, keeping unknown or nonfinite measures as null."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _safe_json(value: Any):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return _float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def load_manifest(path: Path) -> list[CaseSpec]:
    """Validate the manifest and both stable IDs before executing anything."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema") != SCHEMA_VERSION or not isinstance(obj.get("cases"), list):
        raise ValueError("unsupported generator manifest schema")
    cases, seen = [], set()
    for index, item in enumerate(obj["cases"]):
        spec = case_from_dict(item["spec"])
        if (
            item.get("scenario_id") != spec.scenario_id
            or item.get("case_id") != spec.case_id
        ):
            raise ValueError(f"manifest entry {index} has inconsistent IDs")
        if spec.case_id in seen:
            raise ValueError(f"duplicate case ID: {spec.case_id}")
        seen.add(spec.case_id)
        cases.append(spec)
    return cases


def _seed_for(spec: CaseSpec, seed: int, role: str) -> int:
    raw = f"GBRUN-v1:{role}:{spec.case_id}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _exception(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:1600],
        "traceback": "".join(traceback.format_exception(exc))[-6000:],
    }


def _array(value, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.broadcast_to(arr, (size,))
    if arr.shape != (size,):
        raise ValueError(f"expected {(size,)}, got {arr.shape}")
    return arr


def _likelihood_metrics(model, generated) -> dict:
    """Observed-data log score, never the unavailable latent-data likelihood."""
    if generated.observations.ndim == 1:
        ll = _array(model.logpdf(generated.observations), generated.spec.n)
    else:
        limits = generated.observations
        exact = limits[:, 0] == limits[:, 1]
        ll = np.empty(generated.spec.n)
        if exact.any():
            ll[exact] = _array(model.logpdf(limits[exact, 0]), int(exact.sum()))
        if (~exact).any():
            lo, hi = limits[~exact, 0], limits[~exact, 1]
            # Prefer whichever pair of tails avoids subtracting quantities
            # very close to 1. Both are the same probability mathematically.
            cl = _array(model.cdf(lo), len(lo))
            ch = _array(model.cdf(hi), len(hi))
            sl = _array(model.sf(lo), len(lo))
            sh = _array(model.sf(hi), len(hi))
            diff_cdf = ch - cl
            diff_sf = sl - sh
            mass = np.where(cl > 0.5, diff_sf, diff_cdf)
            with np.errstate(divide="ignore", invalid="ignore"):
                ll[~exact] = np.log(mass)
    valid = np.isfinite(ll)
    weights = generated.weights
    value = float(np.average(ll, weights=weights)) if valid.all() else None
    return {
        "observed_mean_log_likelihood": _float(value),
        "unresolved_observation_mass_count": int((~valid).sum()),
        "realized_censored_fraction": float(generated.censored.mean()),
        "weighted": weights is not None,
    }


def _validation_metrics(model, spec: CaseSpec, seed: int, holdout_size: int) -> dict:
    """Independent holdout scores and math checks; no stochastic pass cutoff."""
    rng = np.random.default_rng(_seed_for(spec, seed, "holdout"))
    x = np.asarray(target_sample(spec.distribution, holdout_size, rng), dtype=float)
    if x.shape != (holdout_size,) or not np.isfinite(x).all():
        raise ValueError("nonfinite holdout reference samples")
    x.sort()
    values = {}
    logfit = _array(model.logpdf(x), holdout_size)
    values["holdout_mean_log_density"] = (
        _float(np.mean(logfit)) if np.isfinite(logfit).all() else None
    )
    values["holdout_nonfinite_log_density_count"] = int((~np.isfinite(logfit)).sum())
    fitted_cdf = _array(model.cdf(x), holdout_size)
    values["cdf_outside_unit_interval_count"] = int(
        np.sum(
            ~np.isfinite(fitted_cdf) | (fitted_cdf < -1e-10) | (fitted_cdf > 1 + 1e-10)
        )
    )
    values["cdf_monotonicity_max_violation"] = _float(
        max(0.0, float(np.max(-np.diff(fitted_cdf))))
    )
    # One-sample empirical KS is descriptive, not a failure criterion.
    rank = np.arange(1, holdout_size + 1, dtype=float) / holdout_size
    values["empirical_ks"] = _float(
        max(
            np.max(np.abs(fitted_cdf - rank)),
            np.max(np.abs(fitted_cdf - (rank - 1 / holdout_size))),
        )
    )
    try:
        truth_cdf = _array(target_cdf(spec.distribution, x), holdout_size)
        truth_pdf = _array(target_pdf(spec.distribution, x), holdout_size)
    except NotImplementedError:
        values["oracle"] = "unavailable_convolution"
        values["oracle_cdf_max_abs_error"] = None
        values["oracle_cdf_mean_abs_error"] = None
        values["holdout_logscore_gap_vs_truth"] = None
    else:
        values["oracle"] = "independent_analytic_or_scipy"
        diffs = np.abs(fitted_cdf - truth_cdf)
        values["oracle_cdf_max_abs_error"] = _float(np.max(diffs))
        values["oracle_cdf_mean_abs_error"] = _float(np.mean(diffs))
        with np.errstate(divide="ignore", invalid="ignore"):
            logtrue = np.log(truth_pdf)
        values["holdout_logscore_gap_vs_truth"] = (
            _float(np.mean(logtrue - logfit))
            if np.isfinite(logtrue - logfit).all()
            else None
        )
    p = np.linspace(0.02, 0.98, 17)
    q = _array(model.ppf(p), len(p))
    values["ppf_nonfinite_count"] = int((~np.isfinite(q)).sum())
    if np.isfinite(q).all():
        values["ppf_monotonicity_max_violation"] = _float(
            max(0.0, float(np.max(-np.diff(q))))
        )
        p_back = _array(model.cdf(q), len(q))
        values["cdf_ppf_max_abs_error"] = _float(np.max(np.abs(p - p_back)))
        density = _array(model.pdf(q), len(q))
        values["negative_pdf_count"] = int(np.sum(density < 0))
    else:
        values.update(
            ppf_monotonicity_max_violation=None,
            cdf_ppf_max_abs_error=None,
            negative_pdf_count=None,
        )
    return values


def run_case(spec: CaseSpec, *, seed: int = 42, holdout_size: int = 128) -> dict:
    """Run one numerical fit, preserving separate sampling/fitting/scoring states."""
    if holdout_size < 16:
        raise ValueError("holdout_size must be >= 16")
    spec.validate()
    base = {
        "result_schema": RESULT_SCHEMA,
        "case_id": spec.case_id,
        "scenario_id": spec.scenario_id,
        "seed": seed,
        "strata": spec.strata(),
        "status": None,
        "fit_completed": False,
        "fit_converged": None,
        "timings_s": {},
        "fit_diagnostics": None,
        "spectral_diagnostics": None,
        "likelihood": None,
        "validation": None,
        "error": None,
    }
    wall = time.perf_counter()
    stage = "sampling_error"
    try:
        start = time.perf_counter()
        generated = materialize(spec, seed=seed)
        base["timings_s"]["sampling"] = time.perf_counter() - start
        stage = "fit_error"
        from gibbus import Distribution

        kw = generated.fit_kwargs
        kw["rng"] = _seed_for(spec, seed, "fit") % (2**32)
        start = time.perf_counter()
        model = Distribution().fit(generated.observations, **kw)
        base["timings_s"]["fit"] = time.perf_counter() - start
        base["fit_completed"] = True
        stage = "diagnostics_error"
        start = time.perf_counter()
        fit_diag = model.fit_diagnostics
        spectral_diag = model.spectral_diagnostics
        base["fit_diagnostics"] = _safe_json(fit_diag)
        base["spectral_diagnostics"] = _safe_json(spectral_diag)
        base["fit_converged"] = fit_diag.get("converged")
        base["fitted_components"] = int(model.n_components)
        base["likelihood"] = _likelihood_metrics(model, generated)
        base["validation"] = _validation_metrics(model, spec, seed, holdout_size)
        base["timings_s"]["evaluation"] = time.perf_counter() - start
        base["status"] = SUCCESS
    except Exception as exc:
        base["status"] = stage
        base["error"] = _exception(exc)
    finally:
        base["timings_s"]["total"] = time.perf_counter() - wall
    return _safe_json(base)


def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=".gibbus-runner-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _load_existing(path: Path) -> dict[str, dict]:
    results = {}
    if path.exists():
        for line_num, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["case_id"] in results:
                raise ValueError(f"duplicate case ID in results at line {line_num}")
            results[row["case_id"]] = row
    return results


def _execute(
    spec: CaseSpec,
    *,
    manifest_path: Path,
    index: int,
    seed: int,
    holdout_size: int,
    timeout_s: float,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="gibbus-run-") as temp:
        output = Path(temp) / "result.json"
        cmd = [
            sys.executable,
            "-m",
            "tests.casegen.runner",
            "--worker",
            "--manifest",
            str(manifest_path),
            "--index",
            str(index),
            "--seed",
            str(seed),
            "--holdout",
            str(holdout_size),
            "--result",
            str(output),
        ]
        env = os.environ.copy()
        for name in (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = "1"
        started = time.perf_counter()
        try:
            done = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
            duration = time.perf_counter() - started
            if output.is_file():
                result = json.loads(output.read_text(encoding="utf-8"))
                if result.get("case_id") != spec.case_id:
                    raise ValueError("worker case ID mismatch")
                result["worker_wall_s"] = _float(duration)
                return result
            why = f"worker exit code {done.returncode}; stderr: {done.stderr.decode(errors='replace')[-1800:]}"
            return {
                "result_schema": RESULT_SCHEMA,
                "case_id": spec.case_id,
                "scenario_id": spec.scenario_id,
                "seed": seed,
                "strata": spec.strata(),
                "status": "worker_error",
                "fit_completed": False,
                "fit_converged": None,
                "worker_wall_s": _float(duration),
                "error": {"type": "WorkerProcessError", "message": why},
            }
        except subprocess.TimeoutExpired:
            return {
                "result_schema": RESULT_SCHEMA,
                "case_id": spec.case_id,
                "scenario_id": spec.scenario_id,
                "seed": seed,
                "strata": spec.strata(),
                "status": "timeout",
                "fit_completed": False,
                "fit_converged": None,
                "worker_wall_s": _float(time.perf_counter() - started),
                "error": {
                    "type": "TimeoutExpired",
                    "message": f"wall-time limit {timeout_s:g}s exceeded",
                },
            }


def _summarize_group(rows: list[dict]):
    status = Counter(r["status"] for r in rows)
    fit_complete = sum(r.get("fit_completed", False) for r in rows)
    converged = sum(r.get("fit_converged") is True for r in rows)
    wall = [r.get("worker_wall_s") for r in rows if r.get("worker_wall_s") is not None]
    return {
        "count": len(rows),
        "status_counts": dict(sorted(status.items())),
        "fit_completed": fit_complete,
        "fit_converged": converged,
        "fit_convergence_fraction_of_attempted": (
            _float(converged / len(rows)) if rows else None
        ),
        "worker_wall_s_sum": _float(sum(wall)),
    }


def summarize(rows: list[dict], *, manifest_total: int | None = None) -> dict:
    """Summarize completed attempts without treating error metrics as pass/fail."""
    groups = {}
    for key in ("fit_category", "composition", "support", "observation", "degree"):
        by = defaultdict(list)
        for row in rows:
            by[row.get("strata", {}).get(key, "unknown")].append(row)
        groups[key] = {v: _summarize_group(r) for v, r in sorted(by.items())}
    return {
        "result_schema": RESULT_SCHEMA,
        "attempted": len(rows),
        "manifest_total": manifest_total,
        "overall": _summarize_group(rows),
        "groups": groups,
        "interpretation": "Fit completion/convergence, numerical metrics and data-model mismatch are separate; no accuracy acceptance thresholds applied.",
    }


def _write_summary(dir_path: Path, results: dict[str, dict], manifest_total: int):
    rows = [results[k] for k in sorted(results)]
    summary = summarize(rows, manifest_total=manifest_total)
    _atomic_json(dir_path / "summary.json", summary)
    lines = [
        "# Case runner results",
        "",
        f"Attempted: {summary['attempted']} / {manifest_total}",
        f"Fit completed: {summary['overall']['fit_completed']}",
        f"Optimizer reported converged: {summary['overall']['fit_converged']}",
        "",
        "## Results by fit category",
        "",
        "| Category | Attempted | Fit completed | Converged | Outcomes |",
        "|---|---:|---:|---:|---|",
    ]
    for name, group in summary["groups"]["fit_category"].items():
        outcomes = ", ".join(
            f"{name}:{n}" for name, n in group["status_counts"].items()
        )
        lines.append(
            f"| {name} | {group['count']} | {group['fit_completed']} | {group['fit_converged']} | {outcomes} |"
        )
    lines.extend(
        [
            "",
            "## Important distinctions",
            "",
            "An `ok` row means fitting and metric collection ran without an exception; it is not a statistical accuracy guarantee.",
            "Numerical quality (CDF errors, spectral mass defect, inverse consistency) must be analyzed independently.",
            "Convolution targets have no analytic/SciPy CDF oracle; oracle-specific measures are null.",
            "Intervals are scored using their observed probability mass, not latent unobserved values.",
            "",
        ]
    )
    (dir_path / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _implementation_fingerprint() -> dict:
    """Bind a result journal to its actual Python and compiled implementation."""
    from importlib.util import find_spec

    package_spec = find_spec("gibbus")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ImportError("the Gibbus package is not importable")
    package_root = Path(next(iter(package_spec.submodule_search_locations))).resolve()
    paths = sorted(
        p
        for p in package_root.rglob("*")
        if p.is_file() and p.suffix in (".py", ".pyx", ".pxd", ".so", ".pyd")
    )
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(package_root)).encode("utf-8"))
        h.update(hashlib.sha256(path.read_bytes()).digest())
    runner_root = Path(__file__).resolve().parent
    code = hashlib.sha256()
    for name in (
        "runner.py",
        "cases.py",
        "distributions.py",
        "catalog.py",
        "__init__.py",
    ):
        code.update(name.encode("utf-8"))
        code.update(hashlib.sha256((runner_root / name).read_bytes()).digest())
    return {
        "gibbus_code_sha256": h.hexdigest(),
        "casegen_runner_sha256": code.hexdigest(),
    }


def _run_batch(args) -> int:
    manifest_path = args.manifest.resolve()
    cases = load_manifest(manifest_path)
    if not cases:
        raise ValueError("manifest contains no cases")
    selected = [
        (i, c)
        for i, c in enumerate(cases)
        if args.case_id is None or c.case_id == args.case_id
    ]
    if not selected:
        raise ValueError("requested case ID is not in manifest")
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("selected no cases")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config = {
        "result_schema": RESULT_SCHEMA,
        "manifest_sha256": manifest_hash,
        "seed": args.seed,
        "holdout": args.holdout,
        "timeout_s": args.timeout,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        **_implementation_fingerprint(),
    }
    try:
        from scipy import __version__ as scipy_version

        config["scipy"] = scipy_version
    except ImportError:
        pass
    if config_path.exists():
        prev = json.loads(config_path.read_text())
        for k in (
            "result_schema",
            "manifest_sha256",
            "seed",
            "holdout",
            "python",
            "platform",
            "numpy",
            "scipy",
            "gibbus_code_sha256",
            "casegen_runner_sha256",
        ):
            if prev.get(k) != config[k]:
                raise ValueError(
                    f"cannot resume run: {k} differs; use a new output directory"
                )
    else:
        _atomic_json(config_path, config)
    output_path = output_dir / "results.jsonl"
    existing = _load_existing(output_path)
    valid_ids = {c.case_id for c in cases}
    if any(k not in valid_ids for k in existing):
        raise ValueError("result file contains case IDs not in this manifest")
    pending = []
    for index, c in selected:
        if c.case_id in existing and not (
            args.rerun_failed and existing[c.case_id].get("status") != SUCCESS
        ):
            continue
        pending.append((index, c))
    if args.dry_run:
        print(
            f"Manifest has {len(cases)} cases; selected {len(selected)}, pending {len(pending)}"
        )
        return 0
    print(
        f"Case runner: {len(pending)} pending / {len(selected)} selected, jobs={args.jobs}",
        flush=True,
    )

    # An interrupted batch remains valid JSONL, with every completed row
    # flushed to disk immediately. Re-runs replace failed rows atomically by
    # rewriting the journal, while ordinary new rows append.
    def persist(result):
        key = result["case_id"]
        existed = key in existing
        existing[key] = result
        if not existed:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        else:
            tmp = output_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for k in sorted(existing):
                    f.write(
                        json.dumps(existing[k], sort_keys=True, allow_nan=False) + "\n"
                    )
            os.replace(tmp, output_path)

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    _execute,
                    c,
                    manifest_path=manifest_path,
                    index=i,
                    seed=args.seed,
                    holdout_size=args.holdout,
                    timeout_s=args.timeout,
                ): c
                for i, c in pending
            }
            for future in as_completed(futures):
                c = futures[future]
                try:
                    res = future.result()
                except Exception as exc:
                    res = {
                        "result_schema": RESULT_SCHEMA,
                        "case_id": c.case_id,
                        "scenario_id": c.scenario_id,
                        "seed": args.seed,
                        "strata": c.strata(),
                        "status": "worker_error",
                        "fit_completed": False,
                        "fit_converged": None,
                        "error": _exception(exc),
                    }
                persist(res)
                print(
                    f"[{len(existing)}/{len(cases)}] {c.case_id} {res['status']} ",
                    flush=True,
                )
                _write_summary(output_dir, existing, len(cases))
    except KeyboardInterrupt:
        _write_summary(output_dir, existing, len(cases))
        print(
            "Interrupted; completed cases persisted and may be resumed.",
            file=sys.stderr,
        )
        return 130
    summary = _write_summary(output_dir, existing, len(cases))
    print(
        f"Recorded {summary['attempted']}/{len(cases)}; fit complete {summary['overall']['fit_completed']}; "
        + f"converged {summary['overall']['fit_converged']}. Output: {output_dir}"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("case_results"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", type=str, help="replay only this case ID")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="sample/holdout/fit seed (selection seed remains in manifest)",
    )
    parser.add_argument("--holdout", type=int, default=128)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds per case, subprocess-enforced",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # Worker mode is an internal subprocess contract, not a user-facing API.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.holdout < 16
        or args.timeout <= 0
        or args.jobs < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("holdout >=16, timeout >0, jobs >=1 and limit >=1 are required")
    if args.worker:
        if args.result is None or args.index is None:
            parser.error("internal worker needs --index and --result")
        items = load_manifest(args.manifest)
        try:
            spec = items[args.index]
        except IndexError:
            parser.error("worker manifest index outside range")
        _atomic_json(
            args.result, run_case(spec, seed=args.seed, holdout_size=args.holdout)
        )
        return 0
    return _run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
