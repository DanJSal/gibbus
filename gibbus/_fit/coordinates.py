"""Runtime coefficient coordinates with directly evaluated polynomial recurrences.

A plan changes coordinates inside the existing column span. Gates, endpoint logs,
shared coefficients and tail requests remain structural properties of that span.
Only a constant potential gauge may be added. Durable models stay canonical.
"""

import math
from dataclasses import dataclass, replace
from fractions import Fraction as F

import numpy as np

from .._model.basis import _restrict_to


class _CoordinateMismatch(ArithmeticError):
    """A runtime coordinate conversion failed likelihood accounting."""


@dataclass(frozen=True)
class _PolynomialCoordinates:
    family: str
    degree: int
    origin: float
    scale: float
    alpha: float
    norms: np.ndarray
    reference: np.ndarray
    recurrence: np.ndarray
    polynomial_map: np.ndarray
    transform: np.ndarray
    transform_low: np.ndarray
    column_degree: np.ndarray
    column_fn: np.ndarray

    def _wide_transform(self):
        return self.transform.astype(np.longdouble) + self.transform_low.astype(
            np.longdouble
        )

    def to_canonical(self, coefficients):
        return np.asarray(
            self._wide_transform() @ np.asarray(coefficients, dtype=np.longdouble),
            dtype=float,
        )

    def from_canonical(self, coefficients):
        """Back substitution in the degree-ordered, block-triangular map."""
        result = np.asarray(coefficients, dtype=np.longdouble).copy()
        matrix = self._wide_transform()
        # Column positions need not be degree ordered in a general basis.
        order = np.argsort(self.column_degree, kind="stable")
        for j in order[::-1]:
            result[j] /= matrix[j, j]
            result -= matrix[:, j] * result[j] * (np.arange(len(result)) != j)
        return np.asarray(result, dtype=float)


def _recurrence(family, degree, alpha=0.0):
    """Normalized three-term recurrence and its exact binary64 polynomial map."""
    norms = np.ones(degree + 1)
    if family == "laguerre":
        for n in range(1, degree + 1):
            norms[n] = norms[n - 1] * math.sqrt((n + alpha) / n)
    elif family == "hermite":
        for n in range(1, degree + 1):
            norms[n] = norms[n - 1] * math.sqrt(n)
    raw = [[F(1)]]
    coefficients = []
    for n in range(degree):
        if family == "laguerre":
            a, b, c = (
                -F(1, n + 1),
                (2 * n + F(alpha) + 1) / (n + 1),
                (n + F(alpha)) / (n + 1),
            )
        elif family == "chebyshev":
            a, b, c = F(1 if n == 0 else 2), F(0), F(0 if n == 0 else 1)
        else:
            a, b, c = F(1), F(0), F(n)
        p = [F(0)] * (n + 2)
        for k, value in enumerate(raw[n]):
            p[k] += b * value
            p[k + 1] += a * value
        if n:
            for k, value in enumerate(raw[n - 1]):
                p[k] -= c * value
        raw.append(p)
        coefficients.append(
            (
                float(a * F(float(norms[n])) / F(float(norms[n + 1]))),
                float(b * F(float(norms[n])) / F(float(norms[n + 1]))),
                (
                    float(c * F(float(norms[n - 1])) / F(float(norms[n + 1])))
                    if n
                    else 0.0
                ),
            )
        )
    polynomials = [[v / F(float(norms[n])) for v in p] for n, p in enumerate(raw)]
    return norms, np.ascontiguousarray(coefficients), polynomials


def _polynomial_coordinates(model, points=None, weights=None, start=None, movable=None):
    """Construct support-appropriate coordinates without enlarging a gated span.

    Complete degree prefixes with identical gates can be reparameterized.
    Incomplete or coupled columns retain their direct canonical evaluator;
    converting those columns alone would enlarge the requested span or cause
    cancellation in a recurrence expansion of an isolated monomial.
    """
    if model.polynomial_coordinates is not None:
        return model
    degree = max((f.degree for f in model.functions if f.kind == "monomial"), default=0)
    if degree < 1:
        return model
    request, geometry = model.request, model.geometry
    lower, upper = request.lower, request.upper
    alpha = 0.0
    if math.isfinite(lower) and math.isfinite(upper):
        family, origin, scale = (
            "chebyshev",
            geometry.center,
            min(geometry.scale, upper / 2 - lower / 2),
        )
    elif math.isfinite(lower) or math.isfinite(upper):
        family = "laguerre"
        origin = lower if math.isfinite(lower) else upper
        side = "lower" if math.isfinite(lower) else "upper"
        if start is not None:
            for j, column in enumerate(model.columns):
                if (
                    column.nonnegative
                    and model.functions[column.terms[0][0]].side == side
                ):
                    alpha = max(0.0, float(start[j]))
        distance = geometry.scale
        if points is not None:
            distance = float(
                np.average(np.abs(np.asarray(points) - origin), weights=weights)
            ) / (alpha + 1)
        scale = distance if math.isfinite(lower) else -distance
    else:
        family, origin, scale = "hermite", geometry.center, geometry.scale
    if not math.isfinite(scale) or scale == 0 or not math.isfinite(origin):
        return model
    norms, recurrence, polynomials = _recurrence(family, degree, alpha)
    offset = (F(geometry.center) - F(origin)) / F(scale)
    slope = F(geometry.scale) / F(scale)
    expanded = []
    for p in polynomials:
        expanded.append(
            [
                sum(
                    p[j] * math.comb(j, k) * offset ** (j - k) * slope**k
                    for j in range(k, len(p))
                )
                for k in range(len(p))
            ]
        )
    d = model.n_coefficients
    exact = [[F(int(i == j)) for j in range(d)] for i in range(d)]
    constants = [F(0)] * d
    column_degree = np.zeros(d, dtype=np.intc)
    column_fn = np.zeros(d, dtype=np.intc)
    groups = {}
    movable = set(range(d)) if movable is None else set(movable)
    for j, column in enumerate(model.columns):
        column_fn[j] = column.terms[0][0]
        if j not in movable or len(column.terms) != 1:
            continue
        fn, gate = column.terms[0]
        function = model.functions[fn]
        if function.kind == "monomial" and function.degree > 0:
            groups.setdefault(gate, {})[function.degree] = j
    reference = np.array([float(sum(p)) for p in polynomials])
    for gate, columns in groups.items():
        n = 0
        while n + 1 in columns:
            n += 1
        for power in range(1, n + 1):
            col = columns[power]
            column_degree[col] = power
            for k in range(1, power + 1):
                exact[columns[k]][col] = expanded[power][k]
            # Only an ungated constant is a harmless potential gauge.
            if gate == (F(1), F(0), F(0)):
                constants[col] = expanded[power][0] - F(float(reference[power]))
    if not np.any(column_degree):
        return model
    transformed = _restrict_to(model, list(map(list, zip(*exact))), model.face)
    full = [list(row) for row in transformed.coefficient_map_exact]
    zero = next(
        f.index for f in model.functions if f.kind == "monomial" and f.degree == 0
    )
    for j, value in enumerate(constants):
        full[3 * zero][j] += value
    # Convert each polynomial/gate slot to the recurrence exactly before rounding.
    mapped = [[[F(0)] * d for _ in range(3)] for _ in range(degree + 1)]
    indices = {f.degree: f.index for f in model.functions if f.kind == "monomial"}
    for gate in range(3):
        for col in range(d):
            remainder = [full[3 * indices[n] + gate][col] for n in range(degree + 1)]
            for n in range(degree, -1, -1):
                value = remainder[n] / expanded[n][n]
                mapped[n][gate][col] = value
                for k, v in enumerate(expanded[n]):
                    remainder[k] -= value * v
    transform = np.array(exact, dtype=float)
    low = np.array(
        [
            [float(v - F(float(transform[i, j]))) for j, v in enumerate(row)]
            for i, row in enumerate(exact)
        ]
    )
    plan = _PolynomialCoordinates(
        family,
        degree,
        origin,
        scale,
        alpha,
        norms,
        reference,
        recurrence,
        np.ascontiguousarray(np.array(mapped, dtype=float).reshape(-1, d)),
        transform,
        low,
        column_degree,
        column_fn,
    )
    return replace(
        transformed,
        coefficient_map_exact=tuple(map(tuple, full)),
        coefficient_map=np.array(full, dtype=float),
        polynomial_coordinates=plan,
        columns=tuple(
            replace(
                c,
                terms=tuple(
                    (f.index, tuple(full[3 * f.index + k][j] for k in range(3)))
                    for f in model.functions
                    if any(full[3 * f.index + k][j] for k in range(3))
                ),
            )
            for j, c in enumerate(transformed.columns)
        ),
    )


@dataclass
class _CoordinateBinding:
    original: object
    objective: object
    original_blocks: object
    blocks: object
    start: object
    transform: np.ndarray
    inverse: np.ndarray

    def candidates(self, canonical_start):
        """Try the recurrence representation, then the same family canonically.

        The second representation is only needed after a numerical failure,
        unresolved optimization, or failed output validation. Both use the
        identical controller and acceptance rules.
        """
        yield self
        if self.original is not self.objective:
            identity = np.eye(self.original_blocks.n)
            yield _CoordinateBinding(
                self.original,
                self.original,
                self.original_blocks,
                self.original_blocks,
                canonical_start,
                identity,
                identity,
            )

    def restore(self, result, curvature_options):
        """Validate the canonical likelihood and shape before returning a fit."""
        from .basis_constraints import _ConstraintRows
        from .basis_newton import _face_from, _JointCurvature
        from .curvature_search import _CurvatureOptions

        if self.original is self.objective:
            return result
        beta = result.params
        # Block maps use extended accumulation, including their low parts.
        theta = np.asarray(
            self.transform.astype(np.longdouble) @ beta.astype(np.longdouble),
            dtype=float,
        )
        for k, model in enumerate(self.blocks.models):
            plan = model.polynomial_coordinates
            if plan is not None:
                indices = self.blocks.index[k]
                theta[indices] = plan.to_canonical(beta[indices])
        zero_logs = set(result.face.zero_logs)
        for k, model in enumerate(self.original_blocks.models):
            for j, column in enumerate(model.columns):
                index = self.original_blocks.index[k][j]
                if column.nonnegative and theta[index] == 0.0:
                    zero_logs.add(int(index))
        face = _face_from(
            self.original_blocks, result.face.suppressed, tuple(sorted(zero_logs))
        )
        if face is None:
            raise RuntimeError(
                "numerical coordinates returned an improper canonical face"
            )
        canonical_objective = self.original
        canonical_curvature = _JointCurvature(self.original_blocks)
        if face.equalities.size:
            canonical_basis = face.basis(self.original_blocks.n)
            canonical_objective = canonical_objective.on_face(canonical_basis)
            canonical_curvature = canonical_curvature.on_face(canonical_basis)
        evaluation = canonical_objective(theta)
        errors = 0.0
        for state in (evaluation.state, result.evaluation.state):
            if state is not None:
                errors += state.rel_error + state.cond_error
        if evaluation.state is None:
            # Joint objectives expose per-component normalization contexts.
            # Recover their numerical error estimates, not an ad hoc NLL slack.
            from .._model.basis_normalizer import _basis_state

            for objective, blocks, coefficients in (
                (canonical_objective, self.original_blocks, theta),
                (
                    (
                        self.objective.on_face(result.face.basis(self.blocks.n))
                        if result.face.equalities.size
                        else self.objective
                    ),
                    self.blocks,
                    beta,
                ),
            ):
                for k, context in enumerate(objective.contexts):
                    state = _basis_state(
                        context, blocks.part(k, coefficients), objective.options
                    )
                    errors += state.rel_error + state.cond_error
        tolerance = 32 * (errors + np.finfo(float).eps * max(1.0, abs(evaluation.nll)))
        if (
            not math.isfinite(evaluation.nll)
            or abs(evaluation.nll - result.objective_value) > tolerance
        ):
            raise _CoordinateMismatch(
                "numerical and canonical likelihoods disagree at the fitted point"
            )
        rows = result.rows.rows @ self.inverse
        norms = np.linalg.norm(rows, axis=1)
        rows /= np.where(norms > 0, norms, 1)[:, None]
        separation = canonical_curvature.separate(
            theta, curvature_options or _CurvatureOptions()
        )
        status = result.status
        if (
            status in ("converged", "converged_approximately")
            and not separation.feasible
        ):
            status = "uncertified"
        families = sorted(
            {
                m.polynomial_coordinates.family
                for m in self.blocks.models
                if m.polynomial_coordinates is not None
            }
        )
        return replace(
            result,
            status=status,
            params=theta,
            evaluation=evaluation,
            objective_value=float(evaluation.nll),
            rows=_ConstraintRows(rows, result.rows.kinds),
            face=face,
            final_separation=separation,
            coordinate_system="+".join(families),
        )


def _prepare_coordinates(objective, blocks, start):
    """Bind single and coupled objectives to the same coordinate protocol."""
    from .basis_newton import _Blocks

    original_blocks = _Blocks.single(objective.model) if blocks is None else blocks
    identity = np.eye(original_blocks.n)
    unchanged = _CoordinateBinding(
        objective,
        objective,
        original_blocks,
        original_blocks,
        start,
        identity,
        identity,
    )
    if not hasattr(objective, "coordinate_observations"):
        return unchanged
    if any(m.polynomial_coordinates is not None for m in original_blocks.models):
        return unchanged
    counts = np.bincount(
        np.concatenate(original_blocks.index), minlength=original_blocks.n
    )
    models, transform = [], identity.copy()
    for k, model in enumerate(original_blocks.models):
        observations = objective.coordinate_observations(k)
        if observations is None:
            return unchanged
        points, weights = observations
        indices = original_blocks.index[k]
        movable = np.flatnonzero(counts[indices] == 1)
        numerical = _polynomial_coordinates(
            model,
            points,
            weights,
            None if start is None else np.asarray(start)[indices],
            movable,
        )
        models.append(numerical)
        if numerical.polynomial_coordinates is not None:
            transform[np.ix_(indices, indices)] = (
                numerical.polynomial_coordinates.transform
            )
    if all(a is b for a, b in zip(models, original_blocks.models)):
        return unchanged
    numerical_blocks = replace(original_blocks, models=tuple(models))
    numerical_objective = objective.in_coordinates(numerical_blocks)
    inverse = identity.copy()
    for k, model in enumerate(models):
        plan = model.polynomial_coordinates
        if plan is not None:
            indices = numerical_blocks.index[k]
            inverse[np.ix_(indices, indices)] = np.column_stack(
                [plan.from_canonical(column) for column in np.eye(len(indices))]
            )
    numerical_start = None if start is None else np.asarray(start, dtype=float).copy()
    if numerical_start is not None:
        for k, model in enumerate(models):
            plan = model.polynomial_coordinates
            if plan is not None:
                indices = numerical_blocks.index[k]
                numerical_start[indices] = plan.from_canonical(
                    np.asarray(start)[indices]
                )
    return _CoordinateBinding(
        objective,
        numerical_objective,
        original_blocks,
        numerical_blocks,
        numerical_start,
        transform,
        inverse,
    )
