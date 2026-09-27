"""Reverse stress testing inside a Mahalanobis ball.

The plausible set is

    (x − μ)ᵀ Σ⁻¹ (x − μ) ≤ c².

For a linear portfolio loss L(x) = −wᵀ x the worst point on that ellipsoid is
analytic:

    x* = μ − c Σ w / √(wᵀ Σ w).

A general loss is searched with a Mahalanobis-sphere design plus SLSQP.
The plausibility score is the Gaussian chi-square tail P(D² ≥ d²), so a
larger score is a more central scenario. This is a model of the supplied
covariance, not a market probability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

Array = NDArray[np.float64]
LossFn = Callable[[Array], float]


@dataclass(frozen=True)
class ReverseStressResult:
    scenario: Array
    loss: float
    mahalanobis_distance: float
    plausibility_score: float
    gaussian_density_ratio: float
    radius: float
    method: str
    on_boundary: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario": [float(value) for value in self.scenario],
            "loss": self.loss,
            "mahalanobis_distance": self.mahalanobis_distance,
            "plausibility_score": self.plausibility_score,
            "gaussian_density_ratio": self.gaussian_density_ratio,
            "radius": self.radius,
            "method": self.method,
            "on_boundary": self.on_boundary,
        }


def _as_cov(mu: Array, cov: Array) -> tuple[Array, Array]:
    center = np.asarray(mu, dtype=float).reshape(-1)
    matrix = np.asarray(cov, dtype=float)
    dim = center.size
    if matrix.shape != (dim, dim) or dim < 1:
        raise ValueError("mu and cov dimensions must match")
    if not np.isfinite(center).all() or not np.isfinite(matrix).all():
        raise ValueError("mu and cov must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("cov must be symmetric")
    if float(np.linalg.eigvalsh(matrix)[0]) <= 0.0:
        raise ValueError("cov must be positive definite")
    return np.asarray(center, dtype=np.float64), np.asarray(matrix, dtype=np.float64)


def mahalanobis_distance(x: Array, mu: Array, cov: Array) -> float:
    """Distance of ``x`` from ``mu`` under ``cov``."""
    point = np.asarray(x, dtype=float).reshape(-1)
    center, matrix = _as_cov(mu, cov)
    if point.shape != center.shape or not np.isfinite(point).all():
        raise ValueError("x must be a finite vector matching mu")
    diff = point - center
    solved = np.linalg.solve(matrix, diff)
    value = float(diff @ solved)
    return float(np.sqrt(max(value, 0.0)))


def plausibility_score(distance: float, n_factors: int) -> float:
    """Chi-square tail P(χ²_k ≥ distance²). One at the center, smaller outward."""
    if not np.isfinite(distance) or distance < 0.0 or n_factors < 1:
        raise ValueError("distance must be non-negative and n_factors >= 1")
    return float(stats.chi2.sf(distance**2, df=n_factors))


def default_radius(n_factors: int, contour: float = 0.05) -> float:
    """Radius whose Gaussian plausibility score equals ``contour``.

    ``contour=0.05`` is the boundary of the 95 percent ellipsoid.
    """
    if n_factors < 1 or not np.isfinite(contour) or not 0.0 < contour < 1.0:
        raise ValueError("n_factors >= 1 and contour in (0, 1)")
    return float(np.sqrt(stats.chi2.isf(contour, df=n_factors)))


def _pack(
    x: Array, mu: Array, cov: Array, loss: float, radius: float, method: str
) -> ReverseStressResult:
    distance = mahalanobis_distance(x, mu, cov)
    return ReverseStressResult(
        scenario=np.asarray(x, dtype=np.float64),
        loss=float(loss),
        mahalanobis_distance=distance,
        plausibility_score=plausibility_score(distance, int(np.asarray(mu).size)),
        gaussian_density_ratio=float(np.exp(-0.5 * distance**2)),
        radius=float(radius),
        method=method,
        on_boundary=bool(abs(distance - radius) <= 1e-5 * max(1.0, radius)),
    )


def worst_linear_scenario(
    weights: Array, mu: Array, cov: Array, radius: float
) -> ReverseStressResult:
    """Analytic worst point for loss = −wᵀ x on the Mahalanobis ball."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    center, matrix = _as_cov(mu, cov)
    if w.shape != center.shape or not np.isfinite(w).all():
        raise ValueError("weights must be a finite vector matching mu")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive")
    variance = float(w @ matrix @ w)
    if variance <= 0.0:
        raise ValueError("weights are in the null space of cov")
    scale = math_sqrt(variance)
    scenario = center - radius * (matrix @ w) / scale
    loss = float(-w @ scenario)
    return _pack(scenario, center, matrix, loss, radius, "analytic_linear")


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def reverse_stress(
    loss_fn: LossFn,
    mu: Array,
    cov: Array,
    radius: float,
    *,
    n_directions: int = 64,
    seed: int = 0,
) -> ReverseStressResult:
    """Maximize ``loss_fn(x)`` on the Mahalanobis ball of the given radius.

    The search evaluates the center, a deterministic eigen-direction set, and
    ``n_directions`` random boundary points, then runs SLSQP from the best
    feasible start. The returned point is the best feasible point found.
    """
    if not callable(loss_fn):
        raise TypeError("loss_fn must be callable")
    fn = cast(LossFn, loss_fn)
    center, matrix = _as_cov(mu, cov)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive")
    if isinstance(n_directions, bool) or not isinstance(n_directions, int) or n_directions < 1:
        raise ValueError("n_directions must be a positive integer")
    dim = center.size
    chol = np.linalg.cholesky(matrix)
    rng = np.random.default_rng(seed)

    def loss_of(point: Array) -> float:
        value = float(fn(np.asarray(point, dtype=np.float64)))
        if not np.isfinite(value):
            raise ValueError("loss_fn returned a non-finite value")
        return value

    candidates = [center.copy()]
    eigvals, eigvecs = np.linalg.eigh(matrix)
    for k in range(dim):
        direction = eigvecs[:, k] * np.sqrt(float(eigvals[k]))
        candidates.append(center + radius * direction)
        candidates.append(center - radius * direction)
    for _ in range(n_directions):
        gaussian = rng.standard_normal(dim)
        norm = float(np.linalg.norm(gaussian))
        if norm == 0.0:
            continue
        candidates.append(center + chol @ (radius * gaussian / norm))

    best_point = center.copy()
    best_loss = loss_of(best_point)
    for candidate in candidates:
        value = loss_of(candidate)
        if value > best_loss:
            best_loss = value
            best_point = np.asarray(candidate, dtype=float)

    def objective(point: Array) -> float:
        return -loss_of(np.asarray(point, dtype=float))

    def constraint(point: Array) -> float:
        return float(radius**2 - mahalanobis_distance(point, center, matrix) ** 2)

    result = optimize.minimize(
        objective,
        best_point,
        method="SLSQP",
        constraints={"type": "ineq", "fun": constraint},
        options={"maxiter": 200, "ftol": 1e-12},
    )
    if result.success and np.isfinite(result.x).all():
        refined = np.asarray(result.x, dtype=float)
        if constraint(refined) >= -1e-7:
            refined_loss = loss_of(refined)
            if refined_loss > best_loss:
                best_point = refined
                best_loss = refined_loss
    distance = mahalanobis_distance(best_point, center, matrix)
    if distance > radius * (1.0 + 1e-6):
        raise RuntimeError("reverse stress left the Mahalanobis ball")
    return _pack(best_point, center, matrix, best_loss, radius, "sphere_search_slsqp")


def sample_mean_cov(panel: Array, ridge: float = 0.0) -> tuple[Array, Array]:
    """Column mean and covariance. Optional ridge is explicit, never silent."""
    data = np.asarray(panel, dtype=float)
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] < 1:
        raise ValueError("panel must have at least 3 rows")
    if not np.isfinite(data).all():
        raise ValueError("panel must be finite")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    mean = np.asarray(data.mean(axis=0), dtype=np.float64)
    cov = np.cov(data, rowvar=False, ddof=1)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    if ridge > 0.0:
        cov = cov + ridge * np.eye(cov.shape[0])
    return mean, np.asarray(cov, dtype=np.float64)
