"""One-dimensional partial-dependence curves.

For feature ``j`` and grid value ``v``::

    PD_j(v) = mean_i predict(x_i with x_ij := v)

computed directly — no external dependency. The default grid is the feature's
empirical quantiles between ``quantile_clip`` so outliers cannot compress the
informative region. Fully deterministic: no randomness anywhere in the path.

For matrix-valued predictions (quantile heads) the PD is computed per output
column and also averaged to a scalar curve for ranking purposes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.research.explainability.scoring import PredictFn


@dataclass(frozen=True)
class PartialDependenceCurve:
    """One feature's partial-dependence curve on a fixed grid."""

    feature: str
    grid: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]  # per grid point: per output column
    n_rows: int

    @property
    def mean_curve(self) -> tuple[float, ...]:
        """Grid-point PD averaged over output columns (point heads: identity)."""
        if not self.values:
            return ()
        arr = np.asarray(self.values, dtype=float)
        return tuple(float(v) for v in arr.mean(axis=1).tolist())


def _default_grid(
    column: NDArray[np.float64],
    grid_points: int,
    quantile_clip: tuple[float, float],
) -> NDArray[np.float64]:
    finite = np.asarray(column, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, grid_points)
    lo, hi = quantile_clip
    edges = np.quantile(finite, np.linspace(lo, hi, grid_points))
    grid = np.unique(edges)
    if grid.size == 1:
        # Constant feature: a degenerate single-point grid is honest.
        return grid
    return grid


def partial_dependence_1d(
    predict: PredictFn,
    x: NDArray[np.float64],
    feature_index: int,
    feature_name: str | None = None,
    *,
    grid: NDArray[np.float64] | None = None,
    grid_points: int = 21,
    quantile_clip: tuple[float, float] = (0.01, 0.99),
) -> PartialDependenceCurve:
    """Compute the 1-D partial dependence of ``predict`` on column ``feature_index``.

    Every eval row contributes equally; non-finite feature values are frozen
    to the grid value like any other row (predict's own NaN policy applies).
    """
    xx = np.asarray(x, dtype=float)
    if xx.ndim != 2 or xx.shape[0] == 0:
        raise ValueError("x must be a non-empty 2-D array")
    n_rows, n_features = xx.shape
    j = int(feature_index)
    if not 0 <= j < n_features:
        raise ValueError(f"feature_index {j} out of range for {n_features} features")
    if grid_points < 2:
        raise ValueError("grid_points must be >= 2")
    lo, hi = quantile_clip
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("quantile_clip must satisfy 0 <= lo < hi <= 1")

    points = (
        np.unique(np.asarray(grid, dtype=float).ravel())
        if grid is not None
        else _default_grid(xx[:, j], int(grid_points), (float(lo), float(hi)))
    )
    if points.size == 0:
        raise ValueError("empty partial-dependence grid")
    if not np.isfinite(points).all():
        raise ValueError("partial-dependence grid must be finite")

    rows: list[tuple[float, ...]] = []
    for value in points.tolist():
        x_frozen = xx.copy()
        x_frozen[:, j] = float(value)
        preds = np.asarray(predict(x_frozen), dtype=float)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if preds.ndim != 2 or preds.shape[0] != n_rows:
            raise ValueError(f"predict returned unexpected shape {preds.shape}")
        with np.errstate(invalid="ignore"):
            col_means = np.nanmean(preds, axis=0)
        rows.append(tuple(float(v) for v in col_means.tolist()))
    name = feature_name if feature_name is not None else f"x{j}"
    return PartialDependenceCurve(
        feature=str(name),
        grid=tuple(float(v) for v in points.tolist()),
        values=tuple(rows),
        n_rows=int(n_rows),
    )


def partial_dependence_top_k(
    predict: PredictFn,
    x: NDArray[np.float64],
    feature_names: list[str],
    top_features: list[str],
    *,
    grid_points: int = 21,
    quantile_clip: tuple[float, float] = (0.01, 0.99),
) -> list[PartialDependenceCurve]:
    """PD curves for each named top feature, preserving ``top_features`` order."""
    xx = np.asarray(x, dtype=float)
    if xx.ndim != 2:
        raise ValueError("x must be a 2-D array")
    if len(feature_names) != xx.shape[1]:
        raise ValueError("feature_names length must match x columns")
    index = {name: j for j, name in enumerate(feature_names)}
    curves: list[PartialDependenceCurve] = []
    for name in top_features:
        if name not in index:
            raise ValueError(f"unknown feature {name!r} for partial dependence")
        curves.append(
            partial_dependence_1d(
                predict,
                xx,
                index[name],
                name,
                grid_points=grid_points,
                quantile_clip=quantile_clip,
            )
        )
    return curves
