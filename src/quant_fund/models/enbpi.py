"""Ensemble batch prediction intervals (EnbPI) for time series.

Xu & Xie (2021), "Conformal prediction interval for dynamic time-series",
ICML, PMLR 139:11559–11569, and the journal form in Xu & Xie (2023),
"Conformal prediction for time series", IEEE TPAMI (arXiv:2010.09107v15,
Algorithm 1). EnbPI wraps a fixed bootstrap ensemble: leave-one-out (LOO)
residuals on the training rows, then a prediction interval whose center is
an aggregate of those LOO predictors and whose width is a β-optimized
empirical quantile of the residual list. Residuals slide forward in batches
of size ``s`` once feedback arrives, without refitting.

This module is model-agnostic. Callers fit the bootstrap models and pass
their in-sample prediction matrix plus the in-bag mask. The interval math
is the journal algorithm:

    β̂ = argmin_{β ∈ [0, α]} ( q_{1-α+β}(ε) − q_β(ε) )
    C(x) = [ f̂(x) + q_β̂(ε),  f̂(x) + q_{1-α+β̂}(ε) ]

with ε the signed LOO residuals y − f̂_{−i}(x). Quantiles are the linear
(numpy type-7) empirical quantile. No Sharpe / P&L content.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IndexArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

__all__ = [
    "EnbPI",
    "EnbPIInterval",
    "block_bootstrap_indices",
    "in_bag_mask",
    "leave_one_out_predictions",
    "optimal_beta",
    "prediction_interval",
    "signed_residuals",
]


@dataclass(frozen=True)
class EnbPIInterval:
    """One EnbPI interval around ``point``."""

    lower: float
    upper: float
    point: float
    width_lower: float
    width_upper: float
    beta: float


def _as_finite_vector(values: Array | list[float], name: str) -> Array:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be non-empty and finite")
    return np.asarray(arr, dtype=np.float64)


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _check_aggregate(aggregate: str) -> str:
    if aggregate not in ("mean", "median"):
        raise ValueError("aggregate must be 'mean' or 'median'")
    return aggregate


def empirical_quantile(samples: Array, level: float) -> float:
    """Linear empirical quantile of ``samples`` at ``level`` ∈ [0, 1]."""
    lvl = float(level)
    if not np.isfinite(lvl) or lvl < 0.0 or lvl > 1.0:
        raise ValueError("quantile level must be in [0, 1]")
    data = _as_finite_vector(samples, "samples")
    return float(np.quantile(data, lvl, method="linear"))


def optimal_beta(residuals: Array, alpha: float, n_grid: int = 25) -> float:
    """Grid-search β ∈ [0, α] minimizing the asymmetric interval width.

    Xu & Xie (2023) Algorithm 1, line 13. Ties break toward the smaller β.
    """
    a = _check_alpha(alpha)
    grid_n = int(n_grid)
    if grid_n < 2:
        raise ValueError("n_grid must be at least 2")
    data = _as_finite_vector(residuals, "residuals")
    grid = np.linspace(0.0, a, grid_n)
    widths = np.empty(grid_n, dtype=float)
    for i, beta in enumerate(grid):
        upper = empirical_quantile(data, 1.0 - a + float(beta))
        lower = empirical_quantile(data, float(beta))
        widths[i] = upper - lower
    return float(grid[int(np.argmin(widths))])


def prediction_interval(
    point: float,
    residuals: Array,
    alpha: float = 0.1,
    n_grid: int = 25,
) -> EnbPIInterval:
    """β-optimized EnbPI interval centered at ``point`` (Algorithm 1, lines 13–16)."""
    p = float(point)
    if not np.isfinite(p):
        raise ValueError("point must be finite")
    a = _check_alpha(alpha)
    data = _as_finite_vector(residuals, "residuals")
    beta = optimal_beta(data, a, n_grid=n_grid)
    width_lower = empirical_quantile(data, beta)
    width_upper = empirical_quantile(data, 1.0 - a + beta)
    return EnbPIInterval(
        lower=p + width_lower,
        upper=p + width_upper,
        point=p,
        width_lower=width_lower,
        width_upper=width_upper,
        beta=beta,
    )


def block_bootstrap_indices(
    n: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    block_length: int = 1,
) -> IndexArray:
    """Bootstrap row indices of shape ``(n_bootstrap, n)``.

    ``block_length=1`` draws indices independently with replacement (Algorithm 1,
    line 2). ``block_length>1`` is the non-overlapping block bootstrap described
    in the prose of Xu & Xie (2023): contiguous blocks of that length, leftover
    tail dropped, blocks drawn with replacement and concatenated, then truncated
    to length ``n``.
    """
    length = int(n)
    draws = int(n_bootstrap)
    block = int(block_length)
    if length < 1 or draws < 1:
        raise ValueError("n and n_bootstrap must be positive")
    if block < 1 or block > length:
        raise ValueError("block_length must be in [1, n]")
    if block == 1:
        idx = rng.integers(0, length, size=(draws, length))
        return np.asarray(idx, dtype=np.int64)
    n_blocks = length // block
    starts = np.arange(n_blocks, dtype=np.int64) * block
    n_draw = int(np.ceil(length / block))
    chosen = rng.choice(n_blocks, size=(draws, n_draw), replace=True)
    out = np.empty((draws, length), dtype=np.int64)
    for b in range(draws):
        pieces = [np.arange(starts[j], starts[j] + block, dtype=np.int64) for j in chosen[b]]
        out[b] = np.concatenate(pieces)[:length]
    return out


def in_bag_mask(indices: IndexArray, n: int) -> BoolArray:
    """Boolean mask ``(n, n_bootstrap)``; True where row i was drawn into bootstrap b."""
    idx = np.asarray(indices)
    if idx.ndim != 2:
        raise ValueError("indices must have shape (n_bootstrap, n_draw)")
    length = int(n)
    if length < 1:
        raise ValueError("n must be positive")
    if np.any(idx < 0) or np.any(idx >= length):
        raise ValueError("bootstrap indices fall outside [0, n)")
    mask = np.zeros((length, idx.shape[0]), dtype=bool)
    for b in range(idx.shape[0]):
        mask[idx[b], b] = True
    return mask


def leave_one_out_predictions(
    predictions: Array,
    in_bag: Array,
    aggregate: str = "mean",
) -> Array:
    """LOO ensemble prediction at each training row.

    ``predictions[i, b]`` is bootstrap model b evaluated at row i.
    ``in_bag[i, b]`` is True when row i was used to fit model b. The LOO
    aggregate uses only models with ``in_bag[i, b]`` False (Algorithm 1,
    line 7). A row with no out-of-bag model fails closed.
    """
    how = _check_aggregate(aggregate)
    preds = np.asarray(predictions, dtype=float)
    bag = np.asarray(in_bag, dtype=bool)
    if preds.ndim != 2 or bag.shape != preds.shape:
        raise ValueError("predictions and in_bag must share shape (n, n_bootstrap)")
    if preds.shape[0] < 1 or preds.shape[1] < 1:
        raise ValueError("predictions must be non-empty")
    if not np.all(np.isfinite(preds)):
        raise ValueError("predictions must be finite")
    n_rows = preds.shape[0]
    out = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        oob = ~bag[i]
        if not bool(np.any(oob)):
            raise ValueError(f"row {i} has no out-of-bag bootstrap model")
        vals = preds[i, oob]
        out[i] = float(np.mean(vals) if how == "mean" else np.median(vals))
    return out


def signed_residuals(y: Array, loo_predictions: Array) -> Array:
    """Signed LOO residuals ε = y − f̂_{−i}(x) (Algorithm 1, line 8)."""
    target = _as_finite_vector(y, "y")
    pred = _as_finite_vector(loo_predictions, "loo_predictions")
    if target.shape != pred.shape:
        raise ValueError("y and loo_predictions must have the same length")
    return np.asarray(target - pred, dtype=np.float64)


class EnbPI:
    """Online residual window for EnbPI intervals.

    ``residuals`` is the ordered training LOO residual list. ``observe`` buffers
    feedback residuals ``y − loo_prediction`` and, once ``batch_size`` of them
    have arrived, slides them in one-for-one (Algorithm 1, lines 17–20): drop
    the oldest residual, append the new one, repeated ``batch_size`` times.
    ``interval`` uses the residual list as it stands; a partial batch does not
    move the window.
    """

    def __init__(
        self,
        residuals: Array,
        alpha: float = 0.1,
        batch_size: int = 1,
        n_grid: int = 25,
    ) -> None:
        self._residuals = _as_finite_vector(residuals, "residuals")
        self.alpha = _check_alpha(alpha)
        batch = int(batch_size)
        if batch < 1:
            raise ValueError("batch_size must be positive")
        grid_n = int(n_grid)
        if grid_n < 2:
            raise ValueError("n_grid must be at least 2")
        self.batch_size = batch
        self.n_grid = grid_n
        self._pending: list[float] = []

    @property
    def residuals_(self) -> Array:
        """Copy of the current ordered residual window."""
        return self._residuals.copy()

    @property
    def n_pending(self) -> int:
        """Feedback residuals buffered but not yet slid into the window."""
        return len(self._pending)

    def interval(self, point: float) -> EnbPIInterval:
        """Interval at ``point`` from the current residual window."""
        return prediction_interval(point, self._residuals, self.alpha, n_grid=self.n_grid)

    def observe(self, y: float, loo_prediction: float) -> None:
        """Record one feedback residual; slide the window when the batch fills."""
        actual = float(y)
        pred = float(loo_prediction)
        if not np.isfinite(actual) or not np.isfinite(pred):
            raise ValueError("y and loo_prediction must be finite")
        self._pending.append(actual - pred)
        if len(self._pending) < self.batch_size:
            return
        for resid in self._pending:
            self._residuals = np.concatenate(
                [self._residuals[1:], np.asarray([resid], dtype=np.float64)]
            )
        self._pending.clear()
