"""EnbPI — ensemble batch prediction intervals for time series.

Xu & Xie (2021, ICML, PMLR 139, pp. 11559-11569; 2023, IEEE TPAMI 45(10),
"Conformal prediction for time series", arXiv:2010.09107). Distribution-free
prediction intervals for dependent sequences without exchangeability:

1. Train ``n_estimators`` bootstrap models on block-bootstrap resamples of the
   training window (circular blocks; Politis & Romano 1992 — plain i.i.d.
   resampling is the paper's default and is recovered with ``block_size=1``).
2. For every training point, the LEAVE-ONE-OUT (LOO) ensemble prediction is
   the aggregate (mean or median) of the models whose bootstrap sample did
   NOT include that point; the LOO residual is
   ``eps_i = |y_i - f^{-i}(x_i)|``.
3. For a test point, predict with the aggregate of all models and widen by
   the (1 - alpha) empirical quantile of the most recent ``n_train``
   residuals. The residual window slides: once ``y_{t}`` is observed, its
   out-of-sample residual replaces the oldest, so intervals track
   non-stationary error scales (Algorithm 1, lines 10-16). No model refits
   are required online.

The paper's Theorem 1 gives approximate marginal coverage
``P(y in C) >= 1 - alpha - O(sqrt(log(T)/T)) - O(delta)`` under strong
mixing of the errors and an ensemble-estimation error bound. This module
does not depend on the base learner: pass any object exposing
``fit(X, y)`` / ``predict(X)`` (sklearn contract) via ``model_factory``.

Fail-closed: too few training points, ``alpha`` out of range, degenerate
LOO sets (a point covered by every bootstrap sample when
``n_estimators`` is tiny) all raise. Honesty: only coverage / width
diagnostics are exposed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.conformal import conformal_quantile

Array = NDArray[np.float64]

__all__ = ["EnbPI", "EnbPIResult", "Regressor", "circular_block_bootstrap_indices"]


class Regressor(Protocol):
    def fit(self, X: Array, y: Array) -> object: ...

    def predict(self, X: Array) -> Array: ...


def circular_block_bootstrap_indices(
    n: int, block_size: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Circular block-bootstrap indices of length ``n`` (Politis & Romano 1992)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if block_size == 1:
        return rng.integers(0, n, size=n).astype(np.int64)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_size)
    idx = (starts[:, None] + offsets[None, :]).reshape(-1) % n
    return idx[:n].astype(np.int64)


@dataclass(frozen=True)
class EnbPIResult:
    lower: Array
    upper: Array
    point: Array
    coverage: float
    mean_width: float


class EnbPI:
    """EnbPI with sliding residual window and LOO ensemble aggregation.

    Parameters
    ----------
    model_factory:
        Zero-arg callable returning a fresh unfitted regressor.
    n_estimators:
        Number of bootstrap models B (paper default 20-30).
    alpha:
        Target miscoverage in (0, 1).
    block_size:
        Circular block length for the bootstrap; 1 = i.i.d. bootstrap.
    aggregate:
        'mean' or 'median' ensemble aggregation (phi in the paper).
    seed:
        RNG seed for bootstrap draws.
    """

    def __init__(
        self,
        model_factory: Callable[[], Regressor],
        n_estimators: int = 20,
        alpha: float = 0.1,
        block_size: int = 1,
        aggregate: str = "mean",
        seed: int = 0,
    ) -> None:
        if n_estimators < 2:
            raise ValueError("n_estimators must be >= 2")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if aggregate not in ("mean", "median"):
            raise ValueError("aggregate must be 'mean' or 'median'")
        self.model_factory = model_factory
        self.n_estimators = int(n_estimators)
        self.alpha = float(alpha)
        self.block_size = int(block_size)
        self.aggregate = aggregate
        self.seed = int(seed)
        self._models: list[Regressor] = []
        self._residuals: Array = np.empty(0)
        self._n_train = 0

    def _agg(self, preds: Array, axis: int = 0) -> Array:
        if self.aggregate == "median":
            return np.asarray(np.median(preds, axis=axis), dtype=float)
        return np.asarray(np.mean(preds, axis=axis), dtype=float)

    def fit(self, X: Array, y: Array) -> EnbPI:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n, p)")
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y length mismatch")
        min_n = max(10, int(np.ceil(1.0 / self.alpha)) + 1)
        if n < min_n:
            raise ValueError(f"need at least {min_n} training points for alpha={self.alpha}")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("X and y must be finite")

        rng = np.random.default_rng(self.seed)
        in_bag = np.zeros((self.n_estimators, n), dtype=bool)
        preds = np.empty((self.n_estimators, n), dtype=float)
        models: list[Regressor] = []
        for b in range(self.n_estimators):
            idx = circular_block_bootstrap_indices(n, self.block_size, rng)
            in_bag[b, idx] = True
            m = self.model_factory()
            m.fit(X[idx], y[idx])
            preds[b] = np.asarray(m.predict(X), dtype=float).ravel()
            models.append(m)

        oob = ~in_bag
        if np.any(oob.sum(axis=0) == 0):
            raise ValueError(
                "some training points appear in every bootstrap sample; "
                "increase n_estimators or reduce block_size"
            )
        masked = np.where(oob, preds, np.nan)
        if self.aggregate == "median":
            loo = np.nanmedian(masked, axis=0)
        else:
            loo = np.nanmean(masked, axis=0)
        self._residuals = np.abs(y - loo)
        self._models = models
        self._n_train = n
        return self

    @property
    def residuals(self) -> Array:
        return self._residuals.copy()

    def _check_fitted(self) -> None:
        if not self._models:
            raise RuntimeError("EnbPI is not fitted")

    def predict_point(self, X: Array) -> Array:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n, p)")
        preds = np.stack([np.asarray(m.predict(X), dtype=float).ravel() for m in self._models])
        return self._agg(preds, axis=0)

    def predict_interval(self, X: Array) -> tuple[Array, Array, Array]:
        """Return (lower, upper, point) using the current residual window."""
        point = self.predict_point(X)
        w = conformal_quantile(self._residuals, self.alpha)
        return point - w, point + w, point

    def update(self, X_new: Array, y_new: Array) -> Array:
        """Slide the residual window with newly observed (X, y); returns new residuals."""
        self._check_fitted()
        y_new = np.asarray(y_new, dtype=float).ravel()
        point = self.predict_point(X_new)
        if point.shape[0] != y_new.shape[0]:
            raise ValueError("X_new and y_new length mismatch")
        new_res = np.abs(y_new - point)
        if not np.all(np.isfinite(new_res)):
            raise ValueError("y_new must be finite")
        self._residuals = np.concatenate([self._residuals, new_res])[-self._n_train :]
        return new_res

    def predict_online(self, X: Array, y: Array, batch_size: int = 1) -> EnbPIResult:
        """Sequential Algorithm 1: predict a batch, then reveal its labels and slide."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")
        if X.shape[0] == 0:
            raise ValueError("no test points")
        n = X.shape[0]
        lower = np.empty(n)
        upper = np.empty(n)
        point = np.empty(n)
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            lo, hi, pt = self.predict_interval(X[sl])
            lower[sl], upper[sl], point[sl] = lo, hi, pt
            self.update(X[sl], y[sl])
        cov = float(np.mean((y >= lower) & (y <= upper)))
        return EnbPIResult(
            lower=lower,
            upper=upper,
            point=point,
            coverage=cov,
            mean_width=float(np.mean(upper - lower)),
        )
