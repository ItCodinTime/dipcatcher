"""EnbPI — ensemble batch prediction intervals for time series.

Xu & Xie (2021, ICML, PMLR 139, pp. 11559-11569; 2023, IEEE TPAMI 45(10),
"Conformal prediction for time series", arXiv:2010.09107). Distribution-free
prediction intervals for dependent sequences without exchangeability:

1. Train ``n_estimators`` bootstrap models on block-bootstrap resamples of the
   training window (circular blocks; Politis & Romano 1992 — plain i.i.d.
   resampling is the paper's default and is recovered with ``block_size=1``).
2. For every training point, the leave-one-out (LOO) ensemble prediction is
   the aggregate (mean or median) of the models whose bootstrap sample did
   not include that point. The LOO residual is signed,
   ``eps_i = y_i - f^{-i}(x_i)``.
3. For a test point, predict with the aggregate of all models and add the
   narrowest empirical band ``[q_beta, q_{1-alpha+beta}]`` of the most recent
   ``n_train`` signed residuals (Algorithm 1: beta in ``[0, alpha]`` minimises
   width). Once ``y_t`` is observed, its out-of-sample residual replaces the
   oldest, so the band tracks non-stationary errors. No model refits are
   required online.

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

Array = NDArray[np.float64]

__all__ = [
    "EnbPI",
    "EnbPIResult",
    "Regressor",
    "circular_block_bootstrap_indices",
    "enbpi_residual_bounds",
]


class Regressor(Protocol):
    def fit(self, X: Array, y: Array) -> object: ...

    def predict(self, X: Array) -> Array: ...


def circular_block_bootstrap_indices(
    n: int, block_size: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Circular block-bootstrap indices of length ``n`` (Politis & Romano 1992)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if block_size < 1 or block_size > n:
        raise ValueError("block_size must be in [1, n]")
    if block_size == 1:
        return rng.integers(0, n, size=n).astype(np.int64)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_size)
    idx = (starts[:, None] + offsets[None, :]).reshape(-1) % n
    return idx[:n].astype(np.int64)


def enbpi_residual_bounds(residuals: Array, alpha: float) -> tuple[float, float]:
    """Narrowest ``[q_beta, q_{1-alpha+beta}]`` band of signed residuals.

    On the empirical measure this is the shortest window of
    ``ceil(n * (1 - alpha))`` order statistics (Xu & Xie, Algorithm 1, the
    beta minimisation). The returned offsets are added to the point forecast.
    """
    ordered = np.sort(np.asarray(residuals, dtype=float).ravel())
    if ordered.size == 0 or not np.all(np.isfinite(ordered)):
        raise ValueError("residuals must be non-empty and finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = int(ordered.size)
    cover = int(np.ceil(n * (1.0 - alpha)))
    cover = min(max(cover, 1), n)
    starts = ordered[: n - cover + 1]
    ends = ordered[cover - 1 :]
    i = int(np.argmin(ends - starts))
    return float(starts[i]), float(ends[i])


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
        self._n_features = 0

    def _agg(self, preds: Array, axis: int = 0) -> Array:
        if self.aggregate == "median":
            return np.asarray(np.median(preds, axis=axis), dtype=float)
        return np.asarray(np.mean(preds, axis=axis), dtype=float)

    def _agg_masked(self, preds: Array, mask: NDArray[np.bool_]) -> Array:
        """Aggregate along models, ignoring entries where ``mask`` is false."""
        masked = np.where(mask, preds, np.nan)
        if self.aggregate == "median":
            return np.asarray(np.nanmedian(masked, axis=0), dtype=float)
        return np.asarray(np.nanmean(masked, axis=0), dtype=float)

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
            pred = np.asarray(m.predict(X), dtype=float).ravel()
            if pred.shape != (n,) or not np.all(np.isfinite(pred)):
                raise ValueError("base learner predict must return n finite values")
            preds[b] = pred
            models.append(m)

        oob = ~in_bag
        if np.any(oob.sum(axis=0) == 0):
            raise ValueError(
                "some training points appear in every bootstrap sample; "
                "increase n_estimators or reduce block_size"
            )
        loo = self._agg_masked(preds, oob)
        residuals = y - loo
        if not np.all(np.isfinite(residuals)):
            raise ValueError("LOO residuals are not finite")
        self._residuals = residuals
        self._models = models
        self._n_train = n
        self._n_features = X.shape[1]
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
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError("X must be 2-D with the training feature count")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        preds = np.stack([np.asarray(m.predict(X), dtype=float).ravel() for m in self._models])
        if preds.shape[1] != X.shape[0] or not np.all(np.isfinite(preds)):
            raise ValueError("base learner predict must return n finite values")
        return self._agg(preds, axis=0)

    def predict_interval(self, X: Array) -> tuple[Array, Array, Array]:
        """Return (lower, upper, point) from the current signed-residual window."""
        point = self.predict_point(X)
        lo, hi = enbpi_residual_bounds(self._residuals, self.alpha)
        return point + lo, point + hi, point

    def update(self, X_new: Array, y_new: Array) -> Array:
        """Slide the residual window with newly observed (X, y); returns new residuals."""
        self._check_fitted()
        y_new = np.asarray(y_new, dtype=float).ravel()
        point = self.predict_point(X_new)
        if point.shape[0] != y_new.shape[0]:
            raise ValueError("X_new and y_new length mismatch")
        new_res = y_new - point
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
