"""Quantile regression forests (Meinshausen 2006).

Meinshausen (2006, JMLR 7, pp. 983-999, "Quantile Regression Forests"):
a random forest defines, for every test point x, a weight vector over the
training responses

    w_i(x) = (1/B) * sum_b  1{X_i in leaf_b(x)} / |leaf_b(x)|,

and the conditional CDF estimate ``F_hat(y | x) = sum_i w_i(x) 1{Y_i <= y}``.
Conditional quantiles are the generalized inverse of ``F_hat``. Standard
random-forest regression is the mean of that same distribution, so a QRF is
consistent (Theorem 1) whenever the forest weights concentrate. The forest
here is sklearn's ``RandomForestRegressor`` (for tree growing and leaf
lookup via ``apply``); all distributional logic is in-tree and pure numpy.

``leaf_mode``:

* ``"all"`` — Meinshausen's original: every training point in a leaf
  counts, including the tree's own bootstrap sample (biased toward the
  training response of x when x is a training point).
* ``"oob"`` — only out-of-bag training points contribute to each tree's
  weights (Athey, Tibshirani & Wager 2019 "honest" flavour; Johnson 2024
  quantile-forest). Used for in-sample PIT diagnostics.

Fail-closed: quantile levels outside (0, 1), unfitted use, dimensionality
mismatch, non-finite inputs all raise. Honesty: pinball/CRPS-style outputs
only via quant_fund.metrics.scoring.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

Array = NDArray[np.float64]

__all__ = ["QuantileRegressionForest", "weighted_quantiles"]


def weighted_quantiles(values: Array, weights: Array, taus: Array) -> Array:
    """Generalized-inverse quantiles of a discrete weighted distribution.

    ``values`` (n,), ``weights`` (m, n) rows summing to 1, ``taus`` (k,) →
    (m, k). Uses the smallest y with ``F_hat(y) >= tau`` (Meinshausen eq. 3).
    """
    values = np.asarray(values, dtype=float).ravel()
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    taus = np.asarray(taus, dtype=float).ravel()
    if taus.size == 0 or np.any(taus <= 0.0) or np.any(taus >= 1.0):
        raise ValueError("taus must be non-empty and in (0, 1)")
    if weights.shape[1] != values.shape[0]:
        raise ValueError("weights columns must match values length")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    row_sum = weights.sum(axis=1)
    if np.any(~np.isfinite(row_sum)) or np.any(row_sum <= 0.0):
        raise ValueError("each weight row must have positive finite mass")
    order = np.argsort(values, kind="stable")
    sorted_v = values[order]
    cdf = np.cumsum(weights[:, order] / row_sum[:, None], axis=1)
    cdf[:, -1] = 1.0
    out = np.empty((weights.shape[0], taus.size), dtype=float)
    for i in range(weights.shape[0]):
        pos = np.searchsorted(cdf[i], taus, side="left")
        out[i] = sorted_v[np.minimum(pos, sorted_v.size - 1)]
    return out


class QuantileRegressionForest:
    def __init__(
        self,
        n_estimators: int = 200,
        min_samples_leaf: int = 5,
        max_features: float | int | str | None = 1.0,
        max_depth: int | None = None,
        leaf_mode: str = "all",
        seed: int = 0,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if leaf_mode not in ("all", "oob"):
            raise ValueError("leaf_mode must be 'all' or 'oob'")
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_depth = max_depth
        self.leaf_mode = leaf_mode
        self.seed = int(seed)
        self._forest: RandomForestRegressor | None = None
        self._y: Array = np.empty(0)
        self._train_leaves: NDArray[np.int64] = np.empty((0, 0), dtype=np.int64)
        self._in_bag: NDArray[np.bool_] = np.empty((0, 0), dtype=bool)
        self._n_features = 0

    def fit(self, X: Array, y: Array) -> QuantileRegressionForest:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n, p)")
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X and y length mismatch")
        if n < 2 * self.min_samples_leaf:
            raise ValueError("too few samples for min_samples_leaf")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("X and y must be finite")
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_depth=self.max_depth,
            bootstrap=True,
            random_state=self.seed,
            n_jobs=1,
        )
        forest.fit(X, y)
        samples = getattr(forest, "estimators_samples_", None)
        if not isinstance(samples, list) or len(samples) != self.n_estimators:
            raise RuntimeError("bootstrap sample indices are unavailable; QRF needs bootstrap=True")
        train_leaves = np.asarray(forest.apply(X), dtype=np.int64)  # (n, B)
        in_bag = np.zeros((n, self.n_estimators), dtype=bool)
        for b, idx in enumerate(samples):
            in_bag[np.asarray(idx, dtype=np.int64), b] = True
        self._forest = forest
        self._y = y.copy()
        self._n_features = int(X.shape[1])
        self._train_leaves = train_leaves
        self._in_bag = in_bag
        return self

    def _check(self, X: Array) -> Array:
        if self._forest is None:
            raise RuntimeError("QuantileRegressionForest is not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError("X must be 2-D with the training feature count")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        return X

    def weights(self, X: Array) -> Array:
        """Forest weights w_i(x) over training responses; shape (m, n_train)."""
        X = self._check(X)
        assert self._forest is not None
        test_leaves = np.asarray(self._forest.apply(X), dtype=np.int64)  # (m, B)
        n = self._y.shape[0]
        m = X.shape[0]
        w = np.zeros((m, n), dtype=float)
        # Each tree adds 1/|leaf| on eligible training rows that share x's leaf.
        # OOB mode drops in-bag rows before that count (honest weights).
        eligible = ~self._in_bag if self.leaf_mode == "oob" else np.ones_like(self._in_bag)
        for b in range(self.n_estimators):
            same = test_leaves[:, [b]] == self._train_leaves[:, b]
            same &= eligible[:, b]
            counts = same.sum(axis=1)
            valid = counts > 0
            if not np.any(valid):
                continue
            w[valid] += same[valid] / counts[valid, None].astype(float)
        tree_mass = w.sum(axis=1, keepdims=True)
        if np.any(tree_mass[:, 0] <= 0.0):
            raise ValueError("a test point has no eligible training neighbours in any tree")
        return w / tree_mass

    def predict_quantiles(self, X: Array, taus: Array) -> Array:
        w = self.weights(X)
        return weighted_quantiles(self._y, w, np.asarray(taus, dtype=float))

    def predict_mean(self, X: Array) -> Array:
        w = self.weights(X)
        return np.asarray(w @ self._y, dtype=float)

    def predict_cdf(self, X: Array, y_grid: Array) -> Array:
        """F_hat(y | x) evaluated on ``y_grid``; shape (m, len(y_grid))."""
        w = self.weights(X)
        y_grid = np.asarray(y_grid, dtype=float).ravel()
        if y_grid.size == 0 or not np.all(np.isfinite(y_grid)):
            raise ValueError("y_grid must be non-empty and finite")
        ind = (self._y[None, :] <= y_grid[:, None]).astype(float)  # (g, n)
        return np.asarray(w @ ind.T, dtype=float)

    def pit(self, X: Array, y: Array) -> Array:
        """Randomized PIT of ``y`` under the forest CDF (uniform if calibrated)."""
        w = self.weights(X)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != w.shape[0]:
            raise ValueError("X and y length mismatch")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite")
        below = (self._y[None, :] < y[:, None]).astype(float)
        at = (self._y[None, :] == y[:, None]).astype(float)
        f_minus = (w * below).sum(axis=1)
        mass = (w * at).sum(axis=1)
        rng = np.random.default_rng(self.seed)
        return np.asarray(f_minus + rng.uniform(size=y.shape[0]) * mass, dtype=float)
