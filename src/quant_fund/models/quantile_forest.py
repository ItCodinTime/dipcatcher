"""Quantile regression forests (QRF): conditional distributions from random forests.

Meinshausen, N. (2006), "Quantile Regression Forests", *Journal of Machine
Learning Research* 7:983–999, https://jmlr.org/papers/v7/meinshausen06a.html.
A random forest partitions the covariate space into terminal nodes
R(x; θ_t) per tree t; instead of averaging only the y-mean in the leaf (the
classical regression forest), QRF uses the full empirical distribution of
training responses in the leaf. The estimated conditional CDF is the forest
average (Meinshausen 2006, Eq. 2)

    F̂(y | x) = Σ_{i=1}^{n} w_i(x) 1{y_i ≤ y} / Σ_{i=1}^{n} w_i(x),
    w_i(x) = (1/T) Σ_t 1{X_i ∈ R(x; θ_t)},

i.e. each training sample is weighted by the frequency with which it shares a
terminal node with the query point x, across the T fitted trees. The τ-quantile
is the (left-continuous) inverse F̂^{-1}(τ | x) = inf{y : F̂(y | x) ≥ τ}.
Meinshausen (2006, Sec. 2.2) allows general terminal-node weight sequences
(the paper's theoretical treatment uses weights ω over the nodes of a single
tree with the constraint Σ_i ω_i(x) = 1); the implementation here uses the
canonical EQUAL weight 1/T per tree — the Monte-Carlo estimate of the
expectation E_Θ[1{X_i ∈ R(x)}] over tree randomness, and the weighting that
matches RandomForestRegressor's own prediction aggregation (uniform over
estimators). Data-driven omega weighting (e.g. leaf-size-dependent weights)
is deliberately omitted. Consistency F̂(y|x) → F(y|x) for the underlying
random-forest partition is Theorem 1 of Meinshausen (2006) under the standard
assumptions.

sklearn API used (pinned idiom, sklearn 1.9.x): the ensemble-level
``RandomForestRegressor.apply(X) -> (n_samples, n_estimators)`` of int leaf
node ids, which internally dispatches to ``DecisionTreeRegressor.apply(X)``
per fitted tree in ``estimators_`` — the documented way to recover the
terminal node of every sample in every tree. It is called once at fit time on
the training design (storing the per-tree leaf id of every training sample)
and once per ``predict_quantiles`` call on the query design; the co-leaf
indicator matrix between queries and training samples is then a vectorized
leaf-id equality test per tree, averaged with the equal 1/T tree weight.

Output grids are made non-crossing by row-wise rearrangement
(Chernozhukov, Fernández-Val, Galichon, 2010, Ann. Statist. 38;
quant_fund.metrics.scoring.rearrange_quantiles). Honesty: only quantile,
pinball, and interval-width diagnostics are exposed (AGENTS.md honesty
contract — no Sharpe/Sortino/P&L content).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from quant_fund.metrics.scoring import rearrange_quantiles

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = ["QuantileRegressionForest"]

_MIN_TRAIN_N = 30


def _as_2d_finite(x: Array, name: str) -> Array:
    m = np.asarray(x, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"{name} must be a 2d design matrix")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must contain only finite values (NaN/inf rejected)")
    return m


def _as_1d_finite(y: Array, n: int) -> Array:
    v = np.asarray(y, dtype=float).reshape(-1)
    if v.size != n:
        raise ValueError(f"y must have length {n} (one response per design row)")
    if not np.all(np.isfinite(v)):
        raise ValueError("y must contain only finite values (NaN/inf rejected)")
    return v


def _check_levels(levels: Array) -> Array:
    lev = np.asarray(levels, dtype=float).reshape(-1)
    if lev.size == 0:
        raise ValueError("levels must be non-empty")
    if not np.all(np.isfinite(lev)) or not np.all((lev > 0.0) & (lev < 1.0)):
        raise ValueError("levels must lie strictly inside (0, 1)")
    if lev.size > 1 and np.any(np.diff(lev) <= 0.0):
        raise ValueError("levels must be strictly increasing")
    return lev


class QuantileRegressionForest:
    """Quantile regression forest: per-query conditional CDF from an RF.

    Parameters
    ----------
    n_estimators:
        Number of trees T in the underlying RandomForestRegressor.
    max_features:
        'sqrt' (default), 'log2', or None; forwarded to sklearn and
        validated fail-closed.
    min_samples_leaf:
        Minimum samples per terminal node; enforces the leaf empirical
        distributions used by the weighting above to be averages over at
        least this many responses.
    max_depth:
        None (grow to purity / min_samples_leaf) or a positive int.
    random_state:
        Seed or numpy Generator for the forest; None is allowed but the
        class is deterministic only for a fixed value.

    Notes
    -----
    ``fit`` requires n >= 30 training rows (fail-closed small-sample guard)
    and stores the (n, T) training leaf-id matrix from
    ``RandomForestRegressor.apply``. ``predict_quantiles(X, levels)`` returns
    an ``(m, len(levels))`` grid whose column j is the F̂^{-1}(levels_j | x)
    inverse-CDF quantile, equal-tree-weighted across trees and rearranged to
    be non-crossing. Calling ``predict_quantiles`` before ``fit`` raises
    RuntimeError.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str | int | float | None = "sqrt",
        min_samples_leaf: int = 5,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if (
            isinstance(min_samples_leaf, bool)
            or not isinstance(min_samples_leaf, int)
            or min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be a positive integer")
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
        ):
            raise ValueError("max_depth must be None or a positive integer")
        if isinstance(max_features, str) and max_features not in ("sqrt", "log2"):
            raise ValueError("max_features must be 'sqrt', 'log2', None, or a positive number")
        if max_features is None or isinstance(max_features, str):
            pass
        elif (
            isinstance(max_features, bool)
            or not isinstance(max_features, (int, float))
            or max_features <= 0
        ):
            raise ValueError("max_features must be 'sqrt', 'log2', None, or a positive number")
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._forest: RandomForestRegressor | None = None
        self._leaf_train: IntArray | None = None
        self._order: NDArray[np.intp] | None = None
        self._y_sorted: Array | None = None
        self._n_features: int | None = None

    def fit(self, X: Array, y: Array) -> QuantileRegressionForest:
        """Fit the underlying forest and store per-tree training leaf ids."""
        x = _as_2d_finite(X, "X")
        v = _as_1d_finite(y, x.shape[0])
        if x.shape[0] < _MIN_TRAIN_N:
            raise ValueError(
                f"QRF fit requires n >= {_MIN_TRAIN_N} training rows, got {x.shape[0]}"
            )
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        forest.fit(x, v)
        # sklearn idiom (pinned): ensemble-level apply returns (n, T) leaf
        # node ids, one column per fitted tree in forest.estimators_.
        self._leaf_train = np.asarray(forest.apply(x), dtype=np.int64)
        order = np.argsort(v)
        self._order = order.astype(np.intp)
        self._y_sorted = v[order]
        self._n_features = x.shape[1]
        self._forest = forest
        return self

    def predict_quantiles(self, X: Array, levels: Array) -> Array:
        """Non-crossing conditional quantile grid, shape (m, len(levels))."""
        if (
            self._forest is None
            or self._leaf_train is None
            or self._order is None
            or self._y_sorted is None
        ):
            raise RuntimeError("predict_quantiles called before fit")
        lev = _check_levels(levels)
        x = _as_2d_finite(X, "X")
        if x.shape[1] != self._n_features:
            raise ValueError(f"X has {x.shape[1]} features; fit used {self._n_features}")
        # Leaf id of every query in every tree, (m, T), same pinned idiom.
        leaf_query = np.asarray(self._forest.apply(x), dtype=np.int64)
        leaf_train = self._leaf_train
        y_sorted = self._y_sorted
        n = y_sorted.size
        # w_i(x) = (1/T) * #{trees t : leaf_query[x, t] == leaf_train[i, t]},
        # computed as the mean of per-tree co-leaf indicator matrices.
        weights = np.zeros((x.shape[0], n), dtype=float)
        for t in range(self.n_estimators):
            weights += leaf_query[:, t : t + 1] == leaf_train[:, t].reshape(1, -1)
        weights /= float(self.n_estimators)
        # Column order must align with ascending responses for the cumsum
        # below to be the CDF of the weighted empirical distribution.
        # Each row sums to the average co-leaf leaf size ((1/T) Σ_t
        # |leaf_t(x)|), NOT 1 — normalize against the row total, which is
        # exactly the denominator in F̂(y|x) = Σ_i w_i 1{y_i ≤ y} / Σ_i w_i.
        cum = np.cumsum(weights[:, self._order], axis=1)
        total = cum[:, -1]
        cols = []
        for tau in lev:
            # First index where the row-wise CDF reaches tau (rows sorted):
            # equivalent to searchsorted(cum_row, tau, side='left').
            idx = np.count_nonzero(cum < tau * total[:, None], axis=1)
            cols.append(y_sorted[np.minimum(idx, n - 1)])
        grid = np.stack(cols, axis=1)
        return rearrange_quantiles(grid)
