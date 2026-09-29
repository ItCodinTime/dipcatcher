"""Compatibility API for the canonical quantile-regression forest.

Historically this module and :mod:`quant_fund.models.qrf` contained separate
implementations of Meinshausen's forest weights. The implementations could
silently disagree because this module normalized raw co-leaf counts while the
canonical implementation gives every tree equal mass and normalizes within
each leaf. This module now preserves the older constructor contract while
delegating all fitting and distribution logic to ``models.qrf``.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.qrf import Array
from quant_fund.models.qrf import QuantileRegressionForest as _CanonicalQRF

__all__ = ["QuantileRegressionForest"]

_MIN_TRAIN_N = 30


def _check_levels(levels: Array) -> Array:
    values = np.asarray(levels, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("levels must be non-empty")
    if not np.all(np.isfinite(values)) or not np.all((values > 0.0) & (values < 1.0)):
        raise ValueError("levels must lie strictly inside (0, 1)")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("levels must be strictly increasing")
    return values


class QuantileRegressionForest(_CanonicalQRF):
    """Backward-compatible facade over the single canonical QRF engine.

    ``random_state`` maps to the canonical ``seed`` argument. The historical
    30-row small-sample guard and strict increasing-level validation remain in
    place for callers of this import path.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str | int | float | None = "sqrt",
        min_samples_leaf: int = 5,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int):
            raise ValueError("n_estimators must be a positive integer")
        if isinstance(min_samples_leaf, bool) or not isinstance(min_samples_leaf, int):
            raise ValueError("min_samples_leaf must be a positive integer")
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
        ):
            raise ValueError("max_depth must be None or a positive integer")
        if isinstance(max_features, str) and max_features not in {"sqrt", "log2"}:
            raise ValueError("max_features must be 'sqrt', 'log2', None, or a positive number")
        if (
            max_features is not None
            and not isinstance(max_features, str)
            and (
                isinstance(max_features, bool)
                or not isinstance(max_features, int | float)
                or max_features <= 0
            )
        ):
            raise ValueError("max_features must be 'sqrt', 'log2', None, or a positive number")
        super().__init__(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_depth=max_depth,
            seed=0 if random_state is None else random_state,
        )
        self.random_state = random_state

    def fit(self, X: Array, y: Array) -> QuantileRegressionForest:
        design = np.asarray(X, dtype=float)
        if design.ndim == 2 and design.shape[0] < _MIN_TRAIN_N:
            raise ValueError(
                f"QRF fit requires n >= {_MIN_TRAIN_N} training rows, got {design.shape[0]}"
            )
        super().fit(design, y)
        return self

    def predict_quantiles(self, X: Array, levels: Array) -> Array:
        if self._forest is None:
            raise RuntimeError("predict_quantiles called before fit")
        return super().predict_quantiles(X, _check_levels(levels))
