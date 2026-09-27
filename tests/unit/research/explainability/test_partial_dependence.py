"""Partial-dependence curves on synthetic heads (SYNTHETIC labels)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.explainability import (
    partial_dependence_1d,
    partial_dependence_top_k,
)

from .conftest import FEATURES


def _linear_predict(coef):
    w = np.asarray(coef, dtype=float)

    def predict(x):
        return np.asarray(x, dtype=float) @ w

    return predict


def test_pd_recovers_linear_slope() -> None:
    """PD_j(v) of an additive linear model is affine in v with slope w_j."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    predict = _linear_predict([1.5, -2.0, 0.0])
    curve = partial_dependence_1d(predict, x, 1, "w_neg", grid_points=9)
    grid = np.asarray(curve.grid)
    vals = np.asarray(curve.mean_curve)
    # Exact for a linear model: PD(v) = w_j * v + const. Slope must match.
    slope, _intercept = np.polyfit(grid, vals, 1)
    assert slope == pytest.approx(-2.0, abs=1e-10)
    # Monotone decreasing for a negative coefficient.
    assert np.all(np.diff(vals) <= 0.0)


def test_pd_is_deterministic_and_grid_is_quantile() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 2))
    predict = _linear_predict([1.0, 0.5])
    first = partial_dependence_1d(predict, x, 0, "a", grid_points=11)
    second = partial_dependence_1d(predict, x, 0, "a", grid_points=11)
    assert first == second
    assert len(first.grid) == 11
    assert len(first.values) == len(first.grid)
    assert first.n_rows == 300
    # Quantile grid clips to the empirical range.
    assert min(first.grid) >= float(np.min(x[:, 0])) - 1e-9
    assert max(first.grid) <= float(np.max(x[:, 0])) + 1e-9


def test_pd_constant_feature_single_point() -> None:
    x = np.column_stack([np.ones(50), np.linspace(-1, 1, 50)])
    predict = _linear_predict([2.0, 1.0])
    curve = partial_dependence_1d(predict, x, 0, "const", grid_points=7)
    assert curve.grid == (1.0,)
    assert len(curve.values) == 1


def test_pd_top_k_preserves_order_and_names() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(120, len(FEATURES)))
    predict = _linear_predict(np.linspace(1.0, 0.1, len(FEATURES)))
    curves = partial_dependence_top_k(
        predict, x, FEATURES, ["signal_core", "mom_20"], grid_points=5
    )
    assert [c.feature for c in curves] == ["signal_core", "mom_20"]
    with pytest.raises(ValueError, match="unknown feature"):
        partial_dependence_top_k(predict, x, FEATURES, ["not_a_feature"])


def test_pd_input_validation() -> None:
    predict = _linear_predict([1.0])
    x = np.ones((10, 1))
    with pytest.raises(ValueError, match="out of range"):
        partial_dependence_1d(predict, x, 3)
    with pytest.raises(ValueError, match="grid_points"):
        partial_dependence_1d(predict, x, 0, grid_points=1)
    with pytest.raises(ValueError, match="quantile_clip"):
        partial_dependence_1d(predict, x, 0, quantile_clip=(0.9, 0.1))
    with pytest.raises(ValueError, match="finite"):
        partial_dependence_1d(predict, x, 0, grid=np.asarray([0.0, np.inf]))
    with pytest.raises(ValueError, match="non-empty"):
        partial_dependence_1d(predict, np.empty((0, 1)), 0)
