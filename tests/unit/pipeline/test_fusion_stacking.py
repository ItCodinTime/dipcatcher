"""Leakage and finite-data tests for the research-only fusion stacker."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.fusion.engine import cross_fitted_ridge_stack


def _fixture() -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    x = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0]])
    y = 2.0 * x[:, 0] - x[:, 1] + 0.5
    folds = [
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4])),
    ]
    return x, y, folds


def test_cross_fitted_stack_preserves_warmup_and_returns_finite_fit() -> None:
    x, y, folds = _fixture()
    result = cross_fitted_ridge_stack(x, y, folds, alpha=0.1)

    assert result.oof_predictions.shape == (5,)
    assert np.isnan(result.oof_predictions[:2]).all()
    assert np.isfinite(result.oof_predictions[2:]).all()
    assert np.array_equal(result.fold_ids, np.array([-1, -1, 0, 0, 1]))
    assert np.isfinite(result.weights).all()
    assert np.isfinite(result.intercept)
    assert result.alpha == 0.1


@pytest.mark.parametrize(
    ("folds", "message"),
    [
        ([(np.array([0, 1]), np.array([1, 2]))], "overlap"),
        (
            [
                (np.array([0, 1]), np.array([2, 3])),
                (np.array([0, 1, 2]), np.array([3, 4])),
            ],
            "exactly once",
        ),
        ([(np.array([0, 1]), np.array([8]))], "out of bounds"),
    ],
)
def test_cross_fitted_stack_rejects_unsafe_folds(
    folds: list[tuple[np.ndarray, np.ndarray]], message: str
) -> None:
    x, y, _ = _fixture()
    with pytest.raises(ValueError, match=message):
        cross_fitted_ridge_stack(x, y, folds)


def test_cross_fitted_stack_rejects_nonfinite_inputs_and_unscored_data() -> None:
    x, y, folds = _fixture()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cross_fitted_ridge_stack(x, y, folds)

    with pytest.raises(ValueError, match="each stacker fold"):
        cross_fitted_ridge_stack(
            np.array([[1.0], [2.0], [3.0]]),
            np.array([1.0, 2.0, 3.0]),
            [(np.array([0]), np.array([1, 2]))],
        )
