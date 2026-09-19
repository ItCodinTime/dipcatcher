"""Wave 24: Newey–West / DM / BH / PBO / moments edge fixtures (research diagnostics only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.inference import (
    benjamini_hochberg,
    diebold_mariano,
    newey_west_se,
    newey_west_variance,
)
from quant_fund.metrics.overfitting import (
    moments_from_returns,
    probability_of_backtest_overfitting,
)


def test_newey_west_empty_and_short_are_nan() -> None:
    assert math.isnan(newey_west_variance(np.array([])))
    assert math.isnan(newey_west_se(np.array([])))
    assert math.isnan(newey_west_se(np.array([1.0])))
    assert math.isnan(newey_west_se(np.array([1.0, 2.0])))
    # n=3 is the first finite path
    se3 = newey_west_se(np.array([1.0, 2.0, 3.0]), lags=0)
    assert math.isfinite(se3) and se3 > 0.0
    # Non-finite inputs dropped before length check
    assert math.isnan(newey_west_se(np.array([1.0, np.nan, 2.0])))


def test_diebold_mariano_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="align"):
        diebold_mariano(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
    # Short but equal → inconclusive (n < 3 after finite filter via mean_tstat)
    res = diebold_mariano(np.array([1.0, 2.0]), np.array([1.5, 2.5]), name_a="a", name_b="b")
    assert res.preferred == "inconclusive"
    assert math.isnan(res.statistic)
    assert math.isnan(res.p_value)
    assert res.n == 2


def test_benjamini_hochberg_empty_and_all_nan() -> None:
    reject, cutoff = benjamini_hochberg(np.array([]))
    assert reject.dtype == bool
    assert reject.size == 0
    assert cutoff == 0.0

    reject_nan, cutoff_nan = benjamini_hochberg(np.array([np.nan, np.nan, np.nan]))
    assert reject_nan.tolist() == [False, False, False]
    assert cutoff_nan == 0.0

    # All large p → no rejects
    reject_hi, cutoff_hi = benjamini_hochberg(np.array([0.5, 0.6, 0.9]), alpha=0.05)
    assert not reject_hi.any()
    assert cutoff_hi == 0.0

    with pytest.raises(ValueError, match="p_values"):
        benjamini_hochberg(np.array([0.01, 1.2]))
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg(np.array([0.01]), alpha=0.0)


def test_pbo_empty_mismatch_and_1d_are_nan() -> None:
    assert math.isnan(probability_of_backtest_overfitting(np.array([]), np.array([])))
    assert math.isnan(
        probability_of_backtest_overfitting(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    )
    # Shape mismatch
    assert math.isnan(
        probability_of_backtest_overfitting(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.0, 2.0]]),
        )
    )
    # Too few trials (need >=2 columns) or too few splits
    assert math.isnan(
        probability_of_backtest_overfitting(
            np.array([[1.0], [2.0]]),
            np.array([[0.5], [0.1]]),
        )
    )
    assert math.isnan(
        probability_of_backtest_overfitting(
            np.array([[1.0, 2.0]]),
            np.array([[0.5, 0.1]]),
        )
    )
    # All-nan IS rows → no flags → nan (research diagnostic; not a live Sharpe claim)
    assert math.isnan(
        probability_of_backtest_overfitting(
            np.full((3, 2), np.nan),
            np.ones((3, 2)),
        )
    )
    # Partial paths are refused rather than silently shrinking the PBO sample.
    partial = np.ones((3, 2))
    partial[1, 0] = np.nan
    assert math.isnan(probability_of_backtest_overfitting(partial, np.ones((3, 2))))


def test_moments_from_returns_short_and_degenerate() -> None:
    assert all(math.isnan(x) for x in moments_from_returns(np.array([])))
    assert all(math.isnan(x) for x in moments_from_returns(np.array([1.0, 2.0, 3.0])))
    sig, skew, kurt = moments_from_returns(np.array([1.0, 2.0, 3.0, 4.0]))
    assert math.isfinite(sig) and sig > 0.0
    assert math.isfinite(skew) and math.isfinite(kurt)
    # Constant series → (0, 0, 3) by convention
    assert moments_from_returns(np.array([1.0, 1.0, 1.0, 1.0])) == (0.0, 0.0, 3.0)
    # Non-finite dropped before length check
    assert all(math.isnan(x) for x in moments_from_returns(np.array([1.0, np.nan, 2.0, np.inf])))
