"""DayWave6: PBO residual honesty — known fraction + shape edges (research-diagnostic only).

Invalid / thin inputs must return NaN (never silent 0.0). Valid matrices may
legitimately yield PBO ∈ {0, fraction, 1}. Not a live Sharpe / P&L claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.overfitting import probability_of_backtest_overfitting


def test_pbo_known_fraction_half() -> None:
    """Hand-checkable CSCV PBO = 2/4.

    Four splits, two trials. IS-best is always trial 0 (higher IS Sharpe).
    OOS: splits 0–1 place trial 0 strictly below the OOS median → flag 1;
    splits 2–3 place trial 0 strictly above the OOS median → flag 0.
    PBO = mean(flags) = 0.5.
    """
    is_s = np.array(
        [
            [2.0, 1.0],
            [2.0, 1.0],
            [2.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    oos = np.array(
        [
            [0.0, 1.0],  # 0.0 < median(0.0, 1.0)=0.5
            [0.1, 0.9],  # 0.1 < 0.5
            [0.8, 0.2],  # 0.8 > 0.5
            [0.7, 0.3],  # 0.7 > 0.5
        ],
        dtype=float,
    )
    pbo = probability_of_backtest_overfitting(is_s, oos)
    assert pbo == pytest.approx(0.5, rel=0, abs=0.0)


def test_pbo_known_zero_and_one_valid() -> None:
    """Valid matrices may return exact 0.0 or 1.0 — not an invalid-path artifact."""
    # Always overfit: IS-best (col 0) always OOS-below-median
    is_bad = np.array([[3.0, 0.0], [4.0, 1.0], [2.5, 0.5], [5.0, 0.1]], dtype=float)
    oos_bad = np.array([[-1.0, 1.0], [-0.5, 0.8], [-2.0, 0.4], [-0.2, 0.3]], dtype=float)
    assert probability_of_backtest_overfitting(is_bad, oos_bad) == 1.0

    # Never overfit: IS-best always OOS-above-median
    is_ok = np.array([[3.0, 0.0], [4.0, 1.0], [2.5, 0.5], [5.0, 0.1]], dtype=float)
    oos_ok = np.array([[1.0, -1.0], [0.8, -0.5], [0.4, -2.0], [0.3, -0.2]], dtype=float)
    assert probability_of_backtest_overfitting(is_ok, oos_ok) == 0.0


def test_pbo_invalid_shapes_are_nan_not_zero() -> None:
    """1×N / N×1 / single split / mismatch / 3D / empty → NaN (never silent 0.0)."""
    cases = [
        # empty
        (np.array([]), np.array([])),
        # 1-D
        (np.array([1.0, 2.0]), np.array([1.0, 2.0])),
        # single split (1×N)
        (np.array([[1.0, 2.0, 3.0]]), np.array([[0.1, 0.2, 0.3]])),
        # N×1 insufficient trials
        (np.array([[1.0], [2.0], [3.0]]), np.array([[0.1], [0.2], [0.3]])),
        # (2,1) too few trials
        (np.array([[1.0], [2.0]]), np.array([[0.5], [0.1]])),
        # mismatched shapes
        (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0]])),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        # 3-D
        (np.ones((2, 2, 2)), np.ones((2, 2, 2))),
        # zero-width / zero-height 2-D
        (np.empty((0, 3)), np.empty((0, 3))),
        (np.empty((4, 0)), np.empty((4, 0))),
        (np.empty((0, 0)), np.empty((0, 0))),
    ]
    for is_s, oos in cases:
        pbo = probability_of_backtest_overfitting(is_s, oos)
        assert math.isnan(pbo), f"expected NaN for shapes {getattr(is_s, 'shape', None)}"
        # Guard against accidental float(False)/mean([])→0 regressions
        assert pbo != 0.0


def test_pbo_all_nan_and_partial_nonfinite_are_nan_not_zero() -> None:
    """All-NaN matrix, all-NaN row, or any non-finite cell → refuse whole matrix."""
    ones = np.ones((3, 2))
    assert math.isnan(probability_of_backtest_overfitting(np.full((3, 2), np.nan), ones))
    assert math.isnan(probability_of_backtest_overfitting(ones, np.full((3, 2), np.nan)))

    partial_is = ones.copy()
    partial_is[1, :] = np.nan  # all-NaN row
    assert math.isnan(probability_of_backtest_overfitting(partial_is, ones))

    partial_cell = ones.copy()
    partial_cell[0, 1] = np.nan
    assert math.isnan(probability_of_backtest_overfitting(partial_cell, ones))

    with_inf = ones.copy()
    with_inf[2, 0] = np.inf
    assert math.isnan(probability_of_backtest_overfitting(ones, with_inf))

    for is_s, oos in (
        (np.full((3, 2), np.nan), ones),
        (partial_is, ones),
        (partial_cell, ones),
        (ones, with_inf),
    ):
        pbo = probability_of_backtest_overfitting(is_s, oos)
        assert math.isnan(pbo) and pbo != 0.0


def test_pbo_oos_tie_at_median_is_not_overfit_flag() -> None:
    """Strict < median: OOS-best equal to median does not count as overfitting."""
    # Two trials; OOS values equal → median = that value; best OOS == med → flag 0
    is_s = np.array([[2.0, 1.0], [3.0, 0.5]], dtype=float)
    oos = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
    assert probability_of_backtest_overfitting(is_s, oos) == 0.0
