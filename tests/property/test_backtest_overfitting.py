"""Properties of the backtest-overfitting layer. Research diagnostics only."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.metrics.overfitting import (
    deflated_sharpe,
    probability_of_backtest_overfitting,
)
from quant_fund.validation.cpcv import combinatorial_purged_indices


@given(
    observed=st.floats(min_value=-1.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    length=st.integers(min_value=8, max_value=400),
    skew=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    kurtosis=st.floats(min_value=1.5, max_value=12.0, allow_nan=False, allow_infinity=False),
    n_trials=st.integers(min_value=1, max_value=40),
    var_score=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_deflated_probability_does_not_exceed_undeflated(
    observed: float,
    length: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    var_score: float,
) -> None:
    """DSR is PSR at a hurdle ≥ 0, so it cannot exceed PSR against 0."""
    from quant_fund.metrics.overfitting import probabilistic_sharpe

    psr = probabilistic_sharpe(observed, 0.0, length, skew, kurtosis)
    dsr = deflated_sharpe(observed, length, skew, kurtosis, n_trials, var_score)
    if math.isfinite(psr) and math.isfinite(dsr):
        assert 0.0 <= dsr <= psr + 1e-9
        assert 0.0 <= psr <= 1.0


@given(
    data=st.lists(
        st.lists(
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=4,
        ),
        min_size=2,
        max_size=6,
    )
)
@settings(max_examples=30, deadline=None)
def test_pbo_stays_inside_the_unit_interval(data: list[list[float]]) -> None:
    width = min(len(row) for row in data)
    matrix = np.asarray([row[:width] for row in data], dtype=float)
    pbo = probability_of_backtest_overfitting(matrix, matrix[::-1])
    if math.isfinite(pbo):
        assert 0.0 <= pbo <= 1.0


@given(
    n=st.integers(min_value=16, max_value=36),
    horizon=st.integers(min_value=0, max_value=3),
    embargo=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=25, deadline=None)
def test_purge_never_leaks_overlapping_labels(n: int, horizon: int, embargo: int) -> None:
    """A kept train index never shares a label window or an embargo bar with a test block."""
    folds = combinatorial_purged_indices(n, 4, 2, label_horizon=horizon, embargo=embargo)
    bounds = [int(i * n / 4) for i in range(5)]
    groups = [set(range(bounds[i], bounds[i + 1])) for i in range(4)]
    for train, test in folds:
        train_set = set(train.tolist())
        test_set = set(test.tolist())
        assert train_set.isdisjoint(test_set)
        test_groups = [group for group in groups if group & test_set]
        for index in train_set:
            for point in test_set:
                assert not (index < point <= index + horizon)
                assert index != point
            for group in test_groups:
                lo, hi = min(group), max(group)
                assert not (hi < index <= hi + embargo)
                assert not (lo - embargo <= index < lo)
