"""Quantile-bandit internals: date keys, Cholesky fallback, refit guards, 1-D select.

Complements test_quantile_bandit_extremes (public edges) with the private-helper
branches and posterior guards that previously had no coverage.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from quant_fund.models.quantile_bandit import (
    QuantileThompson,
    _draw_gaussian,
    _ordered_groups,
    _sort_value,
    _topk_indices,
)


def test_sort_value_handles_numeric_and_datetime_inputs() -> None:
    assert _sort_value(np.datetime64("2024-01-02")) < _sort_value(np.datetime64("2024-01-03"))
    assert _sort_value(5) < _sort_value(6)
    assert _sort_value(5.5) < _sort_value(6.5)
    assert _sort_value(float("nan"))[0] == 1
    assert _sort_value(datetime(2024, 1, 2, tzinfo=UTC))[0] == 0


def test_sort_value_falls_back_to_text_when_timestamp_raises() -> None:
    class _BrokenStamp:
        def timestamp(self) -> float:
            raise OSError("no clock")

        def __str__(self) -> str:
            return "123.5"

    assert _sort_value(_BrokenStamp()) == (0, 123.5)


def test_sort_value_non_numeric_text_sorts_last() -> None:
    class _Text:
        def __str__(self) -> str:
            return "not-a-number"

    ordered = sorted([_Text(), 1.0], key=_sort_value)
    assert ordered[0] == 1.0  # numeric keys sort before non-numeric ones


def test_ordered_groups_preserves_first_seen_order_key() -> None:
    keys, order = _ordered_groups([3.0, 1.0, 3.0, 2.0])
    assert keys[0] == "3.0"
    assert order == ["1.0", "2.0", "3.0"]


def test_draw_gaussian_falls_back_to_mean_on_indefinite_precision() -> None:
    prec = np.array([[0.0, 1.0], [1.0, 0.0]])  # not positive definite, jitter cannot fix
    mean = np.array([0.5, -0.5])
    draw = _draw_gaussian(mean, prec, np.random.default_rng(0))
    assert np.allclose(draw, mean)


def test_topk_indices_clips_and_empty() -> None:
    assert _topk_indices(np.array([]), 3).size == 0
    idx = _topk_indices(np.array([0.1, 0.9, 0.5]), 2)
    assert set(idx.tolist()) == {1, 2}


def test_mean_theta_requires_initialized_posterior() -> None:
    model = QuantileThompson(seed=1)
    with pytest.raises(RuntimeError, match="no initialized posterior"):
        model._mean_theta()


def test_refit_requires_dimension_and_empty_history_is_noop() -> None:
    model = QuantileThompson(seed=1)
    model._dirty = True
    with pytest.raises(RuntimeError, match="no initialized feature dimension"):
        model._refit()

    model = QuantileThompson(seed=1)
    model._ensure_dim(2)
    model._dirty = True
    model._refit()  # no rows yet: leaves beta/state untouched
    assert model._beta is not None  # initialized zeros from _ensure_dim


def test_select_accepts_single_context_vector() -> None:
    model = QuantileThompson(seed=2)
    chosen = model.select(np.array([1.0, 2.0, 3.0]), k=1)
    assert chosen.shape == (1,)
    assert 0 <= int(chosen[0]) < 3


def test_select_without_fitted_state_fails_closed() -> None:
    model = QuantileThompson(seed=2)
    model._ensure_dim(2)
    model._beta = None
    model._prec = None
    with pytest.raises(RuntimeError, match="refit produced incomplete state"):
        model.select(np.array([[1.0, 0.0]]), k=1)


def test_update_requires_initialized_posterior() -> None:
    model = QuantileThompson(seed=2)
    model._ensure_dim(2)
    model._a = None
    with pytest.raises(RuntimeError, match="no initialized posterior"):
        model.update(np.array([1.0, 0.0]), reward=1.0)


def test_run_panel_skips_dates_with_too_few_finite_rows() -> None:
    model = QuantileThompson(seed=3)
    x = np.array([[1.0], [2.0], [np.nan], [3.0]])
    y = np.array([0.1, 0.2, 0.3, np.nan])
    dates = np.array([1.0, 1.0, 1.0, 1.0])
    trace = model.run_panel(x, y, dates, k=2)
    assert trace.dates == []  # 1 finite row < max(2k, 4)
    assert trace.policy_reward.size == 0


def test_run_panel_oracle_column_changes_regret_reference() -> None:
    rng = np.random.default_rng(7)
    n_dates, n_names = 12, 10
    x = rng.normal(size=(n_dates * n_names, 3))
    y = x[:, 0] + rng.normal(0, 0.1, size=n_dates * n_names)
    dates = np.repeat(np.arange(n_dates, dtype=float), n_names)
    oracle = x[:, 0]
    model = QuantileThompson(seed=4)
    trace = model.run_panel(x, y, dates, k=3, oracle=oracle)
    assert len(trace.dates) > 0
    assert trace.oracle_reward.shape == trace.policy_reward.shape
    assert trace.cumulative_regret.shape == trace.policy_reward.shape
