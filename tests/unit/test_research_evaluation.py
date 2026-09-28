import numpy as np
import pytest

from quant_fund.metrics.research_evaluation import (
    compare_research_paths,
    serial_adjusted_sharpe,
    walk_forward_allocations,
)


def test_serial_adjustment_ar1_population_limit():
    rng = np.random.default_rng(71)
    r = np.empty(50000)
    r[0] = 0
    for t in range(1, len(r)):
        r[t] = 0.7 * r[t - 1] + rng.normal()
    result = serial_adjusted_sharpe(r + 0.2, periods=252, max_lag=30)
    expected = 1 + 2 * sum((1 - k / 252) * 0.7**k for k in range(1, 31))
    assert result["variance_inflation"] == pytest.approx(expected, rel=0.1)
    assert result["serial_adjusted"] < result["iid"]


def test_serial_no_lags_and_invalid():
    result = serial_adjusted_sharpe(np.array([1.0, 2.0, 3.0, 4.0]), max_lag=0)
    assert result["iid"] == result["serial_adjusted"]
    with pytest.raises(ValueError):
        serial_adjusted_sharpe(np.ones(30))
    with pytest.raises(ValueError):
        serial_adjusted_sharpe(np.array([0.0, 1.0, np.nan]), max_lag=0)


def test_self_financing_entry_rotation_and_weight_drift():
    # Enter 100% asset 0 then rotate to asset 1; fee = 10% one-way.
    calls = 0

    def allocator(history, previous):
        nonlocal calls
        calls += 1
        return np.array([1.0, 0.0]) if calls == 1 else np.array([0.0, 1.0])

    path = walk_forward_allocations(np.zeros((4, 2)), allocator, lookback=2, fee_rate=0.1)
    assert path.cost_fraction[0] == pytest.approx(0.1 / 1.1)
    assert path.cost_fraction[1] == pytest.approx(0.2 / 1.1)
    np.testing.assert_allclose(path.net_returns, -path.cost_fraction)


def test_causality_and_callback_cannot_mutate_input():
    r = np.arange(40, dtype=float).reshape(20, 2) / 1000
    orig = r.copy()

    def allocator(history, previous):
        w = np.array([0.5, 0.0]) if history[-1, 0] > 0.01 else np.array([0.0, 0.5])
        history[:] = -1
        previous[:] = 0
        return w

    p = walk_forward_allocations(r, allocator, lookback=3, fee_rate=0.001)
    r2 = r.copy()
    r2[10:] = 0.9
    p2 = walk_forward_allocations(r2, allocator, lookback=3, fee_rate=0.001)
    np.testing.assert_array_equal(r, orig)
    np.testing.assert_array_equal(p.weights[:8], p2.weights[:8])
    np.testing.assert_array_equal(p.net_returns[:7], p2.net_returns[:7])


def test_drift_avoids_spurious_turnover():
    seen = []

    def allocator(h, prev):
        seen.append(prev.copy())
        return np.array([0.5, 0.0]) if len(seen) == 1 else prev

    r = np.zeros((4, 2))
    r[2, 0] = 0.2
    p = walk_forward_allocations(r, allocator, lookback=2, fee_rate=0)
    assert seen[1][0] == pytest.approx(0.6 / 1.1)
    assert p.turnover[1] == pytest.approx(0)


def test_comparison_retains_trials_and_never_promotes():
    rng = np.random.default_rng(22)
    base = rng.normal(0, 0.01, 100)
    candidates = np.column_stack(
        [base + rng.normal(0, 0.002, 100), base + rng.normal(0.001, 0.002, 100)]
    )
    report = compare_research_paths(base, candidates, n_boot=99, seed=7)
    assert report["n_trials_supplied"] == 2
    assert report["promotable"] is False
    assert 0 <= report["reality_check"]["p_value"] <= 1
    assert report == compare_research_paths(base, candidates, n_boot=99, seed=7)
    with pytest.raises(ValueError):
        compare_research_paths(base[:-1], candidates)
