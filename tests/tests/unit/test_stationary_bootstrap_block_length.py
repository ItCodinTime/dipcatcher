"""Stationary bootstrap (Politis–Romano) and Politis–White block length.

Research-diagnostic machinery only — never a live Sharpe / P&L claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.inference import optimal_block_length, stationary_bootstrap_indices


def test_stationary_bootstrap_shape_range_and_determinism() -> None:
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(120, 40, 6.0, rng)
    assert idx.shape == (40, 120)
    assert idx.dtype == np.intp
    assert int(idx.min()) >= 0 and int(idx.max()) < 120
    again = stationary_bootstrap_indices(120, 40, 6.0, np.random.default_rng(0))
    assert np.array_equal(idx, again)


def test_stationary_bootstrap_block_structure() -> None:
    """Continuation frequency matches the geometric block mean (1 - 1/L)."""
    rng = np.random.default_rng(1)
    n, mean_block = 2000, 20.0
    idx = stationary_bootstrap_indices(n, 200, mean_block, rng)
    continuation = np.mean(idx[:, 1:] == (idx[:, :-1] + 1) % n)
    assert abs(float(continuation) - (1.0 - 1.0 / mean_block)) < 0.01
    # mean_block = 1 → i.i.d. draws: continuation ≈ 1/n
    idx1 = stationary_bootstrap_indices(500, 200, 1.0, np.random.default_rng(2))
    cont1 = np.mean(idx1[:, 1:] == (idx1[:, :-1] + 1) % 500)
    assert float(cont1) < 0.01


def test_stationary_bootstrap_fail_closed() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="non-negative"):
        stationary_bootstrap_indices(-1, 10, 5.0, rng)
    with pytest.raises(ValueError, match="n_boot"):
        stationary_bootstrap_indices(50, 0, 5.0, rng)
    with pytest.raises(ValueError, match="finite"):
        stationary_bootstrap_indices(50, 10, float("nan"), rng)
    with pytest.raises(ValueError, match=r"\[1, n\]"):
        stationary_bootstrap_indices(50, 10, 0.5, rng)
    with pytest.raises(ValueError, match=r"\[1, n\]"):
        stationary_bootstrap_indices(50, 10, 51.0, rng)
    assert stationary_bootstrap_indices(0, 3, 1.0, rng).shape == (3, 0)


def test_optimal_block_length_white_noise_small() -> None:
    rng = np.random.default_rng(3)
    b = optimal_block_length(rng.normal(size=4000))
    assert np.isfinite(b)
    assert 1.0 <= b <= 4.0


def test_optimal_block_length_increases_with_dependence() -> None:
    rng = np.random.default_rng(4)
    n = 4000
    e = rng.normal(size=n)
    weak = np.zeros(n)
    strong = np.zeros(n)
    for t in range(1, n):
        weak[t] = 0.2 * weak[t - 1] + e[t]
        strong[t] = 0.85 * strong[t - 1] + e[t]
    b_weak = optimal_block_length(weak)
    b_strong = optimal_block_length(strong)
    assert np.isfinite(b_weak) and np.isfinite(b_strong)
    assert b_strong > b_weak
    assert b_strong > 10.0


def test_optimal_block_length_edges() -> None:
    assert optimal_block_length(np.ones(500)) == 1.0  # constant → no dependence
    assert np.isnan(optimal_block_length(np.arange(5, dtype=float)))  # n < 10
    assert np.isnan(optimal_block_length(np.array([])))
    # Non-finite values are dropped before the estimate
    mixed = np.concatenate([np.full(5, np.nan), np.random.default_rng(5).normal(size=500)])
    assert np.isfinite(optimal_block_length(mixed))
