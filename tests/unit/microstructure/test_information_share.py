"""Tests for models/information_share.py — Hasbrouck IS, Gonzalo-Granger."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.information_share import gonzalo_granger, hasbrouck_is


def _leader_follower(n: int = 800, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = 100.0 + np.cumsum(rng.standard_normal(n))  # common random walk
    p_a = w + 0.05 * rng.standard_normal(n)  # immediate incorporation
    p_b = np.concatenate([[w[0]], w[:-1]]) + 0.05 * rng.standard_normal(n)  # lags
    return p_a, p_b


def test_leader_gets_higher_is() -> None:
    a, b = _leader_follower()
    out = hasbrouck_is(a, b)
    lo_a, hi_a = out["is_a"]
    lo_b, hi_b = out["is_b"]
    # A leads: its midpoint share should dominate B's
    assert (lo_a + hi_a) / 2 > (lo_b + hi_b) / 2
    # midpoints of the two ordering bounds sum to 1
    assert (lo_a + hi_a + lo_b + hi_b) / 2 == pytest.approx(1.0, abs=1e-9)


def test_bounds_ordered() -> None:
    a, b = _leader_follower(seed=11)
    out = hasbrouck_is(a, b)
    assert out["is_a"][0] <= out["is_a"][1]
    assert out["is_b"][0] <= out["is_b"][1]


def test_gg_weights_sum_to_one() -> None:
    a, b = _leader_follower(seed=21)
    out = gonzalo_granger(a, b)
    assert out["w_a"] + out["w_b"] == pytest.approx(1.0)
    # GG weights are a linear (not convex) combination — just finiteness
    assert np.isfinite(out["w_a"]) and np.isfinite(out["w_b"])


def test_symmetric_markets_split() -> None:
    rng = np.random.default_rng(3)
    n = 1000
    w = 100.0 + np.cumsum(rng.standard_normal(n))
    a = w + 0.1 * rng.standard_normal(n)
    b = w + 0.1 * rng.standard_normal(n)
    out = hasbrouck_is(a, b)
    mid_a = float(out["is_a"].mean())
    assert 0.2 < mid_a < 0.8  # no systematic leader -> roughly balanced


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        hasbrouck_is(np.arange(10.0), np.arange(10.0))
    with pytest.raises(ValueError):
        hasbrouck_is(np.full(50, np.nan), np.arange(50.0))
    with pytest.raises(ValueError):
        gonzalo_granger(np.arange(10.0), np.arange(10.0))
    # constant series -> degenerate design
    with pytest.raises(ValueError):
        hasbrouck_is(np.full(100, 50.0), np.full(100, 51.0))
