"""Canon tests: diff-in-diff, TWFE DiD, interrupted time series."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.causal_panel import (
    diff_in_diff,
    interrupted_time_series,
    twfe_did,
)


def test_diff_in_diff_known_answer() -> None:
    rng = np.random.default_rng(3)
    n = 400
    treated = np.repeat([1.0, 0.0], n // 2)
    post = np.tile([0.0, 1.0], n // 2)
    # E[y] = 1 + 2*post + 0.5*treated + 1.5*treated*post
    y = 1.0 + 2.0 * post + 0.5 * treated + 1.5 * treated * post + rng.normal(0, 0.1, n)
    out = diff_in_diff(y, treated, post)
    assert out["att"] == pytest.approx(1.5, abs=0.1)
    assert out["se"] > 0


def test_diff_in_diff_zero_effect() -> None:
    rng = np.random.default_rng(9)
    treated = rng.integers(0, 2, 500).astype(float)
    post = rng.integers(0, 2, 500).astype(float)
    y = rng.normal(0, 1, 500)
    out = diff_in_diff(y, treated, post)
    assert out["p_value"] > 0.01


def test_diff_in_diff_validation() -> None:
    y = np.ones(20)
    with pytest.raises(ValueError):
        diff_in_diff(y, np.ones(20), np.ones(20))  # binary check fails
    with pytest.raises(ValueError):
        diff_in_diff(y, np.array([0.5] * 20), np.zeros(20))
    with pytest.raises(ValueError):
        diff_in_diff(y, np.zeros(20), np.zeros(20))  # all same cell empty
    with pytest.raises(ValueError):
        diff_in_diff(np.array([np.nan] * 20), np.zeros(20), np.zeros(20))


def test_twfe_did_recovers_att() -> None:
    rng = np.random.default_rng(7)
    units = np.repeat(np.arange(20), 10)
    times = np.tile(np.arange(10), 20)
    treated_post = ((units < 10) & (times >= 5)).astype(float)
    unit_fe = rng.normal(0, 1, 20)[units]
    time_fe = rng.normal(0, 0.3, 10)[times]
    y = unit_fe + time_fe + 2.0 * treated_post + rng.normal(0, 0.1, 200)
    out = twfe_did(y, units, times, treated_post)
    assert out["att"] == pytest.approx(2.0, abs=0.15)
    assert out["se"] > 0


def test_interrupted_time_series_breaks() -> None:
    rng = np.random.default_rng(11)
    t_len, tau = 120, 70
    t = np.arange(t_len, dtype=float)
    y = 0.05 * t + 3.0 * (t >= tau) + 0.4 * np.maximum(t - tau, 0) + rng.normal(0, 0.2, t_len)
    out = interrupted_time_series(y, tau)
    assert out["level_change"] == pytest.approx(3.0, abs=0.5)
    assert out["slope_change"] == pytest.approx(0.4, abs=0.2)
    assert out["fitted"].shape == (t_len,)


def test_interrupted_time_series_no_break() -> None:
    rng = np.random.default_rng(13)
    t = np.arange(100, dtype=float)
    y = 1.0 + 0.1 * t + rng.normal(0, 0.3, 100)
    out = interrupted_time_series(y, 50)
    assert abs(out["level_change_t"]) < 5.0


def test_its_validation() -> None:
    y = np.arange(20.0)
    with pytest.raises(ValueError):
        interrupted_time_series(np.arange(8.0), 4)
    with pytest.raises(ValueError):
        interrupted_time_series(y, 1)
    with pytest.raises(ValueError):
        interrupted_time_series(y, 18)
    with pytest.raises(ValueError):
        interrupted_time_series(np.full(30, np.nan), 15)


def test_twfe_validation() -> None:
    with pytest.raises(ValueError):
        twfe_did(np.ones(10), np.arange(10), np.arange(10), np.zeros(10))
    with pytest.raises(ValueError):
        twfe_did(np.full(10, np.nan), np.arange(10), np.arange(10), np.zeros(10))
