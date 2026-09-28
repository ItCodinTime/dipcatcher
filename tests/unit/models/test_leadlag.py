"""Canon tests: lead-lag analysis + Hayashi-Yoshida."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.leadlag import (
    cross_correlation,
    hayashi_yoshida,
    lead_lag_adjacency,
    lead_lag_matrix,
)


def test_cross_correlation_self_peak_at_zero() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200)
    cc = cross_correlation(x, x, 10)
    assert cc[10] == pytest.approx(1.0, abs=1e-12)
    assert np.abs(cc).max() == pytest.approx(1.0)


def test_cross_correlation_delayed() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 400)
    y = np.roll(x, 3)  # y lags x by 3 -> corr(x_t, y_{t+3})... y_t = x_{t-3}
    cc = cross_correlation(x, y, 8)
    # corr(x_t, y_{t+ell}) = corr(x_t, x_{t+ell-3}) peaks at ell=3
    assert np.argmax(cc) == 8 + 3


def test_lead_lag_detects_leader() -> None:
    rng = np.random.default_rng(7)
    t_len = 600
    leader = rng.normal(0, 1, t_len)
    follower = np.roll(leader, 2) * 0.8 + rng.normal(0, 0.4, t_len)
    panel = np.column_stack([leader, follower])
    out = lead_lag_matrix(panel, max_lag=6)
    assert out["scores"][0, 1] > 0.2  # series 0 leads 1
    assert out["scores"][1, 0] < -0.2
    assert out["peak_lag"][0, 1] == 2


def test_lead_lag_adjacency_shape() -> None:
    rng = np.random.default_rng(9)
    panel = rng.normal(0, 1, (400, 4))
    adj = lead_lag_adjacency(panel, max_lag=4, quantile=0.95)
    assert adj.shape == (4, 4)
    assert set(np.unique(adj)) <= {0, 1}
    assert np.all(np.diag(adj) == 0)


def test_hayashi_yoshida_synchronous_matches() -> None:
    rng = np.random.default_rng(11)
    t = np.arange(1.0, 101.0)
    x = np.cumsum(rng.normal(0, 1, 100))
    y = x + rng.normal(0, 0.01, 100)
    out = hayashi_yoshida(t, x, t, y)
    assert out["corr"] == pytest.approx(1.0, abs=0.05)


def test_hayashi_yoshida_async_subsampled() -> None:
    rng = np.random.default_rng(13)
    t_full = np.arange(1.0, 201.0)
    x = np.cumsum(rng.normal(0, 1, 200))
    # observe x every step, y every other step — overlap pairs still carry
    # most of the covariance
    t_y = t_full[::2]
    y = x[::2]
    out = hayashi_yoshida(t_full, x, t_y, y)
    assert out["cov"] > 0
    assert out["n_overlap"] >= 99


def test_hayashi_yoshida_no_overlap() -> None:
    t1 = np.arange(0.0, 5.0)
    t2 = np.arange(100.0, 105.0)
    out = hayashi_yoshida(t1, np.arange(5.0), t2, np.arange(5.0))
    assert out["n_overlap"] == 0
    assert out["cov"] == 0.0


def test_leadlag_validation() -> None:
    x = np.arange(50.0)
    with pytest.raises(ValueError):
        cross_correlation(x, x, 40)  # max_lag too big
    with pytest.raises(ValueError):
        cross_correlation(np.zeros(50), np.zeros(50), 5)  # degenerate
    with pytest.raises(ValueError):
        lead_lag_matrix(np.ones((20, 4)), 3)  # T < 30
    with pytest.raises(ValueError):
        hayashi_yoshida(np.array([2.0, 1.0, 0.5]), np.ones(3), np.arange(3.0), np.ones(3))
    with pytest.raises(ValueError):
        hayashi_yoshida(np.arange(3.0), np.ones(4), np.arange(3.0), np.ones(3))
