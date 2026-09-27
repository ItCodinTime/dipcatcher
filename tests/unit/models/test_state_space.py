"""State-space battery: Kalman recovery, OU MLE, dynamic hedge ratio."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.state_space import (
    kalman_filter,
    kalman_hedge_ratio,
    kalman_smoother,
    local_level_mle,
    local_linear_trend_mle,
    ou_expected_excursion,
    ou_mle,
)


def _local_level(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = np.cumsum(rng.normal(0.0, 0.05, size=n))
    y = mu + rng.normal(0.0, 0.5, size=n)
    return mu, y


def _ou(n: int = 2000, theta: float = 0.3, mu: float = 5.0, sigma: float = 0.2, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    return x


def test_kalman_filter_recovers_level() -> None:
    mu, y = _local_level()
    out = kalman_filter(
        y.reshape(-1, 1),
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[0.05**2]]),
        np.array([[0.5**2]]),
    )
    est = out["x_filt"][:, 0]
    assert np.mean(np.abs(est - mu)) < np.mean(np.abs(y - mu))
    assert np.isfinite(out["loglik"][0])


def test_kalman_smoother_improves() -> None:
    mu, y = _local_level(seed=2)
    filt = kalman_filter(
        y.reshape(-1, 1),
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[0.05**2]]),
        np.array([[0.5**2]]),
    )
    sm = kalman_smoother(filt, np.array([[1.0]]))
    err_f = np.mean(np.abs(filt["x_filt"][:, 0] - mu))
    err_s = np.mean(np.abs(sm["x_smooth"][:, 0] - mu))
    assert err_s <= err_f + 1e-9


def test_local_level_mle_variance_recovery() -> None:
    _, y = _local_level(seed=3)
    fit = local_level_mle(y)
    assert fit["sigma_obs"] == pytest.approx(0.5, rel=0.5)
    assert fit["converged"] == 1.0


def test_local_linear_trend_recovers_slope() -> None:
    rng = np.random.default_rng(4)
    trend = np.cumsum(0.1 + 0.01 * rng.standard_normal(300))
    y = trend + rng.normal(0.0, 0.3, size=300)
    fit = local_linear_trend_mle(y)
    assert fit["slope_final"] == pytest.approx(0.1, abs=0.1)


def test_kalman_hedge_ratio_tracks_beta() -> None:
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.standard_normal(400))
    beta_path = np.linspace(0.5, 1.5, 400)
    y = 0.3 + beta_path * x + 0.3 * rng.standard_normal(400)
    out = kalman_hedge_ratio(x, y, delta=1e-4)
    # Filtered beta should track the drifting true beta.
    assert np.corrcoef(out["beta"][100:], beta_path[100:])[0, 1] > 0.5
    assert abs(out["beta"][-1] - 1.5) < 0.5


def test_ou_mle_recovery() -> None:
    x = _ou(theta=0.3, mu=5.0, sigma=0.2)
    fit = ou_mle(x)
    assert fit["theta"] == pytest.approx(0.3, rel=0.25)
    assert fit["mu"] == pytest.approx(5.0, abs=0.05)
    assert fit["half_life"] == pytest.approx(np.log(2) / 0.3, rel=0.3)


def test_ou_half_life_scaling() -> None:
    fast = ou_mle(_ou(theta=1.0, seed=6))
    slow = ou_mle(_ou(theta=0.1, seed=6))
    assert fast["half_life"] < slow["half_life"]


def test_ou_expected_excursion_monotone() -> None:
    x = _ou(theta=0.5, mu=0.0, sigma=1.0)
    e1 = ou_expected_excursion(x, 1.0)
    e2 = ou_expected_excursion(x, 2.0)
    assert 0.0 < e1 < e2


def test_state_space_fail_closed() -> None:
    with pytest.raises(ValueError):
        kalman_filter(np.ones(1), np.eye(1), np.eye(1), np.eye(1), np.eye(1))  # T < 2
    with pytest.raises(ValueError):
        kalman_filter(
            np.random.default_rng(0).normal(size=(20, 1)),
            np.array([[1.0, 0.0]]),  # F wrong shape vs H
            np.array([[1.0]]),
            np.array([[0.1]]),
            np.array([[0.1]]),
        )
    with pytest.raises(ValueError):
        local_level_mle(np.ones(4))
    with pytest.raises(ValueError):
        kalman_hedge_ratio(np.ones(50), np.ones(50), delta=0.0)
    with pytest.raises(ValueError):
        ou_mle(np.ones(100))
    with pytest.raises(ValueError):
        ou_mle(np.random.default_rng(0).standard_normal(100), dt=-1.0)
