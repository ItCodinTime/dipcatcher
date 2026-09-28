"""Hedge-ratio estimators: OLS known answer + Kalman convergence."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.pairs.hedge import kalman_hedge_ratio, ols_hedge_ratio


class TestOLS:
    def test_known_answer(self):
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.normal(size=400))
        y = 0.5 + 2.0 * x + rng.normal(scale=0.05, size=400)
        alpha, beta = ols_hedge_ratio(y, x)
        assert alpha == pytest.approx(0.5, abs=0.02)
        assert beta == pytest.approx(2.0, abs=0.02)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            ols_hedge_ratio(np.ones(5), np.ones(5))
        with pytest.raises(ValueError):
            ols_hedge_ratio(np.arange(20.0), np.arange(19.0))
        with pytest.raises(ValueError):
            ols_hedge_ratio(np.full(20, np.nan), np.arange(20.0))


class TestKalman:
    def test_converges_to_static_beta(self):
        rng = np.random.default_rng(1)
        x = np.cumsum(rng.normal(size=600))
        y = 0.2 + 1.5 * x + rng.normal(scale=0.05, size=600)
        path = kalman_hedge_ratio(y, x, q=1e-4)
        assert path.shape == (600,)
        assert path[-1] == pytest.approx(1.5, abs=0.1)
        # The burn-in OLS start is already close; the filter should not diverge.
        assert np.isfinite(path).all()

    def test_tracks_time_varying_beta(self):
        rng = np.random.default_rng(2)
        n = 800
        x = np.cumsum(rng.normal(size=n))
        true_beta = np.concatenate([np.full(n // 2, 1.0), np.full(n - n // 2, 2.0)])
        y = true_beta * x + rng.normal(scale=0.05, size=n)
        path = kalman_hedge_ratio(y, x, q=1e-2)
        early = float(np.median(path[: n // 2]))
        late = float(path[-1])
        assert early == pytest.approx(1.0, abs=0.4)
        assert late == pytest.approx(2.0, abs=0.4)
        assert late > early  # the filter actually adapted

    def test_deterministic(self):
        rng = np.random.default_rng(3)
        x = np.cumsum(rng.normal(size=300))
        y = 1.2 * x + rng.normal(scale=0.1, size=300)
        a = kalman_hedge_ratio(y, x)
        b = kalman_hedge_ratio(y, x)
        np.testing.assert_array_equal(a, b)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            kalman_hedge_ratio(np.ones(20), np.arange(20.0), q=-1.0)
        with pytest.raises(ValueError):
            kalman_hedge_ratio(np.ones(20), np.arange(20.0), r=0.0)
        with pytest.raises(ValueError):
            kalman_hedge_ratio(np.ones(20), np.arange(20.0), alpha=np.inf)
