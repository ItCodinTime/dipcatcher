"""Tests for microstructure liquidity estimators."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.features.liquidity import (
    amivest_ratio,
    effective_tick,
    fht_cost,
    glosten_harris,
    hasbrouck_lambda,
    lot_spread,
    pastor_stambaugh_gamma,
    turnover_volatility,
    zero_freq,
)


class TestAmivestZeros:
    def test_amivest_ratio(self):
        rng = np.random.default_rng(0)
        r = rng.normal(scale=0.01, size=300)
        vol = rng.uniform(1e5, 2e5, 300)
        ratio = amivest_ratio(r, vol)
        # Amivest = sum(V)/sum(|r|) = mean(V)/mean(|r|).
        assert ratio == pytest.approx(float(np.mean(vol) / np.mean(np.abs(r))))
        # More volume for same returns -> more liquid.
        assert amivest_ratio(r, vol * 10) > ratio * 9

    def test_zero_freq(self):
        r = np.array([0.0, 0.0, 0.01, -0.02] * 25)
        assert zero_freq(r) == pytest.approx(0.5)
        with pytest.raises(ValueError):
            zero_freq(np.ones(5))

    def test_lot_spread_positive(self):
        rng = np.random.default_rng(1)
        r = rng.normal(scale=0.02, size=400)
        r[rng.random(400) < 0.3] = 0.0  # 30% zeros
        out = lot_spread(r)
        assert out["spread"] > 0
        assert out["zero_share"] == pytest.approx(0.3, abs=0.05)

    def test_fht_increases_with_zeros(self):
        rng = np.random.default_rng(2)
        r1 = rng.normal(scale=0.02, size=400)
        r2 = r1.copy()
        r2[rng.random(400) < 0.4] = 0.0
        assert fht_cost(r2)["fht"] > fht_cost(r1)["fht"]


class TestReversalImpact:
    def test_ps_gamma_negative_for_reversal(self):
        rng = np.random.default_rng(3)
        n = 500
        vol = rng.uniform(1e4, 1e5, n)
        r = np.zeros(n)
        u = rng.normal(scale=0.01, size=n)
        for t in range(1, n):
            # Strong reversal proportional to signed volume.
            r[t] = -0.0000005 * np.sign(r[t - 1]) * vol[t - 1] + u[t]
            if t > 1:
                r[t] += 0.0
        out = pastor_stambaugh_gamma(r, vol)
        assert out["gamma"] < 0

    def test_hasbrouck_lambda_positive(self):
        rng = np.random.default_rng(4)
        n = 600
        q = rng.choice([-1.0, 1.0], n) * rng.uniform(100, 1000, n)
        dp = 0.0004 * np.sign(q) * np.sqrt(np.abs(q)) + rng.normal(scale=0.01, size=n)
        out = hasbrouck_lambda(dp, q)
        assert abs(out["lambda"] - 0.0004) < 0.0002
        assert out["t"] > 2.0

    def test_glosten_harris(self):
        rng = np.random.default_rng(5)
        n = 800
        q = rng.choice([-1.0, 1.0], n)
        vol = rng.uniform(100, 500, n)
        dp = 0.005 * q + 0.00001 * q * vol + rng.normal(scale=0.01, size=n)
        out = glosten_harris(dp, q, vol)
        assert abs(out["transitory"] - 0.005) < 0.003
        assert out["adverse"] > 0
        assert 0 < out["spread_est"] < 0.03

    def test_failclosed(self):
        with pytest.raises(ValueError):
            pastor_stambaugh_gamma(np.ones(5), np.ones(5))
        with pytest.raises(ValueError):
            glosten_harris(np.ones(50), np.zeros(50))
        with pytest.raises(ValueError):
            hasbrouck_lambda(np.ones(50), np.ones(40))


class TestTickTurnover:
    def test_effective_tick(self):
        rng = np.random.default_rng(6)
        # Prices clustered on a 0.25 grid -> larger effective tick than
        # uniformly distributed prices (Holden is upward-biased on exact
        # grids by construction; ordering is the testable property).
        p_cluster = np.round(rng.uniform(10, 20, 400) / 0.25) * 0.25
        p_unif = rng.uniform(10, 20, 400)
        out = effective_tick(p_cluster)
        assert out["probs"][2] > 0.9  # 0.25 grid prob high
        assert effective_tick(p_cluster)["tick"] > 5.0 * effective_tick(p_unif)["tick"]
        assert 0.2 < out["tick"] < 0.7

    def test_turnover_volatility(self):
        rng = np.random.default_rng(7)
        vol = rng.uniform(1e5, 1.1e5, 100)
        out = turnover_volatility(vol)
        assert out["turnover_cv"] < 0.05
        vol2 = np.concatenate([vol, vol * 5.0])
        assert turnover_volatility(vol2)["turnover_cv"] > out["turnover_cv"]

    def test_failclosed(self):
        with pytest.raises(ValueError):
            effective_tick(-np.ones(50))
        with pytest.raises(ValueError):
            turnover_volatility(np.zeros(30))
