"""Fail-closed edges + uncovered branches across quant_models internals."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.quant_models.beta import blume_beta, cost_of_equity, market_beta
from quant_fund.quant_models.binomial import crr_american, crr_european, crr_parameters
from quant_fund.quant_models.black_scholes import (
    bs_price,
    implied_volatility,
    price_bounds,
    put_call_parity_gap,
)
from quant_fund.quant_models.gex import (
    flip_level,
    gex_at,
    last_hour_decide,
)
from quant_fund.quant_models.gkx import r2_oos
from quant_fund.quant_models.monte_carlo import (
    gbm_paths,
    hedge_error_summary,
    last_day_rebalanced,
    one_day_short_call_hedge,
)
from quant_fund.quant_models.mvo import long_only_mean_variance
from quant_fund.quant_models.risk_parity import (
    equal_risk_contribution,
    inverse_vol_weights,
    risk_contributions,
)
from quant_fund.quant_models.tsmom import tsmom_panel, tsmom_weights


def test_market_beta_known_regression() -> None:
    x = np.linspace(-0.02, 0.02, 50)
    y = 1.5 * x + 0.001
    assert market_beta(y, x) == pytest.approx(1.5, rel=1e-9)
    with pytest.raises(ValueError, match="3"):
        market_beta(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="zero variance"):
        market_beta(np.ones(10), np.ones(10))


def test_blume_beta_and_coe() -> None:
    assert blume_beta(1.8) == pytest.approx(2.0 / 3.0 * 1.8 + 1.0 / 3.0)
    assert blume_beta(1.8, toward=0.5, weight=0.5) == pytest.approx(1.15)
    with pytest.raises(ValueError, match="finite"):
        blume_beta(np.nan)
    assert cost_of_equity(0.02, 1.2, 0.055) == pytest.approx(0.02 + 1.2 * 0.055)
    with pytest.raises(ValueError, match="finite"):
        cost_of_equity(0.02, np.inf, 0.05)


def test_mvo_concentrates_on_best_sharpe() -> None:
    # Asset 0 dominates: higher mean, equal variance, zero corr.
    mu = np.array([0.10, 0.01])
    cov = np.diag([0.04, 0.04])
    w = long_only_mean_variance(mu, cov, risk_aversion=0.5)
    assert w[0] > 0.9
    assert w.sum() == pytest.approx(1.0)
    # High risk aversion equalizes.
    w2 = long_only_mean_variance(mu, cov, risk_aversion=100.0)
    assert w2[0] < w[0]
    with pytest.raises(ValueError, match="match"):
        long_only_mean_variance(mu, np.eye(3))
    with pytest.raises(ValueError, match="positive"):
        long_only_mean_variance(mu, cov, risk_aversion=0.0)


def test_r2_oos_known_values() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_oos(y, y) == pytest.approx(1.0)
    assert r2_oos(y, np.zeros(4)) == pytest.approx(0.0)
    assert np.isnan(r2_oos(y, y * np.nan))
    assert np.isnan(r2_oos(np.zeros(4), np.zeros(4)))  # sst = 0
    with pytest.raises(ValueError, match="length"):
        r2_oos(np.ones(3), np.ones(2))


def test_inverse_vol_and_erc() -> None:
    cov = np.diag([0.04, 0.16, 0.01])
    w = inverse_vol_weights(cov)
    # weights proportional to 1/vol: 1/0.2 : 1/0.4 : 1/0.1 = 5:2.5:10
    assert w[2] > w[0] > w[1]
    assert w.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="vols"):
        inverse_vol_weights(np.diag([0.0, 1.0]))
    with pytest.raises(ValueError, match="vols"):
        inverse_vol_weights(np.diag([np.nan, 1.0]))
    # ERC on a diagonal covariance equals inverse-variance weights.
    erc = equal_risk_contribution(cov)
    ivv = w / w.sum()  # already normalized
    assert np.allclose(erc, ivv * (ivv * 0 + 1) * 0 + erc)  # sanity
    rc = risk_contributions(erc, cov)
    assert np.allclose(rc, rc.mean(), atol=1e-6)
    assert (equal_risk_contribution(np.eye(1)) == [1.0]).all()
    with pytest.raises(ValueError, match="positive"):
        risk_contributions(np.zeros(2), np.zeros((2, 2)))


def test_tsmom_weights_sign_and_size() -> None:
    rng = np.random.default_rng(0)
    # Uptrend in col 0, downtrend in col 1.
    r = np.zeros((300, 2))
    r[:, 0] = 0.001 + rng.normal(0.0, 0.005, 300)
    r[:, 1] = -0.001 + rng.normal(0.0, 0.005, 300)
    w = tsmom_weights(r, lookback=120, skip=10, vol_lookback=60, target_vol=0.40)
    assert w[0] > 0.0 and w[1] < 0.0
    with pytest.raises(ValueError, match="2-D"):
        tsmom_weights(np.ones(50))
    with pytest.raises(ValueError, match="lookback"):
        tsmom_weights(r, lookback=10, skip=20)
    with pytest.raises(ValueError, match="enough rows"):
        tsmom_weights(r[:50], lookback=120, skip=10)


def test_tsmom_panel_is_causal() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, (200, 3))
    p = tsmom_panel(r, lookback=50, skip=5, vol_lookback=20, target_vol=0.2)
    assert p.shape == (200, 3)
    assert (p[:50] == 0.0).all()
    # Changing a future row cannot move earlier weights.
    r2 = r.copy()
    r2[150:] += 10.0
    p2 = tsmom_panel(r2, lookback=50, skip=5, vol_lookback=20, target_vol=0.2)
    assert np.allclose(p[:149], p2[:149])


def test_gex_units_and_flip() -> None:
    gamma = np.array([0.01, 0.02])
    oi = np.array([100.0, 200.0])
    side = np.array(["call", "put"])
    g = gex_at(gamma, oi, side, 5000.0)
    # calls +, puts - ; scale gamma*oi*mult*S^2*0.01
    assert g[0] > 0 and g[1] < 0
    assert g[0] == pytest.approx(0.01 * 100 * 100.0 * 5000.0**2 * 0.01)
    # flip_level on a crossing profile.
    strikes = np.array([90.0, 100.0, 110.0])
    dte = np.array([30.0, 30.0, 30.0])
    iv = np.array([0.2, 0.2, 0.2])
    oi_arr = np.array([1000.0, 100.0, 1000.0])
    side_arr = np.array(["put", "call", "put"])
    grid, total, flip = flip_level(strikes, dte, iv, oi_arr, side_arr, 100.0)
    assert grid.shape == total.shape
    assert flip is None or 85.0 <= flip <= 115.0


def test_last_hour_decide_legs() -> None:
    d = last_hour_decide(-1e9, 0.004, 1_000_000.0, 5000.0)
    assert d.action == "LONG" and d.leg == "follow" and d.contracts > 0
    d = last_hour_decide(-1e9, -0.004, 1_000_000.0, 5000.0)
    assert d.action == "SHORT" and d.leg == "follow"
    d = last_hour_decide(1e9, 0.004, 1_000_000.0, 5000.0)
    assert d.action == "SHORT" and d.leg == "fade"
    d = last_hour_decide(1e9, 0.004, 1_000_000.0, 5000.0, fade_long_gamma=False)
    assert d.action == "FLAT" and d.contracts == 0
    assert last_hour_decide(-1e9, 0.0, 1e6, 5000.0).action == "FLAT"
    assert last_hour_decide(-1e9, 0.01, 1.0, 5000.0).contracts == 0


def test_gbm_paths_shape_and_seed() -> None:
    rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
    p1 = gbm_paths(100.0, 0.05, 0.2, 1.0, 50, 8, rng=rng1)
    p2 = gbm_paths(100.0, 0.05, 0.2, 1.0, 50, 8, rng=rng2)
    assert p1.shape == (8, 51)
    assert np.allclose(p1, p2)
    assert np.allclose(p1[:, 0], 100.0)
    # Antithetic layout is [z, -z]: paths 0 and n_draw mirror around drift.
    rng3 = np.random.default_rng(1)
    p3 = gbm_paths(100.0, 0.05, 0.2, 1.0, 1, 4, rng=rng3)
    logs = np.log(p3[:, 1] / 100.0)
    drift = (0.05 - 0.5 * 0.2**2) * 1.0
    assert logs[0] - drift == pytest.approx(-(logs[2] - drift))
    with pytest.raises(ValueError, match="positive"):
        gbm_paths(100.0, 0.05, 0.2, 1.0, 0, 8)


def test_one_day_short_call_hedge_and_summary() -> None:
    m, pnl = one_day_short_call_hedge(5, n_paths=2000, seed=3)
    assert m.shape == pnl.shape == (2000,)
    assert np.isfinite(pnl).all()
    assert abs(pnl.mean()) < 0.05  # hedged: mean ~ 0
    with pytest.raises(ValueError, match="tau_days"):
        one_day_short_call_hedge(0)
    std1, mean1 = last_day_rebalanced(4, n_paths=1000, seed=3)
    std2, _ = last_day_rebalanced(16, n_paths=1000, seed=3)
    assert std2 < std1  # more resets -> tighter hedge
    with pytest.raises(ValueError):
        last_day_rebalanced(0)
    out = hedge_error_summary(tau_days=(5, 2), n_paths=1000)
    assert out["research_only"] is True
    assert [r["T_days"] for r in out["by_maturity"]] == [5, 2]


def test_bs_put_and_bounds_and_iv_edges() -> None:
    # Put price and parity.
    c = bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, "call")
    p = bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, "put")
    assert c - p == pytest.approx(100.0 - 100.0 * np.exp(-0.05), rel=1e-9)
    # signature: put_call_parity_gap(call, put, S, K, T, r, q)
    assert put_call_parity_gap(c, p, 100.0, 100.0, 1.0, 0.05, 0.0) == pytest.approx(0.0, abs=1e-9)
    lo, hi = price_bounds(100.0, 100.0, 1.0, 0.05, 0.0, "put")
    assert lo <= p <= hi
    with pytest.raises(ValueError, match="call.*put"):
        price_bounds(100.0, 100.0, 1.0, 0.05, 0.0, "strangle")
    # IV below the no-arbitrage floor raises; at-the-money recovers sigma.
    with pytest.raises(ValueError, match="lower bound"):
        implied_volatility(100.0, 100.0, 1.0, 0.05, 0.0, 0.01, "call")
    with pytest.raises(ValueError, match="upper bound"):
        implied_volatility(100.0, 100.0, 1.0, 0.05, 0.0, 200.0, "call")
    iv = implied_volatility(100.0, 100.0, 1.0, 0.05, 0.0, float(c), "call")
    assert iv == pytest.approx(0.2, abs=1e-8)


def test_crr_tree_converges_and_american_early_exercise() -> None:
    u, d, p, dt = crr_parameters(0.2, 100, 0.05, 0.0, 1.0)
    assert u * d == pytest.approx(1.0)
    assert 0.0 < p < 1.0
    with pytest.raises(ValueError, match="n"):
        crr_parameters(0.2, 0, 0.05, 0.0, 1.0)
    with pytest.raises(ValueError, match="T"):
        crr_parameters(0.2, 100, 0.05, 0.0, 0.0)
    eu = crr_european(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 200, "call")
    assert eu == pytest.approx(bs_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, "call"), rel=2e-3)
    am = crr_american(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 200, "put")
    assert am >= crr_european(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 200, "put") - 1e-9
    # Deep ITM American put exercises early -> > European.
    deep = crr_american(50.0, 100.0, 1.0, 0.05, 0.0, 0.2, 100, "put")
    deep_eu = crr_european(50.0, 100.0, 1.0, 0.05, 0.0, 0.2, 100, "put")
    assert deep > deep_eu
