"""Sleeve-policy overlays: trailing stats, ridge policy, gates, bandits."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.hedge_lab.sleeve_policy import (
    apply_sleeve_policy,
    etf_dual_momentum,
    fit_sleeve_policy,
    lead_rule_returns,
    policy_snapshot,
    sleeve_features,
    trailing_mean,
    trailing_vol,
    ucb_then_freeze,
)


def test_trailing_mean_warmup_and_values() -> None:
    r = np.arange(1.0, 6.0)
    m = trailing_mean(r, 3)
    assert np.isnan(m[:2]).all()
    assert m[2] == pytest.approx(2.0)
    assert m[4] == pytest.approx(4.0)
    # NaN inside the window -> NaN (no partial-window fill).
    r2 = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    assert np.isnan(trailing_mean(r2, 3)[2])


def test_trailing_vol_ddof1() -> None:
    r = np.array([1.0, 3.0, 2.0, 4.0])
    v = trailing_vol(r, 3)
    assert np.isnan(v[0]) and np.isnan(v[1])
    assert v[2] == pytest.approx(np.std([1.0, 3.0, 2.0], ddof=1))


def test_sleeve_features_shape_and_mismatch() -> None:
    spy = np.random.default_rng(0).normal(0.0, 0.01, 100)
    book = np.random.default_rng(1).normal(0.0, 0.01, 100)
    f = sleeve_features(spy, book)
    assert f.shape == (100, 4)
    # col 1 is the 63-bar mean -> NaN until row 62; col 0 (21-bar) until row 20.
    assert np.isnan(f[:62, 1]).all() and np.isfinite(f[62, 1])
    assert np.isnan(f[:20, 0]).all() and np.isfinite(f[20, 0])
    with pytest.raises(ValueError, match="align"):
        sleeve_features(spy, book[:-1])


def test_fit_and_apply_sleeve_policy_picks_best_sleeve() -> None:
    rng = np.random.default_rng(0)
    t = 400
    # Feature row predicts sleeve 2 > others in-sample.
    x = np.column_stack([rng.normal(size=t) for _ in range(4)])
    sleeves = np.zeros((t, 3))
    sleeves[:, 0] = rng.normal(0.0, 0.01, t)
    sleeves[:, 1] = rng.normal(0.0, 0.01, t)
    sleeves[:, 2] = 0.5 * x[:, 0] + rng.normal(0.0, 0.01, t)
    train = np.zeros(t, dtype=bool)
    train[80:300] = True
    intercept, slopes = fit_sleeve_policy(x, sleeves, train)
    assert intercept.shape == (3,) and slopes.shape == (4, 3)
    pnl, actions = apply_sleeve_policy(x, sleeves, intercept, slopes)
    assert pnl.shape == (t,) and actions.shape == (t,)
    # Rows outside training still traded by the frozen map, but a feature row
    # of all-NaN always picks cash (action 0).
    x_bad = x.copy()
    x_bad[350] = np.nan
    _, actions_bad = apply_sleeve_policy(x_bad, sleeves, intercept, slopes)
    assert actions_bad[351] == 0


def test_fit_sleeve_policy_needs_30_rows_and_shapes() -> None:
    x = np.random.default_rng(0).normal(size=(40, 4))
    y = np.random.default_rng(1).normal(size=(40, 3))
    train = np.zeros(40, dtype=bool)
    train[5:20] = True
    with pytest.raises(ValueError, match="30"):
        fit_sleeve_policy(x, y, train)
    with pytest.raises(ValueError, match="align"):
        fit_sleeve_policy(x, y[:, :2], np.ones(40, dtype=bool))


def test_apply_sleeve_policy_charges_switch_cost() -> None:
    t = 5
    x = np.zeros((t, 1))
    y = np.zeros((t, 3))
    intercept = np.array([0.0, 0.0, 1.0])
    slopes = np.zeros((1, 3))
    pnl, actions = apply_sleeve_policy(x, y, intercept, slopes, one_way_cost=0.01)
    # First switch cash->sleeve2 costs 0.01, then holds.
    assert actions.tolist() == [0, 2, 2, 2, 2]
    assert pnl[1] == pytest.approx(-0.01)
    assert pnl[2] == pytest.approx(0.0)


def test_lead_rule_returns_switches_on_lagged_signal() -> None:
    n = 30
    xlk_excess = np.full(n, 0.01)  # positive -> wants QQQ
    qqq = np.full(n, 0.02)
    tlt = np.full(n, -0.01)
    pnl = lead_rule_returns(xlk_excess, qqq, tlt, window=5, one_way_cost=0.0)
    assert pnl[0] == 0.0
    # After warmup, every bar earns the QQQ return.
    assert np.all(pnl[6:] == pytest.approx(0.02))
    # Negative XLK excess but rising TLT -> holds TLT.
    pnl2 = lead_rule_returns(-np.ones(n) * 0.01, qqq, np.full(n, 0.005), window=5, one_way_cost=0.0)
    assert np.all(pnl2[6:] == pytest.approx(0.005))


def test_policy_snapshot_counts() -> None:
    a = np.array([0, 0, 1, 1, 2, 2, 2])
    out = policy_snapshot(a)
    assert out["action_counts"] == {"cash": 2, "spy": 2, "topk": 3}
    assert out["live_pnl_claim"] is False


def test_etf_dual_momentum_prefers_defensive_when_risky_weak() -> None:
    n = 300
    # Defensive rises steadily; risky names fall.
    px = np.zeros((n, 3))
    px[:, 0] = 100.0 * np.cumprod(1.0 + np.full(n, 0.001))
    px[:, 1] = 100.0 * np.cumprod(1.0 + np.full(n, -0.001))
    px[:, 2] = 100.0 * np.cumprod(1.0 + np.full(n, -0.001))
    r = etf_dual_momentum(px, defensive=0, lookback=60, skip=5, rebalance_every=5, one_way_cost=0.0)
    assert r.shape == (n,)
    tail = r[-50:]
    # Should be riding the defensive asset => its daily return.
    assert np.mean(tail) == pytest.approx(0.001, abs=2e-4)


def test_etf_dual_momentum_bad_defensive() -> None:
    with pytest.raises(ValueError, match="defensive"):
        etf_dual_momentum(np.ones((10, 2)), defensive=5)
    with pytest.raises(ValueError, match="T, N"):
        etf_dual_momentum(np.ones(10), defensive=0)


def test_ucb_then_freeze_holds_best_arm() -> None:
    rng = np.random.default_rng(0)
    t = 300
    sleeves = np.column_stack([rng.normal(-0.001, 0.005, t), rng.normal(0.002, 0.005, t)])
    train = np.zeros(t, dtype=bool)
    train[:150] = True
    pnl, frozen = ucb_then_freeze(sleeves, train, one_way_cost=0.0)
    assert frozen == 1
    assert pnl.shape == (t,)
    # Holdout pnl equals arm-1 rewards once frozen.
    assert np.allclose(pnl[200:], sleeves[200:, 1])


def test_ucb_then_freeze_shape_check() -> None:
    with pytest.raises(ValueError, match="align"):
        ucb_then_freeze(np.zeros((10, 2)), np.zeros(5, dtype=bool))
