"""Causal sleeve gate and linear policy. Selection fit must ignore the future."""

from __future__ import annotations

import numpy as np

from quant_fund.hedge_lab.sleeve_policy import (
    apply_sleeve_policy,
    fit_sleeve_policy,
    lead_rule_returns,
    sleeve_features,
)


def test_policy_fit_ignores_rows_after_the_cut() -> None:
    rng = np.random.default_rng(0)
    t = 200
    spy = rng.normal(0.0004, 0.01, size=t)
    book = spy + rng.normal(0.0, 0.005, size=t)
    cash = np.zeros(t)
    sleeves = np.column_stack([cash, spy, book])
    features = sleeve_features(spy, book)
    train = np.zeros(t, dtype=bool)
    train[:120] = True
    intercept, slopes = fit_sleeve_policy(features, sleeves, train)
    shocked = sleeves.copy()
    shocked[150:, 2] += 0.05
    intercept2, slopes2 = fit_sleeve_policy(features, shocked, train)
    assert np.allclose(intercept, intercept2)
    assert np.allclose(slopes, slopes2)
    pnl, actions = apply_sleeve_policy(features, sleeves, intercept, slopes)
    pnl2, _ = apply_sleeve_policy(features, shocked, intercept, slopes)
    # Bar 140's action uses features through 139, which do not contain the shock.
    assert int(actions[140]) == int(apply_sleeve_policy(features, shocked, intercept, slopes)[1][140])
    assert pnl.shape == (t,)
    assert float(pnl[140]) == float(pnl2[140])


def test_ucb_freeze_does_not_learn_the_holdout() -> None:
    from quant_fund.hedge_lab.sleeve_policy import ucb_then_freeze

    rng = np.random.default_rng(1)
    sleeves = rng.normal(0.0, 0.01, size=(80, 3))
    sleeves[:50, 2] += 0.02
    train = np.zeros(80, dtype=bool)
    train[:50] = True
    first, arm = ucb_then_freeze(sleeves, train, one_way_cost=0.0)
    shocked = sleeves.copy()
    shocked[50:, 0] += 0.05
    second, arm2 = ucb_then_freeze(shocked, train, one_way_cost=0.0)
    assert arm == arm2
    assert np.allclose(first[:50], second[:50])

    xlk = np.full(40, 0.001)
    qqq = np.full(40, 0.002)
    tlt = np.full(40, -0.001)
    base = lead_rule_returns(xlk, qqq, tlt, window=5, one_way_cost=0.0)
    shocked = qqq.copy()
    shocked[-1] = 0.2
    alt = lead_rule_returns(xlk, shocked, tlt, window=5, one_way_cost=0.0)
    assert np.allclose(base[:-1], alt[:-1])
    assert float(alt[-1]) != float(base[-1])
