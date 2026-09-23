"""Tests for models/egarch.py — Nelson EGARCH + GJR-GARCH."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.egarch import egarch_fit, gjr_garch_fit, news_impact_curve


def _gjr_returns(n: int = 1500, seed: int = 0) -> np.ndarray:
    """Leverage-effect GJR data: gamma > 0 makes down moves raise vol."""
    rng = np.random.default_rng(seed)
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = 0.5
    r[0] = np.sqrt(s2[0]) * rng.standard_normal()
    a, g, b, w = 0.02, 0.12, 0.9, 0.05 * 0.5
    for t in range(1, n):
        s2[t] = w + (a + g * float(r[t - 1] < 0)) * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_normal()
    return r


def test_gjr_recovers_leverage() -> None:
    r = _gjr_returns()
    out = gjr_garch_fit(r)
    assert out["gamma"] > 0.02  # leverage present
    assert out["persistence"] < 1.0
    s2 = np.asarray(out["sig2"])
    assert (s2 > 0).all()


def test_gjr_vol_tracks() -> None:
    r = _gjr_returns(seed=1)
    out = gjr_garch_fit(r)
    vol = np.asarray(out["vol"])
    win = 20
    roll = np.array([np.std(r[i - win : i]) for i in range(win, r.size)])
    assert np.corrcoef(vol[win:], roll)[0, 1] > 0.5


def test_egarch_runs_and_asymmetry() -> None:
    r = _gjr_returns(seed=2)
    out = egarch_fit(r)
    s2 = np.asarray(out["sig2"])
    assert (s2 > 0).all() and np.isfinite(s2).all()
    assert out["persistence"] < 1.0
    # alpha should be negative (negative z raises log-vol)
    assert out["alpha"] < 0.05


def test_news_impact_curve_shape() -> None:
    r = _gjr_returns(seed=3)
    fit = gjr_garch_fit(r)
    shocks = np.linspace(-4, 4, 41)
    nic = news_impact_curve(fit, "gjr", shocks)
    # asymmetric: response to -3 exceeds response to +3
    i_neg = np.argmin(np.abs(shocks + 3))
    i_pos = np.argmin(np.abs(shocks - 3))
    assert nic[i_neg] > nic[i_pos]


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        gjr_garch_fit(np.random.default_rng(0).standard_normal(50))
    with pytest.raises(ValueError):
        egarch_fit(np.full(200, np.nan))
    fit = gjr_garch_fit(_gjr_returns(seed=5))
    with pytest.raises(ValueError):
        news_impact_curve(fit, "bogus", np.linspace(-1, 1, 5))
    with pytest.raises(ValueError):
        news_impact_curve(fit, "gjr", np.array([]))
