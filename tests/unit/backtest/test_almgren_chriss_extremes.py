"""Almgren–Chriss / TWAP trajectory and expected-shortfall edge fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.execution.almgren_chriss import (
    almgren_chriss_trajectory,
    expected_shortfall_ac,
    front_loaded_trajectory,
    slice_trades,
    twap_trajectory,
)


def test_ac_n_slices_zero_raises() -> None:
    with pytest.raises(ValueError, match="n_slices"):
        almgren_chriss_trajectory(10.0, 0, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0)


def test_ac_n_slices_negative_raises() -> None:
    with pytest.raises(ValueError, match="n_slices"):
        almgren_chriss_trajectory(10.0, -3, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0)


def test_ac_tau_nonpositive_raises() -> None:
    with pytest.raises(ValueError, match="tau"):
        almgren_chriss_trajectory(
            10.0, 4, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0, tau=0.0
        )
    with pytest.raises(ValueError, match="tau"):
        almgren_chriss_trajectory(
            10.0, 4, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0, tau=-1.0
        )


def test_ac_zero_qty_all_slices() -> None:
    for n in (1, 2, 7):
        h = almgren_chriss_trajectory(0.0, n, sigma=0.02, eta=1e-6, gamma=0.0, risk_aversion=1e-3)
        assert h.shape == (n + 1,)
        assert np.allclose(h, 0.0)
        assert np.allclose(slice_trades(h), 0.0)


def test_ac_n_slices_one_twap_edge() -> None:
    h = twap_trajectory(50.0, 1)
    assert h.shape == (2,)
    assert h[0] == pytest.approx(50.0)
    assert h[-1] == pytest.approx(0.0)
    trades = slice_trades(h)
    assert trades.shape == (1,)
    assert trades[0] == pytest.approx(50.0)


def test_trajectory_trades_sum_to_qty() -> None:
    qty = 123.0
    for n in (1, 3, 10):
        for maker in (
            lambda q, ns: twap_trajectory(q, ns),
            lambda q, ns: front_loaded_trajectory(q, ns),
            lambda q, ns: almgren_chriss_trajectory(
                q, ns, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=5e-3
            ),
        ):
            h = maker(qty, n)
            trades = slice_trades(h)
            assert trades.sum() == pytest.approx(qty, rel=0, abs=1e-9)
            assert h[0] == pytest.approx(qty)
            assert h[-1] == pytest.approx(0.0)


def test_front_loaded_vs_twap_shape() -> None:
    n = 10
    qty = 100.0
    tw = twap_trajectory(qty, n)
    fl = front_loaded_trajectory(qty, n)
    tw_tr = slice_trades(tw)
    fl_tr = slice_trades(fl)
    # TWAP: equal slices
    assert np.allclose(tw_tr, qty / n)
    # Front-loaded first slice larger; remaining after first smaller
    assert fl_tr[0] >= tw_tr[0] - 1e-9
    assert fl[1] <= tw[1] + 1e-9
    # Front-loaded is monotone nonincreasing holdings
    assert np.all(np.diff(fl) <= 1e-12)


def test_risk_aversion_monotone_front_load() -> None:
    slow = almgren_chriss_trajectory(100, 12, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0)
    mid = almgren_chriss_trajectory(100, 12, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=1e-4)
    fast = almgren_chriss_trajectory(100, 12, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=1e-2)
    assert fast[1] <= mid[1] + 1e-8
    assert mid[1] <= slow[1] + 1e-8


def test_expected_shortfall_ac_zero_parent() -> None:
    h = np.zeros(5)
    tr = slice_trades(h)
    out = expected_shortfall_ac(h, tr, arrival=10.0, eta=1e-4, gamma=0.0, sigma=0.02)
    assert out["expected_is"] == 0.0
    assert out["variance_is"] == 0.0
    assert out["expected_cost"] == 0.0


def test_expected_shortfall_ac_invalid_lengths() -> None:
    h = twap_trajectory(10.0, 4)
    with pytest.raises(ValueError, match="trades length"):
        expected_shortfall_ac(h, np.ones(2), arrival=10.0, eta=1e-4, gamma=0.0, sigma=0.02)
    with pytest.raises(ValueError, match="holdings length"):
        expected_shortfall_ac(
            np.array([1.0]), np.array([]), arrival=10.0, eta=1e-4, gamma=0.0, sigma=0.02
        )


def test_expected_shortfall_ac_invalid_tau_and_nan() -> None:
    h = twap_trajectory(10.0, 4)
    tr = slice_trades(h)
    with pytest.raises(ValueError, match="tau"):
        expected_shortfall_ac(h, tr, arrival=10.0, eta=1e-4, gamma=0.0, sigma=0.02, tau=0.0)
    with pytest.raises(ValueError, match="finite"):
        expected_shortfall_ac(h, tr, arrival=np.nan, eta=1e-4, gamma=0.0, sigma=0.02)


def test_expected_shortfall_temp_impact_monotone_in_eta() -> None:
    h = twap_trajectory(100.0, 8)
    tr = slice_trades(h)
    low = expected_shortfall_ac(h, tr, arrival=10.0, eta=1e-6, gamma=0.0, sigma=0.02)
    high = expected_shortfall_ac(h, tr, arrival=10.0, eta=1e-3, gamma=0.0, sigma=0.02)
    assert high["expected_cost"] >= low["expected_cost"] - 1e-12


def test_expected_shortfall_variance_monotone_in_sigma() -> None:
    h = twap_trajectory(100.0, 8)
    tr = slice_trades(h)
    low = expected_shortfall_ac(h, tr, arrival=10.0, eta=1e-4, gamma=0.0, sigma=0.01)
    high = expected_shortfall_ac(h, tr, arrival=10.0, eta=1e-4, gamma=0.0, sigma=0.05)
    assert high["variance_is"] >= low["variance_is"] - 1e-12


def test_expected_shortfall_front_loaded_higher_temp_than_twap() -> None:
    """With pure temporary impact, concentrating trades raises temp cost (sum v^2)."""
    qty, n = 100.0, 10
    tw_h = twap_trajectory(qty, n)
    fl_h = front_loaded_trajectory(qty, n)
    tw = expected_shortfall_ac(
        tw_h, slice_trades(tw_h), arrival=10.0, eta=1e-3, gamma=0.0, sigma=0.02
    )
    fl = expected_shortfall_ac(
        fl_h, slice_trades(fl_h), arrival=10.0, eta=1e-3, gamma=0.0, sigma=0.02
    )
    assert fl["expected_cost"] >= tw["expected_cost"] - 1e-12
