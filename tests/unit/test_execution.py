import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.config.models import CostConfig
from quant_fund.execution.almgren_chriss import (
    almgren_chriss_trajectory,
    expected_shortfall_ac,
    slice_trades,
)
from quant_fund.execution.costs import (
    commission_cost,
    half_spread_cost,
    sqrt_impact,
    total_cost,
)


def test_zero_quantity_zero_impact() -> None:
    assert sqrt_impact(0.0, 10.0, 1e6, 0.02, 0.1) == 0.0
    c = total_cost(0.0, 10.0, 1e6, 0.02, CostConfig())
    assert c["total"] == 0.0


def test_costs_reject_invalid_market_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        sqrt_impact(1.0, 0.0, 1e6, 0.02, 0.1)
    with np.testing.assert_raises(ValueError):
        total_cost(1.0, 10.0, 0.0, 0.02, CostConfig())
    with np.testing.assert_raises(ValueError):
        sqrt_impact(1.0, 10.0, 1e6, np.nan, 0.1)
    with np.testing.assert_raises(ValueError):
        total_cost(1.0, 10.0, 1e6, -0.01, CostConfig())


def test_larger_order_not_cheaper() -> None:
    a = sqrt_impact(100, 10, 1e6, 0.02, 0.1)
    b = sqrt_impact(400, 10, 1e6, 0.02, 0.1)
    assert b >= a


def test_ac_zero_qty() -> None:
    h = almgren_chriss_trajectory(0.0, 5, sigma=0.02, eta=1e-6, gamma=0.0, risk_aversion=1e-6)
    assert np.allclose(h, 0.0)


def test_ac_front_load_with_risk_aversion() -> None:
    slow = almgren_chriss_trajectory(100, 10, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0)
    fast = almgren_chriss_trajectory(100, 10, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=1e-2)
    # remaining after first slice should be smaller when more risk averse (more sold immediately)
    assert fast[1] <= slow[1] + 1e-8


def test_ac_higher_temp_impact_slower() -> None:
    aggressive = almgren_chriss_trajectory(
        100, 8, sigma=0.02, eta=1e-6, gamma=0.0, risk_aversion=1e-3
    )
    sticky = almgren_chriss_trajectory(100, 8, sigma=0.02, eta=1e-3, gamma=0.0, risk_aversion=1e-3)
    trades_a = slice_trades(aggressive)
    trades_s = slice_trades(sticky)
    assert abs(trades_a[0]) >= abs(trades_s[0]) - 1e-6


def test_expected_is_zero_parent() -> None:
    h = np.zeros(6)
    t = slice_trades(h)
    out = expected_shortfall_ac(h, t, arrival=10, eta=1e-6, gamma=0.0, sigma=0.02)
    assert out["expected_cost"] == 0.0


@given(st.floats(1, 1e4))
@settings(max_examples=20)
def test_impact_monotone_property(q: float) -> None:
    c1 = sqrt_impact(q, 10, 1e7, 0.02, 0.1)
    c2 = sqrt_impact(q * 2, 10, 1e7, 0.02, 0.1)
    assert c2 >= c1 - 1e-9


def test_half_spread_cost_closed_form() -> None:
    assert half_spread_cost(100_000.0, 5.0) == pytest.approx(50.0)
    assert half_spread_cost(-100_000.0, 5.0) == pytest.approx(50.0)  # abs notional
    assert half_spread_cost(0.0, 10.0) == 0.0
    assert half_spread_cost(1e6, 0.0) == 0.0


def test_commission_cost_closed_form() -> None:
    assert commission_cost(50_000.0, 2.0) == pytest.approx(10.0)
    assert commission_cost(-50_000.0, 2.0) == pytest.approx(10.0)
    assert commission_cost(0.0, 3.0) == 0.0


def test_scalar_cost_helpers_reject_invalid_rates() -> None:
    with pytest.raises(ValueError):
        half_spread_cost(100.0, -1.0)
    with pytest.raises(ValueError):
        commission_cost(100.0, float("inf"))


def test_total_cost_splits_spread_and_commission() -> None:
    cfg = CostConfig(commission_bps=1.0, half_spread_bps=2.0, impact_y=0.0, bps_per_turnover=0.0)
    out = total_cost(100.0, 10.0, 1e9, 0.02, cfg)  # notional=1000
    assert out["commission"] == pytest.approx(0.1)
    assert out["spread"] == pytest.approx(0.2)
    assert out["impact"] == pytest.approx(0.0)
    assert out["total"] == pytest.approx(0.3)
