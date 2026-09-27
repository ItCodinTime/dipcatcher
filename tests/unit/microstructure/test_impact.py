"""Tests for market impact and implementation shortfall."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.execution.impact import (
    arrival_price_slippage,
    permanent_impact,
    perold_shortfall,
    pov_schedule,
    pow_law_total_impact,
    propagator_kernel,
    propagator_price_path,
    required_participation,
    sqrt_impact_bps,
    vwap_slippage,
)


class TestImpactLaws:
    def test_sqrt_law(self):
        # upsilon*sigma*sqrt(0.01) = 0.6*0.02*0.1 = 0.0012
        assert abs(sqrt_impact_bps(0.01, 0.02, 0.6) - 0.0012) < 1e-12
        assert sqrt_impact_bps(0.0, 0.02) == 0.0

    def test_permanent_linear(self):
        # gamma*sigma*Q/V = 0.5*0.02*0.1 = 0.001
        assert abs(permanent_impact(1e5, 1e6, 0.02, 0.5) - 0.001) < 1e-12

    def test_power_law_monotone(self):
        i1 = pow_law_total_impact(1e4, 1e6, 0.02)
        i2 = pow_law_total_impact(1e5, 1e6, 0.02)
        i3 = pow_law_total_impact(5e5, 1e6, 0.02)
        assert i1 < i2 < i3

    def test_failclosed(self):
        with pytest.raises(ValueError):
            sqrt_impact_bps(-0.1, 0.02)
        with pytest.raises(ValueError):
            permanent_impact(1.0, 0.0, 0.02)
        with pytest.raises(ValueError):
            pow_law_total_impact(1.0, 1.0, 0.02, exponent=0.0)


class TestPropagator:
    def test_kernel_decay(self):
        g = propagator_kernel(np.arange(10.0), beta=0.5)
        assert g[0] == 1.0
        assert np.all(np.diff(g) < 0.0)
        assert g[-1] < 0.35

    def test_price_path_buy_pushes_up(self):
        q = np.zeros(20)
        q[0] = 100.0  # one buy then nothing
        dp = propagator_price_path(q, beta=0.5, impact_scale=1e-4)
        assert dp[0] > 0.0
        # Impact decays: displacement shrinks over time (transient impact).
        assert dp[-1] < dp[0]
        assert np.all(dp >= 0.0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            propagator_kernel(np.array([-1.0]))
        with pytest.raises(ValueError):
            propagator_price_path(np.array([]))


class TestSchedules:
    def test_pov_fills_exactly(self):
        v = np.full(10, 1000.0)
        s = pov_schedule(2000.0, v, participation=0.25)
        assert abs(s.sum() - 2000.0) < 1e-9
        # Never exceeds 25% of volume in any bar.
        assert np.all(s <= 0.25 * v + 1e-9)

    def test_pov_partial_fill(self):
        v = np.full(5, 100.0)
        s = pov_schedule(1000.0, v, participation=0.5)
        assert s.sum() <= 1000.0
        assert s.sum() == 250.0  # 0.5*100*5 caps out

    def test_required_participation(self):
        v = np.full(10, 100.0)
        p = required_participation(500.0, v)
        assert abs(p - 0.5) < 1e-12
        assert required_participation(1000.0, v) == 1.0
        # Infeasible: order exceeds horizon volume.
        with pytest.raises(ValueError):
            required_participation(500.0, v, max_bars=2)
        with pytest.raises(ValueError):
            required_participation(2000.0, v)
        with pytest.raises(ValueError):
            pov_schedule(1.0, v, participation=1.5)


class TestSlippage:
    def test_vwap_slippage_buy(self):
        p = np.array([100.0, 101.0])
        q = np.array([50.0, 50.0])
        slip = vwap_slippage(p, q, market_vwap=100.0)
        assert slip == pytest.approx(0.005)

    def test_vwap_slippage_sell_signed(self):
        p = np.array([101.0, 101.0])
        q = np.array([-50.0, -50.0])
        slip = vwap_slippage(p, q, market_vwap=100.0)
        # Sold above VWAP -> negative cost for the seller.
        assert slip == pytest.approx(-0.01)

    def test_arrival(self):
        p = np.array([101.0])
        q = np.array([10.0])
        assert arrival_price_slippage(p, q, 100.0) == pytest.approx(0.01)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            vwap_slippage(np.array([100.0]), np.array([0.0]), 100.0)
        with pytest.raises(ValueError):
            arrival_price_slippage(np.array([100.0]), np.array([1.0, 2.0]), 100.0)


class TestPerold:
    def test_full_fill_no_opportunity(self):
        out = perold_shortfall(
            decision_price=100.0,
            exec_prices=np.array([101.0, 102.0]),
            exec_qty=np.array([50.0, 50.0]),
            unfilled_qty=0.0,
            cancel_price=105.0,
            fees=0.0,
        )
        # exec_vwap = 101.5 -> execution = 1.5%, opportunity = 0
        assert out["execution"] == pytest.approx(0.015)
        assert out["opportunity"] == 0.0
        assert out["total"] == pytest.approx(0.015)
        assert out["fill_rate"] == 1.0

    def test_partial_fill_decomposes(self):
        out = perold_shortfall(
            decision_price=100.0,
            exec_prices=np.array([101.0]),
            exec_qty=np.array([50.0]),
            unfilled_qty=50.0,
            cancel_price=110.0,
            fees=10.0,
        )
        # execution: 1% * 0.5 = 0.5%; opportunity: 10% * 0.5 = 5%
        # fees: 10 / (100*100) = 0.1%
        assert out["execution"] == pytest.approx(0.005)
        assert out["opportunity"] == pytest.approx(0.05)
        assert out["fees"] == pytest.approx(0.001)
        assert out["total"] == pytest.approx(0.056)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            perold_shortfall(0.0, np.array([1.0]), np.array([1.0]), 0.0, 1.0)
        with pytest.raises(ValueError):
            perold_shortfall(100.0, np.array([101.0]), np.array([0.0]), 0.0, 105.0)
