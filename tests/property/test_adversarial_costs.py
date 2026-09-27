"""Cost, impact, and slippage identities."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from quant_fund.config.models import CostConfig
from quant_fund.execution.costs import sqrt_impact, total_cost
from quant_fund.execution.impact import pov_schedule, sqrt_impact_bps, vwap_slippage
from quant_fund.execution.implementation_shortfall import fill_shortfall
from tests.property._profiles import adversarial_settings

_rate = st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False)
_price = st.floats(min_value=0.05, max_value=5_000.0, allow_nan=False, allow_infinity=False)
_adv = st.floats(min_value=10.0, max_value=1e9, allow_nan=False, allow_infinity=False)
_sigma = st.floats(min_value=0.0, max_value=1.5, allow_nan=False, allow_infinity=False)
_qty = st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False)


def _cost_config(**overrides: float) -> CostConfig:
    raw = {
        "commission_bps": 1.0,
        "half_spread_bps": 2.0,
        "impact_y": 0.1,
        "bps_per_turnover": 0.5,
        "participation_limit": 1.0,
    }
    raw.update(overrides)
    return CostConfig(**raw)


@given(qty=_qty, price=_price, adv=_adv, sigma=_sigma, commission=_rate, spread=_rate, impact=_rate)
@adversarial_settings()
def test_total_cost_adds_up_and_ignores_sign(
    qty: float,
    price: float,
    adv: float,
    sigma: float,
    commission: float,
    spread: float,
    impact: float,
) -> None:
    cfg = _cost_config(
        commission_bps=commission,
        half_spread_bps=spread,
        impact_y=impact,
        bps_per_turnover=0.0,
    )
    left = total_cost(qty, price, adv, sigma, cfg)
    right = total_cost(-qty, price, adv, sigma, cfg)
    parts = ("commission", "spread", "impact", "turnover_bps")
    assert left["total"] == pytest.approx(sum(float(left[name]) for name in parts))
    assert left["total"] == pytest.approx(float(right["total"]))
    assert float(left["total"]) >= 0.0


@given(
    qty=st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
    price=_price,
    adv=_adv,
    sigma=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    upsilon=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@adversarial_settings()
def test_sqrt_impact_monotone_in_size_vol_and_coefficient(
    qty: float,
    price: float,
    adv: float,
    sigma: float,
    upsilon: float,
) -> None:
    base = sqrt_impact(qty, price, adv, sigma, upsilon)
    assert sqrt_impact(qty * 2.0, price, adv, sigma, upsilon) + 1e-9 >= base
    assert sqrt_impact(qty, price, adv, sigma * 1.5, upsilon) + 1e-9 >= base
    assert sqrt_impact(qty, price, adv, sigma, upsilon * 1.5) + 1e-9 >= base
    assert sqrt_impact(0.0, price, adv, sigma, upsilon) == 0.0


@given(
    qty=st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
    price=_price,
    adv=_adv,
    sigma=_sigma,
    scale=st.floats(min_value=0.25, max_value=8.0, allow_nan=False, allow_infinity=False),
)
@adversarial_settings()
def test_dollar_impact_scales_when_price_and_adv_scale(
    qty: float,
    price: float,
    adv: float,
    sigma: float,
    scale: float,
) -> None:
    upsilon = 0.2
    base = sqrt_impact(qty, price, adv, max(sigma, 1e-6), upsilon)
    scaled = sqrt_impact(qty, price * scale, adv * scale, max(sigma, 1e-6), upsilon)
    assert scaled == pytest.approx(base * scale, rel=1e-9, abs=1e-8)


@given(
    qty=_qty,
    price=_price,
    adv=_adv,
    low=_rate,
    high=_rate,
)
@adversarial_settings()
def test_higher_commission_never_lowers_total_cost(
    qty: float,
    price: float,
    adv: float,
    low: float,
    high: float,
) -> None:
    assume(high >= low)
    cheap = total_cost(qty, price, adv, 0.2, _cost_config(commission_bps=low, impact_y=0.0))
    rich = total_cost(qty, price, adv, 0.2, _cost_config(commission_bps=high, impact_y=0.0))
    assert float(rich["total"]) + 1e-9 >= float(cheap["total"])


@given(qty=_qty, price=_price, adv=_adv)
@adversarial_settings()
def test_frictionless_cost_is_zero(qty: float, price: float, adv: float) -> None:
    cfg = _cost_config()
    cfg.frictionless = True
    out = total_cost(qty, price, adv, 0.2, cfg)
    assert float(out["total"]) == 0.0
    assert out["label"] == "FRICTIONLESS RESEARCH ONLY"


@given(
    quantity=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    participation=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
    volumes=st.lists(
        st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    ),
)
@adversarial_settings()
def test_pov_schedule_never_overfills(
    quantity: float,
    participation: float,
    volumes: list[float],
) -> None:
    forecast = np.asarray(volumes, dtype=float)
    schedule = pov_schedule(quantity, forecast, participation)
    cap = participation * forecast
    assert schedule.shape == forecast.shape
    assert np.all(schedule >= -1e-9)
    assert np.all(schedule <= cap + 1e-6)
    assert float(schedule.sum()) == pytest.approx(
        min(quantity, float(cap.sum())), rel=1e-9, abs=1e-6
    )


@given(
    price=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    premium=st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False),
)
@adversarial_settings()
def test_buy_slippage_is_nonnegative_when_paying_up(
    price: float,
    qty: float,
    premium: float,
) -> None:
    paid = price * (1.0 + premium)
    slip = vwap_slippage(np.array([paid]), np.array([qty]), price)
    assert slip == pytest.approx(premium, rel=1e-9, abs=1e-12)
    assert vwap_slippage(np.array([price]), np.array([qty]), price) == pytest.approx(0.0)


@given(
    decision=st.floats(min_value=1.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    drift=st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=0.2, max_value=50.0, allow_nan=False, allow_infinity=False),
    fee=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
)
@adversarial_settings()
def test_implementation_shortfall_adds_and_signs_the_buy(
    decision: float,
    drift: float,
    qty: float,
    fee: float,
) -> None:
    assume(decision * (1.0 + drift) > 0.0)
    exec_price = decision * (1.0 + drift)
    out = fill_shortfall(
        side_sign=1.0,
        quantity=qty,
        decision_price=decision,
        exec_price=exec_price,
        fee=fee,
    )
    assert out["total_is"] == pytest.approx(out["drift"] + out["explicit"])
    assert out["drift"] == pytest.approx((exec_price - decision) * qty)
    assert sqrt_impact_bps(0.0, 0.2, 0.5) == 0.0
