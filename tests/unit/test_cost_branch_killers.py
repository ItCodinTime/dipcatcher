"""Example tests aimed at surviving mutants in ``execution/costs.py``."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from quant_fund.config.models import CostConfig
from quant_fund.execution.costs import (
    commission_cost,
    half_spread_cost,
    sqrt_impact,
    total_cost,
)

_PRICE = "quantity must be finite and price must be finite and positive"
_ADV = "adv_dollars must be finite and positive"
_UPS = "upsilon must be finite and non-negative"
_SIG = "sigma must be finite and non-negative"
_SPREAD = "notional must be finite and half_spread_bps must be finite and non-negative"
_COMM = "notional must be finite and commission_bps must be finite and non-negative"


def _exact(message: str) -> str:
    """The whole message, so an XX-wrapped mutant does not still match."""
    return rf"^{re.escape(message)}$"


def _raw(**overrides: float | bool) -> CostConfig:
    base: dict[str, float | bool] = {
        "commission_bps": 0.0,
        "half_spread_bps": 0.0,
        "impact_y": 0.0,
        "bps_per_turnover": 0.0,
        "borrow_bps_per_year": 0.0,
        "financing_bps_per_year": 0.0,
        "frictionless": False,
        "participation_limit": 1.0,
    }
    base.update(overrides)
    return CostConfig.model_construct(**base)


def test_scalar_costs_match_the_closed_form_including_sign() -> None:
    assert half_spread_cost(10_000.0, 5.0) == 5.0
    assert half_spread_cost(-10_000.0, 5.0) == 5.0
    assert half_spread_cost(0.0, 5.0) == 0.0
    assert half_spread_cost(10_000.0, 0.0) == 0.0
    assert commission_cost(10_000.0, 2.5) == 2.5
    assert commission_cost(-10_000.0, 2.5) == 2.5
    assert commission_cost(0.0, 2.5) == 0.0
    assert commission_cost(10_000.0, 0.0) == 0.0


def test_scalar_costs_name_the_invalid_input() -> None:
    with pytest.raises(ValueError, match=_exact(_SPREAD)):
        half_spread_cost(float("nan"), 1.0)
    with pytest.raises(ValueError, match=_exact(_SPREAD)):
        half_spread_cost(1.0, float("inf"))
    with pytest.raises(ValueError, match=_exact(_SPREAD)):
        half_spread_cost(1.0, -0.0 - 1e-12)
    with pytest.raises(ValueError, match=_exact(_COMM)):
        commission_cost(float("-inf"), 1.0)
    with pytest.raises(ValueError, match=_exact(_COMM)):
        commission_cost(1.0, -1.0)


def test_sqrt_impact_closed_form_and_zero_shortcuts() -> None:
    # participation = |q|*price / adv; impact = notional * Y * sigma * sqrt(participation)
    got = sqrt_impact(50.0, 4.0, 10_000.0, 0.25, 0.2)
    notional = 200.0
    participation = notional / 10_000.0
    assert got == notional * 0.2 * 0.25 * math.sqrt(participation)
    assert sqrt_impact(-50.0, 4.0, 10_000.0, 0.25, 0.2) == got
    assert sqrt_impact(0.0, 4.0, 10_000.0, 0.25, 0.2) == 0.0
    assert sqrt_impact(50.0, 4.0, 10_000.0, 0.0, 0.2) == 0.0
    assert sqrt_impact(50.0, 4.0, 10_000.0, 0.25, 0.0) == 0.0


def test_sqrt_impact_lifts_a_sub_ulp_adv_floor() -> None:
    """ADV below 1e-12 is floored, so the floor constant is load-bearing."""
    tiny = sqrt_impact(1.0, 1.0, 1e-18, 1.0, 1.0)
    floored = sqrt_impact(1.0, 1.0, 1e-12, 1.0, 1.0)
    assert tiny == floored
    assert tiny == pytest.approx(1.0 * math.sqrt(1.0 / 1e-12))


def test_sqrt_impact_names_each_invalid_argument() -> None:
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        sqrt_impact(float("nan"), 1.0, 10.0, 0.1, 0.1)
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        sqrt_impact(1.0, float("nan"), 10.0, 0.1, 0.1)
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        sqrt_impact(1.0, float("inf"), 10.0, 0.1, 0.1)
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        sqrt_impact(1.0, 0.0, 10.0, 0.1, 0.1)
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        sqrt_impact(1.0, -2.0, 10.0, 0.1, 0.1)
    with pytest.raises(ValueError, match=_exact(_ADV)):
        sqrt_impact(1.0, 1.0, 0.0, 0.1, 0.1)
    with pytest.raises(ValueError, match=_exact(_ADV)):
        sqrt_impact(1.0, 1.0, float("nan"), 0.1, 0.1)
    with pytest.raises(ValueError, match=_exact(_UPS)):
        sqrt_impact(1.0, 1.0, 10.0, 0.1, -0.1)
    with pytest.raises(ValueError, match=_exact(_UPS)):
        sqrt_impact(1.0, 1.0, 10.0, 0.1, float("inf"))
    with pytest.raises(ValueError, match=_exact(_SIG)):
        sqrt_impact(1.0, 1.0, 10.0, -0.1, 0.1)
    with pytest.raises(ValueError, match=_exact(_SIG)):
        sqrt_impact(1.0, 1.0, 10.0, float("nan"), 0.1)


def test_total_cost_parts_and_labels() -> None:
    cfg = _raw(commission_bps=10.0, half_spread_bps=20.0, impact_y=0.0, bps_per_turnover=5.0)
    out = total_cost(-100.0, 2.0, 1e6, 0.0, cfg)
    assert out["commission"] == 0.2
    assert out["spread"] == 0.4
    assert out["impact"] == 0.0
    assert out["turnover_bps"] == 0.1
    assert out["borrow"] == 0.0
    assert out["total"] == pytest.approx(0.7)
    assert out["label"] == "costed"
    free = total_cost(100.0, 2.0, 1e6, 0.2, _raw(frictionless=True, commission_bps=10.0))
    assert free == {
        "commission": 0.0,
        "spread": 0.0,
        "impact": 0.0,
        "borrow": 0.0,
        "turnover_bps": 0.0,
        "total": 0.0,
        "label": "FRICTIONLESS RESEARCH ONLY",
    }


def test_total_cost_rejects_bad_market_and_each_coefficient() -> None:
    ok = _raw()
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        total_cost(float("nan"), 1.0, 10.0, 0.1, ok)
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        total_cost(1.0, float("nan"), 10.0, 0.1, ok)
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        total_cost(1.0, float("inf"), 10.0, 0.1, ok)
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        total_cost(1.0, 0.0, 10.0, 0.1, ok)
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        total_cost(1.0, -1.0, 10.0, 0.1, ok)
    with pytest.raises(ValueError, match=_exact(_ADV)):
        total_cost(1.0, 1.0, -5.0, 0.1, ok)
    with pytest.raises(ValueError, match=_exact(_ADV)):
        total_cost(1.0, 1.0, 0.0, 0.1, ok)
    with pytest.raises(ValueError, match=_exact(_ADV)):
        total_cost(1.0, 1.0, float("nan"), 0.1, ok)
    with pytest.raises(ValueError, match=_exact(_SIG)):
        total_cost(1.0, 1.0, 10.0, -0.2, ok)
    with pytest.raises(ValueError, match=_exact(_SIG)):
        total_cost(1.0, 1.0, 10.0, float("nan"), ok)
    # Sub-dollar ADV is still a valid positive print; only non-positive ADV fails.
    assert total_cost(1.0, 1.0, 0.5, 0.0, ok)["total"] == 0.0
    for name in ("commission_bps", "half_spread_bps", "impact_y", "bps_per_turnover"):
        message = f"{name} must be finite and non-negative"
        with pytest.raises(ValueError, match=_exact(message)):
            total_cost(1.0, 1.0, 10.0, 0.1, _raw(**{name: -1.0}))
        with pytest.raises(ValueError, match=_exact(message)):
            total_cost(1.0, 1.0, 10.0, 0.1, _raw(**{name: float("nan")}))


def test_frictionless_still_validates_before_the_zero_return() -> None:
    free = _raw(frictionless=True)
    with pytest.raises(ValueError, match=_exact(_PRICE)):
        total_cost(1.0, 0.0, 10.0, 0.1, free)
    with pytest.raises(ValueError, match=_exact(_ADV)):
        total_cost(1.0, 1.0, 0.0, 0.1, free)
    with pytest.raises(ValueError, match=_exact(_ADV)):
        total_cost(1.0, 1.0, float("nan"), 0.1, free)
    with pytest.raises(ValueError, match=_exact(_SIG)):
        total_cost(1.0, 1.0, 10.0, float("nan"), free)
    with pytest.raises(ValueError, match=_exact("commission_bps must be finite and non-negative")):
        total_cost(1.0, 1.0, 10.0, 0.1, _raw(frictionless=True, commission_bps=float("inf")))


def test_impact_inside_total_cost_matches_sqrt_impact() -> None:
    cfg = _raw(impact_y=0.3, commission_bps=0.0, half_spread_bps=0.0, bps_per_turnover=0.0)
    out = total_cost(8.0, 5.0, 2_000.0, 0.4, cfg)
    assert out["impact"] == sqrt_impact(8.0, 5.0, 2_000.0, 0.4, 0.3)
    assert out["total"] == out["impact"]
    assert np.isfinite(float(out["total"]))
