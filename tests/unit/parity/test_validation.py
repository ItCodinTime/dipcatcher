"""Fail-closed validation of parity diagnostics and simulation knobs."""

from __future__ import annotations

from dataclasses import replace

import pytest
from quant_fund.config.models import AppConfig

from quant_fund.parity.checker import attribute_pair
from quant_fund.parity.replay import ReplayOptions
from quant_fund.parity.session import MarketSession
from quant_fund.parity.shadow import run_shadow_session
from quant_fund.parity.shortfall import attribute_shortfall
from quant_fund.parity.strategy import FixedWeightStrategy


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume", "adv", "vol_20"])
def test_bar_rejects_infinite_numeric_fields(session: MarketSession, field: str) -> None:
    with pytest.raises(ValueError, match="must be a finite"):
        replace(session.bars[0], **{field: float("inf")})


@pytest.mark.parametrize("tol", [float("inf"), float("-inf"), float("nan"), -1.0])
def test_nonfinite_tolerance_cannot_hide_numeric_divergence(tol: float) -> None:
    with pytest.raises(ValueError, match="tol must be finite"):
        attribute_pair({"open": 1.0}, {"open": 2.0}, tol=tol)


@pytest.mark.parametrize(
    "options",
    [
        ReplayOptions(lot_size=float("inf")),
        ReplayOptions(initial_cash=float("inf")),
        ReplayOptions(pacing="wall_clock", speed=float("inf")),
        ReplayOptions(pacing="unsupported"),
    ],
)
def test_replay_refuses_invalid_options(
    session: MarketSession,
    strategy: FixedWeightStrategy,
    config: AppConfig,
    options: ReplayOptions,
) -> None:
    with pytest.raises(ValueError):
        run_shadow_session(session, strategy, config, options=options)


@pytest.mark.parametrize("value", [-1.0, float("inf")])
def test_cost_override_uses_model_validation(
    session: MarketSession,
    strategy: FixedWeightStrategy,
    config: AppConfig,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        run_shadow_session(
            session,
            strategy,
            config,
            options=ReplayOptions(cost_overrides={"commission_bps": value}),
        )


def test_infinite_injected_fill_price_is_rejected(
    session: MarketSession, strategy: FixedWeightStrategy, config: AppConfig
) -> None:
    def price(**_kwargs: object) -> float:
        return float("inf")

    with pytest.raises(ValueError, match="fill price must be finite"):
        run_shadow_session(session, strategy, config, options=ReplayOptions(fill_price=price))


def test_shortfall_refuses_inconsistent_explicit_cost() -> None:
    fill = {
        "event_time": "2026-01-01",
        "security_id": "SYNTHETIC_A",
        "signed_qty": 1.0,
        "price": 100.0,
        "spread": 2.0,
        "impact": 1.0,
        "explicit_cost": 1.0,
    }
    with pytest.raises(ValueError, match="must cover spread and impact"):
        attribute_shortfall([fill], [], {"SYNTHETIC_A": 100.0})


def test_arrival_shortfall_includes_charged_turnover_cost() -> None:
    fill = {
        "event_time": "2026-01-01",
        "security_id": "SYNTHETIC_A",
        "signed_qty": 1.0,
        "decision_price": 100.0,
        "price": 100.0,
        "fee": 1.0,
        "spread": 2.0,
        "impact": 3.0,
        "explicit_cost": 7.0,
    }
    result = attribute_shortfall([], [fill], {"SYNTHETIC_A": 100.0})
    arrival = result["paper_arrival_shortfall"]
    assert arrival["fee"] == 2.0
    assert arrival["spread_cost"] == 2.0
    assert arrival["impact_cost"] == 3.0
    assert arrival["total_is"] == 7.0
