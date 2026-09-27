"""Fail-closed fixtures for core pydantic / PIT schemas (Wave 30).

PIT assert_pit_safe lookahead is already covered in test_data_pit / test_leakage /
test_config — here we add thin Order/Forecast/error hierarchy edges only.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from quant_fund.schemas.errors import (
    ConfigError,
    DataContractError,
    KillSwitchActive,
    LeakageError,
    OptimizationInfeasible,
    PointInTimeError,
    QuantFundError,
    RiskGateRejected,
)
from quant_fund.schemas.forecast import AssetForecast, MarketState
from quant_fund.schemas.orders import Fill, Order, OrderSide, OrderStatus
from quant_fund.schemas.pit import FeatureIntegrity, assert_pit_safe


def test_asset_forecast_requires_identity_fields() -> None:
    with pytest.raises(ValidationError):
        AssetForecast()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        AssetForecast(security_id="A", symbol="A")  # type: ignore[call-arg]


def test_asset_forecast_rejects_bad_interval_method() -> None:
    asof = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError):
        AssetForecast(
            security_id="A",
            symbol="A",
            asof=asof,
            model_version="v1",
            interval_method="not_a_method",  # type: ignore[arg-type]
        )


def test_asset_forecast_rejects_invalid_interval_contracts() -> None:
    asof = datetime(2020, 1, 2, tzinfo=UTC)
    base = dict(security_id="A", symbol="A", asof=asof, model_version="v1")
    with pytest.raises(ValidationError, match="interval_alpha"):
        AssetForecast(**base, interval_alpha=0.0)
    with pytest.raises(ValidationError, match="matching keys"):
        AssetForecast(**base, interval_lo={"1d": -1.0}, interval_alpha=0.1)
    with pytest.raises(ValidationError, match="finite and ordered"):
        AssetForecast(
            **base,
            interval_lo={"1d": 2.0},
            interval_hi={"1d": 1.0},
            interval_alpha=0.1,
        )


@pytest.mark.parametrize(
    "field", ["probability_positive", "regime_probabilities", "drawdown_probabilities"]
)
def test_asset_forecast_rejects_invalid_probability_maps(field: str) -> None:
    asof = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="probability values"):
        AssetForecast(
            security_id="A",
            symbol="A",
            asof=asof,
            model_version="v1",
            **{field: {"1d": 1.1}},
        )


@pytest.mark.parametrize(
    "quantiles",
    [
        {"5d": {0.0: 0.0}},
        {"5d": {1.0: 0.0}},
        {"5d": {float("nan"): 0.0}},
        {"5d": {0.5: float("nan")}},
    ],
)
def test_asset_forecast_rejects_invalid_quantile_contract(
    quantiles: dict[str, dict[float, float]],
) -> None:
    asof = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="quantile levels and values"):
        AssetForecast(
            security_id="A",
            symbol="A",
            asof=asof,
            model_version="v1",
            quantiles=quantiles,
        )


def test_market_state_requires_asof_and_forecasts_list() -> None:
    with pytest.raises(ValidationError):
        MarketState()  # type: ignore[call-arg]
    asof = datetime(2020, 1, 2, tzinfo=UTC)
    state = MarketState(asof=asof, forecasts=[])
    assert state.forecasts == []
    assert state.notes == []


@pytest.mark.parametrize("condition_number", [0.0, -1.0, float("nan"), float("inf")])
def test_market_state_rejects_invalid_covariance_condition_number(condition_number: float) -> None:
    asof = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="covariance_condition_number"):
        MarketState(
            asof=asof,
            forecasts=[],
            covariance_condition_number=condition_number,
        )


def test_order_requires_core_fields_and_side_enum() -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError):
        Order()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Order(
            order_id="o1",
            security_id="A",
            symbol="A",
            side="hold",  # type: ignore[arg-type]
            quantity=1.0,
            signal_time=t,
            decision_time=t,
            order_time=t,
        )
    ok = Order(
        order_id="o1",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=10.0,
        signal_time=t,
        decision_time=t,
        order_time=t,
    )
    assert ok.status == OrderStatus.NEW


@pytest.mark.parametrize("quantity", [0.0, -1.0, float("nan"), float("inf")])
def test_order_quantity_is_finite_and_strictly_positive(quantity: float) -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="quantity must be finite and strictly positive"):
        Order(
            order_id="o1",
            security_id="A",
            symbol="A",
            side=OrderSide.BUY,
            quantity=quantity,
            signal_time=t,
            decision_time=t,
            order_time=t,
        )


@pytest.mark.parametrize("participation_rate", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_order_participation_rate_is_bounded(participation_rate: float) -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match=r"participation_rate must be finite and in \(0, 1\]"):
        Order(
            order_id="o1",
            security_id="A",
            symbol="A",
            side=OrderSide.BUY,
            quantity=1.0,
            signal_time=t,
            decision_time=t,
            order_time=t,
            participation_rate=participation_rate,
        )


@pytest.mark.parametrize("limit_price", [0.0, -1.0, float("nan"), float("inf")])
def test_order_limit_price_is_finite_and_strictly_positive(limit_price: float) -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="limit_price must be finite and strictly positive"):
        Order(
            order_id="o1",
            security_id="A",
            symbol="A",
            side=OrderSide.BUY,
            quantity=1.0,
            signal_time=t,
            decision_time=t,
            order_time=t,
            limit_price=limit_price,
        )


def test_order_timestamps_are_causally_monotone() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    with pytest.raises(ValidationError, match="signal_time <= decision_time <= order_time"):
        Order(
            order_id="o1",
            security_id="A",
            symbol="A",
            side=OrderSide.BUY,
            quantity=1.0,
            signal_time=t1,
            decision_time=t0,
            order_time=t1,
        )
    with pytest.raises(ValidationError, match="signal_time <= decision_time <= order_time"):
        Order(
            order_id="o2",
            security_id="A",
            symbol="A",
            side=OrderSide.BUY,
            quantity=1.0,
            signal_time=t0,
            decision_time=t1,
            order_time=t0,
        )


def test_order_timestamps_reject_mixed_timezone_awareness() -> None:
    naive = datetime(2020, 1, 2)
    aware = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="share timezone awareness"):
        Order(
            order_id="o1",
            security_id="A",
            symbol="A",
            side=OrderSide.BUY,
            quantity=1.0,
            signal_time=naive,
            decision_time=aware,
            order_time=aware,
        )


def test_fill_requires_core_fields() -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError):
        Fill()  # type: ignore[call-arg]
    fill = Fill(
        fill_id="f1",
        order_id="o1",
        security_id="A",
        quantity=1.0,
        price=100.0,
        fill_time=t,
    )
    assert fill.fee == 0.0
    assert fill.is_partial is False


@pytest.mark.parametrize("quantity", [0.0, -1.0, float("nan"), float("inf")])
def test_fill_quantity_is_finite_and_strictly_positive(quantity: float) -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="quantity must be finite and strictly positive"):
        Fill(
            fill_id="f1",
            order_id="o1",
            security_id="A",
            quantity=quantity,
            price=100.0,
            fill_time=t,
        )


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_fill_price_is_finite_and_strictly_positive(price: float) -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="price must be finite and strictly positive"):
        Fill(
            fill_id="f1",
            order_id="o1",
            security_id="A",
            quantity=1.0,
            price=price,
            fill_time=t,
        )


@pytest.mark.parametrize("field", ["fee", "spread_cost", "impact_cost", "slippage"])
def test_fill_costs_are_finite_and_nonnegative(field: str) -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(ValidationError, match="fill costs must be finite and non-negative"):
        Fill(
            fill_id="f1",
            order_id="o1",
            security_id="A",
            quantity=1.0,
            price=100.0,
            fill_time=t,
            **{field: -1.0},
        )


def test_feature_integrity_lookahead_fail_closed() -> None:
    decision = datetime(2020, 1, 2, tzinfo=UTC)
    future = datetime(2020, 1, 3, tzinfo=UTC)
    with pytest.raises(PointInTimeError):
        FeatureIntegrity(decision_time=decision, max_source_available_time=future)
    with pytest.raises(PointInTimeError):
        assert_pit_safe(future, decision)
    # equal / earlier available is OK
    FeatureIntegrity(decision_time=decision, max_source_available_time=decision)
    assert_pit_safe(decision, decision)


def test_domain_error_hierarchy() -> None:
    assert issubclass(PointInTimeError, QuantFundError)
    assert issubclass(LeakageError, PointInTimeError)
    assert issubclass(OptimizationInfeasible, QuantFundError)
    assert issubclass(RiskGateRejected, QuantFundError)
    assert issubclass(KillSwitchActive, QuantFundError)
    assert issubclass(ConfigError, QuantFundError)
    assert issubclass(DataContractError, QuantFundError)
