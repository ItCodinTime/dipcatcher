"""Pre-trade risk gate. Deterministic. Alpha cannot override."""

from __future__ import annotations

import math

from quant_fund.config.models import AppConfig
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.orders import Order, OrderSide


def resolve_gate_predicted_vol(name_vol: float, market_vol: float | None) -> float:
    """Vol compared to ``max_predicted_vol``.

    A present date-level market overlay (Realized GARCH Parkinson when
    ``vol_realized_garch.joblib`` is available, else return-only GARCH) replaces
    name-level ``vol_20`` for the gate. Impact/cost models keep the name vol.
    ``market_vol is None`` keeps the legacy name-vol comparison.
    """
    if market_vol is None:
        return float(name_vol)
    return float(market_vol)


def check_order(
    order: Order,
    *,
    nav: float,
    price: float,
    current_weight: float,
    gross_after: float,
    net_after: float,
    participation: float,
    predicted_vol: float,
    config: AppConfig,
    price_age_bars: int | None = None,
    model_age_hours: float | None = None,
    market_predicted_vol: float | None = None,
) -> None:
    g = config.risk_gate
    values: list[float] = [
        nav,
        price,
        current_weight,
        gross_after,
        net_after,
        participation,
        predicted_vol,
    ]
    if market_predicted_vol is not None:
        values.append(market_predicted_vol)
    if any(not math.isfinite(float(value)) for value in values):
        raise RiskGateRejected("non-finite pre-trade risk input")
    if nav <= 0.0 or price <= 0.0:
        raise RiskGateRejected("nav and price must be positive")
    if gross_after < 0.0:
        raise RiskGateRejected("gross exposure cannot be negative")
    if participation < 0.0:
        raise RiskGateRejected("participation cannot be negative")
    if predicted_vol < 0.0:
        raise RiskGateRejected("predicted volatility cannot be negative")
    if market_predicted_vol is not None and market_predicted_vol < 0.0:
        raise RiskGateRejected("predicted volatility cannot be negative")
    if not math.isfinite(float(order.quantity)) or order.quantity <= 0.0:
        raise RiskGateRejected("quantity must be finite and strictly positive")
    if price_age_bars is not None:
        if price_age_bars < 0:
            raise RiskGateRejected("price age cannot be negative")
        if price_age_bars > g.stale_price_bars:
            raise RiskGateRejected(f"price is stale: {price_age_bars} bars")
    if model_age_hours is not None:
        if model_age_hours < 0.0:
            raise RiskGateRejected("model age cannot be negative")
        if model_age_hours > g.stale_model_hours:
            raise RiskGateRejected(f"model is stale: {model_age_hours} hours")
    notional = abs(order.quantity) * price
    if notional > g.max_order_notional:
        raise RiskGateRejected(f"order notional {notional} > {g.max_order_notional}")
    signed_qty = float(order.quantity)
    if order.side is OrderSide.SELL:
        signed_qty = -signed_qty
    name_w = abs(current_weight + (signed_qty * price) / max(nav, 1e-12))
    if name_w > g.max_name:
        raise RiskGateRejected(f"name weight {name_w} > {g.max_name}")
    if gross_after > g.max_gross:
        raise RiskGateRejected(f"gross {gross_after} > {g.max_gross}")
    if abs(net_after) > g.max_net:
        raise RiskGateRejected(f"net {net_after} > {g.max_net}")
    if participation > g.max_participation:
        raise RiskGateRejected(f"participation {participation} > {g.max_participation}")
    gate_vol = resolve_gate_predicted_vol(predicted_vol, market_predicted_vol)
    if gate_vol > g.max_predicted_vol:
        raise RiskGateRejected(f"predicted vol {gate_vol} > {g.max_predicted_vol}")
