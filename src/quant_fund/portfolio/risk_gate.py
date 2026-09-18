"""Pre-trade risk gate. Deterministic. Alpha cannot override."""

from __future__ import annotations

import math

from quant_fund.config.models import AppConfig
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.orders import Order, OrderSide


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
) -> None:
    g = config.risk_gate
    values = (nav, price, current_weight, gross_after, net_after, participation, predicted_vol)
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
    if predicted_vol > g.max_predicted_vol:
        raise RiskGateRejected(f"predicted vol {predicted_vol} > {g.max_predicted_vol}")
