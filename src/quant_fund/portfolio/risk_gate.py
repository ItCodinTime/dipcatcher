"""Pre-trade risk gate. Deterministic. Alpha cannot override."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.orders import Order


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
) -> None:
    g = config.risk_gate
    notional = abs(order.quantity) * price
    if notional > g.max_order_notional:
        raise RiskGateRejected(f"order notional {notional} > {g.max_order_notional}")
    name_w = abs(current_weight + (order.quantity * price) / max(nav, 1e-12))
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
