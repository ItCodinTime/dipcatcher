"""Order and fill schemas. Signal, decision, order, and fill times are distinct."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    NEW = "new"
    ACKED = "acked"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class Order(BaseModel):
    order_id: str
    security_id: str
    symbol: str
    side: OrderSide
    quantity: float
    signal_time: datetime
    decision_time: datetime
    order_time: datetime
    status: OrderStatus = OrderStatus.NEW
    parent_id: str | None = None
    participation_rate: float | None = None
    limit_price: float | None = None


class Fill(BaseModel):
    fill_id: str
    order_id: str
    security_id: str
    quantity: float
    price: float
    fill_time: datetime
    fee: float = 0.0
    spread_cost: float = 0.0
    impact_cost: float = 0.0
    slippage: float = 0.0
    is_partial: bool = False
