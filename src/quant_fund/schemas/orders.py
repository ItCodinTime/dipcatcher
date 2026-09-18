"""Order and fill schemas. Signal, decision, order, and fill times are distinct."""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, field_validator, model_validator


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

    @model_validator(mode="after")
    def validate_event_order(self) -> Order:
        """Keep the causal event chain monotone and timezone-consistent."""
        times = (self.signal_time, self.decision_time, self.order_time)
        aware = tuple(value.tzinfo is not None for value in times)
        if len(set(aware)) != 1:
            raise ValueError("order timestamps must share timezone awareness")
        if not self.signal_time <= self.decision_time <= self.order_time:
            raise ValueError(
                "order timestamps must satisfy signal_time <= decision_time <= order_time"
            )
        return self

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, value: float) -> float:
        """Orders are signed by ``side``; quantity itself must be positive."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("quantity must be finite and strictly positive")
        return value

    @field_validator("participation_rate")
    @classmethod
    def validate_participation_rate(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0.0 < value <= 1.0):
            raise ValueError("participation_rate must be finite and in (0, 1]")
        return value

    @field_validator("limit_price")
    @classmethod
    def validate_limit_price(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("limit_price must be finite and strictly positive")
        return value


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

    @field_validator("price")
    @classmethod
    def validate_price(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("price must be finite and strictly positive")
        return value

    @field_validator("fee", "spread_cost", "impact_cost", "slippage")
    @classmethod
    def validate_cost(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("fill costs must be finite and non-negative")
        return value

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, value: float) -> float:
        """Fill quantity is unsigned; direction belongs to the parent order."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("quantity must be finite and strictly positive")
        return value
