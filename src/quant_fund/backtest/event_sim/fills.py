"""Fill models: next-open price, VWAP over a window, and L2 queue position.

Fills never exceed the volume or displayed size passed in. Queue advancement
is deterministic. Cancel/replace resets queue position when the touch moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot


@dataclass(frozen=True)
class BookFill:
    filled: float
    vwap: float
    displayed: float

    @property
    def exhausted_book(self) -> bool:
        return self.filled + 1e-12 >= self.displayed


@dataclass(frozen=True)
class QueueStep:
    filled: float
    queue_ahead: float
    limit_price: float
    replaced: bool


def walk_book(side: str, quantity: float, levels: list[tuple[float, float]]) -> BookFill:
    """Take displayed liquidity from best to worse. ``side`` is ``buy`` or ``sell``."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    qty = float(quantity)
    if not math.isfinite(qty) or qty < 0.0:
        raise ValueError("quantity must be finite and non-negative")
    displayed = 0.0
    for price, size in levels:
        if not math.isfinite(price) or price <= 0.0 or not math.isfinite(size) or size < 0.0:
            raise ValueError("book levels must have a positive price and non-negative size")
        displayed += float(size)
    remaining = qty
    notional = 0.0
    filled = 0.0
    for price, size in levels:
        if remaining <= 1e-15:
            break
        take = min(remaining, float(size))
        filled += take
        notional += take * float(price)
        remaining -= take
    if filled > displayed + 1e-9:
        raise RuntimeError("book walk filled more than displayed size")
    vwap = 0.0 if filled <= 0.0 else notional / filled
    return BookFill(filled=float(filled), vwap=float(vwap), displayed=float(displayed))


def levels_for(snapshot: OrderBookSnapshot, side: str) -> list[tuple[float, float]]:
    book = snapshot.asks if side == "buy" else snapshot.bids
    return [(float(level.price), float(level.size)) for level in book]


def touch(snapshot: OrderBookSnapshot, side: str) -> tuple[float, float]:
    level: BookLevel = snapshot.asks[0] if side == "buy" else snapshot.bids[0]
    return float(level.price), float(level.size)


def advance_queue(
    *,
    side: str,
    remaining: float,
    limit_price: float,
    queue_ahead: float,
    touch_price: float,
    touch_size: float,
    traded_volume: float,
) -> QueueStep:
    """FIFO queue at the touch.

    A price change cancel/replaces to the new touch and joins the back
    (``queue_ahead`` becomes the new displayed size) with no fill on that
    step. Otherwise prints consume the queue, and any fill is capped by the
    order, by traded volume, and by displayed touch size.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    for label, value in (
        ("remaining", remaining),
        ("limit_price", limit_price),
        ("queue_ahead", queue_ahead),
        ("touch_price", touch_price),
        ("touch_size", touch_size),
        ("traded_volume", traded_volume),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if remaining < 0.0 or queue_ahead < 0.0 or touch_size < 0.0 or traded_volume < 0.0:
        raise ValueError("sizes must be non-negative")
    if limit_price <= 0.0 or touch_price <= 0.0:
        raise ValueError("prices must be positive")
    if remaining == 0.0 or touch_size == 0.0:
        return QueueStep(0.0, float(queue_ahead), float(limit_price), False)
    if abs(float(touch_price) - float(limit_price)) > 1e-9:
        return QueueStep(0.0, float(touch_size), float(touch_price), True)
    room = float(touch_size)
    ahead = float(queue_ahead)
    consumed = min(float(traded_volume), ahead + room)
    if consumed <= ahead:
        return QueueStep(0.0, ahead - consumed, float(limit_price), False)
    available = min(room, consumed - ahead, float(traded_volume))
    filled = min(float(remaining), max(0.0, available))
    if filled > room + 1e-9 or filled > float(traded_volume) + 1e-9:
        raise RuntimeError("queue fill exceeded displayed or traded volume")
    return QueueStep(float(filled), 0.0, float(limit_price), False)


def bar_vwap_price(row: dict[str, object]) -> float | None:
    """Volume-weighted proxy for one bar: explicit ``vwap``, else typical price."""

    def _pos(key: str) -> float | None:
        value = row.get(key)
        if value is None:
            return None
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0.0:
            return None
        return number

    explicit = _pos("vwap")
    if explicit is not None:
        return explicit
    high, low, close = _pos("high"), _pos("low"), _pos("close")
    if high is not None and low is not None and close is not None and high >= low:
        return (high + low + close) / 3.0
    open_, close = _pos("open"), _pos("close")
    if open_ is not None and close is not None:
        return (open_ + close) / 2.0
    return open_ or close


def allocate_vwap(
    quantity: float,
    window: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Split ``quantity`` across ``(volume, price)`` bars without exceeding volume.

    Returned quantities are signed like ``quantity``. Empty when nothing trades.
    """
    qty = float(quantity)
    if not math.isfinite(qty):
        raise ValueError("quantity must be finite")
    if qty == 0.0:
        return []
    sign = 1.0 if qty > 0.0 else -1.0
    remaining = abs(qty)
    fills: list[tuple[float, float]] = []
    for volume, price in window:
        vol = float(volume)
        px = float(price)
        if not math.isfinite(vol) or not math.isfinite(px) or px <= 0.0 or vol < 0.0:
            raise ValueError("window volume and price must be finite; price > 0; volume >= 0")
        if remaining <= 1e-15 or vol <= 0.0:
            continue
        take = min(remaining, vol)
        if take > vol + 1e-9:
            raise RuntimeError("vwap slice exceeded bar volume")
        fills.append((sign * take, px))
        remaining -= take
    return fills
