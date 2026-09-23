"""Order book (L2) schemas for candle+LOB research.

Research-only market-structure contracts. Snapshots are point-in-time at
``event_time`` / ``available_time``; never a live fill claim.
"""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class BookLevel(BaseModel):
    """One price level on one side of the book."""

    price: float
    size: float

    @field_validator("price", "size")
    @classmethod
    def _finite_positive(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("price and size must be finite and > 0")
        return float(value)


class OrderBookSnapshot(BaseModel):
    """L2 snapshot aligned to a bar/candle timestamp.

    Bids are best→worse (**strictly** descending price). Asks are best→worse
    (**strictly** ascending). Sizes and prices are finite and > 0.

    Depth honesty: ``depth`` / side lengths ≥ 2 are required for finite
    size slopes, log-price slopes, and mean log tick spacings in book
    metrics. Empty sides and crossed/locked books (best bid ≥ best ask)
    fail closed here. ``book_metrics_from_snapshot`` mirrors
    ``best_bid`` / ``best_ask`` / ``mid`` (== 0.5·(bid+ask)) /
    ``spread`` / ``spread_bps`` identities; metrics ``bid_depth`` /
    ``ask_depth`` sum side sizes.
    """

    security_id: str
    symbol: str = ""
    event_time: datetime
    available_time: datetime
    ingested_time: datetime | None = None
    source: str = "synthetic"
    revision_id: str = "v1"
    bids: list[BookLevel] = Field(default_factory=list)
    asks: list[BookLevel] = Field(default_factory=list)
    depth: int = 5

    @field_validator("security_id", "revision_id", "source")
    @classmethod
    def _non_empty_str(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("security_id, revision_id, and source must be non-empty strings")
        return value.strip()

    @field_validator("event_time", "available_time", "ingested_time")
    @classmethod
    def _tz_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time/available_time/ingested_time must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_book(self) -> OrderBookSnapshot:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if not self.bids or not self.asks:
            raise ValueError("bids and asks must be non-empty")
        if len(self.bids) > self.depth or len(self.asks) > self.depth:
            raise ValueError("side length exceeds declared depth")
        bid_px = [level.price for level in self.bids]
        ask_px = [level.price for level in self.asks]
        if any(bid_px[i] <= bid_px[i + 1] for i in range(len(bid_px) - 1)):
            raise ValueError("bids must be strictly descending price (best-to-worse, no ties)")
        if any(ask_px[i] >= ask_px[i + 1] for i in range(len(ask_px) - 1)):
            raise ValueError("asks must be strictly ascending price (best-to-worse, no ties)")
        if bid_px[0] >= ask_px[0]:
            raise ValueError("crossed or locked book: best bid must be < best ask")
        if self.available_time < self.event_time:
            raise ValueError("available_time must be >= event_time")
        if self.ingested_time is not None and self.ingested_time < self.available_time:
            raise ValueError("ingested_time must be >= available_time")
        if len(bid_px) != len(set(bid_px)) or len(ask_px) != len(set(ask_px)):
            raise ValueError("prices must be unique on each side")
        return self

    @property
    def best_bid(self) -> float:
        return self.bids[0].price

    @property
    def best_ask(self) -> float:
        return self.asks[0].price

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def half_spread(self) -> float:
        return 0.5 * self.spread

    @property
    def effective_spread(self) -> float:
        """Alias for ``spread`` (best_ask − best_bid)."""
        return self.spread

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return 0.0 if mid <= 0.0 else 1e4 * self.spread / mid
