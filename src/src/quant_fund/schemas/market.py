"""Market data schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Bar(BaseModel):
    security_id: str
    symbol: str
    event_time: datetime
    available_time: datetime
    ingested_time: datetime
    source: str
    revision_id: str = "v1"
    open: float
    high: float
    low: float
    close: float
    volume: float
    currency: str = "USD"
    session: str = "rth"


class CorporateAction(BaseModel):
    security_id: str
    event_time: datetime
    available_time: datetime
    ingested_time: datetime
    source: str
    revision_id: str = "v1"
    action_type: str
    factor: float | None = None
    amount: float | None = None
    new_ticker: str | None = None


class SecurityRecord(BaseModel):
    security_id: str
    ticker: str
    name: str
    exchange: str
    currency: str = "USD"
    sector: str = "Unknown"
    industry: str = "Unknown"
    security_type: str = "common_stock"
    valid_from: datetime
    valid_to: datetime | None = None


class UniverseMembership(BaseModel):
    security_id: str
    symbol: str
    effective_from: datetime
    effective_to: datetime | None = None
    sector: str
    industry: str
    exchange: str


class AdjustedBar(Bar):
    close_split_adjusted: float
    close_total_return: float
    open_split_adjusted: float
    high_split_adjusted: float
    low_split_adjusted: float
    split_factor: float = 1.0
    dividend: float = 0.0
