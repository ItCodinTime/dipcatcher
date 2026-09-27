"""Versioned, schema-validated pre-trade configuration and its signed hash.

The signed payload is canonical JSON of the validated model (sorted keys,
tight separators). Every decision records the SHA-256 of that payload and
an HMAC-SHA256 over the same bytes. The HMAC key is not part of the file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LimitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_order_notional: float = Field(gt=0.0)
    max_order_quantity: float = Field(gt=0.0)
    price_collar_bps: float = Field(ge=0.0)
    max_position_notional: float = Field(gt=0.0)
    max_position_quantity: float = Field(gt=0.0)
    max_gross_notional: float = Field(gt=0.0)
    max_net_notional: float = Field(gt=0.0)
    max_name_concentration: float = Field(gt=0.0, le=1.0)
    max_daily_loss_fraction: float = Field(gt=0.0, le=1.0)
    max_trailing_drawdown_fraction: float = Field(gt=0.0, le=1.0)
    max_orders_per_window: int = Field(gt=0, le=4096)
    max_messages_per_window: int = Field(gt=0, le=4096)
    rate_window_ns: int = Field(gt=0)
    duplicate_window_ns: int = Field(gt=0)
    max_reference_age_ns: int = Field(gt=0)
    max_mark_age_ns: int = Field(gt=0)
    pdt_equity_threshold: float = Field(gt=0.0)
    pdt_window_sessions: int = Field(ge=1, le=30)
    pdt_max_day_trades: int = Field(ge=0, le=100)
    settlement_ns: int = Field(ge=0)
    qty_tick: float = Field(gt=0.0)
    price_tick: float = Field(gt=0.0)
    cash_account: bool = True
    restrict_to_settled_cash: bool = True
    max_lots_per_symbol: int = Field(default=8, ge=1, le=32)
    ring_capacity: int = Field(default=256, ge=16, le=4096)
    symbol_capacity: int = Field(default=256, ge=1, le=8192)
    day_trade_capacity: int = Field(default=32, ge=4, le=256)
    unsettled_capacity: int = Field(default=32, ge=4, le=256)

    @field_validator("ring_capacity", "symbol_capacity", "day_trade_capacity", "unsettled_capacity")
    @classmethod
    def _pow2(cls, value: int) -> int:
        if value & (value - 1):
            raise ValueError("capacity must be a power of two")
        return value

    @model_validator(mode="after")
    def _rings_hold_the_window(self) -> LimitConfig:
        if self.max_orders_per_window >= self.ring_capacity:
            raise ValueError("max_orders_per_window must be < ring_capacity")
        if self.max_messages_per_window >= self.ring_capacity:
            raise ValueError("max_messages_per_window must be < ring_capacity")
        return self


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str
    open_minute: int = Field(ge=0, lt=24 * 60)
    close_minute: int = Field(gt=0, le=24 * 60)

    @field_validator("timezone")
    @classmethod
    def _timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value

    @model_validator(mode="after")
    def _session_order(self) -> SessionConfig:
        if self.open_minute >= self.close_minute:
            raise ValueError("session open_minute must be < close_minute")
        return self


class PretradeConfig(BaseModel):
    """Schema version 1. Unknown versions are rejected."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    limits: LimitConfig
    session: SessionConfig


def canonical_config_bytes(config: PretradeConfig) -> bytes:
    payload: Any = config.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sign_config(config: PretradeConfig, hmac_key: bytes) -> tuple[str, str]:
    """Return ``(sha256_hex, hmac_sha256_hex)`` for the canonical payload."""
    if not isinstance(hmac_key, bytes) or len(hmac_key) < 16:
        raise ValueError("hmac_key must be at least 16 bytes")
    blob = canonical_config_bytes(config)
    digest = hashlib.sha256(blob).hexdigest()
    signature = hmac.new(hmac_key, blob, hashlib.sha256).hexdigest()
    return digest, signature


def load_pretrade_config(path: str | Path, *, hmac_key: bytes) -> tuple[PretradeConfig, str, str]:
    """Load YAML, validate the schema, and sign the canonical payload.

    Fails closed: unsafe YAML constructors are not used, extra keys are
    rejected, and a short HMAC key is rejected.
    """
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("pretrade config must be a YAML mapping")
    config = PretradeConfig.model_validate(raw)
    digest, signature = sign_config(config, hmac_key)
    return config, digest, signature
