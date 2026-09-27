"""Shared builders for pre-trade tests. Not collected as tests."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from quant_fund.pretrade.codes import KIND_ORDER
from quant_fund.pretrade.config import LimitConfig, PretradeConfig, SessionConfig
from quant_fund.pretrade.engine import OrderView, PretradeEngine

HMAC_KEY = b"unit-test-hmac-key-32b!!!!!"
TS_NS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def limit_config(**overrides: object) -> LimitConfig:
    base: dict[str, object] = {
        "max_order_notional": 1_000_000.0,
        "max_order_quantity": 10_000.0,
        "price_collar_bps": 100.0,
        "max_position_notional": 5_000_000.0,
        "max_position_quantity": 50_000.0,
        "max_gross_notional": 5_000_000.0,
        "max_net_notional": 5_000_000.0,
        "max_name_concentration": 1.0,
        "max_daily_loss_fraction": 0.2,
        "max_trailing_drawdown_fraction": 0.3,
        "max_orders_per_window": 50,
        "max_messages_per_window": 50,
        "rate_window_ns": 1_000_000_000,
        "duplicate_window_ns": 1_000_000_000,
        "max_reference_age_ns": 60_000_000_000,
        "max_mark_age_ns": 60_000_000_000,
        "pdt_equity_threshold": 25_000.0,
        "pdt_window_sessions": 5,
        "pdt_max_day_trades": 3,
        "settlement_ns": 86_400_000_000_000,
        "qty_tick": 1e-4,
        "price_tick": 1e-4,
        "cash_account": True,
        "restrict_to_settled_cash": True,
        "max_lots_per_symbol": 8,
        "ring_capacity": 64,
        "symbol_capacity": 16,
        "day_trade_capacity": 16,
        "unsettled_capacity": 16,
    }
    base.update(overrides)
    return LimitConfig.model_validate(base)


def make_engine(
    *,
    nav: float = 1_000_000.0,
    cash: float = 1_000_000.0,
    ts_ns: int = TS_NS,
    **overrides: object,
) -> PretradeEngine:
    config = PretradeConfig(
        schema_version=1,
        limits=limit_config(**overrides),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )
    return PretradeEngine(
        config,
        hmac_key=HMAC_KEY,
        initial_nav=nav,
        initial_cash=cash,
        ts_ns=ts_ns,
    )


def arm(
    engine: PretradeEngine,
    symbol: str = "A",
    *,
    ref: float = 100.0,
    pos: float = 0.0,
    ts_ns: int = TS_NS,
    halted: bool = False,
    sho_restricted: bool = False,
    locate_ok: bool = True,
    bid: float = 100.0,
) -> int:
    sid = engine.ensure_symbol(symbol)
    if sid < 0:
        raise RuntimeError("symbol capacity exhausted")
    engine.set_symbol(
        sid,
        pos=pos,
        ref_px=ref,
        ref_ts_ns=ts_ns,
        halted=halted,
        sho_restricted=sho_restricted,
        locate_ok=locate_ok,
        bid=bid,
    )
    return sid


def order(
    symbol_id: int,
    *,
    side: int = 1,
    qty: float = 10.0,
    px: float = 100.0,
    ts_ns: int = TS_NS,
    session_id: int = 10,
    is_limit: int = 1,
    kind: int = KIND_ORDER,
) -> OrderView:
    return OrderView(
        symbol_id=symbol_id,
        side=side,
        qty=qty,
        px=px,
        ts_ns=ts_ns,
        session_id=session_id,
        is_limit=is_limit,
        kind=kind,
    )
