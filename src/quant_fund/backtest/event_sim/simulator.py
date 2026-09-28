"""Event-driven execution simulator (research / backtest only).

Clock phases on each bar, in order: market data, fills at the open, mark,
signal at the close, order submission, exchange arrival. Signal-to-order and
order-to-exchange latency are either bar counts or timedeltas resolved on the
panel calendar. Fills that arrive after that bar's open wait for the next open.

``fill_model="next_open"`` with zero latency, immediate cash, fractional
shares, a $1 minimum, and the legacy cost stack (including the frictionless
stack) follows the same sizing, risk gate, cash update, and close mark as
``run_backtest``. VWAP and L2 queue fills are the non-degenerate extensions.
No function here submits a live order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import polars as pl

from quant_fund.backtest.engine import (
    BacktestResult,
    Book,
    StaleValuationError,
    _build_result,
    _make_order,
    _projected_exposures,
    _valid_price,
    _validate_bar_panel,
    _validate_target_weight_panel,
)
from quant_fund.backtest.event_sim.clock import EventClock, EventKind
from quant_fund.backtest.event_sim.constraints import (
    ConstraintBook,
    below_min_notional,
    round_shares,
    session_date,
)
from quant_fund.backtest.event_sim.costs import (
    FeeSchedule,
    alpaca_equity_schedule,
    quote_execution_cost,
)
from quant_fund.backtest.event_sim.fills import (
    advance_queue,
    allocate_vwap,
    bar_vwap_price,
    levels_for,
    touch,
    walk_book,
)
from quant_fund.config.models import AppConfig, FillConvention
from quant_fund.execution.costs import total_cost
from quant_fund.monitoring.kill_switch import KillSwitch
from quant_fund.pipeline.forecast import (
    MARKET_RISK_OVERLAY_GARCH,
    MARKET_RISK_OVERLAY_REALIZED_GARCH,
    market_risk_overlay_asof,
)
from quant_fund.portfolio.risk_gate import check_order, funded
from quant_fund.risk.overlay import BookRiskOverlay
from quant_fund.schemas.errors import KillSwitchActive, RiskGateRejected
from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot

FillModel = Literal["next_open", "vwap", "l2_queue"]
GateMode = Literal["off", "warn", "block"]


@dataclass
class EventSimSpec:
    """Latency, fill, cost, and constraint controls. Defaults match the daily backtest."""

    signal_to_order_bars: int = 0
    order_to_exchange_bars: int = 0
    signal_to_order: timedelta | None = None
    order_to_exchange: timedelta | None = None
    fill_model: FillModel = "next_open"
    vwap_window_bars: int = 1
    l2_rest_bars: int = 2
    seed: int = 0
    fractional_shares: bool = True
    min_notional: float = 1.0
    settlement_bars: int = 0
    allow_margin: bool = False
    pdt_mode: GateMode = "warn"
    gfv_mode: GateMode = "warn"
    pdt_equity_threshold: float = 25_000.0
    pdt_window_sessions: int = 5
    pdt_max_day_trades: int = 3
    fee_schedule: Literal["bps", "alpaca"] = "bps"
    sec_per_million: float = 27.80
    taf_per_share: float = 0.000166
    taf_min: float = 0.01
    taf_max: float = 8.30
    ac_eta: float = 0.0
    ac_gamma: float = 0.0
    ac_tau: float = 1.0
    spread_from_book: bool = False

    def __post_init__(self) -> None:
        if self.fill_model not in ("next_open", "vwap", "l2_queue"):
            raise ValueError("fill_model must be next_open, vwap, or l2_queue")
        if self.signal_to_order_bars < 0 or self.order_to_exchange_bars < 0:
            raise ValueError("latency bars must be non-negative")
        for label, delay in (
            ("signal_to_order", self.signal_to_order),
            ("order_to_exchange", self.order_to_exchange),
        ):
            if delay is not None and delay < timedelta(0):
                raise ValueError(f"{label} timedelta must be non-negative")
        if self.vwap_window_bars < 1 or self.l2_rest_bars < 0:
            raise ValueError("vwap window must be >= 1 and l2 rest bars >= 0")
        if self.min_notional < 0.0 or not math.isfinite(self.min_notional):
            raise ValueError("min_notional must be finite and non-negative")
        if self.settlement_bars < 0:
            raise ValueError("settlement_bars must be non-negative")
        if self.pdt_mode not in ("off", "warn", "block") or self.gfv_mode not in (
            "off",
            "warn",
            "block",
        ):
            raise ValueError("pdt_mode and gfv_mode must be off, warn, or block")
        if self.fee_schedule not in ("bps", "alpaca"):
            raise ValueError("fee_schedule must be bps or alpaca")
        for label, value in (
            ("ac_eta", self.ac_eta),
            ("ac_gamma", self.ac_gamma),
            ("sec_per_million", self.sec_per_million),
            ("taf_per_share", self.taf_per_share),
            ("taf_min", self.taf_min),
            ("taf_max", self.taf_max),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not math.isfinite(self.ac_tau) or self.ac_tau <= 0.0:
            raise ValueError("ac_tau must be finite and > 0")
        if self.taf_min > self.taf_max:
            raise ValueError("taf_min cannot exceed taf_max")


@dataclass
class EventSimResult:
    result: BacktestResult
    warnings: list[dict[str, object]]
    events: list[dict[str, object]]
    min_cash: float
    cancel_replace_count: int
    day_trade_count: int
    gfv_count: int
    seed: int


@dataclass
class _Resting:
    sid: str
    side: str
    remaining: float
    limit_price: float
    queue_ahead: float
    target_w: float
    signal_index: int
    signal_time: datetime
    decision_price: float | None
    adv: float
    vol: float
    placed_bar: int
    bars_left: int
    kind: str


@dataclass
class _Snap:
    adv: dict[str, float]
    vol: dict[str, float]
    decision: dict[str, float]
    target: dict[str, float]


@dataclass
class _State:
    book: Book
    last_marks: dict[str, float] = field(default_factory=dict)
    mark_ages: dict[str, int] = field(default_factory=dict)
    exec_mark: dict[str, float] = field(default_factory=dict)
    close_mark: dict[str, float] = field(default_factory=dict)
    marked_today: set[str] = field(default_factory=set)
    pre_exec_marks: dict[str, float] = field(default_factory=dict)
    snaps: dict[int, _Snap] = field(default_factory=dict)
    last_target: dict[str, float] = field(default_factory=dict)
    cost_sum: dict[str, float] = field(
        default_factory=lambda: {
            "commission": 0.0,
            "spread": 0.0,
            "impact": 0.0,
            "turnover": 0.0,
        }
    )
    fill_rows: list[dict[str, object]] = field(default_factory=list)
    navs: list[dict[str, object]] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    warnings: list[dict[str, object]] = field(default_factory=list)
    reject_count: int = 0
    cash_reject_count: int = 0
    halt_count: int = 0
    order_seq: int = 0
    traded_turn: float = 0.0
    garch_overlay_dates: int = 0
    realized_garch_overlay_dates: int = 0
    stopped: bool = False
    nav_checked: bool = False
    min_cash: float = 0.0
    cancel_replace_count: int = 0
    resting: dict[str, _Resting] = field(default_factory=dict)
    sessions: list[Any] = field(default_factory=list)
    constraints: ConstraintBook | None = None


def _delay_bars(
    dates: list[datetime],
    origin: int,
    delay: timedelta | None,
    bars: int,
) -> int:
    if delay is None:
        return int(bars)
    target = dates[origin] + delay
    step = 0
    while origin + step < len(dates) and dates[origin + step] < target:
        step += 1
    return step


def _legacy_costs(spec: EventSimSpec) -> bool:
    return (
        spec.fee_schedule == "bps"
        and spec.ac_eta == 0.0
        and spec.ac_gamma == 0.0
        and not spec.spread_from_book
    )


def _schedule_for(spec: EventSimSpec, config: AppConfig) -> FeeSchedule:
    if spec.fee_schedule == "alpaca":
        return alpaca_equity_schedule(
            sec_per_million=spec.sec_per_million,
            taf_per_share=spec.taf_per_share,
            taf_min=spec.taf_min,
            taf_max=spec.taf_max,
        )
    return FeeSchedule(
        name="bps",
        commission_bps=float(config.costs.commission_bps),
        sec_per_million=spec.sec_per_million,
        taf_per_share=spec.taf_per_share,
        taf_min=spec.taf_min,
        taf_max=spec.taf_max,
    )


def _index_rows(
    bars: pl.DataFrame,
) -> tuple[dict[datetime, list[dict[str, Any]]], list[datetime], bool]:
    px = bars.select(
        "security_id",
        "event_time",
        "open",
        "close",
        "close_total_return",
        "volume",
        pl.col("adv")
        if "adv" in bars.columns
        else (pl.col("close") * pl.col("volume")).alias("adv"),
        pl.col("vol_20") if "vol_20" in bars.columns else pl.lit(0.02).alias("vol_20"),
        "source",
        *([pl.col("high")] if "high" in bars.columns else []),
        *([pl.col("low")] if "low" in bars.columns else []),
        *([pl.col("vwap")] if "vwap" in bars.columns else []),
        *[
            pl.col(name)
            for name in (
                "best_bid",
                "best_ask",
                "top_bid_size",
                "top_ask_size",
                "bid_depth",
                "ask_depth",
                "queue_priority_proxy",
                "ask_queue_priority_proxy",
            )
            if name in bars.columns
        ],
    )
    day_rows: dict[datetime, list[dict[str, Any]]] = {}
    for row in px.iter_rows(named=True):
        day_rows.setdefault(row["event_time"], []).append(row)
    dates = sorted(day_rows)
    synthetic = (
        "synthetic" in set(px["source"].drop_nulls().to_list()) if "source" in px.columns else False
    )
    return day_rows, dates, synthetic


def _snapshot_from_row(row: dict[str, Any], half_spread_bps: float) -> OrderBookSnapshot | None:
    when = row["event_time"]
    if not isinstance(when, datetime) or when.tzinfo is None:
        return None
    sid = str(row["security_id"])
    if all(
        name in row and row[name] is not None
        for name in ("best_bid", "best_ask", "top_bid_size", "top_ask_size")
    ):
        try:
            bid = float(row["best_bid"])
            ask = float(row["best_ask"])
            bid_sz = float(row["top_bid_size"])
            ask_sz = float(row["top_ask_size"])
        except (TypeError, ValueError):
            return None
        if bid > 0.0 and ask > bid and bid_sz > 0.0 and ask_sz > 0.0:
            return OrderBookSnapshot(
                security_id=sid,
                symbol=sid,
                event_time=when,
                available_time=when,
                source=str(row.get("source") or "book"),
                bids=[BookLevel(price=bid, size=bid_sz)],
                asks=[BookLevel(price=ask, size=ask_sz)],
                depth=1,
            )
    close = _valid_price(row.get("close"))
    if close is None:
        return None
    volume = row.get("volume")
    try:
        vol = float(volume) if volume is not None else 0.0
    except (TypeError, ValueError):
        vol = 0.0
    half = max(close * float(half_spread_bps) / 1e4 / 2.0, close * 1e-6)
    bid = close - half
    ask = close + half
    if bid <= 0.0 or bid >= ask:
        return None
    size = max(vol * 0.02, 1e-6)
    return OrderBookSnapshot(
        security_id=sid,
        symbol=sid,
        event_time=when,
        available_time=when,
        source="synthetic",
        bids=[BookLevel(price=float(bid), size=float(size))],
        asks=[BookLevel(price=float(ask), size=float(size))],
        depth=1,
    )


def _load_books(
    bars: pl.DataFrame,
    spec: EventSimSpec,
    books: dict[tuple[str, datetime], OrderBookSnapshot] | None,
    rows: dict[datetime, list[dict[str, Any]]],
) -> dict[tuple[str, datetime], OrderBookSnapshot]:
    if books:
        return {(str(sid), when): snap for (sid, when), snap in books.items()}
    if spec.fill_model != "l2_queue":
        return {}
    feature_cols = {"best_bid", "best_ask", "top_bid_size", "top_ask_size"}
    if feature_cols <= set(bars.columns):
        indexed: dict[tuple[str, datetime], OrderBookSnapshot] = {}
        for day in rows.values():
            for row in day:
                snap = _snapshot_from_row(row, 0.0)
                if snap is not None:
                    indexed[(snap.security_id, snap.event_time)] = snap
        return indexed
    if {"high", "low"} <= set(bars.columns):
        from quant_fund.microstructure.synthetic_lob import synthesize_snapshots_from_bars

        indexed = {}
        for snap in synthesize_snapshots_from_bars(bars, seed=int(spec.seed)):
            indexed[(snap.security_id, snap.event_time)] = snap
        return indexed
    indexed = {}
    for day in rows.values():
        for row in day:
            snap = _snapshot_from_row(row, 4.0)
            if snap is not None:
                indexed[(snap.security_id, snap.event_time)] = snap
    return indexed


def _ingest(state: _State, rows: list[dict[str, Any]], *, use_open: bool, stale_limit: int) -> None:
    exec_mark: dict[str, float] = {}
    close_mark = dict(state.last_marks)
    next_ages = dict(state.mark_ages)
    marked: set[str] = set()
    for row in rows:
        sid = str(row["security_id"])
        raw_exec = _valid_price(row["open"] if use_open else row["close"])
        if raw_exec is not None:
            exec_mark[sid] = raw_exec
        total_return_mark = _valid_price(row["close_total_return"])
        fallback_mark = _valid_price(row["close"])
        if total_return_mark is not None:
            close_mark[sid] = total_return_mark
            next_ages[sid] = 0
            marked.add(sid)
        elif fallback_mark is not None:
            close_mark[sid] = fallback_mark
            next_ages[sid] = 0
            marked.add(sid)
    for sid in close_mark:
        if sid not in marked:
            next_ages[sid] = next_ages.get(sid, 0) + 1
    # Marks knowable at execution time are the pre-update marks; valuing a
    # non-executing name at this bar's own close leaks future prices into
    # open-time sizing and gating, mirroring pre_exec_marks in the engine.
    state.pre_exec_marks = dict(state.last_marks)
    state.last_marks = dict(close_mark)
    state.mark_ages = next_ages
    state.exec_mark = exec_mark
    state.close_mark = close_mark
    state.marked_today = marked
    stale_held = {
        sid: state.mark_ages.get(sid)
        for sid, shares in state.book.shares.items()
        if abs(shares) > 1e-12 and (sid not in close_mark or sid not in state.mark_ages)
    }
    stale_held.update(
        {
            sid: state.mark_ages[sid]
            for sid, shares in state.book.shares.items()
            if abs(shares) > 1e-12 and sid in state.mark_ages and state.mark_ages[sid] > stale_limit
        }
    )
    if stale_held:
        details = ", ".join(
            f"{sid}={age if age is not None else 'unknown'}" for sid, age in stale_held.items()
        )
        raise StaleValuationError(
            "held position valuation is stale beyond the configured limit: " + details
        )


def _record_costs(state: _State, costs: dict[str, float | str]) -> None:
    for key in ("commission", "spread", "impact"):
        state.cost_sum[key] += float(costs[key])
    state.cost_sum["turnover"] += float(costs.get("turnover_bps", 0.0))


def _fill_row(
    *,
    fill_time: datetime,
    signal_time: datetime,
    sid: str,
    quantity: float,
    price: float,
    costs: dict[str, float | str],
    decision_price: float | None,
) -> dict[str, object]:
    return {
        "fill_time": fill_time,
        "signal_time": signal_time,
        "security_id": sid,
        "quantity": quantity,
        "price": price,
        "fee": costs["commission"],
        "spread_cost": costs["spread"],
        "impact_cost": costs["impact"],
        "turnover_cost": float(costs.get("turnover_bps", 0.0)),
        "decision_price": decision_price,
    }


def _costs_for(
    spec: EventSimSpec,
    schedule: FeeSchedule,
    config: AppConfig,
    delta: float,
    price: float,
    adv: float,
    sigma: float,
    *,
    book: OrderBookSnapshot | None,
    include_spread: bool,
) -> dict[str, float | str]:
    if _legacy_costs(spec) and include_spread and book is None:
        return total_cost(delta, price, adv, sigma, config.costs)
    half = None
    if spec.spread_from_book and book is not None:
        half = float(book.half_spread)
    return quote_execution_cost(
        delta,
        price,
        adv,
        sigma,
        config.costs,
        schedule=schedule,
        is_sell=delta < 0.0,
        eta=spec.ac_eta,
        gamma=spec.ac_gamma,
        tau=spec.ac_tau,
        half_spread=half,
        include_spread=include_spread,
    )


def _cash_ok(
    state: _State,
    spec: EventSimSpec,
    sid: str,
    delta: float,
    price: float,
    total_trade_cost: float,
    bar_index: int,
) -> bool:
    notional = delta * price
    post = state.book.cash - notional - total_trade_cost
    if spec.settlement_bars == 0:
        if delta > 0 and not funded(state.book.cash, notional + total_trade_cost):
            return False
        return not (not spec.allow_margin and delta <= 0.0 and post < -1e-12)
    assert state.constraints is not None
    if delta > 0:
        return not (
            not spec.allow_margin
            and (notional + total_trade_cost) > state.constraints.buying_power() + 1e-9
        )
    if not spec.allow_margin and post < -1e-12:
        return False
    if state.constraints.gfv_would_block(sid, state.book.shares.get(sid, 0.0), delta, bar_index):
        state.constraints.blocked += 1
        state.constraints.warnings.append(
            {
                "code": "GFV_BLOCK",
                "message": "blocked sale of shares bought with unsettled funds",
                "security_id": sid,
                "bar_index": bar_index,
            }
        )
        return False
    return True


def _move_cash(
    state: _State,
    spec: EventSimSpec,
    sid: str,
    delta: float,
    price: float,
    total_trade_cost: float,
    bar_index: int,
    shares_before: float,
) -> None:
    notional = delta * price
    if spec.settlement_bars == 0:
        state.book.cash -= notional + total_trade_cost
    else:
        assert state.constraints is not None
        need_or_proceeds = -(notional + total_trade_cost)
        if delta > 0:
            ok = state.constraints.apply_buy(sid, delta, notional + total_trade_cost, bar_index)
            if not ok:
                raise RuntimeError("buy passed the cash check and then failed to debit")
            state.book.cash -= notional + total_trade_cost
        else:
            closed = min(shares_before, -delta) if shares_before > 0.0 else 0.0
            state.constraints.consume_lots_for_sell(sid, closed, bar_index)
            state.constraints.apply_sell_proceeds(need_or_proceeds, bar_index)
            state.book.cash -= notional + total_trade_cost
    state.book.shares[sid] = shares_before + delta
    state.min_cash = min(state.min_cash, state.book.cash)


def _note_constraints(
    state: _State,
    spec: EventSimSpec,
    sid: str,
    before: float,
    after: float,
    session: Any,
    nav: float,
) -> None:
    if spec.pdt_mode == "off" or state.constraints is None:
        return
    state.constraints.note_round_trip(sid, before, after, session, nav, list(state.sessions))


def _try_commit(
    state: _State,
    spec: EventSimSpec,
    schedule: FeeSchedule,
    config: AppConfig,
    *,
    sid: str,
    delta: float,
    price: float,
    adv: float,
    vol: float,
    signal_time: datetime,
    exec_time: datetime,
    decision_price: float | None,
    nav: float,
    market_vol: float | None,
    bar_index: int,
    book: OrderBookSnapshot | None,
    include_spread: bool,
    kill: KillSwitch,
) -> bool:
    """One child order through the same gates as the daily backtest. True if it filled."""
    if not spec.fractional_shares:
        delta = round_shares(delta, fractional=False)
    if below_min_notional(delta, price, spec.min_notional):
        return False
    if (
        not np.isfinite(config.costs.participation_limit)
        or not 0 < config.costs.participation_limit <= 1
    ):
        state.reject_count += 1
        return False
    costs = _costs_for(
        spec, schedule, config, delta, price, adv, vol, book=book, include_spread=include_spread
    )
    max_qty = config.costs.participation_limit * (adv / price)
    if abs(delta) > max_qty:
        delta = float(np.sign(delta) * max_qty)
        costs = _costs_for(
            spec, schedule, config, delta, price, adv, vol, book=book, include_spread=include_spread
        )
    if below_min_notional(delta, price, spec.min_notional):
        return False
    try:
        kill.assert_new_orders_allowed()
    except KillSwitchActive:
        state.halt_count += 1
        return False
    nav_prices = {**state.pre_exec_marks, **state.exec_mark}
    current_w, gross_after, net_after = _projected_exposures(
        state.book, nav_prices, sid, delta, nav
    )
    participation = abs(delta) * price / adv
    state.order_seq += 1
    order = _make_order(
        sid=sid,
        delta=float(delta),
        signal_time=signal_time,
        order_time=exec_time,
        order_seq=state.order_seq,
    )
    try:
        check_order(
            order,
            nav=nav,
            price=price,
            current_weight=current_w,
            gross_after=gross_after,
            net_after=net_after,
            participation=participation,
            predicted_vol=vol,
            config=config,
            market_predicted_vol=market_vol,
        )
    except RiskGateRejected:
        state.reject_count += 1
        return False
    session = session_date(exec_time)
    before = state.book.shares.get(sid, 0.0)
    if state.constraints is not None and state.constraints.pdt_would_block(
        sid, before, float(delta), session, nav, list(state.sessions)
    ):
        state.constraints.blocked += 1
        state.warnings.append(
            {
                "code": "PDT_BLOCK",
                "message": "blocked round trip that would exceed the pattern-day-trader allowance",
                "security_id": sid,
                "session": session.isoformat(),
            }
        )
        return False
    total_trade_cost = float(costs["total"])
    if not _cash_ok(state, spec, sid, float(delta), price, total_trade_cost, bar_index):
        state.cash_reject_count += 1
        return False
    _move_cash(state, spec, sid, float(delta), price, total_trade_cost, bar_index, before)
    _note_constraints(state, spec, sid, before, state.book.shares.get(sid, 0.0), session, nav)
    state.traded_turn += abs(float(delta) * price) / max(nav, 1e-12)
    _record_costs(state, costs)
    state.fill_rows.append(
        _fill_row(
            fill_time=exec_time,
            signal_time=signal_time,
            sid=sid,
            quantity=float(delta),
            price=float(price),
            costs=costs,
            decision_price=decision_price,
        )
    )
    return True


def _signal_snapshot(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    decision: dict[str, float] = {}
    advs: dict[str, float] = {}
    vols: dict[str, float] = {}
    for row in rows:
        sid = str(row["security_id"])
        mark = _valid_price(row["close"])
        if mark is not None:
            decision[sid] = mark
        known_adv = _valid_price(row["adv"])
        if known_adv is not None:
            advs[sid] = known_adv
        vols[sid] = _valid_price(row["vol_20"]) or 0.02
    return decision, advs, vols


def _prepare_targets(
    state: _State,
    signal_index: int,
    risk_overlay: BookRiskOverlay | None,
) -> dict[str, float]:
    target_w = dict(state.snaps[signal_index].target)
    if risk_overlay is not None:
        overlay_scale = float(risk_overlay.preview_scale())
        if overlay_scale != 1.0:
            target_w = {key: float(value) * overlay_scale for key, value in target_w.items()}
    for sid in list(target_w):
        if sid not in state.marked_today:
            target_w[sid] = 0.0
    for sid, held_qty in state.book.shares.items():
        if abs(held_qty) > 1e-12 and sid not in state.marked_today:
            target_w[sid] = 0.0
    return target_w


def _count_overlay(state: _State, source: str | None) -> None:
    if source == MARKET_RISK_OVERLAY_REALIZED_GARCH:
        state.realized_garch_overlay_dates += 1
    elif source == MARKET_RISK_OVERLAY_GARCH:
        state.garch_overlay_dates += 1


def _rebalance_next_open(
    state: _State,
    *,
    spec: EventSimSpec,
    schedule: FeeSchedule,
    config: AppConfig,
    signal_index: int,
    exec_time: datetime,
    signal_time: datetime,
    bar_index: int,
    risk_overlay: BookRiskOverlay | None,
    bars: pl.DataFrame,
    kill: KillSwitch,
) -> None:
    """One next-open rebalance. Legacy zero-extra specs follow ``run_backtest``."""
    nav_prices = {**state.pre_exec_marks, **state.exec_mark}
    nav = state.book.nav(nav_prices)
    state.nav_checked = True
    if nav <= 0:
        state.stopped = True
        return
    target_w = _prepare_targets(state, signal_index, risk_overlay)
    snap = state.snaps[signal_index]
    market_vol, overlay_source = market_risk_overlay_asof(config, bars, signal_time)
    _count_overlay(state, overlay_source)
    state.traded_turn = 0.0
    ids = set(state.exec_mark) | set(state.book.shares) | set(target_w)
    costs_cfg = config.costs
    legacy = (
        _legacy_costs(spec)
        and spec.settlement_bars == 0
        and spec.fractional_shares
        and spec.min_notional == 1.0
        and spec.pdt_mode != "block"
        and spec.gfv_mode != "block"
    )
    for sid in sorted(ids):
        price = state.exec_mark.get(sid)
        if price is None:
            continue
        tw = target_w.get(sid, 0.0)
        desired = tw * nav / price
        current = state.book.shares.get(sid, 0.0)
        delta = desired - current
        if not legacy:
            delta = round_shares(delta, fractional=spec.fractional_shares)
        if abs(delta) * price < (1.0 if legacy else spec.min_notional):
            continue
        adv = snap.adv.get(sid)
        if (
            adv is None
            or not np.isfinite(costs_cfg.participation_limit)
            or not 0 < costs_cfg.participation_limit <= 1
        ):
            state.reject_count += 1
            continue
        if legacy:
            costs = total_cost(delta, price, adv, snap.vol.get(sid, 0.02), costs_cfg)
        else:
            costs = _costs_for(
                spec,
                schedule,
                config,
                float(delta),
                price,
                adv,
                snap.vol.get(sid, 0.02),
                book=None,
                include_spread=True,
            )
        max_qty = costs_cfg.participation_limit * (adv / price)
        if abs(delta) > max_qty:
            delta = np.sign(delta) * max_qty
            if legacy:
                costs = total_cost(delta, price, adv, snap.vol.get(sid, 0.02), costs_cfg)
            else:
                costs = _costs_for(
                    spec,
                    schedule,
                    config,
                    float(delta),
                    price,
                    adv,
                    snap.vol.get(sid, 0.02),
                    book=None,
                    include_spread=True,
                )
        try:
            kill.assert_new_orders_allowed()
        except KillSwitchActive:
            state.halt_count += 1
            continue
        current_w, gross_after, net_after = _projected_exposures(
            state.book, nav_prices, sid, float(delta), nav
        )
        participation = abs(delta) * price / adv
        state.order_seq += 1
        order = _make_order(
            sid=sid,
            delta=float(delta),
            signal_time=signal_time,
            order_time=exec_time,
            order_seq=state.order_seq,
        )
        try:
            check_order(
                order,
                nav=nav,
                price=price,
                current_weight=current_w,
                gross_after=gross_after,
                net_after=net_after,
                participation=participation,
                predicted_vol=snap.vol.get(sid, 0.02),
                config=config,
                market_predicted_vol=market_vol,
            )
        except RiskGateRejected:
            state.reject_count += 1
            continue
        notional = delta * price
        total_trade_cost = float(costs["total"])
        if legacy:
            if delta > 0 and not funded(state.book.cash, notional + total_trade_cost):
                state.cash_reject_count += 1
                continue
            state.book.cash -= notional + total_trade_cost
        else:
            session = session_date(exec_time)
            if state.constraints is not None and state.constraints.pdt_would_block(
                sid, current, float(delta), session, nav, list(state.sessions)
            ):
                state.constraints.blocked += 1
                state.warnings.append(
                    {
                        "code": "PDT_BLOCK",
                        "message": "blocked round trip that would exceed the pattern-day-trader allowance",
                        "security_id": sid,
                        "session": session.isoformat(),
                    }
                )
                continue
            if not _cash_ok(state, spec, sid, float(delta), price, total_trade_cost, bar_index):
                state.cash_reject_count += 1
                continue
            before = current
            _move_cash(state, spec, sid, float(delta), price, total_trade_cost, bar_index, before)
            _note_constraints(
                state,
                spec,
                sid,
                before,
                state.book.shares.get(sid, 0.0),
                session_date(exec_time),
                nav,
            )
            state.traded_turn += abs(notional) / max(nav, 1e-12)
            _record_costs(state, costs)
            state.fill_rows.append(
                _fill_row(
                    fill_time=exec_time,
                    signal_time=signal_time,
                    sid=sid,
                    quantity=delta,
                    price=price,
                    costs=costs,
                    decision_price=snap.decision.get(sid),
                )
            )
            continue
        state.book.shares[sid] = current + delta
        state.min_cash = min(state.min_cash, state.book.cash)
        state.traded_turn += abs(notional) / max(nav, 1e-12)
        _record_costs(state, costs)
        state.fill_rows.append(
            _fill_row(
                fill_time=exec_time,
                signal_time=signal_time,
                sid=sid,
                quantity=delta,
                price=price,
                costs=costs,
                decision_price=snap.decision.get(sid),
            )
        )
        if spec.pdt_mode != "off":
            _note_constraints(
                state,
                spec,
                sid,
                current,
                state.book.shares.get(sid, 0.0),
                session_date(exec_time),
                nav,
            )


def _row_on(
    rows: dict[datetime, list[dict[str, Any]]], when: datetime, sid: str
) -> dict[str, Any] | None:
    for row in rows.get(when, []):
        if str(row["security_id"]) == sid:
            return row
    return None


def _rebalance_path(
    state: _State,
    *,
    spec: EventSimSpec,
    schedule: FeeSchedule,
    config: AppConfig,
    signal_index: int,
    exec_time: datetime,
    signal_time: datetime,
    bar_index: int,
    dates: list[datetime],
    rows: dict[datetime, list[dict[str, Any]]],
    books: dict[tuple[str, datetime], OrderBookSnapshot],
    risk_overlay: BookRiskOverlay | None,
    bars: pl.DataFrame,
    kill: KillSwitch,
) -> None:
    nav_prices = {**state.pre_exec_marks, **state.exec_mark}
    nav = state.book.nav(nav_prices)
    state.nav_checked = True
    if nav <= 0:
        state.stopped = True
        return
    target_w = _prepare_targets(state, signal_index, risk_overlay)
    snap = state.snaps[signal_index]
    market_vol, overlay_source = market_risk_overlay_asof(config, bars, signal_time)
    _count_overlay(state, overlay_source)
    state.traded_turn = 0.0
    for sid, resting in list(state.resting.items()):
        same = abs(float(target_w.get(sid, 0.0)) - resting.target_w) <= 1e-12
        if not same:
            state.cancel_replace_count += 1
            state.resting.pop(sid, None)
            state.events.append(
                {
                    "kind": "CANCEL",
                    "bar_index": bar_index,
                    "security_id": sid,
                    "reason": "cancel_replace",
                }
            )
            continue
        _advance_resting(
            state,
            resting,
            spec=spec,
            schedule=schedule,
            config=config,
            exec_time=exec_time,
            bar_index=bar_index,
            rows=rows,
            books=books,
            nav=nav,
            market_vol=market_vol,
            kill=kill,
        )
    ids = set(state.exec_mark) | set(state.book.shares) | set(target_w)
    for sid in sorted(ids):
        if sid in state.resting:
            continue
        price = state.exec_mark.get(sid)
        if price is None:
            continue
        adv = snap.adv.get(sid)
        if adv is None:
            state.reject_count += 1
            continue
        tw = target_w.get(sid, 0.0)
        current = state.book.shares.get(sid, 0.0)
        delta = tw * nav / price - current
        vol = snap.vol.get(sid, 0.02)
        book = books.get((sid, exec_time))
        if spec.fill_model == "vwap":
            window: list[tuple[float, float]] = []
            last = min(len(dates) - 1, bar_index + spec.vwap_window_bars - 1)
            for index in range(bar_index, last + 1):
                row = _row_on(rows, dates[index], sid)
                if row is None:
                    continue
                px = bar_vwap_price(row)
                try:
                    volume = float(row.get("volume") or 0.0)
                except (TypeError, ValueError):
                    volume = 0.0
                if px is None or not math.isfinite(volume) or volume < 0.0:
                    continue
                window.append((volume, px))
            # Size and gate the parent on the open, then keep only what the tape can print.
            capped = delta
            max_qty = config.costs.participation_limit * (adv / price)
            if abs(capped) > max_qty:
                capped = float(np.sign(capped) * max_qty)
            slices = allocate_vwap(capped, window)
            if not slices:
                continue
            first_qty, first_px = slices[0]
            filled = _try_commit(
                state,
                spec,
                schedule,
                config,
                sid=sid,
                delta=float(first_qty),
                price=float(first_px),
                adv=adv,
                vol=vol,
                signal_time=signal_time,
                exec_time=exec_time,
                decision_price=snap.decision.get(sid),
                nav=nav,
                market_vol=market_vol,
                bar_index=bar_index,
                book=book,
                include_spread=True,
                kill=kill,
            )
            if filled and len(slices) > 1:
                signed_left = float(sum(qty for qty, _ in slices[1:]))
                state.resting[sid] = _Resting(
                    sid=sid,
                    side="buy" if signed_left > 0 else "sell",
                    remaining=signed_left,
                    limit_price=float(first_px),
                    queue_ahead=0.0,
                    target_w=float(tw),
                    signal_index=signal_index,
                    signal_time=signal_time,
                    decision_price=snap.decision.get(sid),
                    adv=adv,
                    vol=vol,
                    placed_bar=bar_index,
                    bars_left=spec.vwap_window_bars - 1,
                    kind="vwap",
                )
            continue
        # L2 / queue. Aggressive take cannot exceed displayed size; residual rests.
        if book is None:
            state.warnings.append(
                {
                    "code": "NO_BOOK",
                    "message": "L2 fill skipped; no order book for this bar",
                    "security_id": sid,
                    "bar_index": bar_index,
                }
            )
            continue
        side = "buy" if delta > 0 else "sell"
        max_qty = config.costs.participation_limit * (adv / price)
        if abs(delta) > max_qty:
            delta = float(np.sign(delta) * max_qty)
        walked = walk_book(side, abs(float(delta)), levels_for(book, side))
        take = walked.filled if delta > 0 else -walked.filled
        if walked.filled > 0.0:
            _try_commit(
                state,
                spec,
                schedule,
                config,
                sid=sid,
                delta=float(take),
                price=float(walked.vwap),
                adv=adv,
                vol=vol,
                signal_time=signal_time,
                exec_time=exec_time,
                decision_price=snap.decision.get(sid),
                nav=nav,
                market_vol=market_vol,
                bar_index=bar_index,
                book=book,
                include_spread=False,
                kill=kill,
            )
        residual = abs(delta) - walked.filled
        if residual > 1e-12 and spec.l2_rest_bars > 0 and walked.filled + 1e-9 >= walked.displayed:
            # Book was exhausted; nothing left to queue behind on this snapshot.
            touch_px, touch_sz = touch(book, side)
            # Join the back only when the touch survived. A cleared book rests
            # with queue_ahead 0 at the last walk price so later volume can fill,
            # still capped by that later bar's displayed size.
            if walked.exhausted_book:
                limit_px, ahead = touch_px, 0.0
            else:
                limit_px, ahead = touch_px, touch_sz
            state.resting[sid] = _Resting(
                sid=sid,
                side=side,
                remaining=float(residual if delta > 0 else -residual),
                limit_price=float(limit_px),
                queue_ahead=float(ahead),
                target_w=float(tw),
                signal_index=signal_index,
                signal_time=signal_time,
                decision_price=snap.decision.get(sid),
                adv=adv,
                vol=vol,
                placed_bar=bar_index,
                bars_left=spec.l2_rest_bars,
                kind="l2_queue",
            )


def _advance_resting(
    state: _State,
    resting: _Resting,
    *,
    spec: EventSimSpec,
    schedule: FeeSchedule,
    config: AppConfig,
    exec_time: datetime,
    bar_index: int,
    rows: dict[datetime, list[dict[str, Any]]],
    books: dict[tuple[str, datetime], OrderBookSnapshot],
    nav: float,
    market_vol: float | None,
    kill: KillSwitch,
) -> None:
    if resting.placed_bar == bar_index:
        return
    if resting.bars_left <= 0:
        state.resting.pop(resting.sid, None)
        return
    row = _row_on(rows, exec_time, resting.sid)
    if row is None:
        resting.bars_left -= 1
        if resting.bars_left <= 0:
            state.resting.pop(resting.sid, None)
        return
    try:
        volume = float(row.get("volume") or 0.0)
    except (TypeError, ValueError):
        volume = 0.0
    if resting.kind == "vwap":
        px = bar_vwap_price(row)
        if px is None or not math.isfinite(volume) or volume <= 0.0:
            resting.bars_left -= 1
            if resting.bars_left <= 0:
                state.resting.pop(resting.sid, None)
            return
        sign = 1.0 if resting.remaining > 0 else -1.0
        take = sign * min(abs(resting.remaining), volume)
        ok = _try_commit(
            state,
            spec,
            schedule,
            config,
            sid=resting.sid,
            delta=float(take),
            price=float(px),
            adv=resting.adv,
            vol=resting.vol,
            signal_time=resting.signal_time,
            exec_time=exec_time,
            decision_price=resting.decision_price,
            nav=nav,
            market_vol=market_vol,
            bar_index=bar_index,
            book=None,
            include_spread=True,
            kill=kill,
        )
        if ok:
            resting.remaining -= take
        resting.bars_left -= 1
        if abs(resting.remaining) <= 1e-12 or resting.bars_left <= 0:
            state.resting.pop(resting.sid, None)
        return
    book = books.get((resting.sid, exec_time))
    if book is None or not math.isfinite(volume) or volume < 0.0:
        resting.bars_left -= 1
        if resting.bars_left <= 0:
            state.resting.pop(resting.sid, None)
        return
    touch_px, touch_sz = touch(book, resting.side)
    step = advance_queue(
        side=resting.side,
        remaining=abs(resting.remaining),
        limit_price=resting.limit_price,
        queue_ahead=resting.queue_ahead,
        touch_price=touch_px,
        touch_size=touch_sz,
        traded_volume=max(volume, 0.0),
    )
    if step.replaced:
        state.cancel_replace_count += 1
        resting.limit_price = step.limit_price
        resting.queue_ahead = step.queue_ahead
        state.events.append(
            {
                "kind": "CANCEL",
                "bar_index": bar_index,
                "security_id": resting.sid,
                "reason": "cancel_replace_touch",
            }
        )
    elif step.filled > 0.0:
        sign = 1.0 if resting.remaining > 0 else -1.0
        ok = _try_commit(
            state,
            spec,
            schedule,
            config,
            sid=resting.sid,
            delta=sign * step.filled,
            price=float(step.limit_price),
            adv=resting.adv,
            vol=resting.vol,
            signal_time=resting.signal_time,
            exec_time=exec_time,
            decision_price=resting.decision_price,
            nav=nav,
            market_vol=market_vol,
            bar_index=bar_index,
            book=book,
            include_spread=False,
            kill=kill,
        )
        if ok:
            resting.remaining -= sign * step.filled
            resting.queue_ahead = step.queue_ahead
    else:
        resting.queue_ahead = step.queue_ahead
    resting.bars_left -= 1
    if abs(resting.remaining) <= 1e-12 or resting.bars_left <= 0:
        state.resting.pop(resting.sid, None)


def _mark_equity(
    state: _State,
    *,
    config: AppConfig,
    spec: EventSimSpec,
    exec_time: datetime,
    risk_overlay: BookRiskOverlay | None,
) -> None:
    if state.stopped:
        return
    if not state.nav_checked:
        nav_open = state.book.nav({**state.pre_exec_marks, **state.exec_mark})
        if nav_open <= 0:
            state.stopped = True
            return
    nav_close = state.book.nav(state.close_mark)
    short_notional = sum(
        abs(min(state.book.shares.get(sid, 0.0), 0.0)) * state.close_mark.get(sid, 0.0)
        for sid in state.book.shares
    )
    borrow = short_notional * (config.costs.borrow_bps_per_year / 1e4) / 252.0
    if not config.costs.frictionless:
        if not spec.allow_margin:
            borrow = min(borrow, max(state.book.cash, 0.0))
        state.book.cash -= borrow
        nav_close -= borrow
        state.min_cash = min(state.min_cash, state.book.cash)
    state.navs.append(
        {
            "event_time": exec_time,
            "nav": nav_close,
            "gross": sum(
                abs(state.book.shares.get(sid, 0.0) * state.close_mark.get(sid, 0.0))
                for sid in state.book.shares
            )
            / max(nav_close, 1e-12),
            "net": sum(
                state.book.shares.get(sid, 0.0) * state.close_mark.get(sid, 0.0)
                for sid in state.book.shares
            )
            / max(nav_close, 1e-12),
            "turnover": state.traded_turn,
        }
    )
    if risk_overlay is not None:
        risk_overlay.observe(float(nav_close))


def run_event_backtest(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    spec: EventSimSpec | None = None,
    *,
    initial_nav: float = 1_000_000.0,
    risk_overlay: BookRiskOverlay | None = None,
    books: dict[tuple[str, datetime], OrderBookSnapshot] | None = None,
) -> EventSimResult:
    """Replay ``weights`` through the event clock.

    Research simulator only: ``live_pnl_claim`` stays false. Latency and fill
    models change the path; the zero-latency next-open legacy configuration
    is the daily backtest.
    """
    if spec is None:
        spec = EventSimSpec()
    if not math.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ValueError("initial_nav must be finite and positive")
    if (
        config.execution.allow_close_auction
        or config.execution.fill is not FillConvention.NEXT_OPEN
    ):
        raise ValueError(
            "event simulator requires execution.fill=next_open without a close auction"
        )
    _validate_target_weight_panel(weights)
    _validate_bar_panel(bars)
    day_rows, dates, synthetic = _index_rows(bars)
    weights_by_date: dict[datetime, dict[str, float]] = {}
    for wrow in weights.iter_rows(named=True):
        weights_by_date.setdefault(wrow["event_time"], {})[str(wrow["security_id"])] = float(
            wrow["target_weight"]
        )
    schedule = _schedule_for(spec, config)
    book_index = _load_books(bars, spec, books, day_rows)
    state = _State(book=Book(cash=float(initial_nav)), min_cash=float(initial_nav))
    if spec.settlement_bars > 0 or spec.pdt_mode != "off" or spec.gfv_mode != "off":
        state.constraints = ConstraintBook(
            settlement_bars=spec.settlement_bars,
            allow_margin=spec.allow_margin,
            pdt_mode=spec.pdt_mode,
            gfv_mode=spec.gfv_mode,
            pdt_equity_threshold=spec.pdt_equity_threshold,
            pdt_window_sessions=spec.pdt_window_sessions,
            pdt_max_day_trades=spec.pdt_max_day_trades,
            settled=float(initial_nav),
        )
    kill = KillSwitch(config.kill_switch)
    n = len(dates)
    clock = EventClock(seed=spec.seed)
    for bar_index in range(n):
        clock.schedule(bar_index, EventKind.MARKET, {})
        if bar_index >= 1:
            clock.schedule(bar_index, EventKind.FILL, {})
            clock.schedule(bar_index, EventKind.MARK, {})
    for signal_index in range(max(n - 1, 0)):
        clock.schedule(signal_index, EventKind.SIGNAL, {"signal_index": signal_index})

    def _log(kind: str, bar_index: int, **payload: object) -> None:
        state.events.append(
            {"kind": kind, "bar_index": bar_index, "time": dates[bar_index], **payload}
        )

    def on_market(bar_index: int) -> None:
        _log("MARKET", bar_index)
        if state.stopped:
            return
        session = session_date(dates[bar_index])
        if not state.sessions or state.sessions[-1] != session:
            state.sessions.append(session)
        if state.constraints is not None and spec.settlement_bars > 0:
            state.constraints.mature(bar_index)
        if bar_index == 0:
            return
        state.nav_checked = False
        state.traded_turn = 0.0
        _ingest(
            state,
            day_rows.get(dates[bar_index], []),
            use_open=True,
            stale_limit=config.risk_gate.stale_price_bars,
        )

    def on_signal(bar_index: int, signal_index: int) -> None:
        _log("SIGNAL", bar_index, signal_index=signal_index)
        if state.stopped:
            return
        when = dates[signal_index]
        decision, advs, vols = _signal_snapshot(day_rows.get(when, []))
        if when in weights_by_date:
            state.last_target = dict(weights_by_date[when])
        state.snaps[signal_index] = _Snap(
            adv=advs, vol=vols, decision=decision, target=dict(state.last_target)
        )
        step = _delay_bars(dates, signal_index, spec.signal_to_order, spec.signal_to_order_bars)
        order_bar = signal_index + step
        if order_bar < n:
            clock.schedule(order_bar, EventKind.ORDER, {"signal_index": signal_index})

    def on_order(bar_index: int, signal_index: int) -> None:
        _log("ORDER", bar_index, signal_index=signal_index)
        if state.stopped:
            return
        step = _delay_bars(dates, bar_index, spec.order_to_exchange, spec.order_to_exchange_bars)
        ex_bar = bar_index + step
        if ex_bar < n:
            clock.schedule(ex_bar, EventKind.EXCHANGE, {"signal_index": signal_index})

    def on_exchange(bar_index: int, signal_index: int) -> None:
        _log("EXCHANGE", bar_index, signal_index=signal_index)
        if state.stopped:
            return
        fill_bar = bar_index + 1
        if fill_bar < n:
            clock.schedule(fill_bar, EventKind.FILL, {"signal_index": signal_index})
        else:
            state.events.append(
                {
                    "kind": "CANCEL",
                    "bar_index": bar_index,
                    "signal_index": signal_index,
                    "reason": "expired_after_sample",
                }
            )

    def on_fill(bar_index: int, signal_index: int | None) -> None:
        _log("FILL", bar_index, signal_index=signal_index)
        if state.stopped or signal_index is None:
            return
        if signal_index not in state.snaps:
            return
        exec_time = dates[bar_index]
        signal_time = dates[signal_index]
        if spec.fill_model == "next_open":
            _rebalance_next_open(
                state,
                spec=spec,
                schedule=schedule,
                config=config,
                signal_index=signal_index,
                exec_time=exec_time,
                signal_time=signal_time,
                bar_index=bar_index,
                risk_overlay=risk_overlay,
                bars=bars,
                kill=kill,
            )
        else:
            _rebalance_path(
                state,
                spec=spec,
                schedule=schedule,
                config=config,
                signal_index=signal_index,
                exec_time=exec_time,
                signal_time=signal_time,
                bar_index=bar_index,
                dates=dates,
                rows=day_rows,
                books=book_index,
                risk_overlay=risk_overlay,
                bars=bars,
                kill=kill,
            )

    def handler(event: Any) -> None:
        bar_index = int(event.bar_index)
        payload = event.payload
        if event.kind is EventKind.MARKET:
            on_market(bar_index)
        elif event.kind is EventKind.SIGNAL:
            on_signal(bar_index, int(payload["signal_index"]))
        elif event.kind is EventKind.ORDER:
            on_order(bar_index, int(payload["signal_index"]))
        elif event.kind is EventKind.EXCHANGE:
            on_exchange(bar_index, int(payload["signal_index"]))
        elif event.kind is EventKind.FILL:
            signal_index = payload.get("signal_index")
            if signal_index is None:
                return
            on_fill(bar_index, int(signal_index))
        elif event.kind is EventKind.MARK:
            _log("MARK", bar_index)
            if not state.stopped:
                _mark_equity(
                    state,
                    config=config,
                    spec=spec,
                    exec_time=dates[bar_index],
                    risk_overlay=risk_overlay,
                )

    # Placeholder FILL events carry no signal. The exchange handler schedules
    # the real FILL with a signal index. Both share the phase, FIFO by seq.
    # A placeholder that arrives before the exchange's FILL (it was scheduled
    # at init, so its seq is lower) must not consume the bar. The real FILL
    # is scheduled later and therefore runs after the placeholder in the same
    # phase. ``on_fill`` ignores an empty payload; the later event trades.
    clock.run(handler)

    built = _build_result(
        navs=state.navs,
        fill_rows=state.fill_rows,
        cost_sum=state.cost_sum,
        reject_count=state.reject_count,
        cash_reject_count=state.cash_reject_count,
        halt_count=state.halt_count,
        synthetic=synthetic,
        config=config,
        initial_nav=initial_nav,
    )
    built.metrics["garch_risk_overlay_dates"] = state.garch_overlay_dates
    built.metrics["realized_garch_risk_overlay_dates"] = state.realized_garch_overlay_dates
    built.metrics["execution_sim"] = True
    built.metrics["research_only"] = True
    built.metrics["live_pnl_claim"] = False
    if risk_overlay is not None:
        built.metrics["book_risk_overlay"] = risk_overlay.snapshot()
    warnings = list(state.warnings)
    if state.constraints is not None:
        warnings.extend(state.constraints.warnings)
    gfv_count = 0 if state.constraints is None else state.constraints.gfv_count
    day_trades = 0 if state.constraints is None else len(state.constraints.day_trades)
    built.metrics["execution_warnings"] = len(warnings)
    built.metrics["cancel_replace_count"] = state.cancel_replace_count
    return EventSimResult(
        result=built,
        warnings=warnings,
        events=state.events,
        min_cash=float(state.min_cash),
        cancel_replace_count=state.cancel_replace_count,
        day_trade_count=day_trades,
        gfv_count=gfv_count,
        seed=spec.seed,
    )
