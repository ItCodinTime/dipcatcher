"""Replay one tape through a strategy into :class:`SimulatedBroker` only.

The backtest reference and the shadow runner both call
:func:`replay_session`. The only broker constructed here is
``SimulatedBroker``. Live mode is refused before any order is built.

Fill convention follows the config: next-open (the repository default)
or same-bar close when a close auction is explicitly enabled. The last
next-open decision is recorded and left unfilled, matching the backtest
engine's horizon.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import polars as pl

from quant_fund.config.models import AppConfig, FillConvention, RuntimeMode
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.parity.session import (
    Bar,
    MarketSession,
    bars_to_frame,
    end_marks,
    iter_bar_groups,
    session_fingerprint,
)
from quant_fund.parity.strategy import (
    DecideFn,
    DecisionContext,
    Strategy,
    export_strategy_state,
    load_strategy_state,
    resolve_decide,
    round_weight,
)
from quant_fund.parity.trace import CallTracer, accept_strategy_modules, digest_calls
from quant_fund.schemas.orders import OrderSide

Origin = Literal["backtest", "shadow"]
Pacing = Literal["accelerated", "wall_clock"]

_LEDGER_SCHEMA: dict[str, Any] = {
    "event_time": pl.Datetime(time_zone="UTC"),
    "decision_time": pl.Datetime(time_zone="UTC"),
    "security_id": pl.Utf8,
    "source": pl.Utf8,
    "revision_id": pl.Utf8,
    "available_time": pl.Datetime(time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "raw_weight": pl.Float64,
    "rounded_weight": pl.Float64,
    "lot_size": pl.Float64,
    "commission_bps": pl.Float64,
    "half_spread_bps": pl.Float64,
    "impact_y": pl.Float64,
    "bps_per_turnover": pl.Float64,
    "state_digest": pl.Utf8,
    "code_path_digest": pl.Utf8,
    "restarted": pl.Boolean,
    "restart_generation": pl.Int64,
    "exec_open": pl.Float64,
    "exec_close": pl.Float64,
    "exec_source": pl.Utf8,
    "exec_revision": pl.Utf8,
    "fill_signed_qty": pl.Float64,
    "fill_price": pl.Float64,
    "fill_fee": pl.Float64,
    "fill_spread": pl.Float64,
    "fill_impact": pl.Float64,
    "explicit_cost": pl.Float64,
}

FillPriceFn = Callable[..., float]


class ParityValuationError(RuntimeError):
    """A held name has no mark on this bar. The replay refuses to invent one."""


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    """Knobs that the parity checker is designed to attribute.

    Defaults reproduce a next-open paper replay of the shared strategy.
    Tests inject exactly one difference (lag, lot, costs, fill price,
    restart mutator) and expect the checker to name that cause.
    """

    pacing: Pacing = "accelerated"
    speed: float = 1.0
    decision_lag: timedelta = timedelta(0)
    lot_size: float = 0.0
    fill_price: FillPriceFn | None = None
    restart_after: datetime | None = None
    state_mutator: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    sleeper: Callable[[float], None] | None = None
    cost_overrides: Mapping[str, float] | None = None
    initial_cash: float = 1_000_000.0


@dataclass
class ParityRun:
    """One origin's decision ledger and simulated fills.

    ``terminal_delta`` is the change in marked book value from the starting
    cash. It is a simulated execution diagnostic. It is not a research
    score and not a live P&L claim.
    """

    origin: str
    ledger: pl.DataFrame
    fills: pl.DataFrame
    terminal_delta: float
    initial_cash: float
    final_marked_value: float
    end_marks: dict[str, float]
    call_traces: list[list[str]]
    strategy_module: str
    strategy_qualname: str
    code_path_digest: str
    synthetic: bool
    pacing: str
    tape_fingerprint: str
    live_pnl_claim: bool = False
    research_only: bool = True
    rows: list[dict[str, Any]] = field(default_factory=list)


def _refuse_live(config: AppConfig) -> None:
    mode = config.runtime.mode
    if mode is RuntimeMode.LIVE or bool(config.runtime.allow_live):
        raise RuntimeError(
            "parity replay refuses live mode and never submits real orders; "
            "use the simulated broker"
        )


def _config_for_run(config: AppConfig, options: ReplayOptions) -> AppConfig:
    _refuse_live(config)
    cfg = config.model_copy(deep=True)
    if options.cost_overrides:
        unknown = set(options.cost_overrides) - set(type(cfg.costs).model_fields)
        if unknown:
            raise ValueError(f"unknown cost overrides: {sorted(unknown)}")
        cfg.costs = type(cfg.costs).model_validate(
            {**cfg.costs.model_dump(), **options.cost_overrides}
        )
    _refuse_live(cfg)
    return cfg


def _state_digest(
    cash: float, shares: Mapping[str, float], strategy_state: Mapping[str, Any]
) -> str:
    payload = {
        "cash": float(cash),
        "shares": {key: float(shares[key]) for key in sorted(shares) if float(shares[key]) != 0.0},
        "strategy": dict(strategy_state),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _use_next_open(config: AppConfig) -> bool:
    return (
        config.execution.fill is FillConvention.NEXT_OPEN
        and not config.execution.allow_close_auction
    )


def _default_fill_price(
    *,
    side: str,
    open_px: float,
    close_px: float,
    decision_px: float,
    use_close: bool,
) -> float:
    del side, decision_px
    return close_px if use_close else open_px


@dataclass
class _Pending:
    bar_time: datetime
    decision_time: datetime
    rows: dict[str, dict[str, Any]]
    weights: dict[str, float]
    decision_prices: dict[str, float]
    group: tuple[Bar, ...]


def _blank_fill(row: dict[str, Any]) -> None:
    row["exec_open"] = None
    row["exec_close"] = None
    row["exec_source"] = None
    row["exec_revision"] = None
    row["fill_signed_qty"] = 0.0
    row["fill_price"] = None
    row["fill_fee"] = 0.0
    row["fill_spread"] = 0.0
    row["fill_impact"] = 0.0
    row["explicit_cost"] = 0.0


def _decision_price(pending: _Pending, sid: str, fallback: float) -> float:
    price = pending.decision_prices.get(sid)
    if price is None or price <= 0.0:
        return fallback
    return price


def _execute_pending(
    broker: SimulatedBroker,
    pending: _Pending,
    exec_group: tuple[Bar, ...],
    *,
    use_close: bool,
    fill_price: FillPriceFn,
    fill_rows: list[dict[str, Any]],
) -> None:
    exec_by_id = {bar.security_id: bar for bar in exec_group}
    px = dict(broker.last_marks)
    for bar in exec_group:
        px[bar.security_id] = bar.close if use_close else bar.open
    missing = sorted(
        sid for sid, qty in broker.shares.items() if float(qty) != 0.0 and sid not in px
    )
    if missing:
        raise ParityValuationError("held security has no execution mark: " + ", ".join(missing))
    broker.mark(px)
    nav = broker.nav(px)
    orders = broker.target_to_orders(
        pending.weights,
        px,
        signal_time=pending.bar_time,
        order_time=exec_group[0].event_time,
        nav=nav,
    )
    fills_by_sid: dict[str, list[Any]] = {}
    for order in orders:
        sid = order.security_id
        tape = exec_by_id.get(sid)
        if tape is None:
            continue
        side = "buy" if order.side is OrderSide.BUY else "sell"
        decision_px = _decision_price(pending, sid, tape.close if use_close else tape.open)
        price = float(
            fill_price(
                side=side,
                open_px=tape.open,
                close_px=tape.close,
                decision_px=decision_px,
                use_close=use_close,
            )
        )
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError("fill price must be finite and strictly positive")
        cash_before = broker.cash
        record = broker.submit(
            order,
            price=price,
            nav=nav,
            adv_dollars=tape.adv,
            sigma=tape.vol_20,
            decision_price=decision_px,
        )
        explicit = 0.0
        signed = 0.0
        if record.fill is not None:
            signed = float(record.fill.quantity)
            if order.side is OrderSide.SELL:
                signed = -signed
            # Broker cash change is -(signed_qty * price + explicit costs).
            explicit = -(broker.cash - cash_before) - signed * float(record.fill.price)
            if explicit < -1e-8:
                raise RuntimeError("simulated fill credited costs; refusing to continue")
            explicit = max(0.0, explicit)
            fill_rows.append(
                {
                    "event_time": pending.bar_time,
                    "security_id": sid,
                    "signed_qty": signed,
                    "price": float(record.fill.price),
                    "fee": float(record.fill.fee),
                    "spread": float(record.fill.spread_cost),
                    "impact": float(record.fill.impact_cost),
                    "explicit_cost": explicit,
                    "decision_price": float(decision_px),
                }
            )
        fills_by_sid.setdefault(sid, []).append((record, signed, explicit, tape))

    for sid, row in pending.rows.items():
        tape_row = exec_by_id.get(sid)
        if tape_row is not None:
            row["exec_open"] = tape_row.open
            row["exec_close"] = tape_row.close
            row["exec_source"] = tape_row.source
            row["exec_revision"] = tape_row.revision_id
        matched = fills_by_sid.get(sid, [])
        if not matched or all(item[0].fill is None for item in matched):
            continue
        signed_qty = 0.0
        notional = 0.0
        fee = 0.0
        spread = 0.0
        impact = 0.0
        explicit = 0.0
        for record, signed, cost, _bar in matched:
            if record.fill is None:
                continue
            signed_qty += signed
            notional += abs(signed) * float(record.fill.price)
            fee += float(record.fill.fee)
            spread += float(record.fill.spread_cost)
            impact += float(record.fill.impact_cost)
            explicit += cost
        row["fill_signed_qty"] = signed_qty
        row["fill_price"] = (notional / abs(signed_qty)) if signed_qty != 0.0 else None
        row["fill_fee"] = fee
        row["fill_spread"] = spread
        row["fill_impact"] = impact
        row["explicit_cost"] = explicit


def _pace(
    options: ReplayOptions,
    previous: datetime | None,
    decision_time: datetime,
) -> datetime:
    if options.pacing == "accelerated" or previous is None:
        return decision_time
    if options.pacing != "wall_clock":
        raise ValueError(f"unknown pacing {options.pacing!r}")
    if not math.isfinite(options.speed) or options.speed <= 0.0:
        raise ValueError("wall-clock speed must be finite and strictly positive")
    delta = (decision_time - previous).total_seconds()
    wait = max(0.0, delta / float(options.speed))
    sleeper = options.sleeper
    if sleeper is None:
        import time

        sleeper = time.sleep
    if wait > 0.0:
        sleeper(wait)
    return decision_time


def _restart_broker(
    broker: SimulatedBroker,
    config: AppConfig,
    strategy: Strategy | DecideFn,
    mutator: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> tuple[SimulatedBroker, int]:
    snap = copy.deepcopy(broker.to_dict())
    strategy_state = export_strategy_state(strategy)
    if mutator is not None:
        updated = mutator(snap)
        if not isinstance(updated, dict):
            raise TypeError("state_mutator must return a broker state dict")
        snap = updated
    restored = SimulatedBroker.from_state(config, snap)
    if type(restored) is not SimulatedBroker:
        raise RuntimeError("parity restart must restore a SimulatedBroker")
    load_strategy_state(strategy, strategy_state)
    return restored, 1


def replay_session(
    session: MarketSession | Iterable[Bar],
    strategy: Strategy | DecideFn,
    config: AppConfig,
    *,
    origin: Origin,
    options: ReplayOptions | None = None,
) -> ParityRun:
    """Run ``strategy`` over ``session`` inside a simulated broker.

    ``origin`` is passed through on the decision context and is otherwise
    ignored by the runner. Both origins share this function.
    """
    if origin not in ("backtest", "shadow"):
        raise ValueError("origin must be 'backtest' or 'shadow'")
    opts = options or ReplayOptions()
    if not math.isfinite(opts.lot_size) or opts.lot_size < 0.0:
        raise ValueError("lot_size must be finite and non-negative")
    if not math.isfinite(opts.initial_cash) or opts.initial_cash <= 0.0:
        raise ValueError("initial_cash must be finite and strictly positive")
    if opts.pacing not in ("accelerated", "wall_clock"):
        raise ValueError(f"unknown pacing {opts.pacing!r}")
    if opts.pacing == "wall_clock" and (not math.isfinite(opts.speed) or opts.speed <= 0.0):
        raise ValueError("wall-clock speed must be finite and strictly positive")
    cfg = _config_for_run(config, opts)
    broker = SimulatedBroker(
        config=cfg,
        initial_cash=float(opts.initial_cash),
        slot="parity",
        allow_capital=True,
    )
    if type(broker) is not SimulatedBroker:
        raise RuntimeError("parity replay only constructs SimulatedBroker")

    decide = resolve_decide(strategy)
    strategy_module = str(getattr(decide, "__module__", "") or "")
    strategy_qualname = str(getattr(decide, "__qualname__", "") or "")
    use_next_open = _use_next_open(cfg)
    fill_price = opts.fill_price or _default_fill_price
    bars = session.bars if isinstance(session, MarketSession) else session
    synthetic = isinstance(session, MarketSession) and session.synthetic

    rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    traces: list[list[str]] = []
    seen: list[Bar] = []
    pending: _Pending | None = None
    previous_decision: datetime | None = None
    restart_generation = 0
    restarted_next = False
    restart_done = False
    fingerprint_bars: list[Bar] = []
    mark_ages: dict[str, int] = {}

    def _maybe_restart(after: datetime) -> None:
        nonlocal broker, restart_generation, restarted_next, restart_done, mark_ages
        if restart_done or opts.restart_after is None or after != opts.restart_after:
            return
        broker, step = _restart_broker(broker, cfg, strategy, opts.state_mutator)
        restart_generation += step
        restarted_next = True
        restart_done = True
        # Restored marks have no persisted age. The next decision treats them
        # as just observed, then ages them if later bars omit the name.
        mark_ages = {sid: 0 for sid, qty in broker.shares.items() if float(qty) != 0.0}

    for group in iter_bar_groups(bars):
        fingerprint_bars.extend(group)
        if not synthetic and any(bar.source.lower() == "synthetic" for bar in group):
            synthetic = True
        if use_next_open and pending is not None:
            _execute_pending(
                broker,
                pending,
                group,
                use_close=False,
                fill_price=fill_price,
                fill_rows=fill_rows,
            )
            rows.extend(pending.rows[sid] for sid in sorted(pending.rows))
            _maybe_restart(pending.bar_time)
            pending = None

        seen.extend(group)
        bar_time = group[0].event_time
        decision_time = bar_time + opts.decision_lag
        previous_decision = _pace(opts, previous_decision, decision_time)
        visible = [bar for bar in seen if bar.available_time <= decision_time]
        visible_close: dict[str, float] = {}
        for bar in visible:
            visible_close[bar.security_id] = bar.close
        fresh_ids = {bar.security_id for bar in group if bar.available_time <= decision_time}
        for sid in list(mark_ages):
            mark_ages[sid] = 0 if sid in fresh_ids else mark_ages[sid] + 1
        for sid in fresh_ids:
            mark_ages[sid] = 0
        stale_limit = int(cfg.risk_gate.stale_price_bars)
        stale_held = sorted(
            sid
            for sid, qty in broker.shares.items()
            if float(qty) != 0.0 and mark_ages.get(sid, 0) > stale_limit
        )
        if stale_held:
            raise ParityValuationError(
                "stale valuation mark for held security: " + ", ".join(stale_held)
            )
        mark_px = dict(broker.last_marks)
        mark_px.update(visible_close)
        missing = sorted(
            sid for sid, qty in broker.shares.items() if float(qty) != 0.0 and sid not in mark_px
        )
        if missing:
            raise ParityValuationError(
                "held security has no decision-time mark: " + ", ".join(missing)
            )
        if mark_px:
            broker.mark(mark_px)
            marked = float(broker.nav(mark_px))
        else:
            marked = float(broker.cash)
        strategy_state = export_strategy_state(strategy)
        positions = {sid: float(qty) for sid, qty in broker.shares.items() if float(qty) != 0.0}
        digest = _state_digest(broker.cash, positions, strategy_state)
        ctx = DecisionContext(
            origin=origin,
            decision_time=decision_time,
            bar_time=bar_time,
            history=bars_to_frame(visible),
            positions=positions,
            cash=float(broker.cash),
            marked_value=marked,
            restarted=restarted_next,
            restart_generation=restart_generation,
        )
        with CallTracer(accept_module=accept_strategy_modules(strategy_module)) as tracer:
            decision = decide(ctx)
        if not hasattr(decision, "weights"):
            raise TypeError("strategy must return a TargetDecision")
        weights = {str(sid): float(weight) for sid, weight in decision.weights.items()}
        restarted_next = False
        trace = list(tracer.calls)
        traces.append(trace)
        code_digest = digest_calls(trace)
        last_visible: dict[str, float] = {}
        for bar in visible:
            last_visible[bar.security_id] = bar.close

        pending_rows: dict[str, dict[str, Any]] = {}
        for bar in group:
            raw = float(weights.get(bar.security_id, 0.0))
            pending_rows[bar.security_id] = {
                "event_time": bar_time,
                "decision_time": decision_time,
                "security_id": bar.security_id,
                "source": bar.source,
                "revision_id": bar.revision_id,
                "available_time": bar.available_time,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "raw_weight": raw,
                "rounded_weight": round_weight(raw, opts.lot_size),
                "lot_size": float(opts.lot_size),
                "commission_bps": float(cfg.costs.commission_bps),
                "half_spread_bps": float(cfg.costs.half_spread_bps),
                "impact_y": float(cfg.costs.impact_y),
                "bps_per_turnover": float(cfg.costs.bps_per_turnover),
                "state_digest": digest,
                "code_path_digest": code_digest,
                "restarted": ctx.restarted,
                "restart_generation": int(ctx.restart_generation),
            }
            _blank_fill(pending_rows[bar.security_id])
        # Targets are sized from rounded weights of names on this bar.
        sized = {sid: pending_rows[sid]["rounded_weight"] for sid in pending_rows}
        new_pending = _Pending(
            bar_time=bar_time,
            decision_time=decision_time,
            rows=pending_rows,
            weights=sized,
            decision_prices=last_visible,
            group=group,
        )
        if use_next_open:
            pending = new_pending
            continue
        _execute_pending(
            broker,
            new_pending,
            group,
            use_close=True,
            fill_price=fill_price,
            fill_rows=fill_rows,
        )
        rows.extend(new_pending.rows[sid] for sid in sorted(new_pending.rows))
        _maybe_restart(bar_time)

    if pending is not None:
        rows.extend(pending.rows[sid] for sid in sorted(pending.rows))

    marks = end_marks(fingerprint_bars)
    if marks:
        missing = sorted(
            sid for sid, qty in broker.shares.items() if float(qty) != 0.0 and sid not in marks
        )
        if missing:
            raise ParityValuationError("held security has no terminal mark: " + ", ".join(missing))
        broker.mark(marks)
        final_marked = float(broker.nav(marks))
    else:
        final_marked = float(broker.cash)
    flat_trace = [name for trace in traces for name in trace]
    ledger = (
        pl.DataFrame(rows, schema=_LEDGER_SCHEMA) if rows else pl.DataFrame(schema=_LEDGER_SCHEMA)
    )
    fills = (
        pl.DataFrame(fill_rows)
        if fill_rows
        else pl.DataFrame(
            schema={
                "event_time": pl.Datetime(time_zone="UTC"),
                "security_id": pl.Utf8,
                "signed_qty": pl.Float64,
                "price": pl.Float64,
                "fee": pl.Float64,
                "spread": pl.Float64,
                "impact": pl.Float64,
                "explicit_cost": pl.Float64,
                "decision_price": pl.Float64,
            }
        )
    )
    return ParityRun(
        origin=origin,
        ledger=ledger,
        fills=fills,
        terminal_delta=float(final_marked - opts.initial_cash),
        initial_cash=float(opts.initial_cash),
        final_marked_value=float(final_marked),
        end_marks=marks,
        call_traces=traces,
        strategy_module=strategy_module,
        strategy_qualname=strategy_qualname,
        code_path_digest=digest_calls(flat_trace),
        synthetic=synthetic,
        pacing=opts.pacing,
        tape_fingerprint=session_fingerprint(fingerprint_bars),
        rows=rows,
    )
