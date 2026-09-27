"""NautilusTrader adapter (optional engine).

Wiring: one ``SIM`` venue (CASH account, ``bar_execution=True``), one
``Equity`` per symbol with a micro lot size (``Quantity`` precision 6) so
fractional fills are expressible, and daily ``Bar`` streams per instrument.

Convention mapping:

- ``on_bar`` at bar ``t`` submits MARKET orders for that instrument; with
  ``bar_execution`` the simulated exchange fills them when the next bar
  arrives — at bar ``t+1`` open. This mirrors NEXT_OPEN.
- KNOWN convention difference (``sizing_nav_basis``): nautilus has no
  order-target-percent primitive; the adapter sizes
  ``delta = tw * equity_t / close_t - position`` at *signal-time* marks —
  the same basis backtrader uses. Overnight gaps therefore appear as
  quantity deltas vs the native exec-time sizing.
- Commission maps to the instrument ``taker_fee`` (fraction of notional).

Install caveat: ``nautilus_trader`` pins ``fsspec<=2026.2.0`` while the shared
env carries a newer fsspec; the wheel imports and runs regardless (installed
via ``uv pip install --no-deps``). This is documented in run notes.
"""

from __future__ import annotations

from datetime import UTC
from decimal import Decimal

import polars as pl

from quant_fund.research.parity.canonical import (
    EngineRun,
    empty_trades,
    marks_from_bars,
    positions_from_trades,
)
from quant_fund.research.parity.strategies import StrategyPlan, carried_weight_maps

ENGINE = "nautilus"
VENUE_NAME = "PARSIM"


def run(
    panel: pl.DataFrame,
    plan: StrategyPlan,
    *,
    initial_nav: float = 1_000_000.0,
    commission_bps: float = 0.0,
) -> EngineRun:
    import nautilus_trader
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import (
        AccountType,
        BookType,
        OmsType,
        OrderSide,
        TimeInForce,
    )
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.instruments import CurrencyPair
    from nautilus_trader.model.objects import Currency, Money, Price, Quantity
    from nautilus_trader.trading.strategy import Strategy

    notes: list[str] = []
    dates = sorted(panel["event_time"].unique().to_list())
    symbols = sorted(panel["security_id"].unique().to_list())
    carried = carried_weight_maps(plan.weights, dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    engine = BacktestEngine()
    venue = Venue(VENUE_NAME)
    engine.add_venue(
        venue,
        OmsType.NETTING,
        AccountType.CASH,
        [Money(initial_nav, USD)],
        base_currency=None,
        book_type=BookType.L1_MBP,
        bar_execution=True,
    )
    taker_fee = Decimal(str(float(commission_bps) / 1e4))
    instruments = {}
    bar_types = {}
    for sid in symbols:
        iid = InstrumentId(symbol=Symbol(sid), venue=venue)
        # CurrencyPair is the only nautilus instrument exposing
        # size_precision/size_increment; Equity pins size_precision=0 which
        # would drown every convention diff in integer-lot rounding. The
        # synthetic names are labeled pairs on a SIM venue.
        instruments[sid] = CurrencyPair(
            instrument_id=iid,
            raw_symbol=Symbol(sid),
            base_currency=Currency.from_str(sid),
            quote_currency=USD,
            price_precision=6,
            size_precision=6,
            price_increment=Price.from_str("0.000001"),
            size_increment=Quantity.from_str("0.000001"),
            taker_fee=taker_fee,
            ts_event=0,
            ts_init=0,
        )
        bar_types[sid] = BarType.from_str(f"{sid}.{VENUE_NAME}-1-DAY-LAST-EXTERNAL")
        engine.add_instrument(instruments[sid])

    bars = []
    for row in panel.iter_rows(named=True):
        sid = str(row["security_id"])
        ts = int(row["event_time"].timestamp() * 1e9)
        bars.append(
            Bar(
                bar_type=bar_types[sid],
                open=Price(float(row["open"]), 6),
                high=Price(float(row["high"]), 6),
                low=Price(float(row["low"]), 6),
                close=Price(float(row["close"]), 6),
                volume=Quantity(float(row["volume"]), 6),
                ts_event=ts,
                ts_init=ts,
            )
        )
    engine.add_data(bars)

    # Per-instrument order state; targets carried per bar index.
    state: dict[str, object] = {"denied": 0}

    class ParityStrategy(Strategy):
        def on_start(self) -> None:
            self._targets = carried
            self._sid_of_bt = {str(bt.instrument_id): s for s, bt in bar_types.items()}
            for bt in bar_types.values():
                self.subscribe_bars(bt)

        def on_bar(self, bar: Bar) -> None:
            sid = self._sid_of_bt[str(bar.bar_type.instrument_id)]
            dt = bar.ts_event
            # bar.ts_event ns -> python datetime for the carried lookup
            import pandas as pd

            day = pd.Timestamp(dt, unit="ns", tz="UTC").to_pydatetime()
            i = date_to_idx.get(day)
            if i is None:
                return
            tw = self._targets[i].get(sid, 0.0)
            iid = instruments[sid].id
            close = float(bar.close)
            # Signal-time sizing: equity marked on this bar's close.
            account = self.portfolio.account(venue)
            cash = float(account.balance_total(USD).as_double())
            equity_est = cash
            for s2 in symbols:
                pos = float(self.portfolio.net_position(InstrumentId(
                    symbol=Symbol(s2), venue=venue
                )) or 0.0)
                if pos:
                    last = self.cache.bar(bar_types[s2])
                    mark = float(last.close) if last is not None else close
                    equity_est += pos * mark
            desired = tw * equity_est / close
            current = float(self.portfolio.net_position(iid) or 0.0)
            delta = desired - current
            if abs(delta) < 1e-6:
                return
            order = self.order_factory.market(
                iid,
                OrderSide.BUY if delta > 0 else OrderSide.SELL,
                Quantity(abs(delta), 6),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)

        def on_order_denied(self, event) -> None:
            state["denied"] = int(state["denied"]) + 1

    engine.add_strategy(ParityStrategy())
    engine.run()

    # Extract fills from closed orders.
    from nautilus_trader.model.enums import OrderSide as _NOS

    fills_rows: list[dict] = []
    import pandas as pd

    for order in engine.cache.orders_closed():
        for event in order.events:
            if type(event).__name__ != "OrderFilled":
                continue
            ts = pd.Timestamp(event.ts_event, unit="ns", tz="UTC").to_pydatetime()
            fills_rows.append(
                {
                    "engine": ENGINE,
                    "strategy": plan.name,
                    "security_id": str(event.instrument_id.symbol),
                    "fill_time": ts,
                    "side": 1.0 if event.order_side == _NOS.BUY else -1.0,
                    "quantity": float(event.last_qty.as_double()),
                    "price": float(event.last_px.as_double()),
                    "fee": float(event.commission.as_double())
                    if event.commission is not None
                    else 0.0,
                }
            )
    trades = (
        pl.DataFrame(fills_rows).with_columns(
            pl.col("fill_time").cast(pl.Datetime("us", "UTC"))
        )
        if fills_rows
        else empty_trades()
    )
    positions = positions_from_trades(
        trades, marks_from_bars(panel), engine=ENGINE, strategy=plan.name
    )
    # Per-day NAV: cash ledger from fills + close-marked positions (uniform
    # derivation — nautilus CASH accounts do not emit per-bar equity marks).
    marks = marks_from_bars(panel)
    cash_rows = []
    if trades.height:
        ledger = trades.sort("fill_time").with_columns(
            (pl.col("quantity") * pl.col("side") * pl.col("price") * -1.0 - pl.col("fee")).alias(
                "cash_delta"
            )
        )
        by_day = ledger.group_by("fill_time").agg(pl.col("cash_delta").sum())
        cash = initial_nav
        cash_map: dict[object, float] = {}
        for d in dates:
            row = by_day.filter(pl.col("fill_time") == d)
            if row.height:
                cash += float(row["cash_delta"].sum())
            cash_map[d] = cash
        pos_mv = (
            positions.group_by("event_time")
            .agg(pl.col("market_value").sum().alias("mv"))
        )
        mv_map = dict(
            zip(
                pos_mv["event_time"].to_list(),
                pos_mv["mv"].to_list(),
                strict=True,
            )
        )
        for d in dates:
            cash_rows.append(
                {
                    "engine": ENGINE,
                    "strategy": plan.name,
                    "event_time": d,
                    "nav": cash_map[d] + mv_map.get(d, 0.0),
                    "cash": cash_map[d],
                }
            )
    else:
        for d in dates:
            cash_rows.append(
                {
                    "engine": ENGINE,
                    "strategy": plan.name,
                    "event_time": d,
                    "nav": initial_nav,
                    "cash": initial_nav,
                }
            )
    equity = pl.DataFrame(cash_rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC"))
    )
    account = engine.portfolio.account(venue)
    final_balance = float(account.balance_total(USD).as_double())
    notes += [
        f"nautilus_trader={nautilus_trader.__version__}",
        f"denied_orders={state['denied']}",
        f"final_cash_balance={final_balance:.4f}",
        "equity derived from fills + close marks (CASH account has no per-bar marks)",
        "installed --no-deps: upstream pins fsspec<=2026.2.0, env carries newer",
        "label=SYNTHETIC research-only",
    ]
    return EngineRun(
        engine=ENGINE,
        strategy=plan.name,
        status="ok",
        trades=trades,
        positions=positions,
        equity=equity,
        notes=notes,
        metrics={
            "n_orders": int(trades.height),
            "denied_orders": int(state["denied"]),
            "final_value": float(equity["nav"][-1]) if equity.height else float("nan"),
        },
    )
