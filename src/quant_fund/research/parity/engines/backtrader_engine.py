"""backtrader adapter.

Convention mapping to the native NEXT_OPEN engine:

- ``Strategy.next()`` at bar ``t`` issues ``order_target_percent`` Market
  orders; backtrader fills them at bar ``t+1`` open — the same signal-close ->
  next-open path as the native engine.
- KNOWN convention difference (classified ``sizing_nav_basis`` in the
  reconciler): ``order_target_percent`` sizes against broker value and the
  close price *at order creation* (bar ``t``), while the native engine sizes
  ``tw * nav_exec / open[t+1]`` at execution time. Overnight gaps therefore
  produce small share/P&L deltas in exactly this adapter.
- ``commission`` maps to flat ``commission_bps`` (percent-of-notional, no
  minimum) matching the native ``commission_cost`` formula.
- backtrader is a legacy package: on pandas >= 2 it needs the
  ``Series.iteritems -> Series.items`` compat shim (recorded in notes).
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl

from quant_fund.research.parity.canonical import (
    EngineRun,
    empty_trades,
    marks_from_bars,
    positions_from_trades,
)
from quant_fund.research.parity.strategies import StrategyPlan, carried_weight_maps

ENGINE = "backtrader"


def _pandas_compat() -> list[str]:
    notes = []
    import pandas as pd

    if not hasattr(pd.Series, "iteritems"):
        pd.Series.iteritems = pd.Series.items  # type: ignore[attr-defined]
        notes.append("applied pandas>=2 Series.iteritems->items compat shim")
    return notes


def run(
    panel: pl.DataFrame,
    plan: StrategyPlan,
    *,
    initial_nav: float = 1_000_000.0,
    commission_bps: float = 0.0,
) -> EngineRun:
    import backtrader as bt
    import pandas as pd

    from quant_fund.research.parity.synthetic import wide_ohlc

    notes = _pandas_compat()
    wide = wide_ohlc(panel)
    index: pd.DatetimeIndex = wide["index"]
    symbols: list[str] = wide["symbols"]
    carried = carried_weight_maps(plan.weights, list(index.to_pydatetime()))

    fills: list[dict] = []
    equity_rows: list[dict] = []
    rejected = 0

    class _Rebalance(bt.Strategy):
        params = {"targets": carried, "symbols": symbols}

        def __init__(self) -> None:
            self._bar = -1

        def notify_order(self, order: bt.Order) -> None:
            nonlocal rejected
            if order.status == order.Completed:
                dt = bt.num2date(order.executed.dt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                fills.append(
                    {
                        "engine": ENGINE,
                        "strategy": plan.name,
                        "security_id": order.data._name,
                        "fill_time": dt,
                        "side": 1.0 if order.executed.size > 0 else -1.0,
                        "quantity": abs(float(order.executed.size)),
                        "price": float(order.executed.price),
                        "fee": float(order.executed.comm),
                    }
                )
            elif order.status in (order.Rejected, order.Margin, order.Canceled):
                rejected += 1

        def next(self) -> None:
            self._bar += 1
            i = min(self._bar, len(self.p.targets) - 1)
            book = self.p.targets[i]
            for d in self.datas:
                self.order_target_percent(data=d, target=book.get(d._name, 0.0))
            dt = bt.num2date(self.data.datetime[0])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            equity_rows.append(
                {
                    "engine": ENGINE,
                    "strategy": plan.name,
                    "event_time": dt,
                    "nav": float(self.broker.getvalue()),
                    "cash": float(self.broker.getcash()),
                }
            )

    cerebro = bt.Cerebro(stdstats=False)
    for sid in symbols:
        df = pd.DataFrame(
            {
                "open": np.asarray(wide["open"][sid], dtype=float),
                "high": np.asarray(wide["high"][sid], dtype=float),
                "low": np.asarray(wide["low"][sid], dtype=float),
                "close": np.asarray(wide["close"][sid], dtype=float),
                "volume": 1_000_000.0,
                "openinterest": 0.0,
            },
            index=index,
        )
        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=sid))
    cerebro.broker.setcash(initial_nav)
    cerebro.broker.setcommission(commission=float(commission_bps) / 1e4)
    cerebro.addstrategy(_Rebalance, targets=carried, symbols=symbols)
    cerebro.run()

    trades = (
        pl.DataFrame(fills).with_columns(
            pl.col("fill_time").cast(pl.Datetime("us", "UTC"))
        )
        if fills
        else empty_trades()
    )
    equity = pl.DataFrame(equity_rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC"))
    )
    positions = positions_from_trades(
        trades, marks_from_bars(panel), engine=ENGINE, strategy=plan.name
    )
    import backtrader as _bt

    notes += [
        f"backtrader={_bt.__version__}",
        f"rejected_orders={rejected}",
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
            "rejected_orders": rejected,
            "final_value": float(equity_rows[-1]["nav"]) if equity_rows else float("nan"),
        },
    )
