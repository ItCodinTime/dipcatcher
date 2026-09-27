"""vectorbt adapter (vbt >= 1.x API).

Convention mapping to the native NEXT_OPEN engine:

- A target-weight row decided at close ``t`` is an order arriving at bar
  ``t+1`` -> the dense carried weight frame is shifted one bar and used as
  ``size`` with ``SizeType.TargetPercent``.
- ``price`` = bar open (execution), ``val_price`` = bar open (the portfolio
  value the percent order is sized against) — the native engine also sizes on
  execution-time marks. ``close`` marks the daily equity curve.
- ``fees`` maps to flat commission in percent (native ``commission_bps``).
- Fractional shares (``size_granularity=None``) match the native engine's
  unrounded share accounting.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_fund.research.parity.canonical import (
    EngineRun,
    empty_trades,
    marks_from_bars,
    positions_from_trades,
)
from quant_fund.research.parity.strategies import StrategyPlan, carried_weight_maps

ENGINE = "vectorbt"


def _weight_matrix(
    panel: pl.DataFrame, plan: StrategyPlan, index: "object", symbols: list[str]
) -> "object":
    """Dense (n_days, n_symbols) carried targets shifted one bar for orders."""
    import pandas as pd

    dates = list(index)
    carried = carried_weight_maps(plan.weights, dates)
    dense = np.zeros((len(dates), len(symbols)))
    for i, book in enumerate(carried):
        for j, s in enumerate(symbols):
            dense[i, j] = book.get(s, 0.0)
    shifted = np.vstack([np.full((1, len(symbols)), np.nan), dense[:-1]])
    return pd.DataFrame(shifted, index=index, columns=symbols)


def run(
    panel: pl.DataFrame,
    plan: StrategyPlan,
    *,
    initial_nav: float = 1_000_000.0,
    commission_bps: float = 0.0,
) -> EngineRun:
    import vectorbt as vbt
    from vectorbt.portfolio.enums import Direction, SizeType

    from quant_fund.research.parity.synthetic import wide_ohlc

    wide = wide_ohlc(panel)
    index = wide["index"]
    symbols = wide["symbols"]
    size = _weight_matrix(panel, plan, index, symbols)
    pf = vbt.Portfolio.from_orders(
        close=wide["close"],
        size=size,
        size_type=SizeType.TargetPercent,
        direction=Direction.LongOnly,
        price=wide["open"],
        val_price=wide["open"],
        fees=float(commission_bps) / 1e4,
        fixed_fees=0.0,
        slippage=0.0,
        init_cash=initial_nav,
        cash_sharing=True,
        freq="1D",
    )
    value = np.asarray(pf.value(), dtype=float).reshape(-1)
    cash = np.asarray(pf.cash(), dtype=float).reshape(-1)
    equity = pl.DataFrame(
        {
            "engine": [ENGINE] * len(value),
            "strategy": [plan.name] * len(value),
            "event_time": index.to_pydatetime().tolist(),
            "nav": value.tolist(),
            "cash": cash.tolist(),
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))

    orders = pf.get_orders()
    records = orders.records_arr
    if len(records):
        names = records.dtype.names
        col_idx = names.index("col") if "col" in names else None
        idx_idx = names.index("idx") if "idx" in names else None
        size_idx = names.index("size")
        price_idx = names.index("price")
        side_idx = names.index("side") if "side" in names else None
        fee_idx = names.index("fees") if "fees" in names else None
        tz_index = index.tz_localize("UTC") if index.tz is None else index
        rows = []
        for rec in records:
            r = rec.tolist() if hasattr(rec, "tolist") else tuple(rec)
            ts = tz_index[int(r[idx_idx])]
            sid = symbols[int(r[col_idx])]
            raw_side = int(r[side_idx]) if side_idx is not None else 0
            # vbt OrderSide: 0 = buy, 1 = sell (enums confirmed against orders)
            rows.append(
                {
                    "engine": ENGINE,
                    "strategy": plan.name,
                    "security_id": sid,
                    "fill_time": ts.to_pydatetime(),
                    "side": 1.0 if raw_side == 0 else -1.0,
                    "quantity": abs(float(r[size_idx])),
                    "price": float(r[price_idx]),
                    "fee": float(r[fee_idx]) if fee_idx is not None else 0.0,
                }
            )
        trades = pl.DataFrame(rows).with_columns(
            pl.col("fill_time").cast(pl.Datetime("us", "UTC"))
        )
    else:
        trades = empty_trades()
    positions = positions_from_trades(
        trades, marks_from_bars(panel), engine=ENGINE, strategy=plan.name
    )
    return EngineRun(
        engine=ENGINE,
        strategy=plan.name,
        status="ok",
        trades=trades,
        positions=positions,
        equity=equity,
        notes=[f"vectorbt={vbt.__version__}", "label=SYNTHETIC research-only"],
        metrics={"n_orders": int(len(records)), "final_value": float(value[-1])},
    )
