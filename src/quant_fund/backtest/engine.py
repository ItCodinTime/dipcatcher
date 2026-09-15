"""Event-driven daily backtester. Default fill = next open."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig, FillConvention
from quant_fund.execution.costs import total_cost
from quant_fund.metrics.returns import max_drawdown, sharpe_ratio


@dataclass
class BacktestResult:
    equity: pl.DataFrame
    fills: pl.DataFrame
    metrics: dict[str, float | str | int | bool]
    frictionless: bool
    source_note: str


@dataclass
class Book:
    cash: float
    shares: dict[str, float] = field(default_factory=dict)

    def nav(self, prices: dict[str, float]) -> float:
        pos = sum(self.shares.get(s, 0.0) * prices.get(s, 0.0) for s in self.shares)
        return self.cash + pos


def _valid_price(value: object) -> float | None:
    if value is None:
        return None
    price = float(value)
    return price if np.isfinite(price) and price > 0 else None


def run_backtest(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    initial_nav: float = 1_000_000.0,
) -> BacktestResult:
    """`weights` columns: event_time, security_id, target_weight.

    Target computed from close t is executed at next open (unless close auction enabled).
    """
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
    )
    dates = sorted(px["event_time"].unique().to_list())
    book = Book(cash=initial_nav)
    navs: list[dict] = []
    fill_rows: list[dict] = []
    cost_sum = {"commission": 0.0, "spread": 0.0, "impact": 0.0}
    last_marks: dict[str, float] = {}
    synthetic = (
        "synthetic" in set(px["source"].drop_nulls().to_list()) if "source" in px.columns else False
    )

    use_next_open = (
        config.execution.fill is FillConvention.NEXT_OPEN
        and not config.execution.allow_close_auction
    )

    for i, dt in enumerate(dates[:-1] if use_next_open else dates):
        exec_dt = dates[i + 1] if use_next_open else dt
        day_px = px.filter(pl.col("event_time") == exec_dt)
        day_rows = day_px.iter_rows(named=True)
        exec_mark: dict[str, float] = {}
        close_mark = dict(last_marks)
        advs: dict[str, float] = {}
        vols: dict[str, float] = {}
        for row in day_rows:
            sid = str(row["security_id"])
            raw_exec = _valid_price(row["open"] if use_next_open else row["close"])
            if raw_exec is not None:
                exec_mark[sid] = raw_exec
            total_return_mark = _valid_price(row["close_total_return"])
            fallback_mark = _valid_price(row["close"])
            if total_return_mark is not None:
                close_mark[sid] = total_return_mark
            elif fallback_mark is not None:
                close_mark[sid] = fallback_mark
            advs[sid] = _valid_price(row["adv"]) or 1.0
            vols[sid] = _valid_price(row["vol_20"]) or 0.02
        last_marks = dict(close_mark)
        tgt_rows = weights.filter(pl.col("event_time") == dt)
        nav = book.nav(exec_mark) if exec_mark else book.nav(close_mark)
        if nav <= 0:
            break
        target_w = {
            r["security_id"]: float(r["target_weight"]) for r in tgt_rows.iter_rows(named=True)
        }
        ids = set(exec_mark) | set(book.shares) | set(target_w)
        traded_turn = 0.0
        for sid in ids:
            price = exec_mark.get(sid)
            if price is None:
                # A missing execution bar is not an executable zero price.
                continue
            tw = target_w.get(sid, 0.0)
            desired = tw * nav / price
            current = book.shares.get(sid, 0.0)
            delta = desired - current
            if abs(delta) * price < 1.0:
                continue
            costs = total_cost(delta, price, advs.get(sid, 1.0), vols.get(sid, 0.02), config.costs)
            # participation cap
            max_qty = config.costs.participation_limit * (advs.get(sid, 1.0) / price)
            if abs(delta) > max_qty > 0:
                delta = np.sign(delta) * max_qty
                costs = total_cost(
                    delta, price, advs.get(sid, 1.0), vols.get(sid, 0.02), config.costs
                )
            notional = delta * price
            book.cash -= notional + float(costs["total"])
            book.shares[sid] = current + delta
            # Turnover is based on executed notional, not the requested target
            # change; participation caps can make those materially different.
            traded_turn += abs(notional) / max(nav, 1e-12)
            for k in ("commission", "spread", "impact"):
                cost_sum[k] += float(costs[k])
            fill_rows.append(
                {
                    "fill_time": exec_dt,
                    "signal_time": dt,
                    "security_id": sid,
                    "quantity": delta,
                    "price": price,
                    "fee": costs["commission"],
                    "spread_cost": costs["spread"],
                    "impact_cost": costs["impact"],
                }
            )
        # mark to close
        nav_close = book.nav(close_mark)
        # borrow on shorts
        short_notional = sum(
            abs(min(book.shares.get(s, 0.0), 0.0)) * close_mark.get(s, 0.0) for s in book.shares
        )
        borrow = short_notional * (config.costs.borrow_bps_per_year / 1e4) / 252.0
        if not config.costs.frictionless:
            book.cash -= borrow
            nav_close -= borrow
        navs.append(
            {
                "event_time": exec_dt,
                "nav": nav_close,
                "gross": sum(
                    abs(book.shares.get(s, 0.0) * close_mark.get(s, 0.0)) for s in book.shares
                )
                / max(nav_close, 1e-12),
                "net": sum(book.shares.get(s, 0.0) * close_mark.get(s, 0.0) for s in book.shares)
                / max(nav_close, 1e-12),
                "turnover": traded_turn,
            }
        )
    eq = pl.DataFrame(navs) if navs else pl.DataFrame({"event_time": [], "nav": []})
    if eq.height >= 2:
        rets = eq["nav"].pct_change().drop_nulls().to_numpy()
        sr = sharpe_ratio(rets)
        navs_list = [float(v) for v in eq["nav"].to_list()]
        turns = [float(v) for v in eq["turnover"].to_list()] if "turnover" in eq.columns else [0.0]
        metrics: dict[str, float | str | int | bool] = {
            "total_return": navs_list[-1] / navs_list[0] - 1.0,
            "sharpe": sr["sharpe"],
            "n": sr["n"],
            "max_drawdown": max_drawdown(rets),
            "mean_turnover": float(np.mean(turns)),
            "commission": cost_sum["commission"],
            "spread": cost_sum["spread"],
            "impact": cost_sum["impact"],
            "flag_high_sharpe": sr["flag_high_sharpe"],
        }
    else:
        metrics = {"total_return": 0.0, "sharpe": float("nan"), "n": 0}
    if config.costs.frictionless:
        metrics["label"] = "FRICTIONLESS RESEARCH ONLY"
    note = "SYNTHETIC" if synthetic else "file"
    metrics["data_source"] = note
    return BacktestResult(
        equity=eq,
        fills=pl.DataFrame(fill_rows) if fill_rows else pl.DataFrame(),
        metrics=metrics,
        frictionless=config.costs.frictionless,
        source_note=note,
    )
