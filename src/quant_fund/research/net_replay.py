"""Bounded daily price-return replay for matched research tournaments.

Decisions use completed bars at t; quantities execute at open t+1. This is
independent of backtest.engine, whose legacy execution path is not invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from quant_fund.research.cost_allocation import AllocationConfig, allocate


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    lookback: int = 20
    fraction: float = 0.25
    long_short: bool = False
    allocation: AllocationConfig | None = None

    def validate(self) -> None:
        if not self.name or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in self.name
        ):
            raise ValueError("strategy names must use lowercase letters, digits and underscores")
        if self.family not in {"equal_weight", "momentum", "reversal"}:
            raise ValueError("unsupported strategy family")
        if type(self.lookback) is not int or not 1 <= self.lookback <= 252:
            raise ValueError("lookback must be an integer in [1, 252]")
        if not np.isfinite(self.fraction) or not 0 < self.fraction <= 0.5:
            raise ValueError("fraction must be in (0, 0.5]")
        if type(self.long_short) is not bool or (self.family == "equal_weight" and self.long_short):
            raise ValueError("equal_weight is long-only; long_short must be a boolean")
        if self.allocation is not None:
            if not isinstance(self.allocation, AllocationConfig) or self.family == "equal_weight":
                raise ValueError("allocation requires AllocationConfig and a forecast strategy")
            self.allocation.validate()

    @property
    def required_history(self) -> int:
        return max(20, self.lookback, self.allocation.risk_window if self.allocation else 0)


@dataclass(frozen=True)
class ReplayConfig:
    initial_nav: float = 100_000.0
    gross_limit: float = 0.95
    max_name_weight: float = 0.20
    participation_limit: float = 0.01
    commission_bps: float = 1.0
    half_spread_bps: float = 5.0
    impact_y: float = 0.1
    borrow_apr: float = 0.03
    funding_apr: float = 0.06
    cash_apr: float = 0.0
    max_names: int = 50
    target_buffer: float = 0.01

    def validate(self) -> None:
        for key in ("initial_nav", "gross_limit", "max_name_weight", "participation_limit"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.gross_limit > 2 or self.max_name_weight > 1 or self.participation_limit > 1:
            raise ValueError("gross <= 2, name weight <= 1 and participation <= 1 required")
        for key in (
            "commission_bps",
            "half_spread_bps",
            "impact_y",
            "borrow_apr",
            "funding_apr",
            "cash_apr",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if type(self.max_names) is not int or self.max_names < 2:
            raise ValueError("max_names must be an integer >= 2")
        if not np.isfinite(self.target_buffer) or not 0 < self.target_buffer < 0.5:
            raise ValueError("target_buffer must be in (0, 0.5)")


@dataclass
class MarketPanel:
    dates: list[datetime]
    names: list[str]
    close: np.ndarray
    opening: np.ndarray
    volume: np.ndarray
    known: np.ndarray


def market_panel(frame: pl.DataFrame, *, open_column: str, volume_column: str) -> MarketPanel:
    """Input must already pass real_benchmark._load_bars's timing/price checks."""
    if not {open_column, volume_column}.issubset(frame.columns):
        raise ValueError("execution requires the declared open and volume columns")
    names = sorted(frame["security_id"].unique().to_list())
    dates = frame["event_time"].unique().sort().to_list()

    def matrix(column: str) -> np.ndarray:
        return (
            frame.select("event_time", "security_id", pl.col(column).cast(pl.Float64))
            .pivot(on="security_id", index="event_time", values=column)
            .sort("event_time")
            .select(names)
            .to_numpy()
        )

    return MarketPanel(
        dates,
        names,
        matrix("price"),
        matrix(open_column),
        matrix(volume_column),
        matrix("known") == 1,
    )


def _universe(
    panel: MarketPanel, i: int, history: int, config: ReplayConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(panel.names)
    if i < history:
        return np.zeros(n, dtype=bool), np.zeros(n), np.zeros(n)
    close = panel.close[i - history : i + 1]
    volume = panel.volume[i - 19 : i + 1]
    eligible = (np.isfinite(close) & (close > 0)).all(axis=0) & panel.known[
        i - history : i + 1
    ].all(axis=0)
    eligible &= (np.isfinite(volume) & (volume >= 0)).all(axis=0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        adv = np.mean(panel.close[i - 19 : i + 1] * volume, axis=0)
        sigma = np.std(panel.close[i - 19 : i + 1] / panel.close[i - 20 : i] - 1, axis=0, ddof=1)
    eligible &= np.isfinite(adv) & (adv > 0) & np.isfinite(sigma)
    adv = np.where(eligible, adv, np.nan)
    sigma = np.where(eligible, sigma, np.nan)
    ids = np.flatnonzero(eligible)
    chosen = ids[np.argsort(-adv[ids], kind="stable")[: config.max_names]]
    eligible[:] = False
    eligible[chosen] = True
    return eligible, adv, sigma


def _weights(
    panel: MarketPanel, i: int, eligible: np.ndarray, strategy: Strategy, config: ReplayConfig
) -> np.ndarray:
    weights = np.zeros(len(panel.names))
    ids = np.flatnonzero(eligible)
    if len(ids) < 2:
        return weights
    budget = config.gross_limit * (1 - config.target_buffer)
    name_limit = config.max_name_weight * (1 - config.target_buffer)
    if strategy.family == "equal_weight":
        weights[ids] = min(budget / len(ids), name_limit)
        return weights
    signal = panel.close[i, ids] / panel.close[i - strategy.lookback, ids] - 1
    if strategy.family == "reversal":
        signal = -signal
    ranked = ids[np.argsort(signal, kind="stable")]
    count = max(1, int(len(ids) * strategy.fraction))
    leg = budget / (2 if strategy.long_short else 1)
    size = min(leg / count, name_limit)
    weights[ranked[-count:]] = size
    if strategy.long_short:
        weights[ranked[:count]] = -size
    return weights


def replay(
    panel: MarketPanel,
    strategy: Strategy,
    config: ReplayConfig,
    *,
    start: str,
    end: str,
    history: int,
    impact_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Self-financing shares/cash ledger; terminal exits respect participation.

    Missing marks of held assets fail the trial. Unfilled orders remain visible.
    All prices must use a consistent share basis; dividends/splits are not applied
    here. Financing uses ACT/365 and the preceding open's short market value.
    """
    strategy.validate()
    config.validate()
    if (
        history < strategy.required_history
        or not np.isfinite(impact_multiplier)
        or impact_multiplier <= 0
    ):
        raise ValueError("invalid history or impact multiplier")
    decision_ids = [
        i
        for i, date in enumerate(panel.dates[:-1])
        if start <= date.date().isoformat() and panel.dates[i + 1].date().isoformat() <= end
    ]
    if len(decision_ids) < 2:
        raise ValueError("insufficient execution sessions")
    shares = np.zeros(len(panel.names))
    cash, last_nav = config.initial_nav, config.initial_nav
    previous_open: int | None = None
    ledger, fills, rejections, allocations = [], [], [], []
    rejected = 0
    for i in decision_ids:
        e = i + 1
        terminal = i == decision_ids[-1]
        held = shares != 0
        signal_prices, prices = panel.close[i], panel.opening[e]
        if not (
            np.isfinite(signal_prices[held]) & (signal_prices[held] > 0) & panel.known[i, held]
        ).all():
            raise ValueError("held asset lacks an available signal-close valuation")
        if not (np.isfinite(prices[held]) & (prices[held] > 0)).all():
            raise ValueError("held asset lacks an execution-open valuation")
        signal_nav = cash + float(shares[held] @ signal_prices[held])
        if not np.isfinite(signal_nav) or signal_nav <= 0:
            raise ValueError("nonpositive signal NAV")
        eligible, adv, sigma = _universe(panel, i, history, config)
        target = (
            np.zeros(len(shares)) if terminal else _weights(panel, i, eligible, strategy, config)
        )
        if strategy.allocation is not None and not terminal:
            previous = np.zeros(len(shares))
            previous[held] = shares[held] * signal_prices[held] / signal_nav
            target, diagnostic = _cost_weights(
                panel, i, target, previous, adv, sigma, strategy, config, signal_nav
            )
            allocations.append({"signal_session": panel.dates[i].isoformat(), **diagnostic})
        desired = np.zeros(len(shares))
        active = target != 0
        desired[active] = target[active] * signal_nav / signal_prices[active]
        cash, shares, daily, new_fills, new_rejections = execute_orders(
            names=panel.names,
            desired=desired,
            shares=shares,
            cash=cash,
            prices=prices,
            adv=adv,
            sigma=sigma,
            signal_time=panel.dates[i],
            execution_time=panel.dates[e],
            previous_time=panel.dates[previous_open] if previous_open is not None else None,
            previous_prices=panel.opening[previous_open]
            if previous_open is not None
            else np.zeros(len(shares)),
            last_nav=last_nav,
            config=config,
            eligible_count=int(eligible.sum()),
            terminal=terminal,
            impact_multiplier=impact_multiplier,
        )
        ledger.append(daily)
        fills.extend(new_fills)
        rejections.extend(new_rejections)
        rejected += len(new_rejections)
        last_nav, previous_open = daily["nav"], e
    values = np.array([config.initial_nav, *[row["nav"] for row in ledger]])
    return {
        "status": "completed",
        "daily": ledger,
        "fills": fills,
        "rejections": rejections,
        "allocations": allocations,
        "rejected_orders": rejected,
        "total_return": float(values[-1] / values[0] - 1),
        "max_drawdown": float(np.min(values / np.maximum.accumulate(values) - 1)),
        "terminal_residual_gross": ledger[-1]["gross_market_value"],
        "liquidation_complete": bool(np.abs(shares).max() < 1e-10),
    }


def execute_orders(
    *,
    names: list[str],
    desired: np.ndarray,
    shares: np.ndarray,
    cash: float,
    prices: np.ndarray,
    adv: np.ndarray,
    sigma: np.ndarray,
    signal_time: datetime,
    execution_time: datetime,
    previous_time: datetime | None,
    previous_prices: np.ndarray,
    last_nav: float,
    config: ReplayConfig,
    eligible_count: int,
    terminal: bool = False,
    impact_multiplier: float = 1.0,
) -> tuple[float, np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute frozen quantities with the same accounting for replay and forward shadow."""
    config.validate()
    n = len(names)
    if any(
        np.asarray(v).shape != (n,) for v in (desired, shares, prices, adv, sigma, previous_prices)
    ):
        raise ValueError("execution vectors must match names")
    if not np.isfinite(desired).all() or not np.isfinite(shares).all():
        raise ValueError("quantities must be finite")
    if (
        not np.isfinite([cash, last_nav, impact_multiplier]).all()
        or last_nav <= 0
        or impact_multiplier <= 0
    ):
        raise ValueError("invalid cash, NAV or impact multiplier")
    if execution_time <= signal_time or (
        previous_time is not None and execution_time <= previous_time
    ):
        raise ValueError("execution times must advance")
    held = shares != 0
    if not (np.isfinite(prices[held]) & (prices[held] > 0)).all():
        raise ValueError("held asset lacks an execution-open valuation")
    if (
        previous_time is not None
        and not (np.isfinite(previous_prices[held]) & (previous_prices[held] > 0)).all()
    ):
        raise ValueError("held asset lacks a prior-open valuation")
    shares = shares.copy()
    fills, rejections = [], []
    borrow, financing = 0.0, 0.0
    if previous_time is not None:
        years = (execution_time - previous_time).total_seconds() / (365 * 86400)
        shorts = shares < 0
        borrow = float(-shares[shorts] @ previous_prices[shorts]) * config.borrow_apr * years
        financing = (-cash * config.funding_apr if cash < 0 else -cash * config.cash_apr) * years
        cash -= borrow + financing
    mark = np.where(np.isfinite(prices) & (prices > 0), prices, 0.0)
    costs = {"commission": 0.0, "spread": 0.0, "impact": 0.0}
    deltas = desired - shares
    # Risk-reducing trades first, then new risk; deterministic security order.
    order = sorted(
        np.flatnonzero(np.abs(deltas) > 1e-12),
        key=lambda a: (abs(desired[a]) >= abs(shares[a]), names[a]),
    )
    for a in order:
        price = prices[a]
        if (
            not np.isfinite(price)
            or price <= 0
            or not np.isfinite(adv[a])
            or adv[a] <= 0
            or not np.isfinite(sigma[a])
        ):
            rejections.append(
                {
                    "execution_session": execution_time.isoformat(),
                    "security_id": names[a],
                    "requested_quantity": float(deltas[a]),
                    "reason": "unavailable_price_or_known_liquidity",
                }
            )
            continue
        quantity = float(
            np.sign(deltas[a]) * min(abs(deltas[a]), config.participation_limit * adv[a] / price)
        )
        notional = abs(quantity) * price
        commission = notional * config.commission_bps / 1e4
        spread = notional * config.half_spread_bps / 1e4
        impact = (
            notional * config.impact_y * impact_multiplier * sigma[a] * np.sqrt(notional / adv[a])
        )
        fee = commission + spread + impact
        before_nav = cash + float(shares @ mark)
        after_nav = before_nav - fee
        after_shares = shares.copy()
        after_shares[a] += quantity
        after_cash = cash - quantity * price - fee
        if after_nav <= 0 or not np.isfinite(after_nav):
            raise ValueError("nonpositive NAV after costs")
        before_gross, after_gross = (
            float(np.abs(shares * mark).sum()) / before_nav,
            float(np.abs(after_shares * mark).sum()) / after_nav,
        )
        before_name, after_name = (
            abs(shares[a] * price) / before_nav,
            abs(after_shares[a] * price) / after_nav,
        )
        if (
            (after_gross > config.gross_limit + 1e-10 and after_gross > before_gross + 1e-10)
            or (after_name > config.max_name_weight + 1e-10 and after_name > before_name + 1e-10)
            or (config.gross_limit <= 1 and after_cash < -1e-8)
        ):
            rejections.append(
                {
                    "execution_session": execution_time.isoformat(),
                    "security_id": names[a],
                    "requested_quantity": float(deltas[a]),
                    "reason": "exposure_or_cash_limit",
                }
            )
            continue
        shares, cash = after_shares, after_cash
        costs["commission"] += commission
        costs["spread"] += spread
        costs["impact"] += impact
        fills.append(
            {
                "signal_session": signal_time.isoformat(),
                "execution_session": execution_time.isoformat(),
                "security_id": names[a],
                "quantity": quantity,
                "requested_quantity": float(deltas[a]),
                "unfilled_quantity": float(deltas[a] - quantity),
                "price": float(price),
                "commission": float(commission),
                "spread": float(spread),
                "impact": float(impact),
                "known_adv": float(adv[a]),
                "known_volatility": float(sigma[a]),
            }
        )
    nav = cash + float(shares @ mark)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("nonpositive ending NAV")
    daily = {
        "date": execution_time.isoformat(),
        "nav": nav,
        "cash": cash,
        "net_return": nav / last_nav - 1,
        "gross_market_value": float(np.abs(shares * mark).sum()),
        **costs,
        "borrow": borrow,
        "financing": financing,
        "eligible_names": eligible_count,
        "terminal": terminal,
    }
    return cash, shares, daily, fills, rejections


def _cost_weights(
    panel: MarketPanel,
    i: int,
    ranked: np.ndarray,
    previous: np.ndarray,
    adv: np.ndarray,
    sigma: np.ndarray,
    strategy: Strategy,
    execution: ReplayConfig,
    nav: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Same rank signal/support as the control; size trades from causal estimates.

    Unselected existing positions may only shrink, allowing capacity-limited exits.
    Planning costs stay frozen when replay stresses actual impact costs.
    """
    config = strategy.allocation
    if config is None:
        raise ValueError("cost-aware strategy requires allocation settings")
    ids = np.flatnonzero((ranked != 0) | (previous != 0))
    target = np.zeros(len(previous))
    if not ids.size:
        return target, {"status": "empty_universe", "security_ids": []}
    prices = panel.close[i - config.risk_window : i + 1, ids]
    known = panel.known[i - config.risk_window : i + 1, ids]
    if (
        len(prices) != config.risk_window + 1
        or not known.all()
        or not (np.isfinite(prices) & (prices > 0)).all()
        or not (np.isfinite(adv[ids]) & (adv[ids] > 0) & np.isfinite(sigma[ids])).all()
    ):
        raise ValueError(
            "allocation requires known risk history and liquidity for held/selected assets"
        )
    returns = prices[1:] / prices[:-1] - 1
    sample = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
    covariance = (1 - config.covariance_shrinkage) * sample + config.covariance_shrinkage * np.diag(
        np.diag(sample)
    )
    # A declared daily return proxy, NOT a fitted expected-return estimate.
    alpha = np.expm1(
        np.log(panel.close[i, ids] / panel.close[i - strategy.lookback, ids]) / strategy.lookback
    )
    alpha *= config.alpha_scale * (-1 if strategy.family == "reversal" else 1)
    alpha[ranked[ids] == 0] = 0.0
    # Dispersion proxy; sqrt(n) assumes independence and is not a confidence bound.
    uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0) / strategy.lookback)
    cap = execution.max_name_weight * (1 - execution.target_buffer)
    lower = np.maximum(-cap, np.minimum(previous[ids], np.where(ranked[ids] < 0, -cap, 0)))
    upper = np.minimum(cap, np.maximum(previous[ids], np.where(ranked[ids] > 0, cap, 0)))
    result, diagnostic = allocate(
        alpha,
        covariance,
        uncertainty,
        previous[ids],
        lower,
        upper,
        execution.participation_limit * adv[ids] / nav,
        execution.impact_y * sigma[ids] * np.sqrt(nav / adv[ids]),
        config=config,
        linear_cost=(execution.commission_bps + execution.half_spread_bps) / 1e4,
        gross_limit=execution.gross_limit * (1 - execution.target_buffer),
        name_limit=cap,
        cash_buffer=execution.target_buffer,
        borrow_cost=execution.borrow_apr / 252,
        funding_cost=execution.funding_apr / 252,
        cash_return=execution.cash_apr / 252,
    )
    target[ids] = result
    return target, {
        "security_ids": [panel.names[j] for j in ids],
        "planning_nav": nav,
        **diagnostic,
    }
