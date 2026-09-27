"""Closed forms for NAV, costs, P&L, turnover, splits, and dividends.

The Z3 proofs quantify over these formulas. ``total_cost`` and the paper
broker's cash update are checked against them in the formal test lane.
Proofs are over the real numbers; the implementation is IEEE float.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def split_market_value(shares: float, price: float, factor: float) -> tuple[float, float]:
    """2-for-1 is ``factor=2``: shares scale up, price scales down, value holds."""
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("split factor must be finite and positive")
    return shares * factor, price / factor


def cash_dividend(cash: float, shares: float, price: float, dividend: float) -> tuple[float, float]:
    """Cash dividend per current share. Price drops by the dividend; NAV holds.

    ``dividend`` is cash per share on the same basis as ``shares`` and ``price``.
    """
    if not math.isfinite(dividend) or dividend < 0.0:
        raise ValueError("dividend must be finite and non-negative")
    return cash + shares * dividend, price - dividend


def dividend_yield(
    dividend: float, prev_raw: float, split_factor_prev: float, split_factor_today: float
) -> float:
    """Cash-dividend simple yield per unit of prior wealth.

    Matches ``adjust_prices``: ``dividend * sf_prev / (prev_raw * sf_today)``.
    When the cumulative future-split factors match, this is ``dividend / prev_raw``.
    """
    return dividend * split_factor_prev / (prev_raw * split_factor_today)


def algebraic_total_cost(
    quantity: float,
    price: float,
    adv_dollars: float,
    sigma: float,
    *,
    commission_bps: float,
    half_spread_bps: float,
    impact_y: float,
    bps_per_turnover: float,
    frictionless: bool = False,
) -> dict[str, float]:
    """Closed form of ``quant_fund.execution.costs.total_cost`` (borrow stays 0).

    ``financing_bps_per_year`` is not an input: the cost function does not
    read it.
    """
    if frictionless:
        return {
            "commission": 0.0,
            "spread": 0.0,
            "impact": 0.0,
            "turnover_bps": 0.0,
            "total": 0.0,
        }
    notional = abs(quantity) * price
    commission = notional * commission_bps / 1e4
    spread = notional * half_spread_bps / 1e4
    participation = notional / adv_dollars
    impact = notional * impact_y * sigma * math.sqrt(participation) if notional else 0.0
    turnover = notional * bps_per_turnover / 1e4
    return {
        "commission": commission,
        "spread": spread,
        "impact": impact,
        "turnover_bps": turnover,
        "total": commission + spread + impact + turnover,
    }


def apply_trade(
    qty: float,
    avg: float,
    realized: float,
    signed_qty: float,
    price: float,
    cost: float,
) -> tuple[float, float, float]:
    """Average-cost trade. ``cost`` is expensed immediately into realized P&L.

    Returns ``(qty, avg, realized)``. A same-direction trade updates the
    average. A reducing trade realizes ``(price - avg)`` on the closed
    quantity (sign-flipped for shorts). Crossing zero opens the residual at
    ``price``.
    """
    if signed_qty == 0.0:
        raise ValueError("signed_qty must be non-zero")
    realized = realized - cost
    same_direction = (
        qty == 0.0 or (qty > 0.0 and signed_qty > 0.0) or (qty < 0.0 and signed_qty < 0.0)
    )
    if same_direction:
        new_qty = qty + signed_qty
        new_avg = (abs(qty) * avg + abs(signed_qty) * price) / abs(new_qty)
        return new_qty, new_avg, realized
    close = min(abs(qty), abs(signed_qty))
    if qty > 0.0:
        realized += close * (price - avg)
    else:
        realized += close * (avg - price)
    new_qty = qty + signed_qty
    if new_qty == 0.0:
        return 0.0, 0.0, realized
    if qty * new_qty < 0.0:
        return new_qty, price, realized
    return new_qty, avg, realized


def unrealized_pnl(qty: float, avg: float, mark: float) -> float:
    """``qty * (mark - avg)`` for longs and shorts; zero when flat."""
    return qty * (mark - avg)


@dataclass
class Account:
    """Reference book. Cash moves by ``-(signed_qty * price + cost)``."""

    initial_cash: float
    cash: float | None = None
    realized: float = 0.0
    qty: dict[str, float] = field(default_factory=dict)
    avg: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash is None:
            self.cash = float(self.initial_cash)

    def apply(self, security_id: str, signed_qty: float, price: float, cost: float) -> None:
        current = self.qty.get(security_id, 0.0)
        average = self.avg.get(security_id, 0.0)
        new_qty, new_avg, self.realized = apply_trade(
            current, average, self.realized, signed_qty, price, cost
        )
        self.qty[security_id] = new_qty
        self.avg[security_id] = new_avg
        assert self.cash is not None
        self.cash -= signed_qty * price + cost

    def position_value(self, marks: dict[str, float]) -> float:
        total = 0.0
        for security_id, quantity in self.qty.items():
            if quantity == 0.0:
                continue
            total += quantity * marks[security_id]
        return total

    def nav(self, marks: dict[str, float]) -> float:
        assert self.cash is not None
        return self.cash + self.position_value(marks)

    def unrealized(self, marks: dict[str, float]) -> float:
        total = 0.0
        for security_id, quantity in self.qty.items():
            if quantity == 0.0:
                continue
            total += unrealized_pnl(quantity, self.avg.get(security_id, 0.0), marks[security_id])
        return total

    def identity_gap(self, marks: dict[str, float]) -> float:
        """``nav - (initial_cash + realized + unrealized)``. Zero when the books match."""
        return self.nav(marks) - (self.initial_cash + self.realized + self.unrealized(marks))
