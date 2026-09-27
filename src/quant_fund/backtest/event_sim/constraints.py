"""Cash-account constraints for the research simulator.

T+1 settlement, pattern-day-trader and good-faith-violation checks, fractional
shares, and a minimum notional. Warnings never change fills. Block mode
rejects the offending order. Without margin, buys that would overdraw settled
buying power are rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

Mode = Literal["off", "warn", "block"]


def session_date(when: datetime) -> date:
    return when.date()


def round_shares(quantity: float, *, fractional: bool) -> float:
    if not math.isfinite(quantity):
        raise ValueError("quantity must be finite")
    if fractional:
        return float(quantity)
    return float(math.trunc(quantity))


def below_min_notional(quantity: float, price: float, min_notional: float) -> bool:
    if min_notional < 0.0 or not math.isfinite(min_notional):
        raise ValueError("min_notional must be finite and non-negative")
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError("price must be finite and positive")
    return abs(float(quantity)) * float(price) < float(min_notional)


@dataclass
class TaxLot:
    quantity: float
    settles_bar: int
    unsettled_funded: bool


@dataclass
class ConstraintBook:
    """Settled versus unsettled cash, lots, PDT, and GFV state.

    ``settled + sum(unsettled)`` is economic cash. ``settlement_bars=0`` is
    not used: the simulator keeps the legacy immediate-cash path so the
    zero-latency backtest stays a strict generalization.
    """

    settlement_bars: int
    allow_margin: bool
    pdt_mode: Mode = "warn"
    gfv_mode: Mode = "warn"
    pdt_equity_threshold: float = 25_000.0
    pdt_window_sessions: int = 5
    pdt_max_day_trades: int = 3
    settled: float = 0.0
    unsettled: list[tuple[int, float]] = field(default_factory=list)
    lots: dict[str, list[TaxLot]] = field(default_factory=dict)
    opened_session: dict[str, date] = field(default_factory=dict)
    day_trades: list[date] = field(default_factory=list)
    warnings: list[dict[str, object]] = field(default_factory=list)
    gfv_count: int = 0
    blocked: int = 0

    def __post_init__(self) -> None:
        if self.settlement_bars < 0:
            raise ValueError("settlement_bars must be non-negative")
        if self.pdt_mode not in ("off", "warn", "block"):
            raise ValueError("pdt_mode must be off, warn, or block")
        if self.gfv_mode not in ("off", "warn", "block"):
            raise ValueError("gfv_mode must be off, warn, or block")
        if self.pdt_window_sessions < 1 or self.pdt_max_day_trades < 0:
            raise ValueError("PDT window must be positive and max day trades non-negative")

    def economic_cash(self) -> float:
        return float(self.settled + sum(amount for _, amount in self.unsettled))

    def mature(self, bar_index: int) -> None:
        keep: list[tuple[int, float]] = []
        for available, amount in self.unsettled:
            if available <= bar_index:
                self.settled += amount
            else:
                keep.append((available, amount))
        self.unsettled = keep

    def buying_power(self) -> float:
        return self.economic_cash()

    def _warn(self, code: str, message: str, **extra: object) -> None:
        self.warnings.append({"code": code, "message": message, **extra})

    def recent_day_trades(self, session: date, sessions: list[date]) -> int:
        if not sessions:
            return 0
        window = set(sessions[-self.pdt_window_sessions :])
        return sum(1 for item in self.day_trades if item in window and item <= session)

    def pdt_would_block(
        self,
        sid: str,
        shares_before: float,
        delta: float,
        session: date,
        nav: float,
        sessions: list[date],
    ) -> bool:
        if self.pdt_mode != "block":
            return False
        if nav >= self.pdt_equity_threshold:
            return False
        after = shares_before + delta
        closes_today = abs(shares_before) > 1e-12 and (
            abs(after) <= 1e-12 or shares_before * after < 0.0
        )
        if not closes_today or self.opened_session.get(sid) != session:
            return False
        return self.recent_day_trades(session, sessions) >= self.pdt_max_day_trades

    def gfv_would_block(self, sid: str, shares_before: float, delta: float, bar_index: int) -> bool:
        if self.gfv_mode != "block" or delta >= 0.0 or shares_before <= 1e-12:
            return False
        closing = min(shares_before, -delta)
        left = closing
        for lot in self.lots.get(sid, []):
            if left <= 1e-15:
                break
            take = min(lot.quantity, left)
            if lot.unsettled_funded and bar_index < lot.settles_bar and take > 0.0:
                return True
            left -= take
        return False

    def note_round_trip(
        self,
        sid: str,
        shares_before: float,
        shares_after: float,
        session: date,
        nav: float,
        sessions: list[date],
    ) -> None:
        if self.pdt_mode == "off":
            return
        flat = 1e-12
        before = float(shares_before)
        after = float(shares_after)
        closed = abs(before) > flat and (abs(after) <= flat or before * after < 0.0)
        opened = abs(after) > flat and (abs(before) <= flat or before * after < 0.0)
        if closed and self.opened_session.get(sid) == session:
            self.day_trades.append(session)
            count = self.recent_day_trades(session, sessions)
            if count > self.pdt_max_day_trades:
                self._warn(
                    "PDT",
                    "pattern-day-trader threshold crossed in the rolling session window",
                    security_id=sid,
                    session=session.isoformat(),
                    day_trades=count,
                    nav=float(nav),
                )
        if abs(after) <= flat:
            self.opened_session.pop(sid, None)
        elif opened:
            self.opened_session[sid] = session

    def consume_lots_for_sell(self, sid: str, shares_closed: float, bar_index: int) -> None:
        if shares_closed <= 0.0:
            return
        left = float(shares_closed)
        kept: list[TaxLot] = []
        for lot in self.lots.get(sid, []):
            if left <= 1e-15:
                kept.append(lot)
                continue
            take = min(lot.quantity, left)
            if (
                self.gfv_mode != "off"
                and lot.unsettled_funded
                and bar_index < lot.settles_bar
                and take > 0.0
            ):
                self.gfv_count += 1
                self._warn(
                    "GFV",
                    "sale of shares bought with unsettled funds before settlement",
                    security_id=sid,
                    bar_index=bar_index,
                    quantity=float(take),
                )
            remain = lot.quantity - take
            left -= take
            if remain > 1e-15:
                kept.append(TaxLot(remain, lot.settles_bar, lot.unsettled_funded))
        if kept:
            self.lots[sid] = kept
        else:
            self.lots.pop(sid, None)

    def apply_buy(self, sid: str, quantity: float, need: float, bar_index: int) -> bool:
        """Debit buying power. Return False when the buy would make cash negative."""
        if need < -1e-12:
            raise ValueError("buy cash need must be non-negative")
        if not self.allow_margin and need > self.buying_power() + 1e-9:
            self.blocked += 1
            return False
        from_unsettled = 0.0
        settles = bar_index
        unsettled_funded = False
        if self.settled >= need or self.allow_margin:
            self.settled -= need
        else:
            from_settled = max(self.settled, 0.0)
            from_unsettled = need - from_settled
            self.settled = 0.0
            left = from_unsettled
            kept: list[tuple[int, float]] = []
            for available, amount in self.unsettled:
                if left <= 1e-15:
                    kept.append((available, amount))
                    continue
                take = min(amount, left)
                if take > 0.0:
                    unsettled_funded = True
                    settles = max(settles, available)
                remain = amount - take
                left -= take
                if remain > 1e-12:
                    kept.append((available, remain))
            self.unsettled = kept
        if quantity > 0.0:
            self.lots.setdefault(sid, []).append(
                TaxLot(float(quantity), int(settles), unsettled_funded)
            )
        return True

    def apply_sell_proceeds(self, proceeds: float, bar_index: int) -> None:
        """Park positive sale proceeds in the unsettled queue. Negative proceeds hit settled cash."""
        if proceeds >= 0.0:
            self.unsettled.append((bar_index + self.settlement_bars, float(proceeds)))
        else:
            self.settled += float(proceeds)
            if not self.allow_margin and self.settled < -1e-8:
                self.settled -= proceeds
                raise ValueError("sell costs would drive settled cash negative without margin")
