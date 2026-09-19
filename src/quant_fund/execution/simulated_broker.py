"""Simulated broker for paper / shadow. No vendor fills or live connectivity.

Cash, positions, configurable next-open or same-bar fills, commissions and
impact from the existing ``total_cost`` model. Every order must clear the
kill switch and ``check_order`` risk gate before cash moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np

from quant_fund.config.models import AppConfig
from quant_fund.execution.costs import total_cost
from quant_fund.monitoring.kill_switch import KillSwitch
from quant_fund.portfolio.risk_gate import check_order
from quant_fund.schemas.errors import KillSwitchActive, RiskGateRejected
from quant_fund.schemas.orders import Fill, Order, OrderSide, OrderStatus


class RejectReason(str, Enum):
    KILL_SWITCH = "kill_switch"
    RISK_GATE = "risk_gate"
    ZERO_QTY = "zero_qty"
    MISSING_PRICE = "missing_price"
    INSUFFICIENT_CASH = "insufficient_cash"
    INVALID_MARKET_DATA = "invalid_market_data"


@dataclass
class OrderRecord:
    order: Order
    reject_reason: str | None = None
    fill: Fill | None = None
    slot: str = "champion"  # champion | shadow


@dataclass
class BrokerSnapshot:
    cash: float
    shares: dict[str, float]
    nav: float
    gross: float
    net: float
    asof: datetime | None
    slot: str


@dataclass
class SimulatedBroker:
    """Paper broker with cash/positions. Shadow slot holds weights only (no capital)."""

    config: AppConfig
    initial_cash: float = 1_000_000.0
    slot: str = "champion"
    allow_capital: bool = True
    cash: float = field(init=False)
    shares: dict[str, float] = field(default_factory=dict)
    open_orders: dict[str, Order] = field(default_factory=dict)
    history: list[OrderRecord] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    kill: KillSwitch = field(init=False)
    reject_count: int = 0
    halt_count: int = 0
    risk_gate_reject_count: int = 0
    last_marks: dict[str, float] = field(default_factory=dict)
    # Counts from pre-history state files. Newer files persist full records;
    # legacy files retain their counters so a resumed run remains cumulative.
    _prior_order_count: int = field(default=0, repr=False)
    _prior_fill_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash) if self.allow_capital else 0.0
        self.kill = KillSwitch(self.config.kill_switch)
        if not self.allow_capital:
            self.slot = self.slot or "shadow"

    def nav(self, prices: dict[str, float] | None = None) -> float:
        px = prices or self.last_marks
        for security_id, quantity in self.shares.items():
            if abs(float(quantity)) <= 1e-12:
                continue
            if security_id not in px:
                raise ValueError(f"missing market mark for held security: {security_id}")
            mark = float(px[security_id])
            if not np.isfinite(mark) or mark <= 0.0:
                raise ValueError("valuation marks must be finite and strictly positive")
        pos = sum(self.shares.get(s, 0.0) * px.get(s, 0.0) for s in self.shares)
        return self.cash + pos

    def exposures(self, prices: dict[str, float] | None = None) -> tuple[float, float]:
        px = prices or self.last_marks
        nav = float(self.nav(px))
        if not np.isfinite(nav) or nav < 0.0:
            raise ValueError("broker NAV must be finite and non-negative")
        # A capital-free shadow broker has zero NAV by design; its exposure is
        # zero until it receives a capital allocation. Never floor a negative
        # NAV into a usable denominator.
        if nav == 0.0:
            return 0.0, 0.0
        all_ids = sorted(set(self.shares) | set(px))
        gross = sum(abs(self.shares.get(s, 0.0) * px.get(s, 0.0)) for s in all_ids)
        net = sum(self.shares.get(s, 0.0) * px.get(s, 0.0) for s in all_ids)
        return gross / nav, net / nav

    def snapshot(
        self, asof: datetime | None = None, prices: dict[str, float] | None = None
    ) -> BrokerSnapshot:
        px = prices or self.last_marks
        g, n = self.exposures(px)
        return BrokerSnapshot(
            cash=self.cash,
            shares=dict(self.shares),
            nav=self.nav(px),
            gross=g,
            net=n,
            asof=asof,
            slot=self.slot,
        )

    def mark(self, prices: dict[str, float]) -> None:
        validated: dict[str, float] = {}
        for security_id, value in prices.items():
            if value is None:
                continue
            price = float(value)
            if not np.isfinite(price) or price <= 0.0:
                raise ValueError("market marks must be finite and strictly positive")
            validated[str(security_id)] = price
        self.last_marks.update(validated)

    def submit(
        self,
        order: Order,
        *,
        price: float,
        nav: float | None = None,
        adv_dollars: float = 1.0,
        sigma: float = 0.02,
        price_age_bars: int | None = None,
        model_age_hours: float | None = None,
        market_predicted_vol: float | None = None,
    ) -> OrderRecord:
        """Validate + optionally fill immediately (marketable paper order).

        Shadow / no-capital slots record intent and reject capital moves.
        ``sigma`` is name-level vol for impact costs. ``market_predicted_vol``
        is the optional date-level Realized GARCH or GARCH overlay for
        ``check_order``.
        """
        if not np.isfinite(float(order.quantity)) or float(order.quantity) <= 0.0:
            rec = OrderRecord(
                order=order.model_copy(update={"status": OrderStatus.REJECTED}),
                reject_reason=RejectReason.ZERO_QTY.value,
                slot=self.slot,
            )
            self.reject_count += 1
            self.history.append(rec)
            return rec
        if price is None or not (price > 0):
            rec = OrderRecord(
                order=order.model_copy(update={"status": OrderStatus.REJECTED}),
                reject_reason=RejectReason.MISSING_PRICE.value,
                slot=self.slot,
            )
            self.reject_count += 1
            self.history.append(rec)
            return rec
        if (
            not np.isfinite(float(adv_dollars))
            or float(adv_dollars) <= 0
            or not np.isfinite(float(sigma))
            or float(sigma) < 0
            or (
                market_predicted_vol is not None
                and (not np.isfinite(float(market_predicted_vol)) or float(market_predicted_vol) < 0)
            )
        ):
            rec = OrderRecord(
                order=order.model_copy(update={"status": OrderStatus.REJECTED}),
                reject_reason=RejectReason.INVALID_MARKET_DATA.value,
                slot=self.slot,
            )
            self.reject_count += 1
            self.history.append(rec)
            return rec

        try:
            self.kill.assert_new_orders_allowed()
        except KillSwitchActive:
            self.halt_count += 1
            rec = OrderRecord(
                order=order.model_copy(update={"status": OrderStatus.REJECTED}),
                reject_reason=RejectReason.KILL_SWITCH.value,
                slot=self.slot,
            )
            self.reject_count += 1
            self.history.append(rec)
            return rec

        if not self.allow_capital:
            # Shadow: record intent only — never move cash.
            acked = order.model_copy(update={"status": OrderStatus.ACKED})
            rec = OrderRecord(order=acked, reject_reason=None, slot=self.slot)
            self.open_orders[order.order_id] = acked
            self.history.append(rec)
            return rec

        requested_signed = float(order.quantity)
        if order.side is OrderSide.SELL:
            requested_signed = -requested_signed
        # Validate and execute the participation-capped child, not the
        # uncapped parent order. This matches the backtest engine and allows a
        # large paper target delta to become a safe partial fill.
        max_qty = self.config.costs.participation_limit * (float(adv_dollars) / max(price, 1e-12))
        exec_qty = requested_signed
        if abs(exec_qty) > max_qty > 0:
            exec_qty = (1.0 if exec_qty > 0 else -1.0) * max_qty
        risk_order = order.model_copy(update={"quantity": abs(exec_qty)})
        px_map = dict(self.last_marks)
        px_map[order.security_id] = float(price)
        nav_use = float(nav if nav is not None else self.nav(px_map))
        if not np.isfinite(nav_use) or nav_use <= 0:
            rec = OrderRecord(
                order=order.model_copy(update={"status": OrderStatus.REJECTED}),
                reject_reason=RejectReason.INVALID_MARKET_DATA.value,
                slot=self.slot,
            )
            self.reject_count += 1
            self.history.append(rec)
            return rec
        current_shares = self.shares.get(order.security_id, 0.0)
        current_w = (current_shares * price) / max(nav_use, 1e-12)
        projected = dict(self.shares)
        projected[order.security_id] = current_shares + exec_qty
        all_ids = sorted(set(projected) | set(px_map))
        gross = sum(abs(projected.get(s, 0.0) * px_map.get(s, 0.0)) for s in all_ids)
        net = sum(projected.get(s, 0.0) * px_map.get(s, 0.0) for s in all_ids)
        participation = abs(exec_qty) * price / max(float(adv_dollars), 1e-12)

        try:
            check_order(
                risk_order,
                nav=nav_use,
                price=float(price),
                current_weight=current_w,
                gross_after=gross / max(nav_use, 1e-12),
                net_after=net / max(nav_use, 1e-12),
                participation=participation,
                predicted_vol=float(sigma),
                config=self.config,
                price_age_bars=price_age_bars,
                model_age_hours=model_age_hours,
                market_predicted_vol=market_predicted_vol,
            )
        except RiskGateRejected as exc:
            self.reject_count += 1
            self.risk_gate_reject_count += 1
            rec = OrderRecord(
                order=order.model_copy(update={"status": OrderStatus.REJECTED}),
                reject_reason=f"{RejectReason.RISK_GATE.value}:{exc}",
                slot=self.slot,
            )
            self.history.append(rec)
            return rec

        costs = total_cost(exec_qty, price, adv_dollars, sigma, self.config.costs)
        notional = exec_qty * price
        # Cash check for buys
        if exec_qty > 0 and self.cash < notional + float(costs["total"]):
            self.reject_count += 1
            rec = OrderRecord(
                order=order.model_copy(update={"status": OrderStatus.REJECTED}),
                reject_reason=RejectReason.INSUFFICIENT_CASH.value,
                slot=self.slot,
            )
            self.history.append(rec)
            return rec

        self.cash -= notional + float(costs["total"])
        self.shares[order.security_id] = current_shares + exec_qty
        fill = Fill(
            fill_id=f"fill-{uuid4().hex[:12]}",
            order_id=order.order_id,
            security_id=order.security_id,
            quantity=abs(exec_qty),
            price=float(price),
            fill_time=order.order_time,
            fee=float(costs["commission"]),
            spread_cost=float(costs["spread"]),
            impact_cost=float(costs["impact"]),
            slippage=0.0,
            is_partial=abs(exec_qty) + 1e-12 < abs(requested_signed),
        )
        filled = order.model_copy(update={"status": OrderStatus.FILLED, "quantity": abs(exec_qty)})
        self.fills.append(fill)
        self.open_orders.pop(order.order_id, None)
        rec = OrderRecord(order=filled, fill=fill, slot=self.slot)
        self.history.append(rec)
        self.mark({order.security_id: float(price)})
        return rec

    def target_to_orders(
        self,
        targets: dict[str, float],
        prices: dict[str, float],
        *,
        signal_time: datetime,
        order_time: datetime,
        nav: float | None = None,
        min_notional: float = 1.0,
    ) -> list[Order]:
        """Convert target weights to Order objects (unsigned qty + side)."""
        px = dict(self.last_marks)
        px.update(prices)
        nav_use = float(nav if nav is not None else self.nav(px))
        orders: list[Order] = []
        ids = sorted(set(targets) | set(self.shares))
        for sid in ids:
            price = px.get(sid)
            if price is None or price <= 0:
                continue
            tw = float(targets.get(sid, 0.0))
            desired = tw * nav_use / price
            current = self.shares.get(sid, 0.0)
            delta = desired - current
            if abs(delta) * price < min_notional:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(
                Order(
                    order_id=f"{self.slot}-{uuid4().hex[:10]}",
                    security_id=sid,
                    symbol=sid,
                    side=side,
                    quantity=abs(float(delta)),
                    signal_time=signal_time,
                    decision_time=signal_time,
                    order_time=order_time,
                    status=OrderStatus.NEW,
                )
            )
        # Liquidations must precede purchases: target rotations commonly fund
        # the new leg with proceeds from the old one. Keep the identifier sort
        # deterministic within each side while making the sequence cash-safe.
        orders.sort(key=lambda order: 0 if order.side is OrderSide.SELL else 1)
        return orders

    def cash_nav_identity(self, prices: dict[str, float] | None = None) -> dict[str, float]:
        """Cash + marked positions = NAV. Used by conservation property tests."""
        px = prices or self.last_marks
        pos = sum(self.shares.get(s, 0.0) * px.get(s, 0.0) for s in self.shares)
        nav = self.cash + pos
        return {
            "cash": float(self.cash),
            "position_mv": float(pos),
            "nav": float(nav),
            "residual": float(nav - (self.cash + pos)),
        }

    def reject_accounting(self) -> dict[str, int]:
        """Breakdown of order rejects (kill vs risk_gate vs other)."""
        other = int(self.reject_count) - int(self.halt_count) - int(self.risk_gate_reject_count)
        return {
            "reject_total": int(self.reject_count),
            "kill_switch_halts": int(self.halt_count),
            "risk_gate_rejects": int(self.risk_gate_reject_count),
            "other_rejects": max(other, 0),
        }

    def to_dict(self) -> dict[str, Any]:
        history = [
            {
                "order": record.order.model_dump(mode="json"),
                "reject_reason": record.reject_reason,
                "fill": None if record.fill is None else record.fill.model_dump(mode="json"),
                "slot": record.slot,
            }
            for record in self.history
        ]
        return {
            "slot": self.slot,
            "allow_capital": self.allow_capital,
            "cash": self.cash,
            "shares": dict(self.shares),
            "reject_count": self.reject_count,
            "halt_count": self.halt_count,
            "risk_gate_reject_count": self.risk_gate_reject_count,
            "n_fills": self._prior_fill_count + len(self.fills),
            "n_orders": self._prior_order_count + len(self.history),
            "history": history,
            "fill_convention": self.config.execution.fill.value,
            "kill_state": self.kill.state,
            "last_marks": dict(self.last_marks),
            "initial_cash": float(self.initial_cash),
        }

    @classmethod
    def from_state(cls, config: AppConfig, state: dict[str, Any]) -> SimulatedBroker:
        """Restore cash, positions, and receipt history for paper resume.

        Older state files did not persist records; their counters are retained
        as a baseline while newly generated records are appended normally.
        """
        initial_cash = float(state.get("initial_cash", state.get("cash", 0.0)))
        if not np.isfinite(initial_cash) or initial_cash < 0.0:
            raise ValueError("restored initial cash must be finite and non-negative")
        if "cash" not in state:
            # A persisted book without cash would silently re-grant the initial
            # cash on top of restored positions — fail closed instead.
            raise ValueError("restored broker state missing required 'cash' key")
        broker = cls(
            config=config,
            initial_cash=initial_cash,
            slot=str(state.get("slot", "champion")),
            allow_capital=bool(state.get("allow_capital", True)),
        )
        if "kill_state" in state:
            kill_state = state["kill_state"]
            if not isinstance(kill_state, str):
                raise ValueError("restored kill switch state must be a string")
            try:
                broker.kill.set_state(kill_state)
            except KillSwitchActive as exc:
                raise ValueError("restored kill switch state is invalid") from exc
        broker.cash = float(state["cash"])
        if not np.isfinite(broker.cash):
            raise ValueError("restored broker cash must be finite")
        broker.shares = {str(k): float(v) for k, v in dict(state.get("shares") or {}).items()}
        if any(not np.isfinite(value) for value in broker.shares.values()):
            raise ValueError("restored broker shares must be finite")
        counts = {
            name: int(state.get(name, 0))
            for name in ("reject_count", "halt_count", "risk_gate_reject_count")
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError("restored broker counters must be non-negative")
        broker.reject_count = counts["reject_count"]
        broker.halt_count = counts["halt_count"]
        broker.risk_gate_reject_count = counts["risk_gate_reject_count"]

        raw_history = state.get("history")
        if raw_history is None:
            # Backward-compatible fallback for pre-history state files.
            broker._prior_order_count = int(state.get("n_orders", 0))
            broker._prior_fill_count = int(state.get("n_fills", 0))
            if broker._prior_order_count < 0 or broker._prior_fill_count < 0:
                raise ValueError("restored broker receipt counts must be non-negative")
        elif not isinstance(raw_history, list):
            raise ValueError("restored broker history must be a list")
        else:
            try:
                broker.history = [
                    OrderRecord(
                        order=Order.model_validate(item["order"]),
                        reject_reason=item.get("reject_reason"),
                        fill=(
                            None if item.get("fill") is None else Fill.model_validate(item["fill"])
                        ),
                        slot=str(item.get("slot", broker.slot)),
                    )
                    for item in raw_history
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("restored broker history is invalid") from exc
            seen_fill_ids: set[str] = set()
            broker.fills = []
            for record in broker.history:
                if record.fill is None:
                    continue
                fill_id = str(record.fill.fill_id)
                if fill_id not in seen_fill_ids:
                    seen_fill_ids.add(fill_id)
                    broker.fills.append(record.fill)
            if int(state.get("n_orders", len(broker.history))) != len(broker.history):
                raise ValueError("broker history/order count mismatch")
            if int(state.get("n_fills", len(broker.fills))) != len(broker.fills):
                raise ValueError("broker history/fill count mismatch")

        marks = state.get("last_marks") or {}
        broker.mark({str(k): float(v) for k, v in marks.items()})
        return broker
