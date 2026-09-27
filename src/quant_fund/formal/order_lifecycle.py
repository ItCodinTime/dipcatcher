"""Executable order-lifecycle specification.

Twin of ``spec/tla/OrderLifecycle.tla``. TLC checks the TLA+ model on a
finite integer bound. This module is the same transition relation over
positive quantities and arbitrary fill ids, used as the trace validator.

One order in the TLA model. Traces here are the interleaving product over
order ids (orders do not share state). Cash is an accounting obligation,
not part of this state machine.

Correspondence with ``SimulatedBroker`` (simulation only):

* ``ACKED`` is spec status ``new`` (working, nothing filled).
* ``CANCELLED`` with ``reject_reason == "expired"`` is spec status ``expired``.
  Other cancels are ``canceled``.
* A market fill that participation-caps and does not rest is terminal
  ``filled`` with ``filled <= qty`` (IOC close-out). ``partial`` means the
  residual is still working.
* History rows stamp the child slice ``FILLED`` even when ``is_partial``.
  The trace uses ``fill.is_partial`` and whether a residual remains, not
  that slice status.
* A fill id that was already applied is a stutter (idempotent).
* A new fill id on a terminal order is rejected. The spec action
  ``late_fill`` is the stutter that ignores it.
* Restart of a durable snapshot is ``checkpoint`` then ``crash`` (identity).
  ``crash`` without a new checkpoint drops uncommitted fills. The paper
  ledger writes the snapshot after the step, so observed restarts are the
  identity. The spec still admits the lossy crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TERMINAL = frozenset({"filled", "canceled", "rejected", "expired"})
WORKING = frozenset({"new", "partial"})
EPS = 1e-8


class LifecycleViolation(ValueError):
    """The event is not enabled in the specification."""


@dataclass
class OrderState:
    status: str = "absent"
    qty: float = 0.0
    filled: float = 0.0
    fills: dict[str, float] = field(default_factory=dict)
    s_status: str = "absent"
    s_qty: float = 0.0
    s_filled: float = 0.0
    s_fills: dict[str, float] = field(default_factory=dict)


@dataclass
class Book:
    orders: dict[str, OrderState] = field(default_factory=dict)


@dataclass
class CheckResult:
    ok: bool
    violations: list[str]
    book: Book


def _copy_order(order: OrderState) -> OrderState:
    return OrderState(
        status=order.status,
        qty=order.qty,
        filled=order.filled,
        fills=dict(order.fills),
        s_status=order.s_status,
        s_qty=order.s_qty,
        s_filled=order.s_filled,
        s_fills=dict(order.s_fills),
    )


def _copy_book(book: Book) -> Book:
    return Book(orders={key: _copy_order(order) for key, order in book.orders.items()})


def _le(left: float, right: float) -> bool:
    return left <= right + EPS


def _eq(left: float, right: float) -> bool:
    return abs(left - right) <= EPS


def _snapshot(order: OrderState) -> None:
    order.s_status = order.status
    order.s_qty = order.qty
    order.s_filled = order.filled
    order.s_fills = dict(order.fills)


def _restore(order: OrderState) -> None:
    order.status = order.s_status
    order.qty = order.s_qty
    order.filled = order.s_filled
    order.fills = dict(order.s_fills)


def invariant_violations(order: OrderState) -> list[str]:
    """Safety obligations for one order, including its durable snapshot."""
    found: list[str] = []
    if not _le(order.filled, order.qty):
        found.append("overfilled")
    if not _le(order.s_filled, order.s_qty):
        found.append("stable overfilled")
    if not _eq(order.filled, sum(order.fills.values())):
        found.append("filled is not the sum of applied fills")
    if not _eq(order.s_filled, sum(order.s_fills.values())):
        found.append("stable filled is not the sum of stable fills")
    if order.s_filled > order.filled + EPS:
        found.append("stable filled exceeds volatile filled")
    for fill_id, qty in order.s_fills.items():
        if fill_id not in order.fills or not _eq(order.fills[fill_id], qty):
            found.append(f"stable fill {fill_id} missing from volatile state")
            break
    found.extend(_shape(order.status, order.qty, order.filled, order.fills, "volatile"))
    found.extend(_shape(order.s_status, order.s_qty, order.s_filled, order.s_fills, "stable"))
    return found


def _shape(
    status: str, qty: float, filled: float, fills: dict[str, float], label: str
) -> list[str]:
    if status == "absent":
        if filled > EPS or qty > EPS or fills:
            return [f"{label} absent order is not empty"]
        return []
    if qty <= EPS:
        return [f"{label} {status} order has no authorized quantity"]
    if status == "new" and filled > EPS:
        return [f"{label} new order already filled"]
    if status == "partial" and not (filled > EPS and filled < qty - EPS):
        return [f"{label} partial order filled {filled} of {qty}"]
    if status == "filled" and not (filled > EPS and _le(filled, qty)):
        return [f"{label} filled order filled {filled} of {qty}"]
    if status in {"canceled", "rejected", "expired"} and not _le(filled, qty):
        return [f"{label} {status} order overfilled"]
    if status not in TERMINAL | WORKING | {"absent"}:
        return [f"{label} unknown status {status}"]
    return []


def step(book: Book, event: dict[str, Any]) -> Book:
    """Apply one specification action. Raise ``LifecycleViolation`` if disabled."""
    book = _copy_book(book)
    op = str(event["op"])
    if op == "checkpoint":
        for order in book.orders.values():
            _snapshot(order)
        return book
    if op == "crash":
        for order in book.orders.values():
            _restore(order)
        return book

    order_id = str(event["order_id"])
    order = book.orders.setdefault(order_id, OrderState())
    if op == "submit":
        qty = float(event["qty"])
        if order.status != "absent":
            raise LifecycleViolation(f"{order_id}: submit from {order.status}")
        if not math_positive(qty):
            raise LifecycleViolation(f"{order_id}: submit quantity must be positive")
        order.status = "new"
        order.qty = qty
        return book
    if op == "reject":
        _require_working(order, order_id, op)
        order.status = "rejected"
        return book
    if op == "cancel":
        _require_working(order, order_id, op)
        order.status = "canceled"
        return book
    if op == "expire":
        _require_working(order, order_id, op)
        order.status = "expired"
        return book
    if op == "amend":
        _require_working(order, order_id, op)
        qty = float(event["qty"])
        if not (qty > order.filled + EPS and not _eq(qty, order.qty)):
            raise LifecycleViolation(f"{order_id}: amend to {qty} is not enabled")
        order.qty = qty
        order.status = "new" if order.filled <= EPS else "partial"
        return book
    if op == "fill":
        return _apply_fill(order, order_id, event, book)
    if op == "duplicate_fill":
        fill_id = str(event["fill_id"])
        if fill_id not in order.fills:
            raise LifecycleViolation(f"{order_id}: duplicate of unknown fill {fill_id}")
        return book
    if op == "late_fill":
        if order.status not in TERMINAL:
            raise LifecycleViolation(f"{order_id}: late_fill on {order.status}")
        return book
    raise LifecycleViolation(f"unknown op {op}")


def _apply_fill(order: OrderState, order_id: str, event: dict[str, Any], book: Book) -> Book:
    fill_id = str(event["fill_id"])
    qty = float(event["qty"])
    if fill_id in order.fills:
        return book
    if order.status not in WORKING:
        raise LifecycleViolation(f"{order_id}: fill on {order.status}")
    if qty <= EPS or not _le(order.filled + qty, order.qty):
        raise LifecycleViolation(f"{order_id}: fill {qty} would exceed authorized {order.qty}")
    order.fills[fill_id] = qty
    order.filled += qty
    rest = bool(event.get("rest", False))
    if rest and order.filled < order.qty - EPS:
        order.status = "partial"
    else:
        order.status = "filled"
    return book


def _require_working(order: OrderState, order_id: str, op: str) -> None:
    if order.status not in WORKING:
        raise LifecycleViolation(f"{order_id}: {op} from {order.status}")


def math_positive(value: float) -> bool:
    return value > EPS


def check_trace(events: list[dict[str, Any]]) -> CheckResult:
    """Return the first violation. An empty trace is the absent book."""
    book = Book()
    violations: list[str] = []
    for index, event in enumerate(events):
        try:
            book = step(book, event)
        except LifecycleViolation as exc:
            violations.append(f"event {index} {event.get('op')}: {exc}")
            break
        for order_id, order in book.orders.items():
            for message in invariant_violations(order):
                violations.append(f"event {index} {order_id}: {message}")
        if violations:
            break
    return CheckResult(ok=not violations, violations=violations, book=book)


def _state_key(order: OrderState) -> tuple[Any, ...]:
    return (
        order.status,
        order.qty,
        order.filled,
        tuple(sorted(order.fills.items())),
        order.s_status,
        order.s_qty,
        order.s_filled,
        tuple(sorted(order.s_fills.items())),
    )


def _bounded_events(max_qty: int, max_fill_id: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [{"op": "checkpoint"}, {"op": "crash"}]
    for qty in range(1, max_qty + 1):
        events.append({"op": "submit", "order_id": "A", "qty": qty})
        events.append({"op": "amend", "order_id": "A", "qty": qty})
    for op in ("reject", "cancel", "expire"):
        events.append({"op": op, "order_id": "A"})
    for fill_id in range(1, max_fill_id + 1):
        events.append({"op": "duplicate_fill", "order_id": "A", "fill_id": str(fill_id)})
        events.append({"op": "late_fill", "order_id": "A", "fill_id": str(fill_id)})
        for qty in range(1, max_qty + 1):
            for rest in (False, True):
                events.append(
                    {
                        "op": "fill",
                        "order_id": "A",
                        "fill_id": str(fill_id),
                        "qty": qty,
                        "rest": rest,
                    }
                )
    return events


def _action_violations(before: OrderState, after: OrderState) -> list[str]:
    found: list[str] = []
    if after.filled > before.filled + EPS and after.status not in {"partial", "filled"}:
        found.append("fill moved the order outside partial/filled")
    if before.s_status in TERMINAL and (
        after.s_status != before.s_status
        or not _eq(after.s_filled, before.s_filled)
        or after.s_fills != before.s_fills
    ):
        found.append("durable terminal state changed")
    if (
        before.status in TERMINAL
        and before.status == before.s_status
        and (after.status != before.status or not _eq(after.filled, before.filled))
    ):
        found.append("committed terminal state changed")
    if after.s_filled + EPS < before.s_filled:
        found.append("durable filled decreased")
    for fill_id, qty in before.s_fills.items():
        if fill_id not in after.s_fills or not _eq(after.s_fills[fill_id], qty):
            found.append("durable fill id disappeared")
            break
    return found


def enumerate_safety(max_qty: int = 2, max_fill_id: int = 2) -> tuple[int, list[str]]:
    """Explicit-state check of the executable spec on one integer order.

    Returns ``(states, violations)``. Disabled events are skipped. This is
    the same safety relation TLC checks, on the Python twin, at a small bound.
    """
    if max_qty < 1 or max_fill_id < 1:
        raise ValueError("bounds must be positive")
    start = Book()
    seen: dict[tuple[Any, ...], Book] = {_state_key(OrderState()): start}
    queue: list[Book] = [start]
    events = _bounded_events(max_qty, max_fill_id)
    violations: list[str] = []
    while queue:
        current = queue.pop()
        current_order = current.orders.get("A", OrderState())
        for event in events:
            if event["op"] in {"submit", "amend"} and int(event["qty"]) > max_qty:
                continue
            try:
                nxt = step(current, event)
            except LifecycleViolation:
                continue
            nxt_order = nxt.orders.get("A", OrderState())
            if int(event.get("qty", 0) or 0) > max_qty or nxt_order.qty > max_qty:
                continue
            for message in invariant_violations(nxt_order):
                violations.append(f"{event['op']}: {message}")
            for message in _action_violations(current_order, nxt_order):
                violations.append(f"{event['op']}: {message}")
            if violations:
                return len(seen), violations
            key = _state_key(nxt_order)
            if key not in seen:
                seen[key] = nxt
                queue.append(nxt)
    return len(seen), violations
