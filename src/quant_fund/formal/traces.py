"""Trace extraction from the simulated broker and the paper loop.

The extractor is a best-effort reading of order records. ``TraceSession``
records the requested quantity before ``submit`` mutates it, which the
history row does not retain for an IOC partial (the stored quantity is the
executed child). Both must be behaviours of the spec; the session is the
stricter source.
"""

from __future__ import annotations

import math
from typing import Any

from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.schemas.orders import OrderStatus


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def events_from_order_rows(
    rows: list[dict[str, Any]],
    *,
    open_order_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Map ledger or broker history rows to specification events.

    The first row for an id submits ``quantity`` as authorized. A later
    working row with no fill is an amend of the residual. A partial fill
    rests when another row for the id follows, or the id is still open.
    An IOC partial whose only row is the fill itself cannot recover the
    parent quantity: the trace uses the executed quantity.
    """
    open_ids = open_order_ids or set()
    counts: dict[str, int] = {}
    for row in rows:
        oid = str(row["order_id"])
        counts[oid] = counts.get(oid, 0) + 1
    seen: dict[str, int] = {}
    filled: dict[str, float] = {}
    authorized: dict[str, float] = {}
    events: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["order_id"])
        seen[oid] = seen.get(oid, 0) + 1
        more = seen[oid] < counts[oid]
        status = str(row["status"])
        reason = str(row.get("reject_reason") or "")
        quantity = float(row["quantity"])
        fill_qty = _finite(row.get("fill_qty"))
        fill_id = str(row.get("fill_id") or "")
        is_partial = bool(row.get("is_partial", False))
        if oid not in authorized:
            events.append({"op": "submit", "order_id": oid, "qty": quantity})
            authorized[oid] = quantity
            filled[oid] = 0.0
        if fill_id:
            qty = fill_qty if fill_qty is not None else quantity
            rest = is_partial and (more or oid in open_ids)
            events.append(
                {
                    "op": "fill",
                    "order_id": oid,
                    "fill_id": fill_id,
                    "qty": qty,
                    "rest": rest,
                }
            )
            filled[oid] = filled.get(oid, 0.0) + qty
            continue
        if status == "rejected":
            events.append({"op": "reject", "order_id": oid})
            continue
        if status == "cancelled":
            op = "expire" if reason == "expired" else "cancel"
            events.append({"op": op, "order_id": oid})
            continue
        if status in {"acked", "new", "partial"} and seen[oid] > 1:
            new_auth = filled.get(oid, 0.0) + quantity
            if abs(new_auth - authorized[oid]) > 1e-8:
                events.append({"op": "amend", "order_id": oid, "qty": new_auth})
                authorized[oid] = new_auth
    return events


def events_from_broker(broker: SimulatedBroker) -> list[dict[str, Any]]:
    """Project ``broker.history`` into specification events."""
    rows: list[dict[str, Any]] = []
    for record in broker.history:
        fill = record.fill
        rows.append(
            {
                "order_id": record.order.order_id,
                "status": record.order.status.value,
                "quantity": float(record.order.quantity),
                "reject_reason": "" if record.reject_reason is None else str(record.reject_reason),
                "fill_id": "" if fill is None else str(fill.fill_id),
                "fill_qty": float("nan") if fill is None else float(fill.quantity),
                "is_partial": False if fill is None else bool(fill.is_partial),
            }
        )
    return events_from_order_rows(rows, open_order_ids=set(broker.open_orders))


class TraceSession:
    """Drive a simulated broker and record the spec events it actually takes.

    Requested quantity is logged before ``submit`` replaces it with the
    executed child. This is the conformance source for IOC partials.
    """

    def __init__(self, broker: SimulatedBroker) -> None:
        self.broker = broker
        self.events: list[dict[str, Any]] = []

    def submit(self, order: Any, **kwargs: Any) -> Any:
        self.events.append(
            {"op": "submit", "order_id": order.order_id, "qty": float(order.quantity)}
        )
        record = self.broker.submit(order, **kwargs)
        self._record(record)
        return record

    def process_bar(self, security_id: str, **kwargs: Any) -> list[Any]:
        records = self.broker.process_bar(security_id, **kwargs)
        for record in records:
            self._record(record)
        return records

    def cancel(self, order_id: str) -> Any:
        record = self.broker.cancel_order(order_id)
        self.events.append({"op": "cancel", "order_id": order_id})
        return record

    def amend(self, order_id: str, **kwargs: Any) -> Any:
        before = self.broker.open_orders[order_id].quantity
        record = self.broker.amend_order(order_id, **kwargs)
        quantity = kwargs.get("quantity")
        if quantity is not None and abs(float(quantity) - float(before)) > 1e-8:
            filled = sum(fill.quantity for fill in self.broker.fills if fill.order_id == order_id)
            self.events.append(
                {"op": "amend", "order_id": order_id, "qty": filled + float(quantity)}
            )
        return record

    def restart(self) -> SimulatedBroker:
        """Durable snapshot restore: checkpoint, then crash back onto it."""
        state = self.broker.to_dict()
        self.broker = SimulatedBroker.from_state(self.broker.config, state)
        self.events.append({"op": "checkpoint"})
        self.events.append({"op": "crash"})
        return self.broker

    def _record(self, record: Any) -> None:
        order = record.order
        oid = order.order_id
        if order.status is OrderStatus.REJECTED:
            self.events.append({"op": "reject", "order_id": oid})
            return
        if order.status is OrderStatus.CANCELLED:
            op = "expire" if record.reject_reason == "expired" else "cancel"
            self.events.append({"op": op, "order_id": oid})
            return
        fill = record.fill
        if fill is None:
            return
        rest = bool(fill.is_partial) and oid in self.broker.open_orders
        self.events.append(
            {
                "op": "fill",
                "order_id": oid,
                "fill_id": fill.fill_id,
                "qty": float(fill.quantity),
                "rest": rest,
            }
        )
