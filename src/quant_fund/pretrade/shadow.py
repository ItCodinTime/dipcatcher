"""Shadow adapter. Observes simulated-broker orders and does not change them.

The adapter evaluates each submission the paper broker would send, appends an
allow or deny record, and then calls the original method. Observer failures
are logged as fail-closed denies. They are not raised into the broker.
"""

from __future__ import annotations

import json
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.pretrade.codes import (
    INTERNAL,
    KIND_CANCEL,
    KIND_ORDER,
    KIND_REPLACE,
    UNKNOWN_SYMBOL,
    reason_names,
)
from quant_fund.pretrade.config import PretradeConfig, sign_config
from quant_fund.pretrade.engine import OrderView, PretradeEngine
from quant_fund.schemas.orders import Order, OrderSide

_KIND_NAME = {KIND_ORDER: "order", KIND_CANCEL: "cancel", KIND_REPLACE: "replace"}


def business_session_id(day: date) -> int:
    """Weekday index. Holidays are not removed; Saturday and Sunday share Friday."""
    zero = day.toordinal() - 1
    return (zero // 7) * 5 + min(zero % 7, 4) + 1


def _ts_ns(when: datetime) -> int:
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return int(when.timestamp() * 1_000_000_000)


class ShadowRiskAdapter:
    """In-process shadow log for ``SimulatedBroker`` submissions."""

    def __init__(
        self,
        config: PretradeConfig,
        *,
        hmac_key: bytes,
        log_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self._hmac_key = hmac_key
        self.config_sha256, self.config_hmac_sha256 = sign_config(config, hmac_key)
        self.schema_version = int(config.schema_version)
        self._tz = ZoneInfo(config.session.timezone)
        self.events: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._engines: dict[int, PretradeEngine] = {}
        self._engine_refs: dict[int, weakref.ReferenceType[SimulatedBroker]] = {}
        self._pending: dict[tuple[int, str], tuple[int, float, int, int]] = {}
        self._fh: TextIO | None = None
        if log_path is not None:
            self._fh = Path(log_path).open("a", encoding="utf-8")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def attach(self, broker: SimulatedBroker) -> None:
        """Wrap one broker instance. The original methods still perform the fill."""
        if getattr(broker, "_pretrade_shadow_attached", False):
            raise RuntimeError("broker already has a pretrade shadow adapter")
        if getattr(type(broker), "_pretrade_shadow_patched", False):
            raise RuntimeError("simulated broker class is already observed")
        orig_submit = broker.submit
        orig_cancel = broker.cancel_order
        orig_amend = broker.amend_order
        adapter = self

        def submit(order: Order, **kwargs: Any) -> Any:
            adapter._safe_before(broker, order, kwargs, KIND_ORDER)
            record = orig_submit(order, **kwargs)
            adapter._safe_after(broker, order, record)
            return record

        def cancel_order(order_id: str) -> Any:
            adapter._safe_cancel(broker, order_id)
            return orig_cancel(order_id)

        def amend_order(order_id: str, **kwargs: Any) -> Any:
            adapter._safe_amend(broker, order_id, kwargs)
            return orig_amend(order_id, **kwargs)

        broker.submit = submit  # type: ignore[method-assign]
        broker.cancel_order = cancel_order  # type: ignore[method-assign]
        broker.amend_order = amend_order  # type: ignore[method-assign]
        broker._pretrade_shadow_attached = True  # type: ignore[attr-defined]

    @contextmanager
    def observe_simulated_broker(self) -> Iterator[ShadowRiskAdapter]:
        """Patch ``SimulatedBroker`` for the duration of a paper loop.

        ``run_paper_loop`` constructs its own brokers. The patch sees those
        submissions and restores the original methods before returning.
        """
        if getattr(SimulatedBroker, "_pretrade_shadow_patched", False):
            raise RuntimeError("simulated broker class is already observed")
        orig_submit = SimulatedBroker.submit
        orig_cancel = SimulatedBroker.cancel_order
        orig_amend = SimulatedBroker.amend_order
        adapter = self

        def submit(broker: SimulatedBroker, order: Order, **kwargs: Any) -> Any:
            adapter._safe_before(broker, order, kwargs, KIND_ORDER)
            record = orig_submit(broker, order, **kwargs)
            adapter._safe_after(broker, order, record)
            return record

        def cancel_order(broker: SimulatedBroker, order_id: str) -> Any:
            adapter._safe_cancel(broker, order_id)
            return orig_cancel(broker, order_id)

        def amend_order(broker: SimulatedBroker, order_id: str, **kwargs: Any) -> Any:
            adapter._safe_amend(broker, order_id, kwargs)
            return orig_amend(broker, order_id, **kwargs)

        SimulatedBroker.submit = submit  # type: ignore[method-assign, assignment]
        SimulatedBroker.cancel_order = cancel_order  # type: ignore[method-assign, assignment]
        SimulatedBroker.amend_order = amend_order  # type: ignore[method-assign, assignment]
        SimulatedBroker._pretrade_shadow_patched = True  # type: ignore[attr-defined]
        try:
            yield self
        finally:
            SimulatedBroker.submit = orig_submit  # type: ignore[method-assign]
            SimulatedBroker.cancel_order = orig_cancel  # type: ignore[method-assign]
            SimulatedBroker.amend_order = orig_amend  # type: ignore[method-assign]
            SimulatedBroker._pretrade_shadow_patched = False  # type: ignore[attr-defined]

    def _engine_for(self, broker: SimulatedBroker, ts_ns: int) -> PretradeEngine:
        # CPython ids are reused after garbage collection: a new broker must
        # never bind to a dead broker's engine, so state is dropped on GC.
        key = id(broker)
        engine = self._engines.get(key)
        if engine is not None:
            return engine
        cash = float(broker.cash)
        try:
            nav = float(broker.nav())
        except Exception:
            nav = cash
        if not (nav == nav and nav > 0.0):
            nav = cash if cash > 0.0 else 1.0
        engine = PretradeEngine(
            self.config,
            hmac_key=self._hmac_key,
            initial_nav=nav,
            initial_cash=cash if cash > 0.0 else 0.0,
            ts_ns=ts_ns if ts_ns > 0 else 1,
        )
        self._engines[key] = engine

        def release(_ref: weakref.ReferenceType[SimulatedBroker], k: int = key) -> None:
            self._forget_broker(k)

        self._engine_refs[key] = weakref.ref(broker, release)
        return engine

    def _forget_broker(self, key: int) -> None:
        self._engine_refs.pop(key, None)
        self._engines.pop(key, None)
        for pending_key in [k for k in self._pending if k[0] == key]:
            self._pending.pop(pending_key, None)

    def _sync(self, engine: PretradeEngine, broker: SimulatedBroker, ts_ns: int) -> None:
        marks = broker.last_marks
        securities = set(broker.shares) | set(marks)
        for security in securities:
            sid = engine.ensure_symbol(str(security))
            if sid < 0:
                engine.book.mark_deadline = 0
                continue
            raw_ref = marks.get(security, float("nan"))
            ref = float(raw_ref) if raw_ref is not None else float("nan")
            engine.set_symbol(
                sid,
                pos=float(broker.shares.get(security, 0.0)),
                ref_px=ref,
                ref_ts_ns=ts_ns,
            )
        try:
            nav = float(broker.nav(marks))
        except Exception:
            nav = float("nan")
        engine.update_account(nav=nav, mark_ts_ns=ts_ns if nav == nav and nav > 0.0 else 0)

    def _safe_before(
        self,
        broker: SimulatedBroker,
        order: Order,
        kwargs: dict[str, Any],
        kind: int,
    ) -> None:
        try:
            self._before(broker, order, kwargs, kind)
        except Exception as exc:
            self._fail(order.order_id, str(getattr(order, "security_id", "")), kind, exc)

    def _before(
        self,
        broker: SimulatedBroker,
        order: Order,
        kwargs: dict[str, Any],
        kind: int,
    ) -> None:
        ts_ns = _ts_ns(order.order_time)
        engine = self._engine_for(broker, ts_ns)
        self._sync(engine, broker, ts_ns)
        sid = engine.ensure_symbol(str(order.security_id))
        limit = order.limit_price
        raw_price = kwargs.get("price")
        if limit is not None:
            px = float(limit)
            is_limit = 1
        elif raw_price is not None:
            px = float(raw_price)
            is_limit = 0
        else:
            px = 0.0
            is_limit = 0
        if sid >= 0 and str(order.security_id) not in broker.last_marks and px > 0.0:
            engine.set_symbol(
                sid,
                pos=float(broker.shares.get(order.security_id, 0.0)),
                ref_px=px,
                ref_ts_ns=ts_ns,
            )
        session_id = business_session_id(self._local_date(order.order_time))
        pos_before = engine.book.syms[sid].pos if sid >= 0 else 0.0
        view = OrderView(
            symbol_id=sid,
            side=1 if order.side is OrderSide.BUY else -1,
            qty=float(order.quantity),
            px=px,
            ts_ns=ts_ns,
            session_id=session_id,
            is_limit=is_limit,
            kind=kind,
        )
        decision = engine.decide(view, apply=False)
        self._pending[(id(broker), order.order_id)] = (sid, pos_before, session_id, ts_ns)
        self._emit(
            {
                "mode": "shadow",
                "behavior_changed": False,
                "order_id": order.order_id,
                "symbol": order.security_id,
                "kind": _KIND_NAME.get(kind, "order"),
                "allowed": decision.allowed,
                "reasons": list(decision.reasons),
                "reason_bits": decision.reason_bits,
                "config_sha256": decision.config_sha256,
                "config_hmac_sha256": decision.config_hmac_sha256,
                "schema_version": decision.schema_version,
                "kill_latched": decision.kill_latched,
                "broker_reject_reason": None,
                "broker_status": None,
            }
        )

    def _safe_after(self, broker: SimulatedBroker, order: Order, record: Any) -> None:
        try:
            self._after(broker, order, record)
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def _after(self, broker: SimulatedBroker, order: Order, record: Any) -> None:
        if self.events:
            event = self.events[-1]
            if event.get("order_id") == order.order_id:
                event["broker_reject_reason"] = record.reject_reason
                event["broker_status"] = record.order.status.value
                self._rewrite_last(event)
        pending = self._pending.pop((id(broker), order.order_id), None)
        fill = record.fill
        if fill is None:
            return
        if pending is None:
            return
        sid, pos_before, session_id, _ts = pending
        if sid < 0:
            return
        engine = self._engines[id(broker)]
        engine.note_fill(
            symbol_id=sid,
            side=1 if order.side is OrderSide.BUY else -1,
            qty=float(fill.quantity),
            px=float(fill.price),
            fee=float(fill.fee),
            ts_ns=_ts_ns(fill.fill_time),
            session_id=session_id,
            pos_before=pos_before,
        )

    def _safe_cancel(self, broker: SimulatedBroker, order_id: str) -> None:
        try:
            working = broker.open_orders.get(order_id)
            if working is None:
                self._emit(self._bare("cancel", order_id, "", ("unknown_symbol",), UNKNOWN_SYMBOL))
                return
            self._before(
                broker,
                working,
                {"price": working.limit_price or broker.last_marks.get(working.security_id, 1.0)},
                KIND_CANCEL,
            )
        except Exception as exc:
            self._fail(order_id, "", KIND_CANCEL, exc)

    def _safe_amend(self, broker: SimulatedBroker, order_id: str, kwargs: dict[str, Any]) -> None:
        try:
            working = broker.open_orders.get(order_id)
            if working is None:
                self._emit(self._bare("replace", order_id, "", ("unknown_symbol",), UNKNOWN_SYMBOL))
                return
            data = working.model_dump()
            if "quantity" in kwargs and kwargs["quantity"] is not None:
                data["quantity"] = kwargs["quantity"]
            if "limit_price" in kwargs and kwargs["limit_price"] is not None:
                data["limit_price"] = kwargs["limit_price"]
            amended = Order.model_validate(data)
            price = amended.limit_price or broker.last_marks.get(amended.security_id, 1.0)
            self._before(broker, amended, {"price": price}, KIND_REPLACE)
        except Exception as exc:
            self._fail(order_id, "", KIND_REPLACE, exc)

    def _local_date(self, when: datetime) -> date:
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return when.astimezone(self._tz).date()

    def _fail(self, order_id: str, symbol: str, kind: int, exc: BaseException) -> None:
        self.errors.append(f"{type(exc).__name__}: {exc}")
        self._emit(
            self._bare(
                _KIND_NAME.get(kind, "order"),
                order_id,
                symbol,
                ("internal_error",),
                INTERNAL,
            )
        )

    def _bare(
        self,
        kind: str,
        order_id: str,
        symbol: str,
        reasons: tuple[str, ...],
        bits: int,
    ) -> dict[str, Any]:
        return {
            "mode": "shadow",
            "behavior_changed": False,
            "order_id": order_id,
            "symbol": symbol,
            "kind": kind,
            "allowed": False,
            "reasons": list(reasons),
            "reason_bits": bits,
            "config_sha256": self.config_sha256,
            "config_hmac_sha256": self.config_hmac_sha256,
            "schema_version": self.schema_version,
            "kill_latched": True,
            "broker_reject_reason": None,
            "broker_status": None,
        }

    def _emit(self, event: dict[str, Any]) -> None:
        event["config_sha256"] = event.get("config_sha256") or self.config_sha256
        event["config_hmac_sha256"] = event.get("config_hmac_sha256") or self.config_hmac_sha256
        self.events.append(event)
        if self._fh is not None:
            self._fh.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            self._fh.flush()

    def _rewrite_last(self, event: dict[str, Any]) -> None:
        if self._fh is None:
            return
        # The JSONL stream is append-only. The post-trade broker status is a
        # second line so the pre-trade record is not rewritten in place.
        self._fh.write(
            json.dumps(
                {
                    "mode": "shadow",
                    "order_id": event.get("order_id"),
                    "broker_reject_reason": event.get("broker_reject_reason"),
                    "broker_status": event.get("broker_status"),
                    "config_sha256": self.config_sha256,
                    "kind": "broker_result",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._fh.flush()


def shadow_reason_names(bits: int) -> tuple[str, ...]:
    return reason_names(bits)
