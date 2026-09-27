"""Typed corporate-action event model.

Each event is anchored on its ``ex_date`` — the first session where the
security trades ex (without) the action's entitlement or on the new share
basis. ``announce_date`` is the first date the action was knowable; it is the
point-in-time visibility gate. An event only enters an as-of adjustment when
both ``announce_date <= asof`` (observable) and ``ex_date <= asof``
(effective): a declared-but-not-yet-effective split does not rescale history,
and a back-dated filing cannot rewrite a series queried before it arrived.

Effective ranges: a SPLIT/REVERSE_SPLIT/STOCK_DIVIDEND/CASH_DIVIDEND rescales
every bar strictly before ``ex_date`` (the half-open range ``(-inf, ex_date)``).
SYMBOL_CHANGE renames on ``[ex_date, next change)``. DELISTING and MERGER use
``ex_date`` as the *last* trading day of the security (inclusive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

import polars as pl

from quant_fund.schemas.errors import PointInTimeError


class CorporateActionKind(StrEnum):
    """Supported corporate-action event kinds."""

    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    CASH_DIVIDEND = "cash_dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SYMBOL_CHANGE = "symbol_change"
    DELISTING = "delisting"
    MERGER = "merger"


# Kinds that rescale the share basis multiplicatively (price / ratio).
_SHARE_BASIS_KINDS = frozenset(
    {
        CorporateActionKind.SPLIT,
        CorporateActionKind.REVERSE_SPLIT,
        CorporateActionKind.STOCK_DIVIDEND,
    }
)
# Kinds paid in cash per share (additive amount, proportional price factor).
_CASH_KINDS = frozenset(
    {
        CorporateActionKind.CASH_DIVIDEND,
        CorporateActionKind.SPECIAL_DIVIDEND,
    }
)
# Kinds that end the security's trading life on ``ex_date`` (inclusive).
_TERMINAL_KINDS = frozenset(
    {
        CorporateActionKind.DELISTING,
        CorporateActionKind.MERGER,
    }
)

# Frame vocabulary accepted by ``events_from_frame`` (extends the warehouse
# action_type set in ``data/adapters/parquet.py``).
_FRAME_KIND_MAP = {
    "split": CorporateActionKind.SPLIT,
    "reverse_split": CorporateActionKind.REVERSE_SPLIT,
    "cash_dividend": CorporateActionKind.CASH_DIVIDEND,
    "special_dividend": CorporateActionKind.SPECIAL_DIVIDEND,
    "stock_dividend": CorporateActionKind.STOCK_DIVIDEND,
    "ticker_change": CorporateActionKind.SYMBOL_CHANGE,
    "symbol_change": CorporateActionKind.SYMBOL_CHANGE,
    "delist": CorporateActionKind.DELISTING,
    "delisting": CorporateActionKind.DELISTING,
    "merger": CorporateActionKind.MERGER,
}


def _as_date(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


@dataclass(frozen=True)
class CorporateActionEvent:
    """One corporate action on one security.

    Fields per kind:

    - ``SPLIT``: ``ratio`` > 1 shares multiplier (4.0 for a 4:1 split).
    - ``REVERSE_SPLIT``: ``ratio`` in (0, 1) (0.125 for a 1:8 reverse).
    - ``STOCK_DIVIDEND``: ``ratio`` > 1 (1.05 for a 5% stock dividend).
    - ``CASH_DIVIDEND``/``SPECIAL_DIVIDEND``: ``amount`` >= 0 cash per share;
      ``reference_price`` optionally overrides the pre-ex close used as the
      proportional-adjustment denominator.
    - ``SYMBOL_CHANGE``: ``new_symbol`` non-blank.
    - ``DELISTING``: no extras; ``ex_date`` is the last trading day.
    - ``MERGER``: ``ex_date`` is the target's last trading day; ``amount`` is
      cash per target share and/or ``ratio`` is acquirer shares per target
      share; ``successor_id``/``new_symbol`` point at the surviving security.
      At least one of ``amount``, ``ratio``/``successor_id`` is required.
    """

    security_id: str
    kind: CorporateActionKind
    ex_date: date
    announce_date: date | None = None
    ratio: float | None = None
    amount: float | None = None
    new_symbol: str | None = None
    successor_id: str | None = None
    reference_price: float | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not str(self.security_id).strip():
            raise PointInTimeError("corporate action missing security_id")
        kind = CorporateActionKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "ex_date", _as_date(self.ex_date))
        if self.announce_date is not None:
            object.__setattr__(self, "announce_date", _as_date(self.announce_date))
        if kind in _SHARE_BASIS_KINDS:
            if not _finite(self.ratio) or self.ratio <= 0:  # type: ignore[operator]
                raise PointInTimeError(f"{kind} requires a finite positive ratio")
            if kind is CorporateActionKind.SPLIT and self.ratio <= 1.0:
                raise PointInTimeError("split ratio must be > 1 (use REVERSE_SPLIT otherwise)")
            if kind is CorporateActionKind.REVERSE_SPLIT and self.ratio >= 1.0:
                raise PointInTimeError("reverse_split ratio must be in (0, 1)")
            if kind is CorporateActionKind.STOCK_DIVIDEND and self.ratio <= 1.0:
                raise PointInTimeError("stock_dividend ratio must be > 1")
        if kind in _CASH_KINDS:
            if not _finite(self.amount) or self.amount < 0:  # type: ignore[operator]
                raise PointInTimeError(f"{kind} requires a finite non-negative amount")
            if self.reference_price is not None and (
                not math.isfinite(self.reference_price) or self.reference_price <= 0
            ):
                raise PointInTimeError("reference_price must be finite and positive")
        if kind is CorporateActionKind.SYMBOL_CHANGE:
            if not self.new_symbol or not str(self.new_symbol).strip():
                raise PointInTimeError("symbol_change requires a non-blank new_symbol")
            object.__setattr__(self, "new_symbol", str(self.new_symbol).strip())
        if kind is CorporateActionKind.MERGER:
            has_cash = _finite(self.amount) and self.amount > 0  # type: ignore[operator]
            has_stock = _finite(self.ratio) and self.ratio > 0  # type: ignore[operator]
            if not (has_cash or has_stock or self.successor_id):
                raise PointInTimeError(
                    "merger requires cash amount, share ratio, or successor_id"
                )
            if self.ratio is not None and (not math.isfinite(self.ratio) or self.ratio <= 0):
                raise PointInTimeError("merger ratio must be finite and positive")
            if self.amount is not None and (not math.isfinite(self.amount) or self.amount < 0):
                raise PointInTimeError("merger cash amount must be finite and non-negative")

    @property
    def visible_from(self) -> date:
        """First date the event is observable (announcement, else ex-date)."""
        return self.announce_date if self.announce_date is not None else self.ex_date

    @property
    def is_terminal(self) -> bool:
        return self.kind in _TERMINAL_KINDS

    def known_at(self, asof: date | datetime) -> bool:
        """True when the event is observable at ``asof`` (availability gate)."""
        return self.visible_from <= _as_date(asof)

    def effective_at(self, asof: date | datetime) -> bool:
        """True when the event has taken price effect by ``asof``."""
        return self.known_at(asof) and self.ex_date <= _as_date(asof)

    def applies_to_bar(self, bar_date: date | datetime) -> bool:
        """True when a bar on ``bar_date`` is pre-ex and must be rescaled."""
        return _as_date(bar_date) < self.ex_date


def share_basis_factor(event: CorporateActionEvent) -> float:
    """Price multiplier applied to pre-ex bars for share-basis events.

    A 4:1 split yields 1/4; a 1:8 reverse yields 8.0. Cash-dividend and
    identity events are handled elsewhere (they need a reference price).
    """
    if event.kind in _SHARE_BASIS_KINDS:
        assert event.ratio is not None  # enforced by __post_init__
        return 1.0 / float(event.ratio)
    return 1.0


def volume_factor(event: CorporateActionEvent) -> float:
    """Volume multiplier for pre-ex bars: shares scale with ``ratio``."""
    if event.kind in _SHARE_BASIS_KINDS:
        assert event.ratio is not None
        return float(event.ratio)
    return 1.0


def dividend_amount(event: CorporateActionEvent) -> float:
    """Cash per share paid on ``ex_date`` (0.0 for non-cash kinds)."""
    if event.kind in _CASH_KINDS:
        assert event.amount is not None
        return float(event.amount)
    return 0.0


def events_from_frame(actions: pl.DataFrame, *, strict: bool = True) -> list[CorporateActionEvent]:
    """Adapt a warehouse ``corporate_actions`` frame into typed events.

    Recognized ``action_type`` values: ``split``, ``reverse_split``,
    ``cash_dividend``, ``special_dividend``, ``stock_dividend``,
    ``ticker_change``/``symbol_change``, ``delist``/``delisting``, ``merger``.
    A legacy ``split`` row with ``factor < 1`` is re-typed ``REVERSE_SPLIT``.
    ``available_time`` becomes ``announce_date``. Unknown action types raise
    ``PointInTimeError`` unless ``strict=False`` (skip silently is a leakage
    risk, so strict is the default).
    """
    if actions.is_empty() or "action_type" not in actions.columns:
        return []
    required = {"security_id", "event_time"}
    missing = required - set(actions.columns)
    if missing:
        raise PointInTimeError(f"corporate actions missing required columns: {sorted(missing)}")

    events: list[CorporateActionEvent] = []
    for row in actions.iter_rows(named=True):
        raw_kind = row.get("action_type")
        kind_name = str(raw_kind).strip() if raw_kind is not None else ""
        mapped = _FRAME_KIND_MAP.get(kind_name)
        if mapped is None:
            if strict:
                raise PointInTimeError(f"unknown corporate action_type: {kind_name!r}")
            continue
        ratio = row.get("factor")
        if mapped is CorporateActionKind.SPLIT and ratio is not None and float(ratio) < 1.0:
            mapped = CorporateActionKind.REVERSE_SPLIT
        announce = row.get("available_time")
        amount = row.get("amount")
        successor = row.get("successor_id")
        events.append(
            CorporateActionEvent(
                security_id=str(row["security_id"]),
                kind=mapped,
                ex_date=_as_date(row["event_time"]),
                announce_date=_as_date(announce) if announce is not None else None,
                ratio=float(ratio) if ratio is not None else None,
                amount=float(amount) if amount is not None else None,
                new_symbol=row.get("new_ticker"),
                successor_id=str(successor) if successor is not None else None,
                source=str(row.get("source") or "frame"),
            )
        )
    return events
