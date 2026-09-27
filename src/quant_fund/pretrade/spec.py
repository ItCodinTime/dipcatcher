"""Independent hard-limit specification.

The hot path in ``kernel.py`` inlines the same formulas. Property tests
OR these predicates in shuffled orders and require the kernel's hard-limit
bits to match. Hard limits are inclusive ceilings: a value equal to the
limit is inside it; a value strictly above it is a breach.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_fund.pretrade.codes import (
    COLLAR,
    CONCENTRATION,
    GROSS,
    MAX_NOTIONAL,
    MAX_QTY,
    NET,
    POSITION_NOTIONAL,
    POSITION_QTY,
)


@dataclass(slots=True)
class HardView:
    qty: float
    px: float
    side: int
    is_limit: int
    pos: float
    ref: float
    ref_ok: int
    hi: float
    lo: float
    gross: float
    net: float
    max_qty: float
    max_notional: float
    max_pos_qty: float
    max_pos_notional: float
    max_gross: float
    max_net: float
    name_limit: float


def notional_basis(px: float, ref: float, ref_ok: int) -> float:
    """Conservative order notional price: the larger of the order price and the mark."""
    if ref_ok and px < ref:
        return ref
    return px


def _abs(value: float) -> float:
    return value if value >= 0.0 else -value


def pred_max_qty(view: HardView) -> int:
    if view.qty > view.max_qty:
        return MAX_QTY
    return 0


def pred_max_notional(view: HardView) -> int:
    basis = notional_basis(view.px, view.ref, view.ref_ok)
    if view.qty * basis > view.max_notional:
        return MAX_NOTIONAL
    return 0


def pred_collar(view: HardView) -> int:
    if view.is_limit and view.ref_ok and (view.px > view.hi or view.px < view.lo):
        return COLLAR
    return 0


def _projected(view: HardView) -> tuple[float, float, float]:
    signed = view.qty if view.side > 0 else -view.qty
    new_pos = view.pos + signed
    new_n = new_pos * view.ref
    old_n = view.pos * view.ref
    return new_pos, new_n, old_n


def pred_position_qty(view: HardView) -> int:
    new_pos, _, _ = _projected(view)
    if new_pos > view.max_pos_qty or -new_pos > view.max_pos_qty:
        return POSITION_QTY
    return 0


def pred_position_notional(view: HardView) -> int:
    if not view.ref_ok:
        return 0
    _, new_n, _ = _projected(view)
    if new_n > view.max_pos_notional or -new_n > view.max_pos_notional:
        return POSITION_NOTIONAL
    return 0


def pred_gross(view: HardView) -> int:
    if not view.ref_ok:
        return 0
    _, new_n, old_n = _projected(view)
    gross_after = view.gross - _abs(old_n) + _abs(new_n)
    if gross_after > view.max_gross or gross_after < 0.0:
        return GROSS
    return 0


def pred_net(view: HardView) -> int:
    if not view.ref_ok:
        return 0
    _, new_n, old_n = _projected(view)
    net_after = view.net - old_n + new_n
    if net_after > view.max_net or -net_after > view.max_net:
        return NET
    return 0


def pred_concentration(view: HardView) -> int:
    if not view.ref_ok:
        return 0
    _, new_n, _ = _projected(view)
    if _abs(new_n) > view.name_limit:
        return CONCENTRATION
    return 0


HARD_PREDICATES = (
    pred_max_qty,
    pred_max_notional,
    pred_collar,
    pred_position_qty,
    pred_position_notional,
    pred_gross,
    pred_net,
    pred_concentration,
)


def hard_limit_bits(view: HardView, order: tuple[int, ...] | None = None) -> int:
    """OR hard-limit predicates. ``order`` permutes predicate indices."""
    bits = 0
    if order is None:
        for pred in HARD_PREDICATES:
            bits |= pred(view)
        return bits
    for index in order:
        bits |= HARD_PREDICATES[index](view)
    return bits
