"""Stable reason bits for the pre-trade hot path.

Bits are OR-combined. The allow path returns ``0``. Names are decoded only
when a caller asks for a decision record or a shadow log line.
"""

from __future__ import annotations

KILL = 1 << 0
STALE = 1 << 1
NON_FINITE = 1 << 2
MARKET_CLOSED = 1 << 3
HALTED = 1 << 4
MAX_QTY = 1 << 5
MAX_NOTIONAL = 1 << 6
COLLAR = 1 << 7
POSITION_QTY = 1 << 8
POSITION_NOTIONAL = 1 << 9
GROSS = 1 << 10
NET = 1 << 11
CONCENTRATION = 1 << 12
DUPLICATE = 1 << 13
ORDER_RATE = 1 << 14
MESSAGE_RATE = 1 << 15
REG_SHO = 1 << 16
PDT = 1 << 17
GFV = 1 << 18
DAILY_LOSS = 1 << 19
TRAILING_DD = 1 << 20
INTERNAL = 1 << 21
BUYING_POWER = 1 << 22
STATE_FULL = 1 << 23
UNKNOWN_SYMBOL = 1 << 24
NON_MONOTONIC = 1 << 25
MANUAL = 1 << 26

# Size, price, and exposure ceilings. Regulatory and throttle bits are separate.
HARD_LIMIT_MASK = (
    MAX_QTY | MAX_NOTIONAL | COLLAR | POSITION_QTY | POSITION_NOTIONAL | GROSS | NET | CONCENTRATION
)

# Cancels are risk-reducing. These bits still block them.
CANCEL_BLOCK_MASK = (
    KILL
    | STALE
    | NON_FINITE
    | MESSAGE_RATE
    | INTERNAL
    | STATE_FULL
    | UNKNOWN_SYMBOL
    | NON_MONOTONIC
)

KIND_ORDER = 1
KIND_CANCEL = 2
KIND_REPLACE = 3

REASON_ORDER: tuple[tuple[int, str], ...] = (
    (KILL, "kill_switch"),
    (STALE, "stale_data"),
    (NON_FINITE, "non_finite"),
    (MARKET_CLOSED, "market_closed"),
    (HALTED, "halted"),
    (MAX_QTY, "max_order_quantity"),
    (MAX_NOTIONAL, "max_order_notional"),
    (COLLAR, "price_collar"),
    (POSITION_QTY, "position_quantity"),
    (POSITION_NOTIONAL, "position_notional"),
    (GROSS, "gross_exposure"),
    (NET, "net_exposure"),
    (CONCENTRATION, "concentration"),
    (DUPLICATE, "duplicate_order"),
    (ORDER_RATE, "order_rate"),
    (MESSAGE_RATE, "message_rate"),
    (REG_SHO, "reg_sho"),
    (PDT, "pattern_day_trader"),
    (GFV, "good_faith_violation"),
    (DAILY_LOSS, "daily_loss"),
    (TRAILING_DD, "trailing_drawdown"),
    (INTERNAL, "internal_error"),
    (BUYING_POWER, "buying_power"),
    (STATE_FULL, "state_full"),
    (UNKNOWN_SYMBOL, "unknown_symbol"),
    (NON_MONOTONIC, "non_monotonic_time"),
    (MANUAL, "manual_kill"),
)


def reason_names(bits: int) -> tuple[str, ...]:
    """Decode bits in a fixed order. The order does not depend on check sequence."""
    return tuple(name for bit, name in REASON_ORDER if bits & bit)


def decision_allowed(bits: int, kind: int) -> bool:
    """Orders and replaces pass only with an empty bitset. Cancels use a smaller mask."""
    if kind == KIND_CANCEL:
        return (bits & CANCEL_BLOCK_MASK) == 0
    return bits == 0
