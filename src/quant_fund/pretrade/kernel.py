"""Allocation-free check kernel.

The steady-state allow path writes only into preallocated rings and the
caller's result integer. It does not build strings, lists, dicts, or
decision records. CPython still materializes transient integers for
arithmetic; container growth is reserved for the audit trail, which runs
only when a latch transitions.
"""

from __future__ import annotations

from quant_fund.pretrade.codes import (
    BUYING_POWER,
    COLLAR,
    CONCENTRATION,
    DAILY_LOSS,
    DUPLICATE,
    GFV,
    GROSS,
    HALTED,
    INTERNAL,
    KILL,
    KIND_CANCEL,
    KIND_ORDER,
    KIND_REPLACE,
    MARKET_CLOSED,
    MAX_NOTIONAL,
    MAX_QTY,
    MESSAGE_RATE,
    NET,
    NON_FINITE,
    NON_MONOTONIC,
    ORDER_RATE,
    PDT,
    POSITION_NOTIONAL,
    POSITION_QTY,
    REG_SHO,
    STALE,
    STATE_FULL,
    TRAILING_DD,
)
from quant_fund.pretrade.state import HotBook, SymbolSlot

_EPS = 1e-9
_CASH_EPS = 1e-6


def _fingerprint(book: HotBook, symbol_id: int, side: int, qty: float, px: float) -> int:
    q = int(round(qty / book.qty_tick))
    p = int(round(px / book.price_tick))
    mixed = (symbol_id + 1) * 0x9E3779B1
    mixed ^= (side & 0xFFFF) * 0x85EBCA6B
    mixed ^= q * 0xC2B2AE35
    mixed ^= p * 0x27D4EB2F
    mixed &= 0x7FFFFFFFFFFFFFFF
    if mixed < 2:
        return 2
    return mixed


def _expire_ring(
    stamps: list[int], tail: int, live: int, cap: int, ts: int, window: int
) -> tuple[int, int]:
    guard = 0
    while live > 0 and ts - stamps[tail] > window and guard <= cap:
        tail += 1
        if tail == cap:
            tail = 0
        live -= 1
        guard += 1
    return tail, live


def _push_ring(stamps: list[int], head: int, live: int, cap: int, ts: int) -> tuple[int, int, int]:
    if live >= cap:
        return head, live, 0
    stamps[head] = ts
    head += 1
    if head == cap:
        head = 0
    return head, live + 1, 1


def _hash_remove(h_fp: list[int], h_ts: list[int], mask: int, fp: int) -> None:
    """Backward-shift deletion so probe chains stay intact without tombstones."""
    idx = fp & mask
    n = 0
    limit = mask + 1
    while n < limit:
        if h_fp[idx] == 0:
            return
        if h_fp[idx] == fp:
            break
        idx = (idx + 1) & mask
        n += 1
    else:
        return
    h_fp[idx] = 0
    h_ts[idx] = 0
    hole = idx
    j = idx
    while True:
        j = (j + 1) & mask
        sfp = h_fp[j]
        if sfp == 0:
            return
        home = sfp & mask
        if hole <= j:
            if hole < home <= j:
                continue
        elif hole < home or home <= j:
            continue
        h_fp[hole] = sfp
        h_ts[hole] = h_ts[j]
        h_fp[j] = 0
        h_ts[j] = 0
        hole = j


def _note_duplicate(book: HotBook, fp: int, ts: int) -> int:
    """Return DUPLICATE, STATE_FULL, or 0. Expired fingerprints are removed first."""
    window = book.dup_window
    mask = book.ring_mask
    r_ts = book.r_ts
    r_fp = book.r_fp
    h_fp = book.h_fp
    h_ts = book.h_ts
    tail = book.r_tail
    live = book.r_live
    while live > 0 and ts - r_ts[tail] > window:
        _hash_remove(h_fp, h_ts, mask, r_fp[tail])
        tail = (tail + 1) & mask
        live -= 1
    book.r_tail = tail
    book.r_live = live
    idx = fp & mask
    n = 0
    limit = mask + 1
    while n < limit:
        sfp = h_fp[idx]
        if sfp == 0:
            if live >= book.ring_cap:
                return STATE_FULL
            h_fp[idx] = fp
            h_ts[idx] = ts
            head = book.r_head
            r_fp[head] = fp
            r_ts[head] = ts
            book.r_head = (head + 1) & mask
            book.r_live = live + 1
            return 0
        if sfp == fp and ts - h_ts[idx] <= window:
            return DUPLICATE
        idx = (idx + 1) & mask
        n += 1
    return STATE_FULL


def _pdt_count(book: HotBook, session_id: int) -> int:
    lo = session_id - book.pdt_window + 1
    arr = book.dt_sess
    n = book.dt_n
    count = 0
    i = 0
    while i < n:
        session = arr[i]
        if lo <= session <= session_id:
            count += 1
        i += 1
    return count


def _mature(book: HotBook, ts: int) -> None:
    live = book.uc_live
    if live <= 0:
        return
    tail = book.uc_tail
    mask = book.uc_mask
    amounts = book.uc_amt
    stamps = book.uc_ts
    settled = book.settled
    while live > 0 and stamps[tail] <= ts:
        settled += amounts[tail]
        tail = (tail + 1) & mask
        live -= 1
    book.uc_tail = tail
    book.uc_live = live
    book.settled = settled


def account_message(book: HotBook, ts: int) -> int:
    """Count one message when the order cannot be priced. Fail closed on a full ring."""
    if ts < 0:
        return NON_FINITE
    bits = 0
    if book.last_ts != 0 and ts < book.last_ts:
        bits |= NON_MONOTONIC
        eff = book.last_ts
    else:
        eff = ts
        book.last_ts = ts
    m_tail, m_live = _expire_ring(
        book.mts, book.m_tail, book.m_live, book.ring_cap, eff, book.window_ns
    )
    book.m_tail = m_tail
    book.m_live = m_live
    if m_live >= book.max_msgs:
        bits |= MESSAGE_RATE
    else:
        head, live, ok = _push_ring(book.mts, book.m_head, m_live, book.ring_cap, eff)
        book.m_head = head
        book.m_live = live
        if not ok:
            bits |= STATE_FULL
    if book.killed and (book.kill_reason & INTERNAL):
        bits |= KILL
    return bits


def hot_check(
    book: HotBook,
    sym: SymbolSlot,
    *,
    side: int,
    qty: float,
    px: float,
    ts: int,
    session_id: int,
    is_limit: int,
    kind: int,
    symbol_id: int,
) -> int:
    """Run one check-set. Returns the reason bitset. ``0`` allows an order."""
    if (
        qty != qty
        or px != px
        or qty <= 0.0
        or px <= 0.0
        or (side != 1 and side != -1)
        or kind < KIND_ORDER
        or kind > KIND_REPLACE
        or ts < 0
        or session_id < 0
    ):
        return NON_FINITE

    bits = 0
    if book.last_ts != 0 and ts < book.last_ts:
        bits |= NON_MONOTONIC
        eff = book.last_ts
    else:
        eff = ts
        book.last_ts = ts

    _mature(book, eff)

    if (
        sym.ref_ok == 0
        or eff > sym.deadline
        or eff < sym.ref_ts
        or eff > book.mark_deadline
        or eff < book.mark_ts
    ):
        bits |= STALE
    nav = book.nav
    if nav != nav or not nav > 0.0:
        bits |= NON_FINITE

    if kind != KIND_CANCEL and (
        book.session_open == 0 or eff < book.open_ns or eff >= book.close_ns
    ):
        bits |= MARKET_CLOSED
    if sym.halted and kind != KIND_CANCEL:
        bits |= HALTED

    ref_ok = sym.ref_ok
    ref = sym.ref
    pos = sym.pos
    if kind != KIND_CANCEL:
        if qty > book.max_qty:
            bits |= MAX_QTY
        basis = ref if ref_ok and px < ref else px
        if qty * basis > book.max_notional:
            bits |= MAX_NOTIONAL
        if is_limit and ref_ok and (px > sym.hi or px < sym.lo):
            bits |= COLLAR
        signed = qty if side > 0 else -qty
        new_pos = pos + signed
        if new_pos > book.max_pos_qty or -new_pos > book.max_pos_qty:
            bits |= POSITION_QTY
        if ref_ok:
            old_n = pos * ref
            new_n = new_pos * ref
            if new_n > book.max_pos_notional or -new_n > book.max_pos_notional:
                bits |= POSITION_NOTIONAL
            abs_old = old_n if old_n >= 0.0 else -old_n
            abs_new = new_n if new_n >= 0.0 else -new_n
            gross_after = book.gross - abs_old + abs_new
            net_after = book.net - old_n + new_n
            if gross_after > book.max_gross or gross_after < 0.0:
                bits |= GROSS
            if net_after > book.max_net or -net_after > book.max_net:
                bits |= NET
            if abs_new > book.name_limit:
                bits |= CONCENTRATION
        else:
            new_pos = pos
        # Reg SHO. A cash account cannot open or increase a short. Margin
        # shorts need a locate, and Rule 201 (sho flag) requires a limit at
        # or above the national best bid.
        opening_short = side < 0 and new_pos < -_EPS and new_pos < pos - _EPS
        if opening_short and (
            book.cash_account
            or not sym.locate
            or (sym.sho and (not is_limit or sym.bid <= 0.0 or px + _EPS < sym.bid))
        ):
            bits |= REG_SHO
        if book.restrict_settled and side > 0 and qty * px > book.settled + _CASH_EPS:
            bits |= BUYING_POWER
        if book.pdt_tight and session_id > 0:
            before = pos
            after = new_pos
            closes = (before > _EPS or before < -_EPS) and (
                (after <= _EPS and after >= -_EPS) or before * after < 0.0
            )
            if (
                closes
                and sym.opened_session == session_id
                and _pdt_count(book, session_id) >= book.pdt_max
            ):
                bits |= PDT
        if side < 0 and pos > _EPS:
            left = qty if qty < pos else pos
            nlot = sym.nlot
            lq = sym.lq
            ls = sym.ls
            lu = sym.lu
            k = 0
            while k < nlot and left > _EPS:
                if lu[k] and eff < ls[k] and lq[k] > _EPS:
                    bits |= GFV
                    break
                take = lq[k] if lq[k] < left else left
                left -= take
                k += 1
        if side > 0 and sym.nlot >= book.max_lots:
            bits |= STATE_FULL
        if side < 0 and book.uc_live >= book.uc_cap:
            bits |= STATE_FULL

    if kind != KIND_CANCEL:
        fp = _fingerprint(book, symbol_id, side, qty, px)
        bits |= _note_duplicate(book, fp, eff)

    o_tail, o_live = _expire_ring(
        book.ots, book.o_tail, book.o_live, book.ring_cap, eff, book.window_ns
    )
    book.o_tail = o_tail
    book.o_live = o_live
    if kind != KIND_CANCEL:
        if o_live >= book.max_orders:
            bits |= ORDER_RATE
        else:
            head, live, ok = _push_ring(book.ots, book.o_head, o_live, book.ring_cap, eff)
            book.o_head = head
            book.o_live = live
            if not ok:
                bits |= STATE_FULL

    m_tail, m_live = _expire_ring(
        book.mts, book.m_tail, book.m_live, book.ring_cap, eff, book.window_ns
    )
    book.m_tail = m_tail
    book.m_live = m_live
    if m_live >= book.max_msgs:
        bits |= MESSAGE_RATE
    else:
        head, live, ok = _push_ring(book.mts, book.m_head, m_live, book.ring_cap, eff)
        book.m_head = head
        book.m_live = live
        if not ok:
            bits |= STATE_FULL

    tripped = 0
    if nav == nav and nav > 0.0:
        if nav <= book.daily_floor:
            bits |= DAILY_LOSS
            tripped |= DAILY_LOSS
        if nav <= book.dd_floor:
            bits |= TRAILING_DD
            tripped |= TRAILING_DD
    if tripped and book.killed == 0:
        book.killed = 1
        book.kill_reason = tripped
        book.pending_trip = tripped
    elif tripped:
        book.kill_reason |= tripped

    if book.killed and (kind != KIND_CANCEL or book.kill_reason & INTERNAL):
        bits |= KILL

    return bits
